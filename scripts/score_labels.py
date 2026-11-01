"""Score 0/1 with a fixed wrapper; test scoring requires a frozen configuration."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

WRAPPER = "<think>\n\n</think>\n\n"
ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-path', type=Path, required=True)
    ap.add_argument('--output', type=Path, default=ROOT/'results/label-scoring')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--split', choices=['validation', 'test'], default='validation')
    ap.add_argument('--frozen-config', type=Path)
    ap.add_argument('--adapters', nargs='+', default=['pilot-001','pilot-002'],
                    help='Names of adapter folders under adapters/, or base for no adapter')
    args = ap.parse_args()
    frozen = None
    if args.split == 'test':
        assert args.frozen_config and not args.limit, 'Test requires frozen config and full split'
        frozen = json.loads(args.frozen_config.read_text())
        assert args.adapters == [frozen['adapter']], 'Only the frozen adapter may be tested'
        assert args.model_path.name == frozen['model_revision']
        assert frozen['scoring_protocol'] == 'fixed_empty_think_label_logits_v1'
        assert frozen['wrapper'] == WRAPPER and frozen['context'] == 1024
        assert frozen['comparison'] == 'score >= threshold' and 0 <= frozen['threshold'] <= 1
        if frozen['adapter'] == 'base':
            assert frozen['adapter_sha256'] is None
        else:
            assert sha(ROOT/'adapters'/frozen['adapter']/'adapters.safetensors') == frozen['adapter_sha256']
        assert not args.output.exists(), 'Refuse overwriting an existing test run'
    assert len(set(args.adapters)) == len(args.adapters)
    assert all(Path(name).name == name and name not in ('.','..') for name in args.adapters)
    args.output.mkdir(parents=True, exist_ok=True)
    prep = json.loads((ROOT/'data/manifests/preparation_report.json').read_text())
    if frozen:
        assert frozen['system_prompt'] == prep['system_prompt']
        (args.output/'frozen_config.json').write_bytes(args.frozen_config.read_bytes())
    filename = 'test.jsonl' if args.split == 'test' else 'valid.jsonl'
    path = ROOT/'data/evaluation'/filename
    assert sha(path) == prep['sha256']['evaluation/'+filename]
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    assert len(rows) == len({r['id'] for r in rows}) == (3000 if args.split == 'test' else 1000)
    assert sum(r['label'] for r in rows) == (239 if args.split == 'test' else 79)
    if args.limit:
        rows = rows[:args.limit]
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.utils import load_tokenizer
    tok = load_tokenizer(args.model_path)
    labels = [tok.encode(s, add_special_tokens=False) for s in ('0','1')]
    assert all(len(t)==1 for t in labels)
    label_ids = [x[0] for x in labels]
    def encode(text):
        prompt = tok.apply_chat_template([
            {'role':'system','content':prep['system_prompt']},
            {'role':'user','content':'Comment:\n'+text}],
            tokenize=False, add_generation_prompt=True) + WRAPPER
        return prompt, tok.encode(prompt, add_special_tokens=False)
    prepared=[]
    for row in rows:
        prompt, ids = encode(row['text']); truncated=False
        text_tokens=tok.encode(row['text'], add_special_tokens=False)
        while len(ids)>1023:
            truncated=True
            text_tokens=text_tokens[:max(0,len(text_tokens)-(len(ids)-1023))]
            prompt,ids=encode(tok.decode(text_tokens))
        for s,i in zip(('0','1'),label_ids):
            assert tok.encode(prompt+s, add_special_tokens=False) == ids+[i]
        prepared.append((row,ids,truncated))
    manifest=dict(protocol='fixed_empty_think_label_logits_v1', wrapper=WRAPPER,
        system_prompt=prep['system_prompt'], label_ids=label_ids, context=1024,
        candidate_definition='Common rendered prompt + empty wrapper + one label token; no EOS scoring',
        scoring='float32 softmax over last-position logits for 0 and 1; uncalibrated conditional preference',
        split=args.split, rows=len(rows), positives=sum(r['label'] for r in rows),
        dataset_revision=prep['revision'],
        model_revision=args.model_path.name, truncated=sum(x[2] for x in prepared),
        packages={p:importlib.metadata.version(p) for p in ['mlx','mlx-lm','transformers']},
        script_sha256=sha(Path(__file__)), models={})
    manifest['test_sha256' if frozen else 'validation_sha256'] = sha(path)
    if frozen:
        manifest['frozen_config_sha256'] = sha(args.frozen_config)
        manifest['frozen_threshold'] = frozen['threshold']
    for pilot in args.adapters:
        adapter=ROOT/'adapters'/pilot
        if pilot != 'base':
            config=json.loads((adapter/'adapter_config.json').read_text())
            revision=config.get('base_model_revision',config.get('base_model_commit_hash'))
            assert revision==args.model_path.name
            assert sha(adapter/'chat_template.jinja') == sha(args.model_path/'chat_template.jinja')
        print('Loading '+pilot,flush=True)
        model,_=load(str(args.model_path),adapter_path=None if pilot == 'base' else str(adapter)); model.eval()
        def logits_for(ids):
            return model(mx.array([ids]))[0,-1,:].astype(mx.float32)
        # Independent checks: deterministic evaluation and causal candidate scoring.
        ids=prepared[0][1]
        logits=logits_for(ids); mx.eval(logits)
        repeat=logits_for(ids); mx.eval(repeat)
        assert mx.max(mx.abs(logits-repeat)).item()<1e-5
        lps=[]; causal_diffs=[]
        for token in label_ids:
            candidate=model(mx.array([ids+[token]]))[0,-2,:].astype(mx.float32)
            lp=candidate[token]-mx.logsumexp(candidate)
            lps.append(lp.item())
            causal_diffs.append(abs(lp.item()-(logits[token]-mx.logsumexp(logits)).item()))
        score_check=mx.softmax(logits[mx.array(label_ids)])[1].item()
        full_candidate_score=mx.softmax(mx.array(lps))[1].item()
        # BF16 kernels can vary slightly with sequence length; record actual difference.
        assert abs(score_check-full_candidate_score)<0.02
        mx.synchronize();mx.reset_peak_memory()
        start_all=time.perf_counter()
        with (args.output/f'{pilot}.scores.jsonl').open('w') as f:
            for i,(row,ids,truncated) in enumerate(prepared):
                mx.synchronize(); start=time.perf_counter()
                logits=logits_for(ids)
                two=logits[mx.array(label_ids)]
                probs=mx.softmax(two)
                lps=two-mx.logsumexp(logits)
                mx.eval(probs,lps);mx.synchronize()
                latency=(time.perf_counter()-start)*1000
                score=probs[1].item()
                assert 0<=score<=1 and abs(probs.sum().item()-1)<1e-6
                record=dict(id=row['id'],label=row['label'],score=score,
                    log_probability_0=lps[0].item(),log_probability_1=lps[1].item(),
                    prompt_tokens=len(ids),truncated=truncated,latency_ms=latency)
                f.write(json.dumps(record,allow_nan=False)+'\n');f.flush()
                if (i+1)%100==0: print(f'{pilot}: {i+1}/{len(rows)}',flush=True)
        manifest['models'][pilot]=dict(adapter_sha256=None if pilot == 'base' else sha(adapter/'adapters.safetensors'),
            scores_sha256=sha(args.output/f'{pilot}.scores.jsonl'),
            elapsed_seconds=time.perf_counter()-start_all,mlx_peak_allocated_gib=mx.get_peak_memory()/1024**3,
            deterministic_check=True,candidate_score_difference=abs(score_check-full_candidate_score),
            candidate_logprob_differences=causal_diffs)
        (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        del model,logits,repeat,candidate;gc.collect();mx.clear_cache()
    print('Frozen test adapter scored; no threshold fitting.' if frozen else
          'Selected adapters scored. Test set not read.',flush=True)


if __name__=='__main__':
    main()
