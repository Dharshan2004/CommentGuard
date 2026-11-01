"""Validation-only evaluation: pilot-trained logistic regression, base Qwen, MLX adapter.

Run with an environment containing mlx-lm and scikit-learn. No test-set access.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import re
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline


def read(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rows):
    y = np.array([r['label'] for r in rows])
    p = np.array([r['prediction'] for r in rows])
    cm = [[int(np.sum((y == actual) & (p == predicted))) for predicted in (0, 1, -1)] for actual in (0, 1)]
    return dict(n=len(rows), toxic_support=int(y.sum()), accuracy=accuracy_score(y, p),
                toxic_precision=precision_score(y, p, labels=[1], average='macro', zero_division=0),
                toxic_recall=recall_score(y, p, labels=[1], average='macro', zero_division=0),
                toxic_f1=f1_score(y, p, labels=[1], average='macro', zero_division=0),
                macro_f1=f1_score(y, p, labels=[0, 1], average='macro', zero_division=0),
                average_precision=(average_precision_score(y, [r['score'] for r in rows])
                                   if all(r['score'] is not None for r in rows) else None),
                invalid_outputs=int(np.sum(p == -1)), confusion_matrix=cm,
                raw_format_violations=sum(r.get('raw_format_violation', False) for r in rows),
                confusion_columns=['non-toxic', 'toxic', 'invalid'],
                latency_ms_p50=float(np.percentile([r['latency_ms'] for r in rows], 50)),
                latency_ms_p95=float(np.percentile([r['latency_ms'] for r in rows], 95)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter', type=Path, default=Path('adapters/pilot-001'))
    parser.add_argument('--train-name', default='pilot_train', choices=['pilot_train', 'pilot002_mix25_train'])
    parser.add_argument('--reuse-base', type=Path, help='Prior full evaluation directory; verifies protocol before reusing base predictions')
    parser.add_argument('--model-path', type=Path, required=True, help='Cached exact base snapshot')
    parser.add_argument('--output', type=Path, default=Path('results/validation-pilot-001'))
    parser.add_argument('--limit', type=int, default=0, help='Smoke check only; 0 evaluates all validation rows')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.adapter = args.adapter.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    train_path = root/f'data/evaluation/{args.train_name}.jsonl'
    valid_path = root/'data/evaluation/valid.jsonl'
    train, valid = read(train_path), read(valid_path)
    assert len(train) == 2000 and len(valid) == 1000
    assert len({r['id'] for r in valid}) == 1000
    assert not {r['id'] for r in train} & {r['id'] for r in valid}
    prep = json.loads((root/'data/manifests/preparation_report.json').read_text())
    train_prep = (json.loads((root/'data/manifests/pilot002_mix25_report.json').read_text())
                  if args.train_name == 'pilot002_mix25_train' else prep)
    assert digest(train_path) == train_prep['sha256']['evaluation/'+train_path.name]
    assert digest(valid_path) == prep['sha256']['evaluation/'+valid_path.name]
    if args.limit:
        valid = valid[:args.limit]
    adapter_config = json.loads((args.adapter/'adapter_config.json').read_text())
    assert args.model_path.name == adapter_config['base_model_revision'], 'Use pinned base revision'
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.utils import load_tokenizer

    tokenizer = load_tokenizer(args.model_path)
    system = prep['system_prompt']
    def prompt_for(text):
        return tokenizer.apply_chat_template([
            {'role':'system', 'content':system},
            {'role':'user', 'content':'Comment:\n'+text}],
            tokenize=True, add_generation_prompt=True)

    def prepare(row):
        text = row['text']
        original = prompt_for(text)
        ids = original
        # Trim comment only, retaining system and generation prefix; reserve 8 output tokens.
        if len(ids) > 1016:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            while len(ids) > 1016:
                tokens = tokens[:max(0, len(tokens)-(len(ids)-1016))]
                text = tokenizer.decode(tokens)
                ids = prompt_for(text)
        return dict(row, model_text=text, prompt_ids=ids, truncated=len(original)>1016)
    train = [prepare(r) for r in train]
    valid = [prepare(r) for r in valid]
    label_ids = [tokenizer.encode(s, add_special_tokens=False) for s in ('0','1')]
    assert all(len(x)==1 for x in label_ids)
    label_ids = [x[0] for x in label_ids]
    for row in valid:
        prefix = tokenizer.decode(row['prompt_ids'])
        for label, token in zip(('0','1'), label_ids):
            assert tokenizer.encode(prefix+label, add_special_tokens=False) == row['prompt_ids']+[token]

    manifest = dict(split='validation', train_n=len(train), validation_n=len(valid),
                    dataset_revision=prep['revision'], model_revision=args.model_path.name,
                    train_name=args.train_name, train_positive=sum(r['label'] for r in train),
                    model=adapter_config['base_model_name_or_path'],
                    adapter_sha256=digest(args.adapter/'adapters.safetensors'),
                    train_sha256=digest(train_path), validation_sha256=digest(valid_path),
                    context=1024, max_output_tokens=8, decoding='greedy; fresh cache per example; no tools',
                    score='Logistic predict_proba only; no LLM AP because optional generated wrappers change label position',
                    prediction='Greedy text; strip whitespace and one leading EMPTY think block only; accept exactly 0/1 else -1. Raw format violations also reported.',
                    truncation='Trim comment token suffix only to fit 1016 prompt tokens; shared by all models',
                    truncated_train=sum(r['truncated'] for r in train),
                    truncated_valid=sum(r['truncated'] for r in valid),
                    logistic=dict(ngram_range=[1,2], min_df=2, max_features=50000,
                                  sublinear_tf=True, C=1.0, class_weight='balanced', threshold=0.5,
                                  selection='Fixed before validation; no tuning'),
                    packages={p:importlib.metadata.version(p) for p in ['mlx','mlx-lm','scikit-learn','transformers']})
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    summaries = {}
    baseline = make_pipeline(TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=50000, sublinear_tf=True),
                             LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42))
    start = time.perf_counter()
    baseline.fit([r['model_text'] for r in train], [r['label'] for r in train])
    fit_seconds = time.perf_counter()-start
    rows=[]
    for r in valid:
        start=time.perf_counter(); score=float(baseline.predict_proba([r['model_text']])[0,1])
        rows.append(dict(id=r['id'],label=r['label'],prediction=int(score>=0.5),score=score,
                         latency_ms=(time.perf_counter()-start)*1000))
    def save(name, rows):
        assert [r['id'] for r in rows] == [r['id'] for r in valid]
        with (args.output/f'{name}.predictions.jsonl').open('w') as f:
            for row in rows: f.write(json.dumps(row)+'\n')
        summaries[name] = summarize(rows)
        (args.output/'metrics.json').write_text(json.dumps(summaries,indent=2)+'\n')
        print(name, json.dumps(summaries[name]), flush=True)
    save('logistic_regression',rows)
    summaries['logistic_regression']['fit_seconds']=fit_seconds
    for name, adapter in [('base',None),('pilot',str(args.adapter))]:
        if name == 'base' and args.reuse_base:
            old = json.loads((args.reuse_base/'manifest.json').read_text())
            for key in ['validation_sha256','model_revision','context','max_output_tokens','decoding','prediction','truncation','packages']:
                assert old[key] == manifest[key], f'Cannot reuse base: {key} changed'
            rows = read(args.reuse_base/'base.predictions.jsonl')
            assert [r['label'] for r in rows] == [r['label'] for r in valid]
            save(name, rows)
            manifest['base_reused_from'] = str(args.reuse_base)
            (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            continue
        print(f'Loading {name}',flush=True)
        model, tok = load(str(args.model_path), adapter_path=adapter)
        model.eval()
        assert tok.apply_chat_template([{'role':'system','content':system},{'role':'user','content':'Comment:\nwarmup'}],tokenize=True,add_generation_prompt=True) == prompt_for('warmup')
        sampler=make_sampler(temp=0)
        list(stream_generate(model,tok,prompt_for('warmup'),max_tokens=8,sampler=sampler))
        mx.synchronize(); mx.reset_peak_memory()
        rows=[]
        with (args.output/f'{name}.progress.jsonl').open('w') as progress:
            for i,r in enumerate(valid):
                mx.synchronize(); start=time.perf_counter(); pieces=[]
                for response in stream_generate(model,tok,r['prompt_ids'],max_tokens=8,sampler=sampler):
                    pieces.append(response.text)
                mx.synchronize(); latency=(time.perf_counter()-start)*1000
                output=''.join(pieces).strip()
                normalized=re.sub(r'^<think>\s*</think>\s*', '', output).strip()
                row=dict(id=r['id'],label=r['label'],prediction=int(normalized) if normalized in ('0','1') else -1,
                         score=None,output=output,raw_format_violation=output not in ('0','1'),
                         latency_ms=latency,prompt_tokens=len(r['prompt_ids']))
                rows.append(row); progress.write(json.dumps(row)+'\n'); progress.flush()
                if (i+1)%50==0: print(f'{name}: {i+1}/{len(valid)}',flush=True)
        peak=mx.get_peak_memory()/1024**3
        save(name,rows); summaries[name]['mlx_peak_allocated_gib']=peak
        (args.output/'metrics.json').write_text(json.dumps(summaries,indent=2)+'\n')
        del model, tok; gc.collect(); mx.clear_cache()
    print('Evaluation complete. Test set was not read.',flush=True)


if __name__ == '__main__':
    main()
