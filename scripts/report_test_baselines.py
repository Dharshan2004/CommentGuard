"""Compare frozen test operating points and fixed 0.5; never select on test."""
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score
from select_threshold import metrics

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    sources = [
        ('fine_tuned','test-main5000-step4000','main5000-step4000.scores.jsonl'),
        ('base','test-base-labels','base.scores.jsonl'),
        ('logistic','test-logistic-main5000','logistic.scores.jsonl')]
    results = {}; reference = None; llm_manifest = None
    for name, folder, filename in sources:
        p = ROOT/'results'/folder
        manifest = json.loads((p/'manifest.json').read_text())
        config = json.loads((p/'frozen_config.json').read_text())
        assert sha(p/'frozen_config.json') == manifest['frozen_config_sha256']
        assert manifest['split'] == 'test' and manifest['rows'] == 3000
        assert manifest['test_sha256'] == sha(ROOT/'data/evaluation/test.jsonl')
        if name != 'logistic':
            record = manifest['models'][config['adapter']]
            assert record['adapter_sha256'] == config['adapter_sha256']
            assert manifest['frozen_threshold'] == config['threshold']
            if llm_manifest is None: llm_manifest = manifest
            for key in ['protocol','wrapper','system_prompt','label_ids','context',
                        'candidate_definition','scoring','dataset_revision','model_revision','packages']:
                assert manifest[key] == llm_manifest[key], key
        else:
            record = manifest
            assert sha(p/'model.joblib') == config['model_sha256']
            assert manifest['threshold'] == config['threshold']
        assert sha(p/filename) == record['scores_sha256']
        rows = [json.loads(s) for s in (p/filename).read_text().splitlines()]
        keys = [(r['id'],r['label']) for r in rows]
        if reference is None: reference = keys
        assert keys == reference and len(set(i for i,_ in keys)) == 3000
        y = np.array([r['label'] for r in rows]);scores = np.array([r['score'] for r in rows])
        assert y.sum() == 239 and np.isfinite(scores).all() and ((scores>=0)&(scores<=1)).all()
        results[name] = dict(threshold=config['threshold'],
            validation_selected=metrics(y,scores>=config['threshold']),
            fixed_0_5=metrics(y,scores>=0.5),average_precision=float(average_precision_score(y,scores)),
            latency_ms_p50=float(np.median([r['latency_ms'] for r in rows])),
            latency_ms_p95=float(np.percentile([r['latency_ms'] for r in rows],95)),
            elapsed_seconds=record['elapsed_seconds'],
            manifest_sha256=sha(p/'manifest.json'),scores_sha256=sha(p/filename))
    out = ROOT/'results/test-baseline-comparison'
    (out/'metrics.json').write_text(json.dumps(results,indent=2)+'\n')
    lines = ['# Held-out test baseline comparison','',
        'All models evaluated on the same 3,000 comments (239 toxic). Each primary threshold was fitted on validation, never test. Fine-tuned predictions reuse the previously frozen test run; they were not regenerated or retuned.','',
        '| Model | Rule | Threshold | Precision | Recall | F1 | AP | FP | FN |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for rule in ['validation_selected','fixed_0_5']:
        for name,r in results.items():
            m = r[rule];t = r['threshold'] if rule=='validation_selected' else .5
            lines.append(f"| {name} | {rule} | {t:.9f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {r['average_precision']:.4f} | {m['fp']} | {m['fn']} |")
    ft = results['fine_tuned']['validation_selected']
    base = results['base']['validation_selected']
    lr = results['logistic']['validation_selected']
    lines += ['', '## Did fine-tuning help?', '',
        f"Yes, under these fixed evaluation procedures. Fine-tuned F1 improves by {ft['f1']-base['f1']:.4f} over base Qwen and by {ft['f1']-lr['f1']:.4f} over this logistic baseline. Its AP is also higher, supporting an improvement in ranking rather than merely a favorable test threshold.", '',
        f"Against logistic regression, the fine-tuned model detects {ft['tp']-lr['tp']} more toxic comments and produces {lr['fp']-ft['fp']} fewer false positives. Against base Qwen it produces {base['fp']-ft['fp']} fewer false positives but misses {ft['fn']-base['fn']} more toxic comments. These are different operating points, not exactly recall-matched test results.", '',
        'This provides evidence of improvement over the two measured baselines. It does not establish state-of-the-art performance, superiority over a tuned classical classifier, or the best prompting achievable with base Qwen. The base has a severe toxic-score bias under the fixed-wrapper protocol; its high selected threshold is not a calibrated 99.6% toxicity probability. Fine-tuning helps this protocol, but this is not a comprehensive base-model capability benchmark.', '',
        'The selected model still misses 94 of 239 toxic comments. Retain that limitation alongside the gains, and keep the pre-test selected threshold unchanged.']
    lines += ['', '## Inference timing', '',
        '| Model | Median ms/comment | P95 ms/comment | Scoring elapsed seconds |',
        '|---|---:|---:|---:|']
    for name,r in results.items():
        lines.append(f"| {name} | {r['latency_ms_p50']:.3f} | {r['latency_ms_p95']:.3f} | {r['elapsed_seconds']:.3f} |")
    lines += ['',
        'Timing is sequential local inference on this Mac, not a controlled deployment benchmark. LLM timing includes synchronized forward passes; logistic timing includes per-comment transform/predict. Model loading and initial prompt preparation are excluded. No throughput/SLA claim is implied.', '',
        '## Protocol and limitations', '',
        'Base Qwen uses the exact model revision, prompt, fixed empty-thinking wrapper and two-label softmax as the adapter. Logistic regression uses the existing fixed hyperparameters on the full 5,000-example training pool; vocabulary and IDF are fitted on training only. It receives full text. The selected adapter is checkpoint 4,000, so this is equal training pool, not equal exposure or compute.', '',
        'Both baseline thresholds use the existing maximum-precision subject to validation recall >=60% objective. The fixed-0.5 table is a secondary matched numerical threshold comparison; it is not a calibrated probability comparison. No baseline hyperparameter sweep or test threshold fitting occurred.', '',
        'The fine-tuned test result was already known when baseline evaluation was requested. PROTOCOL.md was recorded before baseline test scoring. This is a subsequent comparison, not a jointly preregistered blind experiment. Results apply to this held-out Civil Comments sample and do not establish live moderation performance. Do not switch thresholds or retrain against these test outcomes without a new development/evaluation plan.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
