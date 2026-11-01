"""Compare main5000 with cached pilot-002 validation scores at fixed 0.5."""
import csv
import json
from pathlib import Path
from sklearn.metrics import average_precision_score
from score_labels import sha
from select_threshold import metrics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/validation-main5000-001'

def main():
    folders = {'pilot-002': ROOT/'results/label-scoring', 'main5000-001': OUT}
    manifests = {n: json.loads((p/'manifest.json').read_text()) for n,p in folders.items()}
    keys = ['protocol','wrapper','system_prompt','label_ids','context','candidate_definition',
            'scoring','split','rows','positives','validation_sha256','dataset_revision','model_revision','packages']
    a,b = manifests.values()
    assert all(a[k] == b[k] for k in keys)
    assert a['rows'] == 1000 and a['positives'] == 79
    records = {}
    summary = {}
    for name,folder in folders.items():
        path = folder/f'{name}.scores.jsonl'
        assert sha(path) == manifests[name]['models'][name]['scores_sha256']
        assert sha(ROOT/'adapters'/name/'adapters.safetensors') == manifests[name]['models'][name]['adapter_sha256']
        records[name] = [json.loads(s) for s in path.read_text().splitlines()]
        rows = records[name]
        y = [r['label'] for r in rows]
        scores = [r['score'] for r in rows]
        summary[name] = dict(threshold=0.5, **metrics(y,[s>=0.5 for s in scores]),
            average_precision=float(average_precision_score(y,scores)))
    old,new = records.values()
    assert [(r['id'],r['label']) for r in old] == [(r['id'],r['label']) for r in new]
    changes = []
    for a,b in zip(old,new):
        pa,pb = int(a['score']>=0.5),int(b['score']>=0.5)
        if pa != pb:
            changes.append(dict(id=a['id'], label=a['label'], pilot_prediction=pa,
                main_prediction=pb, pilot_score=a['score'], main_score=b['score'],
                change='corrected' if pb==a['label'] else 'regressed'))
    (OUT/'comparison_metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (OUT/'paired_changes.jsonl').open('w') as f:
        for row in changes: f.write(json.dumps(row)+'\n')
    lines = ['# Main 5,000-example model vs pilot-002', '',
        'All 1,000 validation comments (79 toxic); fixed label-scoring protocol and threshold 0.5. No threshold tuning or test evaluation.', '',
        '| Model | Precision | Recall | Toxic F1 | Average precision | FP | FN |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for n,m in summary.items():
        lines.append(f"| {n} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['average_precision']:.3f} | {m['fp']} | {m['fn']} |")
    lines += ['', f"Paired prediction changes: {sum(r['change']=='corrected' for r in changes)} corrected and {sum(r['change']=='regressed' for r in changes)} regressed.", '',
        '## Verification and limits', '',
        '- Cached pilot scores reused only after checking protocol, model revision, package versions, validation hash, score hashes, adapter hashes and exact row/label order.',
        '- Main model scored with deterministic-forward and candidate-likelihood checks; see manifest for recorded differences and truncation count.',
        '- The larger dataset retains all pilot-002 examples and the same 25% toxic mix, but uses 5,000 rather than 2,000 optimizer steps. This is not a compute-matched experiment.',
        '- Validation has already informed development decisions. These are development results, not final-test estimates or statistical-significance claims.',
        '- Scores are conditional label preferences, not calibrated harm probabilities. Test remains unevaluated.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2))

if __name__ == '__main__':
    main()
