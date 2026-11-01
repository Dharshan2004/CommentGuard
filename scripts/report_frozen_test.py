"""Report the frozen test operating point without fitting or searching thresholds."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score
from select_threshold import metrics


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scores', type=Path, required=True)
    args = ap.parse_args()
    folder = args.scores
    manifest = json.loads((folder/'manifest.json').read_text())
    config_path = folder/'frozen_config.json'
    config = json.loads(config_path.read_text())
    name = config['adapter']
    assert manifest['split'] == 'test' and manifest['rows'] == 3000
    assert manifest['frozen_config_sha256'] == sha(config_path)
    assert manifest['frozen_threshold'] == config['threshold']
    assert list(manifest['models']) == [name]
    assert manifest['models'][name]['adapter_sha256'] == config['adapter_sha256']
    root = Path(__file__).resolve().parents[1]
    validation = json.loads((root/config['validation_manifest']).read_text())
    for key in ['protocol', 'wrapper', 'system_prompt', 'label_ids', 'context',
                'candidate_definition', 'scoring', 'dataset_revision', 'model_revision', 'packages']:
        assert manifest[key] == validation[key], f'Protocol mismatch: {key}'
    path = folder/f'{name}.scores.jsonl'
    assert sha(path) == manifest['models'][name]['scores_sha256']
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    assert len(rows) == len({r['id'] for r in rows}) == 3000
    truth = [json.loads(s) for s in (root/'data/evaluation/test.jsonl').read_text().splitlines()]
    assert sha(root/'data/evaluation/test.jsonl') == manifest['test_sha256']
    assert [(r['id'], r['label']) for r in rows] == [(r['id'], r['label']) for r in truth]
    y = np.array([r['label'] for r in rows])
    scores = np.array([r['score'] for r in rows])
    assert y.sum() == 239 and np.isfinite(scores).all()
    assert ((scores >= 0) & (scores <= 1)).all()
    predictions = scores >= config['threshold']
    result = dict(adapter=name, threshold=config['threshold'], split='test', n=len(rows),
                  positives=int(y.sum()), **metrics(y, predictions),
                  accuracy=float((predictions == y).mean()),
                  average_precision=float(average_precision_score(y, scores)),
                  truncated=manifest['truncated'],
                  frozen_config_sha256=sha(config_path), scores_sha256=sha(path),
                  threshold_fitted_on_test=False)
    (folder/'metrics.json').write_text(json.dumps(result, indent=2)+'\n')
    with (folder/'predictions.jsonl').open('w') as f:
        for row, pred in zip(rows, predictions):
            f.write(json.dumps(dict(id=row['id'], label=row['label'],
                                   score=row['score'], prediction=int(pred)))+'\n')
    report = f'''# Frozen CommentGuard test evaluation

Adapter: `{name}`. Threshold: `{config['threshold']}` (score >= threshold).
Evaluated all {len(rows):,} held-out examples, including {int(y.sum())} toxic comments.
The model, prompt, scoring protocol and threshold were frozen before this evaluation.

| Metric | Test result |
|---|---:|
| Toxic precision | {result['precision']:.2%} |
| Toxic recall | {result['recall']:.2%} |
| Toxic F1 | {result['f1']:.4f} |
| Average precision | {result['average_precision']:.4f} |
| Macro F1 | {result['macro_f1']:.4f} |
| Accuracy | {result['accuracy']:.2%} |
| True positives | {result['tp']} |
| False positives | {result['fp']} |
| False negatives | {result['fn']} |
| True negatives | {result['tn']} |
| Truncated comments | {result['truncated']} |

No threshold fitting or candidate selection was performed on test. Scores are
uncalibrated two-label preferences, not calibrated toxicity probabilities.
The 60% recall target was used during validation selection; it does not guarantee
60% recall on new data. Test performance describes this held-out Civil Comments
sample, not live reel comments or deployment safety. This run evaluates only the
frozen selected adapter; it does not establish test-set superiority over baselines.

The report verifies the score hash, frozen-config hash, adapter identity, exact
test row/label order, and matching validation/test inference metadata. The saved
configuration is the pre-test snapshot; its historical status remains unchanged.
'''
    (folder/'REPORT.md').write_text(report)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
