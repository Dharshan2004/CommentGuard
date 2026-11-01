"""Render measured threshold-selection results and precision-recall figure."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

root=Path(__file__).resolve().parents[1]
out=root/'results/label-scoring'
results=json.loads((out/'threshold_results.json').read_text())
manifest=json.loads((out/'manifest.json').read_text())
lines=['# Label scoring and threshold selection — validation', '',
 'Scored both saved pilots on all 1,000 validation comments (79 toxic), using a fixed empty thinking wrapper and the relative logits of label tokens 0 and 1. No retraining or test evaluation occurred.', '',
 '## Classification comparison', '',
 '| Pilot | Inference rule | Precision | Recall | Toxic F1 | FP | FN |',
 '|---|---|---:|---:|---:|---:|---:|']
fig,ax=plt.subplots(figsize=(7.5,5.5))
colors=['#3b6fb6','#c95829']
for (name,result),color in zip(results.items(),colors):
    old=json.loads((root/f'results/validation-{name}/metrics.json').read_text())['pilot']
    lines.append(f"| {name} | Previous free generation | {old['toxic_precision']:.3f} | {old['toxic_recall']:.3f} | {old['toxic_f1']:.3f} | {old['confusion_matrix'][0][1]} | {old['confusion_matrix'][1][0]} |")
    for key,label in [('fixed_0_5','Fixed score threshold 0.5'),('out_of_fold','Fold-selected thresholds (OOF)')]:
        m=result[key];lines.append(f"| {name} | {label} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['fp']} | {m['fn']} |")
    rows=[json.loads(s) for s in (out/f'{name}.scores.jsonl').read_text().splitlines()]
    precision,recall,_=precision_recall_curve([r['label'] for r in rows],[r['score'] for r in rows])
    ax.step(recall,precision,where='post',color=color,label=f"{name} · AP {result['average_precision']:.3f}")
    m=result['fixed_0_5'];ax.scatter(m['recall'],m['precision'],color=color,marker='o',s=55,zorder=3)
ax.axhline(.079,color='#777777',linestyle=':',label='Positive prevalence 0.079')
ax.set(xlim=(0,1),ylim=(0,1.03),xlabel='Toxic recall',ylabel='Toxic precision',title='CommentGuard · validation precision–recall')
ax.grid(alpha=.2);ax.legend(loc='lower left');fig.text(.12,.01,'1,000 comments · 79 positive · circles mark threshold 0.5 · development results',fontsize=9)
fig.tight_layout(rect=(0,.03,1,1));fig.savefig(out/'precision_recall.png',dpi=180);fig.savefig(out/'precision_recall.svg');plt.close(fig)
lines+=['', 'OOF thresholds were selected on four folds and evaluated on the fifth, with identical folds across pilots. The selection objective was maximum precision subject to >=60% recall. This constraint applies to the selection folds; held-out recall can fall below it.', '', '## Candidate thresholds fitted on all validation data', '',
 '| Pilot | Average precision | Candidate threshold | Fit precision | Fit recall | Fit F1 |', '|---|---:|---:|---:|---:|---:|']
for name,r in results.items():
    m=r['full_validation_fit_metrics'];lines.append(f"| {name} | {r['average_precision']:.3f} | {r['full_validation_threshold']:.6f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} |")
lines+=['', 'These full-validation fit metrics are selection-optimistic. The saved candidate configurations are provisional; no model or threshold has been deployed or evaluated on the reserved test.', '', '## Fold thresholds', '']
for name,r in results.items():
    lines.append(f"- {name}: "+', '.join(f"{f['threshold']:.6f}" for f in r['folds'])+'.')
lines+=['', '## Validation and limits', '',
 '- Label tokenization checked on every example; deterministic evaluation and full-candidate likelihood checks passed for both pilots.',
 f"- Truncated validation comments: {manifest['truncated']}.",
 '- Scores are uncalibrated conditional preferences. The fixed wrapper is part of this inference protocol, not a generated confidence claim.',
 '- Five-fold selection reduces threshold-fitting optimism but does not undo prior validation use for training decisions and qualitative error review.',
 '- Previous generation metrics use a different inference rule. Keep improvements due to protocol changes distinct from improvements due to training.',
 '- Average precision is sklearn AP, not trapezoidal PR-AUC. No statistical significance claim is made.',
 '- Per-example scores, fold membership, OOF predictions, exhaustive threshold curves, and candidate configurations are saved beside this report.', '',
 '![Validation precision–recall](precision_recall.png)', '',
 'See [implementation protocol](../../docs/LABEL_SCORING.md).']
(out/'REPORT.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[:16]))
