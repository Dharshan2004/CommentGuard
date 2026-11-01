"""Compare completed pilot validation results without reading any dataset text."""
import csv
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
old = root/'results/validation-pilot-001'
new = root/'results/validation-pilot-002'
m1=json.loads((old/'metrics.json').read_text())
m2=json.loads((new/'metrics.json').read_text())
assert 'pilot' in m2 and m2['pilot']['n']==1000
def records(path):
    return [json.loads(s) for s in path.read_text().splitlines()]
p1=records(old/'pilot.predictions.jsonl'); p2=records(new/'pilot.predictions.jsonl')
assert [(r['id'],r['label']) for r in p1] == [(r['id'],r['label']) for r in p2]
assert len({r['id'] for r in p2})==1000
changes = {
 'previous_false_negatives_recovered':sum(a['label']==1 and a['prediction']==0 and b['prediction']==1 for a,b in zip(p1,p2)),
 'previous_true_positives_lost':sum(a['label']==1 and a['prediction']==1 and b['prediction']!=1 for a,b in zip(p1,p2)),
 'new_false_positives':sum(a['label']==0 and a['prediction']==0 and b['prediction']==1 for a,b in zip(p1,p2)),
 'previous_false_positives_fixed':sum(a['label']==0 and a['prediction']==1 and b['prediction']==0 for a,b in zip(p1,p2)),
}
(new/'paired_changes.json').write_text(json.dumps(changes,indent=2)+'\n')
models={'Base Qwen (reused)':m2['base'],'Logistic / pilot-001 data':m1['logistic_regression'],
        'Logistic / pilot-002 data':m2['logistic_regression'],'Pilot-001 / 8% toxic':m1['pilot'],
        'Pilot-002 / 25% toxic':m2['pilot']}
fields=['model','toxic_precision','toxic_recall','toxic_f1','macro_f1','accuracy','invalid_outputs','raw_format_violations']
with (new/'comparison.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
    for name,m in models.items(): w.writerow(dict(model=name,**{k:m[k] for k in fields[1:]}))
lines=['# Pilot-002 validation comparison','',
 'All models evaluated on the same 1,000 validation comments (79 toxic). '
 'Both pilots trained for 2,000 steps from the original pinned Qwen base. '
 'Pilot-002 uses 500 positive and 1,500 negative examples; pilot-001 used 160 and 1,840. '
 'Each logistic baseline is fitted on its matching training set. The test set was not evaluated.', '',
 '| Model | Toxic precision | Toxic recall | Toxic F1 | Macro-F1 | Accuracy |',
 '|---|---:|---:|---:|---:|---:|']
for name,m in models.items():
    lines.append('| '+name+' | '+' | '.join(f'{m[k]:.3f}' for k in fields[1:6])+' |')
lines += ['', '## Errors', '', '| Model | False positives | Missed toxic | Caught toxic | Invalid | Raw format violations |', '|---|---:|---:|---:|---:|---:|']
for name,m in models.items():
    cm=m['confusion_matrix'];lines.append(f"| {name} | {cm[0][1]} | {cm[1][0]} | {cm[1][1]} | {m['invalid_outputs']} | {m['raw_format_violations']} |")
lines+=['', '## Paired prediction changes', '']
for name,val in changes.items(): lines.append(f'- {name.replace("_"," ")}: {val}.')
meets=m2['pilot']['toxic_recall']>=0.4 and m2['pilot']['toxic_precision']>=0.5
lines+=['', f'Previously proposed development target (recall >=40%, precision >=50%): **{"met" if meets else "not met"}**.', '',
 '## Reproducibility and limits', '',
 '- Native exported weights were verified against each Studio training output. The pilot-002 upload matches the generated mix25 JSONL.',
 '- Training metadata is in `training_provenance.json`. Core checked settings match; evaluation frequency changed from 0.25 to 0.1. Both completed 2,000 steps without resuming an adapter.',
 '- Evaluation protocol is unchanged: independent prompts, greedy decoding, 8 output tokens, 1,024 context, no tools. Empty thinking wrappers are stripped for label scoring; raw violations are reported separately.',
 '- Base predictions were reused only after checking validation hash, revision, protocol and software versions. Base timings are from the earlier run, not a fresh benchmark.',
 '- Changing the mix also added positive-example diversity and removed some negative examples; this is not a pure causal test of class prevalence.',
 '- The validation split has been used for development and error analysis. No significance or final-test-performance claim is made.',
 '- See `manifest.json`, `metrics.json` and prediction JSONL files for exact evidence.', '',
 '## Reproduce', '', 'From the repo root, using the Studio Python environment or an equivalent MLX environment:', '',
 '```bash', 'python scripts/evaluate.py --adapter adapters/pilot-002 \\',
 '  --train-name pilot002_mix25_train \\',
 '  --model-path /path/to/snapshots/50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b \\',
 '  --output results/validation-pilot-002 --reuse-base results/validation-pilot-001',
 'python scripts/compare_pilots.py', '```']
(new/'REPORT.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[:14])); print(changes)
