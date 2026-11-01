"""Build a readable report and CSV from a completed validation comparison."""
import csv
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = root/'results/validation-pilot-001'
metrics = json.loads((out/'metrics.json').read_text())
manifest = json.loads((out/'manifest.json').read_text())
assert set(metrics) == {'logistic_regression', 'base', 'pilot'}, 'Evaluation is incomplete'
assert all(m['n'] == 1000 for m in metrics.values())
names = {'logistic_regression':'TF-IDF + logistic regression', 'base':'Base Qwen3-4B', 'pilot':'Pilot Qwen3-4B + LoRA'}
fields = ['model','n','toxic_precision','toxic_recall','toxic_f1','macro_f1','accuracy',
          'invalid_outputs','raw_format_violations','latency_ms_p50','latency_ms_p95']
with (out/'metrics.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
    for name,m in metrics.items(): writer.writerow(dict(model=name,**{k:m[k] for k in fields[1:]}))
lines = ['# CommentGuard pilot validation comparison', '',
         'Evaluated all 1,000 fixed validation comments: 921 non-toxic and 79 toxic. '
         'Logistic regression was fitted on the same 2,000 pilot training comments. '
         'The 3,000-example test set was not read.', '',
         '| Model | Toxic precision | Toxic recall | Toxic F1 | Macro-F1 | Accuracy |',
         '|---|---:|---:|---:|---:|---:|']
for name,m in metrics.items():
    lines.append('| '+names[name]+' | '+' | '.join(f'{m[k]:.3f}' for k in ['toxic_precision','toxic_recall','toxic_f1','macro_f1','accuracy'])+' |')
lines += ['', '## Error counts', '', '| Model | TN | FP | FN | TP | Invalid | Raw format violations |', '|---|---:|---:|---:|---:|---:|---:|']
for name,m in metrics.items():
    a,b=m['confusion_matrix']; lines.append(f"| {names[name]} | {a[0]} | {a[1]} | {b[0]} | {b[1]} | {m['invalid_outputs']} | {m['raw_format_violations']} |")
lines += ['', 'Raw format violations include empty thinking wrappers. Classification metrics strip only a leading empty `<think>...</think>` block and surrounding whitespace. All other malformed outputs remain invalid and count as failures. The same rule is applied to both models.', '', '## Local batch-one inference', '', '| Model | Median ms | p95 ms | MLX peak allocated GiB |', '|---|---:|---:|---:|']
for name,m in metrics.items():
    peak=f"{m['mlx_peak_allocated_gib']:.2f}" if 'mlx_peak_allocated_gib' in m else 'N/A'
    lines.append(f"| {names[name]} | {m['latency_ms_p50']:.2f} | {m['latency_ms_p95']:.2f} | {peak} |")
lines += ['', 'Observed on the user’s Apple M5 Pro, 24 GB unified-memory Mac. These warm sequential measurements are not an isolated production benchmark. LLM timings exclude tokenization/loading; logistic timings include vectorization. MLX allocated memory is not total process or system memory.', '', '## Protocol and limits', '',
          f"- Base revision: `{manifest['model_revision']}`.",
          '- Greedy decoding; maximum 8 output tokens; context budget 1,024; independent prompts; no search or tools.',
          f"- Truncated inputs: {manifest['truncated_train']} training; {manifest['truncated_valid']} validation.",
          '- TF-IDF/logistic settings were fixed in advance; no threshold or hyperparameter search.',
          '- Final pilot checkpoint had already been selected using validation loss; these are development results.',
          '- The parser rule was corrected after a three-row smoke check, before the full comparison.',
          '- There are only 79 toxic validation examples. No statistical significance claim is made.',
          '- Language-model average precision is omitted because optional generated wrappers complicate label-position scoring.',
          '- An always-non-toxic classifier obtains 92.1% accuracy but zero toxic recall.',
          '- Exact source/hash provenance and software versions are in `manifest.json`. Per-example outputs are in the prediction JSONL files.', '',
          'See [evaluation instructions](../../docs/EVALUATION.md) and [the evaluator](../../scripts/evaluate.py).']
(out/'REPORT.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[:11]))
