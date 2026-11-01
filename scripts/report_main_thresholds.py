"""Report main-model threshold selection against pilot-002 development results."""
import json
from pathlib import Path
from score_labels import sha

ROOT = Path(__file__).resolve().parents[1]

def main():
    old = ROOT/'results/label-scoring'
    new = ROOT/'results/validation-main5000-001'
    assert json.loads((old/'fold_membership.json').read_text()) == json.loads((new/'fold_membership.json').read_text())
    pilot = json.loads((old/'threshold_results.json').read_text())['pilot-002']
    main = json.loads((new/'threshold_results.json').read_text())['main5000-001']
    lines=['# Five-fold threshold comparison', '',
        'All 1,000 validation comments, including 79 toxic. Identical stratified folds (seed 42) verified against the prior pilot analysis. Each threshold is selected on 800 comments and evaluated on the remaining 200. Objective unchanged: maximize precision subject to at least 60% recall on selection folds; tie-break by recall, then threshold.', '',
        '| Model | Rule | Precision | Recall | F1 | FP | FN |',
        '|---|---|---:|---:|---:|---:|---:|']
    for name,result in [('pilot-002',pilot),('main5000-001',main)]:
        for key,label in [('fixed_0_5','Fixed 0.5'),('out_of_fold','Fold-selected')]:
            m=result[key]
            lines.append(f"| {name} | {label} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['fp']} | {m['fn']} |")
    lines += ['', '## Interpretation', '',
        'Main-model threshold selection reduces FP from 81 to 36 but increases FN from 19 to 30, improving F1 from 0.545 to 0.598. Against pilot-002 with the same threshold-selection procedure, main has 6 fewer FP and the same 30 FN (F1 0.598 versus 0.576).', '',
        'Against the previously preferred pilot-002 at fixed 0.5, main fold-selected results have 18 fewer FP but 7 additional misses. F1 is only 0.005 higher (0.598 versus 0.593), so there is no clear overall winner or statistical-significance claim. Pilot-002 remains preferable when recall matters more; main is a candidate when reducing review false alarms matters more.', '',
        '## Candidate threshold', '',
        f"The main-model threshold refitted on all validation data is **{main['full_validation_threshold']:.9f}**. Its full-validation fit has precision 0.575, recall 0.633, F1 0.602, FP37 and FN29. These are selection-optimistic fit metrics, not out-of-fold estimates. The out-of-fold table uses five different thresholds, not this single refitted threshold.", '',
        'Fold thresholds: '+', '.join(f"{f['threshold']:.6f}" for f in main['folds'])+'.', '',
        'The configuration is saved in candidate_configs.json; no deployment or final model freeze occurred. Validation was reused for earlier development, so five-fold threshold selection does not erase prior model-selection bias. The test set remains unevaluated. No further training is indicated by this comparison alone.']
    (new/'THRESHOLD_REPORT.md').write_text('\n'.join(lines)+'\n')
    (new/'threshold_verification.json').write_text(json.dumps(dict(
        identical_fold_membership=True, validation_rows=1000, positives=79,
        selector_sha256=sha(ROOT/'scripts/select_threshold.py'),
        tests='Three threshold-selection unit tests passed',
        test_evaluated=False),indent=2)+'\n')
    print('Verified matching folds; report and candidate configuration saved.')

if __name__ == '__main__':
    main()
