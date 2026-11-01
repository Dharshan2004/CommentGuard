"""Aggregate completed validation experiments without selecting on test data."""
import json
import argparse
import hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    sources=[('pilot-002','label-scoring'),('main5000-001','validation-main5000-001'),('main5000-step4000','validation-main5000-step4000')]
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--additional',nargs='*',default=[],help='Additional adapter names with results under validation-NAME')
    args=ap.parse_args()
    sources.extend((n,'validation-'+n) for n in args.additional)
    assert len({n for n,_ in sources})==len(sources)
    summary={}; folds=None; reference=None; reference_rows=None
    keys=['protocol','wrapper','system_prompt','label_ids','context','candidate_definition','scoring','split','rows','positives','validation_sha256','dataset_revision','model_revision','packages']
    for name,folder in sources:
        p=ROOT/'results'/folder
        manifest=json.loads((p/'manifest.json').read_text())
        if reference is None:reference=manifest
        assert all(manifest[k]==reference[k] for k in keys), 'Scoring protocol mismatch'
        scores=p/f'{name}.scores.jsonl'
        assert hashlib.sha256(scores.read_bytes()).hexdigest()==manifest['models'][name]['scores_sha256']
        rows=[(r['id'],r['label']) for r in map(json.loads,scores.read_text().splitlines())]
        if reference_rows is None: reference_rows=rows
        assert rows==reference_rows and len(set(i for i,_ in rows))==1000
        membership=json.loads((p/'fold_membership.json').read_text())
        if folds is None: folds=membership
        assert membership==folds
        summary[name]=json.loads((p/'threshold_results.json').read_text())[name]
    out=ROOT/'results/iterative-selection';out.mkdir(exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Iterative validation experiment ledger','',
        'All rows use the same 1,000 validation comments and scoring protocol. Fold-selected means thresholds fitted on four folds and evaluated on the fifth with identical membership. Objective is precision maximization subject to 60% selection recall, not F1 maximization. Validation has already influenced model development; these are not untouched-test results.', '',
        '| Model | Rule | Precision | Recall | F1 | FP | FN | AP |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for name,r in summary.items():
        for key in ('fixed_0_5','out_of_fold'):
            m=r[key];lines.append(f"| {name} | {key} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['fp']} | {m['fn']} | {r['average_precision']:.4f} |")
    lines+=['','Thresholds refitted on all validation data (selection-optimistic; not the thresholds used for the OOF table):']
    for name,r in summary.items(): lines.append(f"- {name}: {r['full_validation_threshold']:.9f}")
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
if __name__=='__main__':main()
