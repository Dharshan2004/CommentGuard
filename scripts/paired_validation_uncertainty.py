"""Paired stratified bootstrap sensitivity for fixed-0.5 development F1.

Intervals condition on already-selected models; they do not correct selection bias.
"""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def f1(y,p):
    tp=np.sum(p & y,axis=-1); fp=np.sum(p & ~y,axis=-1);fn=np.sum(~p & y,axis=-1)
    return 2*tp/(2*tp+fp+fn)

def main():
    a=[json.loads(s) for s in (ROOT/'results/label-scoring/pilot-002.scores.jsonl').read_text().splitlines()]
    b=[json.loads(s) for s in (ROOT/'results/validation-main5000-step4000/main5000-step4000.scores.jsonl').read_text().splitlines()]
    assert [(r['id'],r['label']) for r in a]==[(r['id'],r['label']) for r in b]
    y=np.array([r['label'] for r in a],dtype=bool)
    pa=np.array([r['score']>=.5 for r in a]);pb=np.array([r['score']>=.5 for r in b])
    rng=np.random.default_rng(42);pos=np.flatnonzero(y);neg=np.flatnonzero(~y)
    indices=np.concatenate([rng.choice(pos,(5000,len(pos))),rng.choice(neg,(5000,len(neg)))],axis=1)
    diffs=f1(y[indices],pb[indices])-f1(y[indices],pa[indices])
    result=dict(comparison='step4000 minus pilot002, both fixed threshold0.5',
        observed_f1_difference=float(f1(y,pb)-f1(y,pa)),
        percentile_95_interval=np.quantile(diffs,[.025,.975]).tolist(),
        resamples=5000,seed=42,method='paired bootstrap within true-label strata, preserving observed prevalence',
        limitation='Conditional development sensitivity only. Models selected using this validation set; interval does not account for selection, label noise, or thread dependence. Not an unbiased generalization/significance claim. No OOF threshold refitting performed in bootstrap.')
    out=ROOT/'results/iterative-selection/paired_uncertainty.json';out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
