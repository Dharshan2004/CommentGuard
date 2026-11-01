"""Five-fold development evaluation of max precision subject to >=60% recall.

Thresholds are selected only on training folds. Full-validation selected rules
are saved as development candidates, not unbiased estimates of final performance.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold


def metrics(y,p):
    tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel()
    return dict(precision=float(precision_score(y,p,zero_division=0)),
        recall=float(recall_score(y,p,zero_division=0)),
        f1=float(f1_score(y,p,zero_division=0)),
        macro_f1=float(f1_score(y,p,average='macro',zero_division=0)),
        tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))


def choose_threshold(y,scores,min_recall=0.6):
    if not 0<min_recall<=1 or not np.any(y):
        raise ValueError('Require positive labels and recall target in (0,1]')
    precision,recall,thresholds=precision_recall_curve(y,scores)
    eligible=np.flatnonzero(recall[:-1]>=min_recall)
    # Deterministic tie-break: precision, then recall, then higher threshold.
    best=max(eligible,key=lambda i:(precision[i],recall[i],thresholds[i]))
    return float(thresholds[best])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scores',type=Path,default=Path(__file__).resolve().parents[1]/'results/label-scoring')
    args=ap.parse_args();folder=args.scores
    manifest=json.loads((folder/'manifest.json').read_text())
    assert manifest['rows']==1000 and manifest['models']
    names=list(manifest['models'])
    import hashlib
    def read(name):
        path=folder/f'{name}.scores.jsonl'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['models'][name]['scores_sha256']
        return [json.loads(s) for s in path.read_text().splitlines()]
    first=read(names[0]); y=np.array([r['label'] for r in first]);ids=[r['id'] for r in first]
    assert len(ids)==len(set(ids))==1000 and y.sum()==79
    folds=list(StratifiedKFold(n_splits=5,shuffle=True,random_state=42).split(np.zeros(len(y)),y))
    summary={}; candidates={}; fold_membership=[]
    for k,(_,test) in enumerate(folds):
        fold_membership.extend(dict(id=ids[i],heldout_fold=k) for i in test)
    (folder/'fold_membership.json').write_text(json.dumps(fold_membership,indent=2)+'\n')
    for name in names:
        rows=read(name)
        assert [(r['id'],r['label']) for r in rows]==list(zip(ids,y.tolist()))
        scores=np.array([r['score'] for r in rows]);assert np.isfinite(scores).all()
        assert ((scores>=0)&(scores<=1)).all()
        oof=np.full(len(y),-1);fold_results=[]
        for k,(train,test) in enumerate(folds):
            assert not set(train)&set(test)
            t=choose_threshold(y[train],scores[train])
            oof[test]=(scores[test]>=t).astype(int)
            fold_results.append(dict(fold=k,threshold=t,selection_n=len(train),heldout_n=len(test),
                selection_metrics=metrics(y[train],scores[train]>=t),
                heldout_metrics=metrics(y[test],oof[test])))
        assert (oof>=0).all()
        final=choose_threshold(y,scores)
        summary[name]=dict(average_precision=float(average_precision_score(y,scores)),
            fixed_0_5=metrics(y,scores>=0.5),out_of_fold=metrics(y,oof),folds=fold_results,
            full_validation_threshold=final,
            full_validation_fit_metrics=metrics(y,scores>=final))
        candidates[name]=dict(status='development candidate; not final-test validated',
            adapter=name,adapter_sha256=manifest['models'][name]['adapter_sha256'],
            model_revision=manifest['model_revision'],scoring_protocol=manifest['protocol'],
            wrapper=manifest['wrapper'],system_prompt=manifest['system_prompt'],context=1024,
            threshold=final,comparison='score >= threshold',
            objective='maximize precision subject to selection recall >=0.60; tie-break recall then threshold')
        with (folder/f'{name}.threshold_curve.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=['threshold','precision','recall','f1','macro_f1','tn','fp','fn','tp']);w.writeheader()
            for t in sorted(set([0.0,0.5,1.0]+scores.tolist())):
                w.writerow(dict(threshold=t,**metrics(y,scores>=t)))
        with (folder/f'{name}.oof_predictions.jsonl').open('w') as f:
            for i in range(len(y)): f.write(json.dumps(dict(id=ids[i],label=int(y[i]),prediction=int(oof[i]),score=float(scores[i])))+'\n')
    (folder/'threshold_results.json').write_text(json.dumps(summary,indent=2)+'\n')
    (folder/'candidate_configs.json').write_text(json.dumps(candidates,indent=2)+'\n')
    for name,result in summary.items():
        print(name,json.dumps({k:v for k,v in result.items() if k!='folds'}))


if __name__=='__main__':
    main()
