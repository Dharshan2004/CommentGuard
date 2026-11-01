"""Fixed logistic baseline; fit on training, threshold on validation, then test."""
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from select_threshold import choose_threshold

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def main():
    out = ROOT/'results/test-logistic-main5000'
    out.mkdir(exist_ok=False)
    prep = json.loads((ROOT/'data/manifests/preparation_report.json').read_text())
    trainprep = json.loads((ROOT/'data/manifests/main5000_mix25_report.json').read_text())
    trainpath = ROOT/'data/evaluation/main5000_mix25_train.jsonl'
    validpath = ROOT/'data/evaluation/valid.jsonl'
    assert sha(trainpath) == trainprep['sha256']['evaluation/main5000_mix25_train.jsonl']
    assert sha(validpath) == prep['sha256']['evaluation/valid.jsonl']
    train, valid = read(trainpath), read(validpath)
    assert len(train) == 5000 and sum(r['label'] for r in train) == 1250
    assert len(valid) == 1000 and sum(r['label'] for r in valid) == 79
    assert not {r['id'] for r in train} & {r['id'] for r in valid}
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True),
        LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42))
    start = time.perf_counter()
    model.fit([r['text'] for r in train], [r['label'] for r in train])
    fit_seconds = time.perf_counter()-start
    assert max(model[-1].n_iter_) < 1000, 'Classifier did not converge'
    vscores = model.predict_proba([r['text'] for r in valid])[:, 1]
    threshold = choose_threshold(np.array([r['label'] for r in valid]), vscores)
    joblib.dump(model, out/'model.joblib')
    config = dict(model='tfidf-logistic', threshold=threshold, comparison='score >= threshold',
        objective='maximize precision subject to validation recall >=0.60; tie-break recall then threshold',
        train_n=5000, train_positive=1250, train_sha256=sha(trainpath),
        validation_sha256=sha(validpath), model_sha256=sha(out/'model.joblib'),
        vectorizer=dict(ngram_range=[1,2], min_df=2, max_features=50000, sublinear_tf=True),
        classifier=dict(C=1.0, class_weight='balanced', max_iter=1000, random_state=42),
        input='Full original comment text; vocabulary and IDF fitted on training only',
        fit_seconds=fit_seconds, dataset_revision=prep['revision'],
        packages={p:importlib.metadata.version(p) for p in ['scikit-learn','numpy','joblib']})
    (out/'frozen_config.json').write_text(json.dumps(config, indent=2)+'\n')
    with (out/'validation.scores.jsonl').open('w') as f:
        for r,s in zip(valid,vscores): f.write(json.dumps(dict(id=r['id'],label=r['label'],score=float(s)))+'\n')
    # Test data is accessed only after training and threshold/config freezing.
    testpath = ROOT/'data/evaluation/test.jsonl'
    assert sha(testpath) == prep['sha256']['evaluation/test.jsonl']
    test = read(testpath)
    assert len(test) == len({r['id'] for r in test}) == 3000
    assert sum(r['label'] for r in test) == 239
    assert not {r['id'] for r in test} & {r['id'] for r in train+valid}
    started = time.perf_counter()
    with (out/'logistic.scores.jsonl').open('w') as f:
        for r in test:
            tick = time.perf_counter()
            s = float(model.predict_proba([r['text']])[0,1])
            f.write(json.dumps(dict(id=r['id'],label=r['label'],score=s,
                                   latency_ms=(time.perf_counter()-tick)*1000))+'\n')
    manifest = dict(split='test',rows=3000,positives=239,test_sha256=sha(testpath),
        frozen_config_sha256=sha(out/'frozen_config.json'),threshold=threshold,
        scores_sha256=sha(out/'logistic.scores.jsonl'),
        elapsed_seconds=time.perf_counter()-started,script_sha256=sha(Path(__file__)))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__ == '__main__':
    main()
