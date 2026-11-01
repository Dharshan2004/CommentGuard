"""Extend pilot-002 to 5,000 unique train-only rows, preserving its 25% toxic mix."""
import json
import random
from datasets import load_dataset
from prepare_data import DATASET, SYSTEM, text_hash, write_jsonl
from prepare_pilot002 import DATA, read, sha


def main():
    prior = json.loads((DATA / 'manifests/preparation_report.json').read_text())
    pilot_report = json.loads((DATA / 'manifests/pilot002_mix25_report.json').read_text())
    for path, digest in pilot_report['sha256'].items():
        assert sha(DATA / path) == digest
    rows = read(DATA / 'evaluation/pilot002_mix25_train.jsonl')
    old_ids = {r['id'] for r in rows}
    indices = {r['text_sha256']: r['source_row'] for r in read(DATA / 'manifests/pilot002_mix25_train_rows.jsonl')}
    protected = {str(p.relative_to(DATA)): sha(p) for p in DATA.rglob('*.jsonl')}
    source = load_dataset(DATASET, revision=prior['revision'])
    heldout = set()
    for split in ('validation', 'test'):
        for text in source[split]['text']:
            if isinstance(text, str) and text.strip():
                heldout.add(text_hash(text))
    print('Held-out text hashes loaded for exclusion only.', flush=True)
    pools = {0: [], 1: []}
    seen, recovered = set(), set()
    for index, row in enumerate(source['train']):
        text = row['text']
        if not isinstance(text, str) or not text.strip():
            continue
        digest = text_hash(text)
        if digest in seen:
            continue
        seen.add(digest)
        if digest in heldout:
            continue
        if digest in old_ids:
            assert indices[digest] == index
            recovered.add(digest)
        else:
            pools[int(row['toxicity'] >= 0.5)].append(index)
    assert recovered == old_ids
    rng = random.Random(42)
    for label, count in ((1, 750), (0, 2250)):
        for index in rng.sample(pools[label], count):
            row = source['train'][index]
            digest = text_hash(row['text'])
            rows.append(dict(id=digest, text=row['text'], label=label, toxicity=row['toxicity']))
            indices[digest] = index
    rng.shuffle(rows)
    ids = {r['id'] for r in rows}
    assert len(rows) == len(ids) == 5000
    assert sum(r['label'] for r in rows) == 1250
    assert old_ids <= ids and not ids & heldout
    for row in rows:
        assert row['id'] == text_hash(row['text'])
        assert row['label'] == int(row['toxicity'] >= 0.5)
    name = 'main5000_mix25_train'
    chat = DATA / 'prepared' / f'{name}.jsonl'
    records = DATA / 'evaluation' / f'{name}.jsonl'
    lineage = DATA / 'manifests' / f'{name}_rows.jsonl'
    messages = [{'messages': [dict(role='system', content=SYSTEM),
        dict(role='user', content='Comment:\n' + r['text']),
        dict(role='assistant', content=str(r['label']))]} for r in rows]
    write_jsonl(chat, messages)
    write_jsonl(records, rows)
    write_jsonl(lineage, [dict(subset=name, source_split='train', source_row=indices[r['id']],
        text_sha256=r['id'], origin='retained_pilot002' if r['id'] in old_ids else 'additional_train') for r in rows])
    assert read(chat) == messages and read(records) == rows
    assert SYSTEM == prior['system_prompt']
    assert all(sha(DATA / path) == digest for path, digest in protected.items())
    report = dict(dataset=DATASET, revision=prior['revision'], seed=42,
        rows=5000, toxic=1250, non_toxic=3750, unique_rows=5000,
        retained_pilot002=2000, added_toxic=750, added_non_toxic=2250,
        sampling='Retain all pilot-002; random.Random(42) sample 750 additional positives then 2250 negatives; shuffle',
        label_rule='toxicity >= 0.5', heldout_full_source_hash_overlap=0,
        existing_jsonl_files_unchanged=True,
        heldout_use='Text hashes for exclusion only; no held-out labels or error mining',
        truncation='None during preparation; use context 1024 in Studio',
        sha256={str(p.relative_to(DATA)): sha(p) for p in (chat, records, lineage)})
    (DATA / 'manifests/main5000_mix25_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
