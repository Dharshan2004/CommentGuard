"""Build the fixed 25%-positive pilot-002 dataset using only source TRAIN rows.

Run from any directory: python scripts/prepare_pilot002.py
Uses the original pinned dataset revision; held-out source text is hashed only
for overlap exclusion, never selected or scored. No validation-error mining.
"""
import hashlib
import json
import random
from pathlib import Path

from datasets import load_dataset
from prepare_data import DATASET, SYSTEM, text_hash, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    prior = json.loads((DATA / "manifests/preparation_report.json").read_text())
    original_path = DATA / "evaluation/pilot_train.jsonl"
    assert sha(original_path) == prior["sha256"]["evaluation/pilot_train.jsonl"]
    old = read(original_path)
    positives = [r for r in old if r["label"] == 1]
    negatives = [r for r in old if r["label"] == 0]
    assert len(positives) == 160 and len(negatives) == 1840
    old_ids = {r["id"] for r in old}
    old_manifest = [r for r in read(DATA / "manifests/selected_rows.jsonl")
                    if r["subset"] == "pilot_train"]
    indices = {r["text_sha256"]: r["source_row"] for r in old_manifest}
    # Snapshot evaluation files to ensure this command never changes them.
    protected = {str(p.relative_to(DATA)): sha(p) for p in
                 [DATA / "prepared/valid.jsonl", DATA / "evaluation/valid.jsonl",
                  DATA / "evaluation/test.jsonl"]}
    source = load_dataset(DATASET, revision=prior["revision"])
    heldout_hashes = set()
    for split in ("test", "validation"):
        for text in source[split]["text"]:
            if isinstance(text, str) and text.strip():
                heldout_hashes.add(text_hash(text))
    print("Built held-out hash exclusion set; no held-out labels used.", flush=True)
    seen_train = set()
    candidates = []
    recovered_old = set()
    for index, row in enumerate(source["train"]):
        text = row["text"]
        if not isinstance(text, str) or not text.strip():
            continue
        digest = text_hash(text)
        if digest in seen_train:
            continue
        seen_train.add(digest)
        if digest in heldout_hashes:
            continue
        if digest in old_ids:
            assert indices[digest] == index
            recovered_old.add(digest)
            continue
        if row["toxicity"] >= 0.5:
            candidates.append(index)
    assert recovered_old == old_ids
    rng = random.Random(42)
    chosen = rng.sample(candidates, 340)
    additions = []
    for index in chosen:
        row = source["train"][index]
        digest = text_hash(row["text"])
        indices[digest] = index
        additions.append(dict(id=digest, text=row["text"], label=1,
                              toxicity=row["toxicity"]))
    retained_negatives = rng.sample(negatives, 1500)
    rows = positives + additions + retained_negatives
    rng.shuffle(rows)
    ids = {r["id"] for r in rows}
    assert len(rows) == len(ids) == 2000
    assert sum(r["label"] for r in rows) == 500
    assert not ids & heldout_hashes
    assert {r["id"] for r in positives} <= ids
    assert len(ids & old_ids) == 1660
    for row in rows:
        assert row["id"] == text_hash(row["text"])
        assert row["label"] == int(row["toxicity"] >= 0.5)
    name = "pilot002_mix25_train"
    chat_path = DATA / "prepared" / f"{name}.jsonl"
    records_path = DATA / "evaluation" / f"{name}.jsonl"
    manifest_path = DATA / "manifests" / f"{name}_rows.jsonl"
    write_jsonl(chat_path, [{"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Comment:\n" + r["text"]},
        {"role": "assistant", "content": str(r["label"])}]} for r in rows])
    write_jsonl(records_path, rows)
    write_jsonl(manifest_path, [dict(subset=name, source_split="train",
        source_row=indices[r["id"]], text_sha256=r["id"],
        origin="retained_pilot001" if r["id"] in old_ids else "additional_train_positive") for r in rows])
    # Independently deserialize and validate what will be uploaded.
    exported = read(chat_path)
    assert len(exported) == 2000
    for chat, record in zip(exported, rows):
        messages = chat["messages"]
        assert [m["role"] for m in messages] == ["system", "user", "assistant"]
        assert messages[0]["content"] == prior["system_prompt"]
        assert messages[1]["content"] == "Comment:\n" + record["text"]
        assert messages[2]["content"] == str(record["label"])
    assert all(sha(DATA / path) == value for path, value in protected.items())
    report = dict(dataset=DATASET, revision=prior["revision"], seed=42,
        sampling="Python random.Random(42): sample 340 source positives, sample 1500 pilot negatives, shuffle",
        label_rule="toxicity >= 0.5", rows=2000, toxic=500, non_toxic=1500,
        retained_original_positives=160, additional_unique_positives=340,
        retained_original_negatives=1500, eligible_new_positive_pool=len(candidates),
        unique_rows=2000, heldout_full_source_hash_overlap=0,
        heldout_files_unchanged=True, heldout_use="Text hashes only; no held-out labels, evaluation or error mining",
        truncation="None; use the existing 1024-token training/inference policy",
        sha256={str(p.relative_to(DATA)): sha(p) for p in [chat_path, records_path, manifest_path]},
        checks="Counts, unique hashes, lineage, label consistency, messages schema and held-out exclusion passed")
    (DATA / "manifests/pilot002_mix25_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
