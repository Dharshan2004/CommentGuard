"""Validate generated exports without printing any comment content."""
import hashlib
import json
import unicodedata
from pathlib import Path

root = Path(__file__).resolve().parents[1]
data = root / "data"
report = json.loads((data / "manifests/preparation_report.json").read_text())
hashes = {}
for name, stats in report["subsets"].items():
    records = [json.loads(line) for line in (data / "evaluation" / f"{name}.jsonl").read_text().splitlines()]
    assert len(records) == stats["rows"]
    for row in records:
        normalized = " ".join(unicodedata.normalize("NFKC", row["text"]).split())
        assert row["id"] == hashlib.sha256(normalized.encode()).hexdigest()
        assert row["label"] == int(row["toxicity"] >= 0.5)
    hashes[name] = {r["id"] for r in records}
    assert len(hashes[name]) == len(records)
    assert sum(r["label"] for r in records) == stats["positive"]
    if name != "test":
        chats = [json.loads(line) for line in (data / "prepared" / f"{name}.jsonl").read_text().splitlines()]
        assert len(chats) == len(records)
        for chat, row in zip(chats, records):
            messages = chat["messages"]
            assert [m["role"] for m in messages] == ["system", "user", "assistant"]
            assert messages[0]["content"] == report["system_prompt"]
            assert messages[1]["content"] == "Comment:\n" + row["text"]
            assert messages[2]["content"] == str(row["label"])
for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
    assert not hashes[a] & hashes[b]
assert hashes["smoke_train"] <= hashes["pilot_train"] <= hashes["train"]
assert not (data / "prepared/test.jsonl").exists()
for filename, expected in report["sha256"].items():
    assert hashlib.sha256((data / filename).read_bytes()).hexdigest() == expected
print("PASS: schemas, row counts, labels, text hashes, file checksums, split isolation, and nested subsets")
