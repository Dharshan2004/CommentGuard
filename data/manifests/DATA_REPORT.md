# Dataset preparation report

Source: `google/civil_comments` at `f2970eb3a55777454c94069077cc8d9b5866312d`.

Seed: 42. Binary label: toxicity >= 0.5.

| Subset | Rows | Toxic | Non-toxic |
|---|---:|---:|---:|
| train | 5000 | 400 | 4600 |
| valid | 1000 | 79 | 921 |
| test | 3000 | 239 | 2761 |
| pilot_train | 2000 | 160 | 1840 |
| smoke_train | 200 | 16 | 184 |

## Full-source cleaning

| Split | Source rows | Empty | Within-split duplicates | Cross-split duplicates | Eligible |
|---|---:|---:|---:|---:|---:|
| test | 97320 | 0 | 517 | 0 | 96803 |
| validation | 97320 | 0 | 515 | 664 | 96141 |
| train | 1804874 | 0 | 26826 | 2246 | 1775802 |

Priority: test, then validation, then train. Exact normalized-text hashes only; near-duplicate and thread leakage remain possible.

Export verification passed: schemas, counts, labels, hashes, checksums, nested training subsets, and zero cross-split hash overlap. No Studio upload or training was performed.

Only aggregate test statistics were inspected. Comments retain their source text; tokenizer truncation is not applied here.
