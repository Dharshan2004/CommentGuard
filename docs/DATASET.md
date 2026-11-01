# Dataset and splits

Source: [google/civil_comments](https://huggingface.co/datasets/google/civil_comments), revision `f2970eb3a55777454c94069077cc8d9b5866312d`. English news-site comments with crowd toxicity annotations; the source reports CC0-1.0. Model weights and project code have separate terms.

The target is `toxicity >= 0.5`. This binary conversion is a modeling choice, not objective harm ground truth or a platform moderation policy.

| Split | Rows | Toxic |
|---|---:|---:|
| Main training pool | 5,000 | 1,250 |
| Validation | 1,000 | 79 |
| Test | 3,000 | 239 |

Preparation removes empty text and exact normalized-text duplicates using NFKC and collapsed whitespace, preserving case. Original model input is retained. Source test has precedence over validation, then train; semantic near-duplicates and comment/thread correlations are not resolved. Conflicting duplicate labels retain the first source row.

The main 25%-toxic pool retains all 2,000 pilot-002 examples and adds 750 toxic plus 2,250 non-toxic source-training examples. The earlier `train.jsonl` has a different class mix: do not concatenate the nested training exports. Seed is 42.

Training JSONL contains system/user/assistant messages. Evaluation JSONL contains id/text/label/toxicity. Only comment text and the fixed instruction enter inference; labels are used to calculate metrics. Test text hashes were used for exclusion during preparation. Test metrics were evaluated only after model/threshold freeze; the baselines were evaluated subsequently. Do not describe the test data as never accessed during preparation.

Full comments and row-level predictions are excluded from Git. Aggregate preparation manifests and hashes remain under data/manifests/. Regenerate comments with the pinned preparation commands in REPRODUCING.md.
