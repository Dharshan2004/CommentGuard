# Results and experiment decisions

## Held-out test

All models were evaluated on the same3,000 comments (239 toxic). Primary thresholds maximize validation precision subject to >=60% validation recall, then remain fixed for test. The fine-tuned test was run first; baseline protocol was recorded before subsequent baseline test scoring. This is not a jointly preregistered blind benchmark.

| Model | Threshold | Precision | Recall | F1 | AP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fine-tuned | .407333434 | .6416 | .6067 | .6237 | .6808 | 81 | 94 |
| Base | .996406376 | .1739 | .6402 | .2735 | .1892 | 727 | 86 |
| Logistic | .489467258 | .2491 | .5858 | .3496 | .3663 | 422 | 99 |

Fine-tuning gained0.3502 F1 over base and0.2741 over the fixed logistic baseline. Against logistic it catches5 more toxic comments with341 fewer false positives. Against base it has646 fewer false positives but8 more misses. Operating points are not exactly recall-matched on test.

At fixed0.5, F1 is .5945 (fine-tuned), .1697 (base), and .3563 (logistic). AP is higher for fine-tuning too, supporting a ranking improvement beyond a chosen threshold. Base Qwen has a severe toxic-label score bias under this fixed-wrapper protocol; this does not measure its best possible prompting performance.

Median local per-comment inference: fine-tuned145.8ms, base128.1ms, logistic0.17ms. These measurements exclude loading, involve differing pipelines, and are not deployment throughput guarantees. The larger model trades much greater cost for quality here.

## Validation development

| Candidate | Fold-selected precision | Recall | F1 | AP |
|---|---:|---:|---:|---:|
| Pilot-002 | .5385 | .6203 | .5765 | .6168 |
| Main final checkpoint | .5765 | .6203 | .5976 | .6833 |
| Main step4,000 | .6667 | .6076 | .6358 | .6943 |
| Two epochs/dropout0, best checkpoint9,000 | .6282 | .6203 | .6242 | .6928 |
| Two epochs/dropout0.1, best checkpoint5,000 | .5444 | .6203 | .5799 | .6424 |

Each fold threshold was fitted on800 validation comments and evaluated on200. Model development/checkpoint selection already reused validation, so cross-fitting thresholds does not make this an unbiased estimate of model selection. There are only79 positive validation examples.

The two-epoch arms used the same fresh base, data, schedule and seed, differing only in dropout. Each checkpoint was preselected by minimum validation token loss before classification scoring. Dropout0.1 added12 false positives at the same30 misses as dropout0. More epochs also change the linear LR schedule relative to the one-epoch run, so epochs alone are not isolated. Token loss is not classification F1. The bounded comparison ended with step4,000 and a full-validation refit threshold, frozen before test.

## Evidence

- [Test comparison JSON](../reports/test-comparison.json): both operating-point rules, AP and timing.
- [Selected-model test JSON](../reports/selected-test.json): exact counts and hashes.
- [Validation comparison JSON](../reports/validation-comparison.json): fold and fit metrics.
- [Frozen model configuration](../reports/frozen-model.json), [base configuration](../reports/frozen-base.json), [logistic configuration](../reports/frozen-logistic.json).
- [Snapshot provenance](../reports/provenance.json): source locations and SHA256 hashes.

Full prediction files, adapter weights and training histories remain local under results/ and adapters/. Published snapshots are verbatim aggregate/config exports; they do not contain comment text. Scripts can regenerate detailed evidence when local inputs/weights are available. Test outcomes must not be used for another threshold search; a new development cycle informed by them needs a fresh final evaluation.
