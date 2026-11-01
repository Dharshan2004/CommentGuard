# CommentGuard

**Classify English comments for toxicity, locally on your Mac.**

CommentGuard runs a fine-tuned Qwen3-4B model on Apple Silicon with MLX. Give it a comment and get a binary label, toxicity score, and decision threshold as JSON. The model runs locally after the first download—no inference API or API key is required.

[**Download model on Hugging Face**](https://huggingface.co/dharshannn/CommentGuard-Qwen3-4B-MLX-LoRA) · [Model card](MODEL_CARD.md) · [Benchmarks](docs/RESULTS.md) · [Contributing](CONTRIBUTING.md)

## Features

- **Local inference:** a 4-bit Qwen3 base with a lightweight MLX LoRA adapter.
- **Simple command line:** pass a comment and receive a machine-readable result.
- **Published weights:** downloads a pinned release from Hugging Face automatically.
- **Documented behavior:** fixed scoring configuration, held-out benchmarks, and reproducible evaluation tools.

## Quick start

Requires an **Apple Silicon Mac** and **Python 3.13** for the recorded inference environment. Tested on an M5 Pro with 24 GB unified memory. Linux/CUDA and Intel Macs are not supported by this inference path.

```bash
git clone https://github.com/Dharshan2004/CommentGuard.git
cd CommentGuard

python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-inference.txt

python scripts/classify.py "Thanks for explaining that clearly."
```

The first run downloads the base model and adapter; allow several GB of disk space. Subsequent runs reuse the Hugging Face cache. The command does not send comment text to an inference service.

Example output:

```json
{
  "label": 0,
  "toxic_score": 0.0035936038475483656,
  "threshold": 0.4073334336280823,
  "truncated": false
}
```

`label` is `0` for non-toxic and `1` for toxic. Scores at or above `threshold` receive label `1`. The score is a relative preference between the two labels, **not a calibrated probability**. `truncated` indicates whether the comment was shortened to fit the context window. Small numerical differences can occur across runtime versions.

## Model

The published [CommentGuard-Qwen3-4B-MLX-LoRA](https://huggingface.co/dharshannn/CommentGuard-Qwen3-4B-MLX-LoRA) release includes adapter weights, tokenizer assets, scoring configuration, an inference example, and evaluation results.

| Component | Details |
|---|---|
| Base | Qwen3-4B-Instruct-2507, 4-bit MLX |
| Adapter | QLoRA, rank 8, alpha 16 |
| Context | 1,024 tokens |
| Training data | 5,000 Civil Comments examples |
| Output | Binary label and score |
| Runtime | MLX on Apple Silicon |

This is a downloadable native MLX adapter, not a merged Transformers model or a hosted API. The command uses the pinned adapter and base revisions in the [release metadata](reports/model-release.json). Ordinary chat generation is a different scoring method and should not be assumed to reproduce the benchmark results.

## Benchmarks

Evaluated on **3,000 held-out Civil Comments examples**, including 239 toxic comments. Each model's threshold was selected on validation and fixed before its test scoring.

| Model | Precision | Recall | F1 | Average precision |
|---|---:|---:|---:|---:|
| Base Qwen, same scoring protocol | 17.4% | 64.0% | 0.274 | 0.189 |
| TF-IDF + logistic regression | 24.9% | 58.6% | 0.350 | 0.366 |
| **CommentGuard** | **64.2%** | **60.7%** | **0.624** | **0.681** |

Compared with this logistic baseline, CommentGuard detected five more toxic comments and produced 341 fewer false positives. Median local scoring latency was approximately 146 ms/comment versus 0.17 ms for logistic regression; these are sequential local measurements, not production throughput guarantees.

Baselines were evaluated after CommentGuard's test result was known and were not exhaustively optimized. Base-model results apply to this prompt and scoring protocol. See [benchmark methodology and full results](docs/RESULTS.md) for thresholds, confusion counts, timing and limitations.

## Limitations

CommentGuard missed **94 of 239 toxic comments** in the test set and incorrectly flagged 81 non-toxic comments. It should not be used as the sole basis for automated enforcement. Performance on other domains, languages, dialects and demographic groups has not been established. Labels reflect crowd annotations in Civil Comments, not a universal definition of harm.

## Training and evaluation

For data preparation, Unsloth Studio configurations, checkpoint selection and evaluation commands, see [the reproduction guide](docs/REPRODUCING.md). The [model card](MODEL_CARD.md) describes the exact inference protocol, and [dataset documentation](docs/DATASET.md) covers labels and splits. Aggregate results and provenance are in `reports/`; datasets and weights are excluded from Git.

```text
scripts/           Classification, data preparation and evaluation commands
configs/training/  Unsloth Studio training configurations
reports/           Aggregate metrics, frozen configurations and release metadata
tests/             CPU-only metric and threshold checks
docs/              Dataset, benchmark and reproduction documentation
```

## Development

CPU checks run without downloading models or datasets:

```bash
python -m pip install -r requirements-test.txt
make test check
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## Acknowledgments

Built with [MLX LM](https://github.com/ml-explore/mlx-lm), [Unsloth](https://github.com/unslothai/unsloth), [Hugging Face Datasets](https://github.com/huggingface/datasets), and [scikit-learn](https://github.com/scikit-learn/scikit-learn), using Qwen3 and Civil Comments.

## License

Project code is [MIT-licensed](LICENSE). The published adapter is Apache-2.0; dataset and base-model terms remain separate.
