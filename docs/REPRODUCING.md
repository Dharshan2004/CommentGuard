# Reproducing CommentGuard

Run commands from the repository root. Recorded data and model revisions matter. No script here promises bit-for-bit GPU training reproduction across Studio/MLX versions. Exact published predictions require the original adapter, available from [Hugging Face](https://huggingface.co/dharshannn/CommentGuard-Qwen3-4B-MLX-LoRA).

## 1. CPU checks

Python3.11+; no dataset or GPU needed:

```bash
python3 -m venv .venv-test
source .venv-test/bin/activate
python -m pip install -r requirements-test.txt
make test check
```

GitHub Actions runs these CPU checks. Local checks have been run; a passing remote workflow is not claimed before push.

## 2. Data preparation

Historical preparation used Python3.9 and scikit-learn1.6.1. Use a separate environment; requirements-lock.txt records that full historical environment. Do not install it together with requirements-eval.txt.

```bash
python3.9 -m venv .venv-data
source .venv-data/bin/activate
python -m pip install -r requirements.txt
python scripts/prepare_data.py --revision f2970eb3a55777454c94069077cc8d9b5866312d
python scripts/verify_data.py
python scripts/prepare_pilot002.py
python scripts/prepare_main25.py
```

Preparation downloads the full source and may take substantial space/time. The scripts verify hashes/overlap and keep held-out records separate. `main5000_mix25_train.jsonl` is the main training pool, not the original differently balanced `train.jsonl`. Generated files are ignored by Git. Existing experiments should not be overwritten casually; use a fresh checkout for a reproduction.

## 3. Train in Unsloth Studio on Apple Silicon

Load `configs/training/QLoRA_Main5000_Config.yaml`. Select `mlx-community/Qwen3-4B-Instruct-2507-4bit`; upload `data/prepared/main5000_mix25_train.jsonl` and `data/prepared/valid.jsonl`. Confirm1epoch, context1024, LR1e-4, batch1, gradient accumulation1, rank8, alpha16, dropout0, warmup50, save500, eval fraction0.1. Model/dataset paths are chosen separately from YAML.

Training is manual in Studio. Save/export the native MLX LoRA checkpoint into `adapters/main5000-step4000/`. The published run selected checkpoint4,000 by its minimum validation token loss. A new training run is a new model: choose its checkpoint on validation and document a different hash rather than relabeling it as the original. Native adapters need their config plus tokenizer/template assets used by the scorer.

The other YAMLs retain smoke/pilot and matched two-epoch experiments. The run monitor scripts have historical run IDs as defaults and depend on Studio's database schema; they are optional local experiment utilities, not the portable training interface.

## 4. Prepare MLX evaluation

Recorded inference environment: Python3.13 on Apple Silicon. Install requirements-eval.txt in a separate compatible environment, or use the Studio environment with those recorded versions. This is not a CUDA/Linux inference setup.

```bash
python3.13 -m venv .venv-eval
source .venv-eval/bin/activate
python -m pip install -r requirements-eval.txt
python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("mlx-community/Qwen3-4B-Instruct-2507-4bit", revision="50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b"))'
```

Set `MODEL_PATH` to the exact snapshot path printed above. Its final directory must be the pinned revision. The scorer checks revision/template identity.

```bash
export MODEL_PATH="/absolute/path/to/snapshots/50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b"
python scripts/score_labels.py --model-path "$MODEL_PATH" --adapters main5000-step4000 --output results/validation-main5000-step4000
python scripts/select_threshold.py --scores results/validation-main5000-step4000
```

`score_labels.py` implements the precise inference protocol in MODEL_CARD.md. `evaluate.py` is the earlier free-generation pilot evaluator, not the final scoring protocol. Do not compare its metrics as though the methods were identical.

## 5. Frozen test evaluation

Download the original adapter into the expected directory:

```bash
python - <<'PYDOWNLOAD'
from huggingface_hub import snapshot_download
snapshot_download("dharshannn/CommentGuard-Qwen3-4B-MLX-LoRA",
                  revision="ef21f5d40dcfb7fb7268b6f3101711281422a2d0",
                  local_dir="adapters/main5000-step4000")
PYDOWNLOAD
```

For the **original adapter only**, verify its hash against reports/frozen-model.json. Test commands require a fresh output directory and reject an incompatible adapter/config. Copy the pre-test snapshot into its expected local location:

```bash
mkdir -p results/iterative-selection
cp reports/frozen-model.json results/iterative-selection/SELECTED_CONFIG.json
python scripts/score_labels.py --model-path "$MODEL_PATH" --adapters main5000-step4000 --split test --frozen-config results/iterative-selection/SELECTED_CONFIG.json --output results/test-main5000-step4000
python scripts/report_frozen_test.py --scores results/test-main5000-step4000
```

The report also needs the validation manifest generated in step4. For a newly trained adapter, freeze its own validation-selected config before test; do not bypass the original hash check or choose a threshold from test metrics.

## 6. Baseline comparison

Use the same prompt/label scoring for base Qwen. Fit its threshold on validation, then freeze the candidate before test:

```bash
python scripts/score_labels.py --model-path "$MODEL_PATH" --adapters base --output results/validation-base-labels
python scripts/select_threshold.py --scores results/validation-base-labels
python - <<'PY'
import json
from pathlib import Path
p=Path('results/validation-base-labels')
c=json.loads((p/'candidate_configs.json').read_text())['base']
c['validation_manifest']='results/validation-base-labels/manifest.json'
(p/'FROZEN_CONFIG.json').write_text(json.dumps(c,indent=2)+'\n')
PY
python scripts/score_labels.py --model-path "$MODEL_PATH" --adapters base --split test --frozen-config results/validation-base-labels/FROZEN_CONFIG.json --output results/test-base-labels
python scripts/evaluate_logistic_test.py
mkdir -p results/test-baseline-comparison
python scripts/report_test_baselines.py
```

Logistic uses fixed historical hyperparameters, fits training-only vocabulary/IDF, then selects its threshold on validation before accessing test. The comparator checks row/label identity, hashes and matching LLM inference metadata. It reports primary validation-selected and secondary fixed0.5 metrics without searching test thresholds.

`make public-results` copies an explicit aggregate/config allowlist into reports/ after all local runs exist. It includes the historical validation experiment ledger; regenerate that ledger with report_experiment_leaderboard.py only if all its required candidates were reproduced. Public snapshots retain their original provenance and should not be overwritten with incomplete reruns.
