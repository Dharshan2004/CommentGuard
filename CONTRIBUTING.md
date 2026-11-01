# Contributing

Contributions to data validation, evaluation, documentation and reproducibility are welcome.

## Development setup

Use Python 3.11 or later for CPU checks:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-test.txt
make test check
```

MLX inference requires Apple Silicon and the separate evaluation environment described in [the reproduction guide](docs/REPRODUCING.md). The CPU checks do not download models or datasets.

## Changes

Keep changes focused and explain the problem, approach and verification in the pull request. Add a regression test when changing metric calculation or threshold selection. For bug reports, include the command, Python/package versions, expected behavior and error message; omit private data and credentials.

Do not commit generated comments, weights, fitted model binaries or row-level predictions. Public results belong in `reports/` and must have reproducible provenance. If you change an evaluation protocol, document the difference rather than comparing incompatible metrics.

## Experiments

Fit preprocessing on training data only. Select models and thresholds on validation, then freeze the configuration before final evaluation. The published test set has already been evaluated: experiments informed by those results need a fresh final evaluation for an independent performance claim.

Report negative results and inference costs alongside quality metrics. Include dataset/model revisions, seeds, configuration and limitations. Avoid overwriting historical aggregate snapshots with incomplete runs.
