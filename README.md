# ml-lab

[![tests](https://github.com/roukaiafadla/ml-lab/actions/workflows/tests.yml/badge.svg)](https://github.com/roukaiafadla/ml-lab/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

Machine-learning algorithms implemented **from scratch with NumPy**. Every project has its own folder with code, unit tests, reproducible experiments and a write-up, and every result is checked against scikit-learn, so correctness is measured, not assumed.

## Projects

| # | Project | What it shows | Key result | Status |
|---|---|---|---|---|
| 01 | [Linear regression](01-linear-regression) | feature scaling, vectorized gradient descent, learning-rate stability, learning curves | Test RMSE 0.7457 vs 0.7456 for scikit-learn (California Housing) | done |
| 02 | Logistic regression | sigmoid, cross-entropy, regularization, precision/recall | | planned |

## Quick start

```bash
git clone https://github.com/roukaiafadla/ml-lab.git
cd ml-lab
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
python -m pytest                   # run all tests
python 01-linear-regression/src/run_experiment.py   # reproduce project 01
```

## Conventions

- One folder per project: `src/`, `tests/`, `notebooks/`, `results/` and its own `README.md`.
- Every algorithm is tested against scikit-learn and against numerical gradients.
- Fixed random seeds, so every number in the READMEs can be reproduced.
- Datasets are never committed; each project README says where they come from.
- Code is linted with [ruff](https://docs.astral.sh/ruff/) and tested on Python 3.11 to 3.13 in CI.

## License

[MIT](LICENSE)
