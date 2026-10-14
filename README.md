# ml-lab

Machine-learning algorithms implemented from scratch with NumPy, each in its own folder with code, tests, results and a write-up. Every project is checked against scikit-learn so correctness is measurable, not assumed.

## Projects

| # | Project | Concepts | Status |
|---|---|---|---|
| 01 | [Linear regression](01-linear-regression) | gradient descent, feature scaling, learning rate, learning curves | code + tests done |
| 02 | Logistic regression | sigmoid, cross-entropy, regularization | planned |

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1      # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest                  # run all tests
```

## Conventions

- One folder per project: `src/`, `tests/`, `notebooks/`, `results/`, and its own `README.md`.
- Datasets are never committed. Each project README says how to get them.
- Fixed random seeds, so results can be reproduced.
