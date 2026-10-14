"""Train on California Housing, compare against scikit-learn, save plots and metrics.

Run from the project folder:  python src/run_experiment.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from linear_regression import LinearRegressionGD, StandardScaler, r2_score, rmse

RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)

SEED = 42
LEARNING_RATES = [0.001, 0.01, 0.1, 0.5]
BEST_LR = 0.1
N_ITERS = 1000


def main():
    data = fetch_california_housing()
    X, y = data.data, data.target  # y = median house value, in $100k
    names = list(data.feature_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    scaler = StandardScaler().fit(X_train)  # train statistics only: no leakage
    Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)

    # --- learning-rate sweep -------------------------------------------------
    models = {}
    for lr in LEARNING_RATES:
        models[lr] = LinearRegressionGD(lr=lr, n_iters=N_ITERS).fit(Xtr, y_train)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for lr, mdl in models.items():
        ax.plot(mdl.history_, label=f"lr = {lr}", linewidth=1.8)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("cost J (log scale)")
    ax.set_title("Learning curves on the training set")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "learning_curve.png", dpi=150)
    plt.close(fig)

    # --- compare best run with scikit-learn ---------------------------------
    gd = models[BEST_LR]
    sk = LinearRegression().fit(Xtr, y_train)

    gd_pred, sk_pred = gd.predict(Xte), sk.predict(Xte)
    metrics = {
        "seed": SEED,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "lr": BEST_LR,
        "n_iters": N_ITERS,
        "gradient_descent": {
            "test_rmse": rmse(y_test, gd_pred),
            "test_r2": r2_score(y_test, gd_pred),
            "final_cost": float(gd.history_[-1]),
        },
        "sklearn": {
            "test_rmse": rmse(y_test, sk_pred),
            "test_r2": r2_score(y_test, sk_pred),
        },
        "max_abs_weight_diff": float(np.max(np.abs(gd.w_ - sk.coef_))),
        "bias_diff": float(abs(gd.b_ - sk.intercept_)),
        "weights": {
            n: {"gd": float(a), "sklearn": float(b)}
            for n, a, b in zip(names, gd.w_, sk.coef_)
        },
        "learning_rate_sweep": {
            str(lr): float(m.history_[-1]) for lr, m in models.items()
        },
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # markdown tables, ready to paste into the README
    g, s = metrics["gradient_descent"], metrics["sklearn"]
    md = [
        "| Model | Test RMSE ($100k) | Test R² |",
        "|---|---|---|",
        f"| Gradient descent (mine, lr={BEST_LR}, {N_ITERS} iters) | {g['test_rmse']:.4f} | {g['test_r2']:.4f} |",
        f"| scikit-learn `LinearRegression` | {s['test_rmse']:.4f} | {s['test_r2']:.4f} |",
        "",
        f"Max |weight difference| vs scikit-learn: {metrics['max_abs_weight_diff']:.2e}  ",
        f"Bias difference: {metrics['bias_diff']:.2e}",
        "",
        "| Learning rate | Final training cost J |",
        "|---|---|",
        *[f"| {lr} | {c:.5f} |" for lr, c in metrics["learning_rate_sweep"].items()],
    ]
    (RESULTS / "results.md").write_text("\n".join(md) + "\n")

    # --- weight comparison plot ---------------------------------------------
    idx = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(idx - 0.2, gd.w_, 0.4, label="gradient descent (mine)")
    ax.bar(idx + 0.2, sk.coef_, 0.4, label="scikit-learn")
    ax.set_xticks(idx)
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_ylabel("weight (standardized features)")
    ax.set_title("Learned weights: mine vs scikit-learn")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "weights_comparison.png", dpi=150)
    plt.close(fig)

    # --- predicted vs actual -------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_test, gd_pred, s=4, alpha=0.35)
    lim = [0, 5.2]
    ax.plot(lim, lim, "k--", linewidth=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("actual ($100k)")
    ax.set_ylabel("predicted ($100k)")
    ax.set_title("Test set: predicted vs actual")
    fig.tight_layout()
    fig.savefig(RESULTS / "predicted_vs_actual.png", dpi=150)
    plt.close(fig)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
