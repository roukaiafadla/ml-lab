"""Train on California Housing, compare against scikit-learn, save plots and metrics.

Run from anywhere:  python 01-linear-regression/src/run_experiment.py

Outputs (all regenerated on every run):
  results/metrics.json, results/results.md, results/*.png
and the block between <!-- RESULTS:START --> and <!-- RESULTS:END --> in the project README.
"""

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from linear_regression import (
    LinearRegressionGD,
    StandardScaler,
    gd_diagnostics,
    r2_score,
    rmse,
)
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

PROJECT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "results"
README = PROJECT / "README.md"
RESULTS.mkdir(exist_ok=True)

SEED = 42
LEARNING_RATES = [0.001, 0.01, 0.1, 0.5]
BEST_LR = 0.1
N_ITERS = 1000
ITER_STUDY = [100, 300, 1000, 3000, 10000]
DIVERGENCE_ITERS = 40


def write_text(path, text):
    """Always UTF-8 with LF line endings (Windows would otherwise write cp1252 + CRLF)."""
    path.write_text(text, encoding="utf-8", newline="\n")


def update_readme(block):
    """Replace the auto-generated results block in the project README."""
    text = README.read_text(encoding="utf-8")
    pattern = re.compile(r"(<!-- RESULTS:START -->).*?(<!-- RESULTS:END -->)", re.DOTALL)
    if not pattern.search(text):
        print("README has no RESULTS markers; skipped README update.")
        return
    write_text(README, pattern.sub(lambda m: f"{m.group(1)}\n{block}\n{m.group(2)}", text))


def sci(x):
    return f"{x:.2e}"


def row(*cells):
    """One markdown table row."""
    return "| " + " | ".join(str(c) for c in cells) + " |"


def main():
    data = fetch_california_housing()
    X, y = data.data, data.target  # y = median house value, in $100k
    names = list(data.feature_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    scaler = StandardScaler().fit(X_train)  # train statistics only: no leakage
    Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)

    # --- why scaling matters: curvature of the cost surface ----------------------------
    diag_raw, diag_scaled = gd_diagnostics(X_train), gd_diagnostics(Xtr)

    # --- learning-rate sweep -----------------------------------------------------------
    models = {
        lr: LinearRegressionGD(lr=lr, n_iters=N_ITERS).fit(Xtr, y_train) for lr in LEARNING_RATES
    }

    # --- stability demo: just below vs just above the limit 2 / lambda_max -------------
    lr_max = diag_scaled["lr_max"]
    demo = {
        f"{f:.1f} x limit (lr = {f * lr_max:.2f})": LinearRegressionGD(
            lr=f * lr_max, n_iters=DIVERGENCE_ITERS
        ).fit(Xtr, y_train)
        for f in (0.9, 1.1)
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for lr, mdl in models.items():
        ax1.plot(mdl.history_, label=f"lr = {lr}", linewidth=1.8)
    ax1.set_yscale("log")
    ax1.set_xlabel("iteration")
    ax1.set_ylabel("cost J (log scale)")
    ax1.set_title("Learning-rate sweep (training set)")
    ax1.grid(alpha=0.3)
    ax1.legend()
    for label, mdl in demo.items():
        ax2.plot(mdl.history_, label=label, linewidth=1.8)
    ax2.set_yscale("log")
    ax2.set_xlabel("iteration")
    ax2.set_title(f"Stability limit: lr < 2/λmax = {lr_max:.2f}")
    ax2.grid(alpha=0.3)
    ax2.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "learning_curve.png", dpi=150)
    plt.close(fig)

    # --- compare best run with scikit-learn and a trivial baseline ---------------------
    gd = models[BEST_LR]
    sk = LinearRegression().fit(Xtr, y_train)
    baseline_pred = np.full_like(y_test, y_train.mean())  # always predict the train mean

    def scores(model_pred_train, model_pred_test):
        return {
            "train_rmse": rmse(y_train, model_pred_train),
            "test_rmse": rmse(y_test, model_pred_test),
            "test_r2": r2_score(y_test, model_pred_test),
        }

    res = {
        "baseline_mean": scores(np.full_like(y_train, y_train.mean()), baseline_pred),
        "gradient_descent": scores(gd.predict(Xtr), gd.predict(Xte)),
        "sklearn": scores(sk.predict(Xtr), sk.predict(Xte)),
    }

    # --- how close does gradient descent get as we give it more iterations? ------------
    iter_study = []
    for n in ITER_STUDY:
        m = LinearRegressionGD(lr=BEST_LR, n_iters=n).fit(Xtr, y_train)
        iter_study.append(
            {
                "n_iters": n,
                "test_rmse": rmse(y_test, m.predict(Xte)),
                "max_abs_weight_diff": float(np.max(np.abs(m.w_ - sk.coef_))),
                "final_cost": float(m.history_[-1]),
            }
        )

    metrics = {
        "seed": SEED,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "lr": BEST_LR,
        "n_iters": N_ITERS,
        **res,
        "gradient_descent_extra": {"final_cost": float(gd.history_[-1])},
        "max_abs_weight_diff": float(np.max(np.abs(gd.w_ - sk.coef_))),
        "bias_diff": float(abs(gd.b_ - sk.intercept_)),
        "weights": {
            n: {"gd": float(a), "sklearn": float(b)}
            for n, a, b in zip(names, gd.w_, sk.coef_, strict=True)
        },
        "learning_rate_sweep": {str(lr): float(m.history_[-1]) for lr, m in models.items()},
        "conditioning": {"raw_features": diag_raw, "standardized_features": diag_scaled},
        "iteration_study": iter_study,
    }
    write_text(RESULTS / "metrics.json", json.dumps(metrics, indent=2) + "\n")

    # --- markdown block: results.md + auto-inserted into the README --------------------
    b, g, s = res["baseline_mean"], res["gradient_descent"], res["sklearn"]
    def model_row(label, r):
        return row(label, f"{r['train_rmse']:.4f}", f"{r['test_rmse']:.4f}", f"{r['test_r2']:.4f}")

    def diag_row(label, d):
        cells = (d["lam_max"], d["lam_min"], d["condition_number"], d["lr_max"])
        return row(label, *(sci(c) for c in cells))

    md = [
        row("Model", "Train RMSE ($100k)", "Test RMSE ($100k)", "Test R²"),
        row("---", "---", "---", "---"),
        model_row("Baseline: always predict the train mean", b),
        model_row(f"**Gradient descent (mine)**, lr={BEST_LR}, {N_ITERS} iters", g),
        model_row("scikit-learn `LinearRegression`", s),
        "",
        f"Max |weight difference| vs scikit-learn: {sci(metrics['max_abs_weight_diff'])} · "
        f"bias difference: {sci(metrics['bias_diff'])}",
        "",
        f"**Learning-rate sweep** (final training cost after {N_ITERS:,} iterations)",
        "",
        row("Learning rate", "Final training cost J"),
        row("---", "---"),
        *[row(lr, f"{c:.5f}") for lr, c in metrics["learning_rate_sweep"].items()],
        "",
        "**Why scaling matters** (curvature of the cost surface; stable only if lr < 2/λmax)",
        "",
        row("Features", "λmax", "λmin", "Condition number", "Largest stable lr"),
        row("---", "---", "---", "---", "---"),
        diag_row("Raw", diag_raw),
        diag_row("Standardized", diag_scaled),
        "",
        f"**Does gradient descent reach the exact solution?** (lr = {BEST_LR})",
        "",
        row("Iterations", "Test RMSE", "Max abs weight diff vs sklearn", "Final cost J"),
        row("---", "---", "---", "---"),
        *[
            row(
                f"{r['n_iters']:,}",
                f"{r['test_rmse']:.5f}",
                sci(r["max_abs_weight_diff"]),
                f"{r['final_cost']:.6f}",
            )
            for r in iter_study
        ],
    ]
    block = "\n".join(md)
    write_text(RESULTS / "results.md", block + "\n")
    update_readme(block)

    # --- weight comparison plot --------------------------------------------------------
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

    # --- predicted vs actual -----------------------------------------------------------
    gd_pred = gd.predict(Xte)
    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    ax.scatter(y_test, gd_pred, s=4, alpha=0.35)
    lim = [min(0, float(gd_pred.min())) - 0.1, 5.3]
    ax.plot(lim, lim, "k--", linewidth=1)
    ax.axvline(5.0, color="crimson", linewidth=1, linestyle=":")
    ax.annotate("true value capped\nat 5.0 ($500k)", xy=(5.0, 0.4), xytext=(3.0, 0.2), fontsize=8,
                color="crimson", arrowprops={"arrowstyle": "->", "color": "crimson"})
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
