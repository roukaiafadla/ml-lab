import numpy as np
import pytest
from linear_regression import (
    LinearRegressionGD,
    StandardScaler,
    gd_diagnostics,
    r2_score,
    rmse,
)
from sklearn.linear_model import LinearRegression


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    true_w, true_b = np.array([2.0, -1.0, 0.5]), 3.0
    y = X @ true_w + true_b + rng.normal(scale=0.05, size=200)
    return X, y


# --- scaler ---------------------------------------------------------------------------


def test_scaler_zero_mean_unit_std(toy):
    X, _ = toy
    Z = StandardScaler().fit_transform(X * 10 + 5)
    assert np.allclose(Z.mean(axis=0), 0, atol=1e-9)
    assert np.allclose(Z.std(axis=0), 1, atol=1e-9)


def test_scaler_uses_train_statistics_only(toy):
    X, _ = toy
    sc = StandardScaler().fit(X[:100])
    assert np.allclose(sc.mean_, X[:100].mean(axis=0))


def test_scaler_constant_column_does_not_blow_up():
    X = np.column_stack([np.ones(10), np.arange(10.0)])
    Z = StandardScaler().fit_transform(X)
    assert np.all(np.isfinite(Z))


def test_scaler_transform_before_fit_raises(toy):
    X, _ = toy
    with pytest.raises(RuntimeError):
        StandardScaler().transform(X)


# --- model ----------------------------------------------------------------------------


def test_cost_decreases_monotonically(toy):
    X, y = toy
    m = LinearRegressionGD(lr=0.1, n_iters=200).fit(X, y)
    assert np.all(np.diff(m.history_) <= 1e-12)


def test_matches_sklearn(toy):
    X, y = toy
    gd = LinearRegressionGD(lr=0.1, n_iters=2000).fit(X, y)
    sk = LinearRegression().fit(X, y)
    assert np.allclose(gd.w_, sk.coef_, atol=1e-4)
    assert gd.b_ == pytest.approx(sk.intercept_, abs=1e-4)


def test_gradients_match_numerical_gradients(toy):
    """Analytic gradients must agree with finite differences of the cost."""
    X, y = toy
    model = LinearRegressionGD()
    model.w_, model.b_ = np.array([0.3, -0.2, 0.8]), 0.1
    dw, db, _ = model._gradients(X, y)

    def cost(w, b):
        e = X @ w + b - y
        return e @ e / (2 * len(y))

    eps = 1e-6
    for j in range(3):
        up, down = model.w_.copy(), model.w_.copy()
        up[j] += eps
        down[j] -= eps
        numeric = (cost(up, model.b_) - cost(down, model.b_)) / (2 * eps)
        assert dw[j] == pytest.approx(numeric, rel=1e-5, abs=1e-8)
    numeric_b = (cost(model.w_, model.b_ + eps) - cost(model.w_, model.b_ - eps)) / (2 * eps)
    assert db == pytest.approx(numeric_b, rel=1e-5, abs=1e-8)


def test_too_large_learning_rate_diverges(toy):
    X, y = toy
    m = LinearRegressionGD(lr=5.0, n_iters=50).fit(X, y)
    assert m.history_[-1] > m.history_[0]


def test_stability_threshold_is_two_over_lambda_max(toy):
    """Just below 2/lambda_max the cost falls; just above it, it blows up."""
    X, y = toy
    lr_max = gd_diagnostics(X)["lr_max"]
    stable = LinearRegressionGD(lr=0.95 * lr_max, n_iters=300).fit(X, y)
    unstable = LinearRegressionGD(lr=1.05 * lr_max, n_iters=300).fit(X, y)
    assert stable.history_[-1] < stable.history_[0]
    assert unstable.history_[-1] > unstable.history_[0]


def test_tol_stops_early_and_history_matches(toy):
    X, y = toy
    m = LinearRegressionGD(lr=0.1, n_iters=5000, tol=1e-9).fit(X, y)
    assert m.n_iter_ < 5000
    assert len(m.history_) == m.n_iter_


def test_predict_before_fit_raises(toy):
    X, _ = toy
    with pytest.raises(RuntimeError):
        LinearRegressionGD().predict(X)


def test_fit_rejects_mismatched_shapes(toy):
    X, y = toy
    with pytest.raises(ValueError):
        LinearRegressionGD().fit(X, y[:-1])


# --- helpers --------------------------------------------------------------------------


def test_diagnostics_scaling_improves_conditioning():
    rng = np.random.default_rng(1)
    X = np.column_stack([rng.normal(size=500) * 1000 + 5000, rng.normal(size=500)])
    raw = gd_diagnostics(X)
    scaled = gd_diagnostics(StandardScaler().fit_transform(X))
    assert scaled["condition_number"] < raw["condition_number"] / 100
    assert scaled["lr_max"] > raw["lr_max"]


def test_metrics():
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, y) == 0.0
    assert r2_score(y, y) == 1.0
    assert r2_score(y, np.full(3, y.mean())) == pytest.approx(0.0)
