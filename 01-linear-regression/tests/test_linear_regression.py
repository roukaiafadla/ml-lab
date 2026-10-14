import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from linear_regression import LinearRegressionGD, StandardScaler, r2_score, rmse


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    true_w, true_b = np.array([2.0, -1.0, 0.5]), 3.0
    y = X @ true_w + true_b + rng.normal(scale=0.05, size=200)
    return X, y


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


def test_too_large_learning_rate_diverges(toy):
    X, y = toy
    m = LinearRegressionGD(lr=5.0, n_iters=50).fit(X, y)
    assert m.history_[-1] > m.history_[0]


def test_metrics():
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, y) == 0.0
    assert r2_score(y, y) == 1.0
    assert r2_score(y, np.full(3, y.mean())) == pytest.approx(0.0)
