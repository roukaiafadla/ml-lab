"""Linear regression from scratch with NumPy: z-score scaling + batch gradient descent."""

import numpy as np


class StandardScaler:
    """Z-score scaling. Fit on train only, then transform train and test."""

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0  # constant column: avoid division by zero
        return self

    def transform(self, X):
        if not hasattr(self, "mean_"):
            raise RuntimeError("Call fit() before transform().")
        return (np.asarray(X, dtype=float) - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class LinearRegressionGD:
    """Ordinary least squares fitted with vectorized batch gradient descent.

    Cost: J(w, b) = (1 / 2m) * sum((X @ w + b - y) ** 2)

    Parameters
    ----------
    lr : float
        Learning rate (alpha).
    n_iters : int
        Maximum number of iterations.
    tol : float or None
        If set, stop early once the cost decreases by less than ``tol`` in one iteration.
        A cost that goes *up* never triggers the stop, so divergence stays visible.

    Attributes (after fit)
    ----------------------
    w_, b_ : learned weights and bias
    history_ : cost J at the start of every iteration
    n_iter_ : number of iterations actually run
    """

    def __init__(self, lr=0.1, n_iters=1000, tol=None):
        self.lr = lr
        self.n_iters = n_iters
        self.tol = tol

    def _gradients(self, X, y):
        """Return (dJ/dw, dJ/db, J) at the current parameters."""
        m = X.shape[0]
        error = X @ self.w_ + self.b_ - y
        cost = (error @ error) / (2 * m)
        return X.T @ error / m, error.mean(), cost

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
            raise ValueError("X must be (m, n) and y must be (m,) with the same m.")

        self.w_ = np.zeros(X.shape[1])
        self.b_ = 0.0
        history = []

        for _ in range(self.n_iters):
            dw, db, cost = self._gradients(X, y)
            history.append(cost)
            if self.tol is not None and len(history) > 1:
                decrease = history[-2] - history[-1]
                if 0 <= decrease < self.tol:
                    break
            # Both gradients were computed from the same (old) parameters,
            # so this is a simultaneous update.
            self.w_ -= self.lr * dw
            self.b_ -= self.lr * db

        self.history_ = np.array(history)
        self.n_iter_ = len(history)
        return self

    def predict(self, X):
        if not hasattr(self, "w_"):
            raise RuntimeError("Call fit() before predict().")
        return np.asarray(X, dtype=float) @ self.w_ + self.b_


def gd_diagnostics(X):
    """Curvature of the cost surface, which decides how gradient descent behaves.

    The Hessian of J is H = Xa^T Xa / m, with Xa = [X, 1] (the column of ones is the bias).
    Gradient descent is stable only if lr < 2 / lambda_max, and its speed is limited by the
    condition number lambda_max / lambda_min: the larger it is, the longer and thinner
    the valley the optimizer has to travel along.
    """
    X = np.asarray(X, dtype=float)
    m = X.shape[0]
    Xa = np.column_stack([X, np.ones(m)])
    eig = np.linalg.eigvalsh(Xa.T @ Xa / m)  # ascending
    lam_min, lam_max = max(float(eig[0]), 1e-300), float(eig[-1])
    return {
        "lam_max": lam_max,
        "lam_min": lam_min,
        "condition_number": lam_max / lam_min,
        "lr_max": 2.0 / lam_max,
    }


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot)
