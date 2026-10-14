"""Linear regression from scratch with NumPy: z-score scaling + batch gradient descent."""

import numpy as np


class StandardScaler:
    """Z-score scaling. Fit on train only, then transform train and test."""

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0  # constant column: avoid division by zero
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class LinearRegressionGD:
    """Ordinary least squares fitted with vectorized batch gradient descent.

    Cost: J(w, b) = (1 / 2m) * sum((X @ w + b - y) ** 2)
    """

    def __init__(self, lr=0.1, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters

    def fit(self, X, y):
        m, n = X.shape
        self.w_ = np.zeros(n)
        self.b_ = 0.0
        self.history_ = np.empty(self.n_iters)

        for i in range(self.n_iters):
            error = X @ self.w_ + self.b_ - y
            self.history_[i] = (error @ error) / (2 * m)
            self.w_ -= self.lr * (X.T @ error) / m
            self.b_ -= self.lr * error.mean()
        return self

    def predict(self, X):
        return X @ self.w_ + self.b_


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot)
