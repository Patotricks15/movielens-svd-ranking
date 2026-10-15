"""Biased matrix factorization trained with regularized SGD.

This is the classical Funk-SVD formulation popularized by the Netflix Prize.
It is intentionally implemented with NumPy rather than a neural framework:
the model is a linear latent-factor estimator with user/item biases.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from recommender.data.splitting import Split


@dataclass
class SVDRecommender:
    """Explicit-feedback matrix factorization used for top-N ranking."""

    n_factors: int = 32
    epochs: int = 30
    learning_rate: float = 0.01
    regularization: float = 0.05
    learning_rate_decay: float = 0.95
    random_state: int = 42
    user_factors: np.ndarray | None = None
    item_factors: np.ndarray | None = None
    user_bias: np.ndarray | None = None
    item_bias: np.ndarray | None = None
    global_mean: float = 0.0
    train_rmse_: list[float] | None = None
    n_users: int = 0
    n_items: int = 0

    def fit(self, split: Split) -> SVDRecommender:
        """Fit latent factors using only the training partition."""
        if self.n_factors < 1 or self.epochs < 1:
            raise ValueError("n_factors and epochs must be positive.")

        frame = split.train
        users = frame["user_index"].to_numpy(dtype=np.int64)
        items = frame["item_index"].to_numpy(dtype=np.int64)
        ratings = frame["rating"].to_numpy(dtype=np.float64)
        if ratings.size == 0:
            raise ValueError("Cannot fit SVD on an empty training partition.")

        self.n_users = split.n_users
        self.n_items = split.n_items
        self.global_mean = float(ratings.mean())
        rng = np.random.default_rng(self.random_state)
        scale = 0.1 / np.sqrt(self.n_factors)
        self.user_factors = rng.normal(0.0, scale, (self.n_users, self.n_factors))
        self.item_factors = rng.normal(0.0, scale, (self.n_items, self.n_factors))
        self.user_bias = np.zeros(self.n_users, dtype=np.float64)
        self.item_bias = np.zeros(self.n_items, dtype=np.float64)
        self.train_rmse_ = []

        order = np.arange(ratings.size)
        lr = self.learning_rate
        for _ in range(self.epochs):
            rng.shuffle(order)
            squared_error = 0.0
            for index in order:
                user = users[index]
                item = items[index]
                prediction = self._predict_known(user, item)
                error = ratings[index] - prediction
                squared_error += error * error

                old_user = self.user_factors[user].copy()
                self.user_bias[user] += lr * (
                    error - self.regularization * self.user_bias[user]
                )
                self.item_bias[item] += lr * (
                    error - self.regularization * self.item_bias[item]
                )
                self.user_factors[user] += lr * (
                    error * self.item_factors[item]
                    - self.regularization * self.user_factors[user]
                )
                self.item_factors[item] += lr * (
                    error * old_user - self.regularization * self.item_factors[item]
                )

            self.train_rmse_.append(float(np.sqrt(squared_error / ratings.size)))
            lr *= self.learning_rate_decay
        return self

    def _check_fitted(self) -> None:
        if any(
            value is None
            for value in (
                self.user_factors,
                self.item_factors,
                self.user_bias,
                self.item_bias,
            )
        ):
            raise RuntimeError("Model is not fitted; call fit() first.")

    def _predict_known(self, user: int, item: int) -> float:
        self._check_fitted()
        assert self.user_factors is not None
        assert self.item_factors is not None
        assert self.user_bias is not None
        assert self.item_bias is not None
        return float(
            self.global_mean
            + self.user_bias[user]
            + self.item_bias[item]
            + self.user_factors[user] @ self.item_factors[item]
        )

    def predict(self, user_index: int, item_index: int) -> float:
        """Predict a rating, falling back to learned biases for cold entities."""
        self._check_fitted()
        if not 0 <= item_index < self.n_items:
            raise IndexError(f"item_index {item_index} is outside the catalogue.")
        assert self.item_bias is not None
        if not 0 <= user_index < self.n_users:
            return float(self.global_mean + self.item_bias[item_index])
        return self._predict_known(user_index, item_index)

    def score_user(self, user_index: int) -> np.ndarray:
        """Score the complete item catalogue for one user."""
        self._check_fitted()
        assert self.user_factors is not None
        assert self.item_factors is not None
        assert self.user_bias is not None
        assert self.item_bias is not None
        if not 0 <= user_index < self.n_users:
            return self.global_mean + self.item_bias.copy()
        return (
            self.global_mean
            + self.user_bias[user_index]
            + self.item_bias
            + self.item_factors @ self.user_factors[user_index]
        )

    def recommend(
        self, user_index: int, k: int = 10, exclude: set[int] | None = None
    ) -> np.ndarray:
        """Return top-scoring unseen item indices."""
        if k < 1:
            raise ValueError("k must be positive.")
        scores = self.score_user(user_index).copy()
        if exclude:
            scores[list(exclude)] = -np.inf
        k = min(k, int(np.isfinite(scores).sum()))
        if k == 0:
            return np.array([], dtype=np.int64)
        top = np.argpartition(-scores, k - 1)[:k]
        return top[np.argsort(-scores[top], kind="stable")]
