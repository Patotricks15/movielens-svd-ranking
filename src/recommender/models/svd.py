"""Classical PureSVD collaborative filtering.

PureSVD applies truncated singular-value decomposition directly to the sparse
user-item interaction matrix. It is linear algebra, not a neural network, and
has no optimizer, hidden layers, or deep-learning dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

from recommender.data.splitting import Split


@dataclass
class SVDRecommender:
    """Low-rank reconstruction of the implicit interaction matrix."""

    n_factors: int = 32
    random_state: int = 42
    user_factors: np.ndarray | None = None
    singular_values: np.ndarray | None = None
    item_factors: np.ndarray | None = None
    popularity: np.ndarray | None = None
    reconstruction_rmse_: float | None = None
    n_users: int = 0
    n_items: int = 0

    def fit(self, split: Split) -> SVDRecommender:
        """Fit a truncated SVD on the binary training matrix."""
        maximum_rank = min(split.n_users, split.n_items) - 1
        if not 1 <= self.n_factors <= maximum_rank:
            raise ValueError(
                f"n_factors must be between 1 and {maximum_rank}, got {self.n_factors}."
            )
        users, items = split.train_matrix_entries()
        matrix = csr_matrix(
            (np.ones(users.size), (users, items)),
            shape=(split.n_users, split.n_items),
            dtype=np.float64,
        )
        user_factors, singular_values, item_factors = svds(
            matrix,
            k=self.n_factors,
            which="LM",
            random_state=self.random_state,
        )
        order = np.argsort(singular_values)[::-1]
        self.user_factors = user_factors[:, order]
        self.singular_values = singular_values[order]
        self.item_factors = item_factors[order, :]
        self.popularity = np.asarray(matrix.sum(axis=0)).ravel()
        self.n_users = split.n_users
        self.n_items = split.n_items

        reconstruction = (
            self.user_factors * self.singular_values
        ) @ self.item_factors
        self.reconstruction_rmse_ = float(
            np.sqrt(np.mean((matrix.toarray() - reconstruction) ** 2))
        )
        return self

    def _check_fitted(self) -> None:
        if any(
            value is None
            for value in (
                self.user_factors,
                self.singular_values,
                self.item_factors,
                self.popularity,
            )
        ):
            raise RuntimeError("Model is not fitted; call fit() first.")

    def score_user(self, user_index: int) -> np.ndarray:
        """Score every item, with popularity as the cold-user fallback."""
        self._check_fitted()
        assert self.user_factors is not None
        assert self.singular_values is not None
        assert self.item_factors is not None
        assert self.popularity is not None
        if not 0 <= user_index < self.n_users:
            return self.popularity.copy()
        return (self.user_factors[user_index] * self.singular_values) @ self.item_factors

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
