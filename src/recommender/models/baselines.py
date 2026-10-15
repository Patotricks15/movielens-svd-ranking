"""Classical baselines for the MovieLens ranking experiment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from recommender.data.splitting import Split


def _top_k(scores: np.ndarray, k: int, exclude: set[int] | None) -> np.ndarray:
    if k < 1:
        raise ValueError("k must be positive.")
    ranked = scores.copy()
    if exclude:
        ranked[list(exclude)] = -np.inf
    k = min(k, int(np.isfinite(ranked).sum()))
    if k == 0:
        return np.array([], dtype=np.int64)
    top = np.argpartition(-ranked, k - 1)[:k]
    return top[np.argsort(-ranked[top], kind="stable")]


@dataclass
class PopularityRecommender:
    item_scores: np.ndarray | None = None
    n_items: int = 0

    def fit(self, split: Split) -> PopularityRecommender:
        self.n_items = split.n_items
        self.item_scores = np.bincount(
            split.train["item_index"].to_numpy(dtype=np.int64),
            minlength=self.n_items,
        ).astype(np.float64)
        return self

    def score_user(self, user_index: int) -> np.ndarray:  # noqa: ARG002
        if self.item_scores is None:
            raise RuntimeError("Model is not fitted; call fit() first.")
        return self.item_scores

    def recommend(
        self, user_index: int, k: int = 10, exclude: set[int] | None = None
    ) -> np.ndarray:
        return _top_k(self.score_user(user_index), k, exclude)


@dataclass
class ItemKNNRecommender:
    top_k_neighbours: int = 50
    shrinkage: float = 10.0
    similarity: np.ndarray | None = None
    user_items: dict[int, np.ndarray] | None = None
    n_items: int = 0

    def fit(self, split: Split) -> ItemKNNRecommender:
        users = split.train["user_index"].to_numpy(dtype=np.int64)
        items = split.train["item_index"].to_numpy(dtype=np.int64)
        matrix = np.zeros((split.n_items, split.n_users), dtype=np.float32)
        matrix[items, users] = 1.0
        cooccurrence = matrix @ matrix.T
        norms = np.sqrt(np.diag(cooccurrence))
        denominator = np.outer(norms, norms) + self.shrinkage
        self.similarity = np.divide(
            cooccurrence,
            denominator,
            out=np.zeros_like(cooccurrence),
            where=denominator > 0,
        )
        np.fill_diagonal(self.similarity, 0.0)
        self.n_items = split.n_items
        if self.top_k_neighbours < self.n_items:
            remove = self.n_items - self.top_k_neighbours
            threshold = np.partition(self.similarity, remove, axis=1)[:, remove]
            self.similarity[self.similarity < threshold[:, None]] = 0.0
        self.user_items = {
            int(user): group["item_index"].to_numpy(dtype=np.int64)
            for user, group in split.train.groupby("user_index")
        }
        return self

    def score_user(self, user_index: int) -> np.ndarray:
        if self.similarity is None or self.user_items is None:
            raise RuntimeError("Model is not fitted; call fit() first.")
        history = self.user_items.get(int(user_index))
        if history is None or history.size == 0:
            return np.zeros(self.n_items, dtype=np.float64)
        return self.similarity[history].sum(axis=0)

    def recommend(
        self, user_index: int, k: int = 10, exclude: set[int] | None = None
    ) -> np.ndarray:
        return _top_k(self.score_user(user_index), k, exclude)


@dataclass
class RandomRecommender:
    random_state: int = 0
    n_items: int = 0

    def fit(self, split: Split) -> RandomRecommender:
        self.n_items = split.n_items
        return self

    def score_user(self, user_index: int) -> np.ndarray:
        return np.random.default_rng(self.random_state + user_index).random(self.n_items)

    def recommend(
        self, user_index: int, k: int = 10, exclude: set[int] | None = None
    ) -> np.ndarray:
        return _top_k(self.score_user(user_index), k, exclude)
