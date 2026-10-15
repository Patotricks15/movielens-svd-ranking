"""Top-N ranking and explicit-rating metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from recommender.data.splitting import Split


class Recommender(Protocol):
    def recommend(
        self, user_index: int, k: int = 10, exclude: set[int] | None = None
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class EvaluationReport:
    model: str
    protocol: str
    k: int
    ndcg: float
    recall: float
    precision: float
    mrr: float
    hit_rate: float
    coverage: float
    n_users: int

    def as_row(self) -> dict[str, float | str | int]:
        return {
            "model": self.model,
            "protocol": self.protocol,
            "k": self.k,
            "ndcg": round(self.ndcg, 4),
            "recall": round(self.recall, 4),
            "precision": round(self.precision, 4),
            "mrr": round(self.mrr, 4),
            "hit_rate": round(self.hit_rate, 4),
            "coverage": round(self.coverage, 4),
            "n_users": self.n_users,
        }


def evaluate(
    model: Recommender,
    split: Split,
    k: int = 10,
    partition: str = "test",
    model_name: str = "model",
) -> EvaluationReport:
    """Evaluate one or more relevant items per user against the full catalogue."""
    if partition not in {"validation", "test"}:
        raise ValueError("partition must be 'validation' or 'test'.")
    frame = getattr(split, partition)
    if frame.empty:
        raise ValueError(f"Partition '{partition}' is empty.")

    histories = split.user_positives()
    relevant = {
        int(user): set(group["item_index"].astype(int))
        for user, group in frame.groupby("user_index")
    }
    ndcg_total = recall_total = precision_total = rr_total = 0.0
    hits = 0
    recommended: set[int] = set()

    for user, targets in relevant.items():
        ranked = model.recommend(user, k=k, exclude=histories.get(user, set()))
        recommended.update(map(int, ranked))
        positions = [position for position, item in enumerate(ranked) if int(item) in targets]
        if positions:
            hits += 1
            rr_total += 1.0 / (positions[0] + 1)
        gains = sum(1.0 / np.log2(position + 2) for position in positions)
        ideal = sum(1.0 / np.log2(position + 2) for position in range(min(len(targets), k)))
        ndcg_total += gains / ideal if ideal else 0.0
        recall_total += len(positions) / len(targets)
        precision_total += len(positions) / k

    n_users = len(relevant)
    return EvaluationReport(
        model=model_name,
        protocol=split.protocol,
        k=k,
        ndcg=ndcg_total / n_users,
        recall=recall_total / n_users,
        precision=precision_total / n_users,
        mrr=rr_total / n_users,
        hit_rate=hits / n_users,
        coverage=len(recommended) / split.n_items,
        n_users=n_users,
    )


def rating_rmse(model: object, split: Split, partition: str = "test") -> float:
    """RMSE for models exposing ``predict(user, item)``."""
    if partition not in {"validation", "test"}:
        raise ValueError("partition must be 'validation' or 'test'.")
    frame = getattr(split, partition)
    if frame.empty:
        raise ValueError(f"Partition '{partition}' is empty.")
    predict = getattr(model, "predict", None)
    if predict is None:
        raise TypeError("rating_rmse requires a model exposing predict().")
    predicted = np.array(
        [
            predict(int(user), int(item))
            for user, item in zip(frame["user_index"], frame["item_index"], strict=True)
        ]
    )
    return float(np.sqrt(np.mean((frame["rating"].to_numpy() - predicted) ** 2)))
