"""Feature pipeline: materializes every declared feature from training data."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from recommender.data.loader import InteractionData
from recommender.data.splitting import Split
from recommender.features import transforms
from recommender.features.registry import validate_registry
from recommender.features.spec import FeatureSpec, load_feature_specs

UNKNOWN_YEAR = 0.0


def resolve_as_of(split: Split) -> float:
    """Derive the point-in-time cut-off implied by a partition.

    For the temporal protocol the cut-off is the first held-out event: every
    training row precedes it by construction. For leave-one-out there is no
    global boundary, so the cut-off is one second past the last training
    event -- the earliest instant at which all of training is strictly past.

    Args:
        split: The partition features will be fitted on.

    Returns:
        A Unix timestamp strictly greater than every training timestamp.
    """
    last_train = float(split.train["timestamp"].max())
    if split.protocol == "temporal" and not split.validation.empty:
        return float(split.validation["timestamp"].min())
    return last_train + 1.0


@dataclass
class FeatureStore:
    """Materialized features, fitted on the training partition only.

    Training and serving both read this object, which is what makes
    offline/online parity testable rather than aspirational.
    """

    user_activity_count: pd.Series
    user_mean_rating: pd.Series
    user_tenure_days: pd.Series
    user_recency_days: pd.Series
    user_genre_affinity: pd.Series
    item_popularity_rank: pd.Series
    item_trailing_popularity: pd.Series
    item_genre_tokens: pd.Series
    item_release_year: pd.Series
    n_users: int
    n_items: int
    as_of: float
    global_rating_mean: float
    specs: list[FeatureSpec] = field(default_factory=list)

    def user_vector(self, user_index: int) -> dict[str, float | str]:
        """Feature values for one user, with declared fills applied.

        Unknown users get neutral values rather than raising: the serving
        path must answer for a user the model has never seen.
        """
        return {
            "user_activity_count": float(self.user_activity_count.get(user_index, 0.0)),
            "user_mean_rating": float(
                self.user_mean_rating.get(user_index, self.global_rating_mean)
            ),
            "user_tenure_days": float(self.user_tenure_days.get(user_index, 0.0)),
            "user_recency_days": float(self.user_recency_days.get(user_index, 0.0)),
            "user_genre_affinity": str(self.user_genre_affinity.get(user_index, "")),
        }

    def item_vector(self, item_index: int) -> dict[str, float | str]:
        """Feature values for one item, with declared fills applied."""
        return {
            "item_popularity_rank": float(self.item_popularity_rank.get(item_index, 1.0)),
            "item_trailing_popularity": float(
                self.item_trailing_popularity.get(item_index, 0.0)
            ),
            "item_genre_tokens": str(self.item_genre_tokens.get(item_index, "")),
            "item_release_year": float(
                self.item_release_year.get(item_index, UNKNOWN_YEAR)
            ),
        }

    def popularity_order(self) -> np.ndarray:
        """Item indices ordered from most to least popular."""
        return np.argsort(self.item_popularity_rank.to_numpy(), kind="stable")

    def to_frames(self) -> dict[str, pd.DataFrame]:
        """Export user-level and item-level features as tidy frames."""
        index = range(self.n_users)
        users = pd.DataFrame(
            {
                "user_index": np.arange(self.n_users),
                "user_activity_count": self.user_activity_count.reindex(index)
                .fillna(0.0)
                .to_numpy(),
                "user_mean_rating": self.user_mean_rating.reindex(index)
                .fillna(self.global_rating_mean)
                .to_numpy(),
                "user_tenure_days": self.user_tenure_days.reindex(index)
                .fillna(0.0)
                .to_numpy(),
                "user_recency_days": self.user_recency_days.reindex(index)
                .fillna(0.0)
                .to_numpy(),
                "user_genre_affinity": self.user_genre_affinity.reindex(index)
                .fillna("")
                .to_numpy(),
            }
        )
        items = pd.DataFrame(
            {
                "item_index": np.arange(self.n_items),
                "item_popularity_rank": self.item_popularity_rank.to_numpy(),
                "item_trailing_popularity": self.item_trailing_popularity.to_numpy(),
                "item_genre_tokens": self.item_genre_tokens.to_numpy(),
                "item_release_year": self.item_release_year.to_numpy(),
            }
        )
        return {"users": users, "items": items}


def build_features(
    split: Split,
    data: InteractionData,
    catalogue: pd.DataFrame,
    spec_path: str | Path | None = None,
) -> FeatureStore:
    """Materialize every declared feature from the training partition.

    Only ``split.train`` reaches the interaction transforms, which is what
    enforces the ``fit_on: train_only`` contract. Catalogue transforms read
    movie metadata directly because that is known before any interaction and
    carries no label signal.

    Args:
        split: Leakage-safe partition.
        data: Indexed interactions, for the item id mapping.
        catalogue: Raw ``movies.csv`` frame.
        spec_path: Optional override of ``conf/features.yaml``.

    Returns:
        A populated :class:`FeatureStore`.
    """
    specs = load_feature_specs(spec_path)
    validate_registry(specs)

    train = split.train
    as_of = resolve_as_of(split)
    params = {spec.transform: spec.params for spec in specs}

    genre_tokens = transforms.item_genre_tokens(
        catalogue,
        data.item_ids,
        separator=params["item_genre_tokens"].get("separator", "|"),
    )
    global_rating_mean = float(train["rating"].mean()) if not train.empty else 0.0

    return FeatureStore(
        user_activity_count=transforms.user_activity_count(
            train,
            log_transform=params["user_activity_count"].get("log_transform", True),
        ),
        user_mean_rating=transforms.user_mean_rating(train),
        user_tenure_days=transforms.user_tenure_days(train),
        user_recency_days=transforms.user_recency_days(train, as_of=as_of),
        user_genre_affinity=transforms.user_genre_affinity(
            train,
            genre_tokens,
            as_of=as_of,
            half_life_days=params["user_genre_affinity"].get("half_life_days", 180.0),
            top_k=params["user_genre_affinity"].get("top_k", 5),
        ),
        item_popularity_rank=transforms.item_popularity_rank(
            train,
            n_items=split.n_items,
            normalize=params["item_popularity_rank"].get("normalize", True),
        ),
        item_trailing_popularity=transforms.item_trailing_popularity(
            train,
            n_items=split.n_items,
            as_of=as_of,
            window_days=params["item_trailing_popularity"].get("window_days", 90.0),
            log_transform=params["item_trailing_popularity"].get("log_transform", True),
        ),
        item_genre_tokens=genre_tokens,
        item_release_year=transforms.item_release_year(
            catalogue,
            data.item_ids,
            fallback=params["item_release_year"].get("fallback", UNKNOWN_YEAR),
        ),
        n_users=split.n_users,
        n_items=split.n_items,
        as_of=as_of,
        global_rating_mean=global_rating_mean,
        specs=specs,
    )
