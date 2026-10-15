"""Dataset partitioning.

Two protocols are implemented deliberately:

``temporal_split``
    A global wall-clock cut-off. This is the protocol the project reports as
    primary, because it is the only one that does not let a model see events
    that postdate another user's held-out interaction.

``leave_last_one_out``
    The per-user protocol most published notebooks use. It is kept so the
    numbers here are comparable with theirs, and so the gap between the two
    is measurable rather than asserted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from recommender.data.loader import InteractionData


@dataclass(frozen=True)
class Split:
    """Train/validation/test partitions over contiguous indices."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    n_users: int
    n_items: int
    protocol: str = "temporal"

    def train_matrix_entries(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(users, items)`` arrays for the training partition."""
        return (
            self.train["user_index"].to_numpy(dtype=np.int64),
            self.train["item_index"].to_numpy(dtype=np.int64),
        )

    def user_positives(self) -> dict[int, set[int]]:
        """Map each user to the set of items seen in training."""
        return {
            int(user): set(map(int, items))
            for user, items in self.train.groupby("user_index")["item_index"]
        }

    def __post_init__(self) -> None:
        if self.train.empty:
            raise ValueError("The training partition is empty.")


def _drop_unseen_users(frame: pd.DataFrame, known_users: set[int]) -> pd.DataFrame:
    """Remove evaluation rows for users with no training history.

    A user absent from training has no history for a collaborative model to
    personalise on, so including them measures the popularity prior rather
    than the model. They are reported separately as the cold-user rate
    instead of being silently folded into the headline metric.
    """
    return frame[frame["user_index"].isin(known_users)]


def temporal_split(
    data: InteractionData,
    train_quantile: float = 0.80,
    validation_quantile: float = 0.90,
) -> Split:
    """Split globally on event time.

    Args:
        data: Interactions, assumed sorted by timestamp.
        train_quantile: Quantile of the timestamp distribution ending training.
        validation_quantile: Quantile ending validation.

    Returns:
        A :class:`Split` with ``protocol="temporal"``.

    Raises:
        ValueError: If the quantiles are not ordered within ``(0, 1)``.
    """
    if not 0 < train_quantile < validation_quantile < 1:
        raise ValueError(
            "Expected 0 < train_quantile < validation_quantile < 1, got "
            f"{train_quantile} and {validation_quantile}."
        )

    frame = data.frame
    train_cutoff = frame["timestamp"].quantile(train_quantile)
    val_cutoff = frame["timestamp"].quantile(validation_quantile)

    # Strict `<` on the training side: an event exactly at the cut-off belongs
    # to the future, not to the history used to predict it.
    train = frame[frame["timestamp"] < train_cutoff]
    validation = frame[(frame["timestamp"] >= train_cutoff) & (frame["timestamp"] < val_cutoff)]
    test = frame[frame["timestamp"] >= val_cutoff]

    known = set(train["user_index"].unique())
    return Split(
        train=train.reset_index(drop=True),
        validation=_drop_unseen_users(validation, known).reset_index(drop=True),
        test=_drop_unseen_users(test, known).reset_index(drop=True),
        n_users=data.n_users,
        n_items=data.n_items,
        protocol="temporal",
    )


def leave_last_one_out(data: InteractionData, min_train_interactions: int = 3) -> Split:
    """Hold out each user's last and second-to-last interaction.

    Args:
        data: Interactions, assumed sorted by timestamp.
        min_train_interactions: Users with fewer remaining interactions than
            this contribute entirely to training and are not evaluated.

    Returns:
        A :class:`Split` with ``protocol="loo"``.
    """
    frame = data.frame.sort_values(["user_index", "timestamp"]).reset_index(drop=True)
    rank_from_end = frame.groupby("user_index").cumcount(ascending=False)
    sizes = frame.groupby("user_index")["item_index"].transform("size")

    eligible = sizes >= min_train_interactions + 2
    test_mask = eligible & (rank_from_end == 0)
    val_mask = eligible & (rank_from_end == 1)

    return Split(
        train=frame[~(test_mask | val_mask)].reset_index(drop=True),
        validation=frame[val_mask].reset_index(drop=True),
        test=frame[test_mask].reset_index(drop=True),
        n_users=data.n_users,
        n_items=data.n_items,
        protocol="loo",
    )


def cold_user_rate(data: InteractionData, split: Split) -> float:
    """Fraction of post-cut-off users that the temporal split had to discard.

    Reported rather than hidden: it is the headline cost of the temporal
    protocol and it is invisible in the ranking metrics themselves.
    """
    frame = data.frame
    train_users = set(split.train["user_index"].unique())
    future = frame[~frame.index.isin(split.train.index)]
    future_users = set(future["user_index"].unique())
    if not future_users:
        return 0.0
    return len(future_users - train_users) / len(future_users)
