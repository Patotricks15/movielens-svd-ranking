"""Load MovieLens ratings and convert them to implicit feedback.

MovieLens ships a real ``timestamp`` column, which is the reason this project
exists alongside the anime BPR one: every temporal claim here is backed by an
actual event time rather than a row-order proxy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"userId", "movieId", "rating", "timestamp"}
MIN_POSITIVE_RATING = 4.0


@dataclass(frozen=True)
class InteractionData:
    """Implicit-feedback interactions with contiguous internal indices.

    Attributes:
        frame: Columns ``user_index``, ``item_index``, ``timestamp``.
        user_ids: Original user ids, positionally aligned to ``user_index``.
        item_ids: Original movie ids, positionally aligned to ``item_index``.
    """

    frame: pd.DataFrame
    user_ids: np.ndarray
    item_ids: np.ndarray

    @property
    def n_users(self) -> int:
        return int(self.user_ids.shape[0])

    @property
    def n_items(self) -> int:
        return int(self.item_ids.shape[0])

    def __len__(self) -> int:
        return int(self.frame.shape[0])


def load_ratings(path: str | Path) -> pd.DataFrame:
    """Read ratings.csv and validate its contract.

    Args:
        path: Path to a MovieLens-format ratings file.

    Returns:
        The raw frame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing or the file is empty.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python scripts/download_data.py --synthetic"
        )

    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{path} contains no rows.")
    return frame


def load_movie_metadata(path: str | Path) -> pd.DataFrame:
    """Read movies.csv (``movieId``, ``title``, ``genres``)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    frame = pd.read_csv(path)
    missing = {"movieId", "title", "genres"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return frame


def build_implicit_feedback(
    ratings: pd.DataFrame,
    min_user_interactions: int = 5,
    min_item_interactions: int = 5,
) -> InteractionData:
    """Binarize ratings, apply an iterative k-core filter and reindex.

    A rating of 4.0 or above is a positive. Lower ratings are *dropped*
    rather than treated as negatives: they signal dislike, which is strictly
    more information than the "unobserved" state the samplers assume, and
    they already fall into the negative pool by virtue of not being positive.

    Args:
        ratings: Raw MovieLens frame.
        min_user_interactions: k-core threshold on the user side.
        min_item_interactions: k-core threshold on the item side.

    Returns:
        Interactions with contiguous indices, sorted by timestamp.

    Raises:
        ValueError: If the thresholds eliminate every interaction.
    """
    frame = ratings.loc[ratings["rating"] >= MIN_POSITIVE_RATING].copy()
    frame = frame.drop_duplicates(subset=["userId", "movieId"], keep="first")

    # Iterative: dropping sparse items can push a user below the user
    # threshold, so a single pass would leave the invariant violated.
    while True:
        user_counts = frame["userId"].value_counts()
        item_counts = frame["movieId"].value_counts()
        keep_users = user_counts[user_counts >= min_user_interactions].index
        keep_items = item_counts[item_counts >= min_item_interactions].index

        filtered = frame[frame["userId"].isin(keep_users) & frame["movieId"].isin(keep_items)]
        if len(filtered) == len(frame):
            frame = filtered
            break
        frame = filtered
        if frame.empty:
            break

    if frame.empty:
        raise ValueError(
            "Core filtering removed every interaction; thresholds "
            f"({min_user_interactions}, {min_item_interactions}) are too strict."
        )

    user_ids = np.sort(frame["userId"].unique())
    item_ids = np.sort(frame["movieId"].unique())
    user_map = {uid: idx for idx, uid in enumerate(user_ids)}
    item_map = {iid: idx for idx, iid in enumerate(item_ids)}

    out = pd.DataFrame(
        {
            "user_index": frame["userId"].map(user_map).astype(np.int64),
            "item_index": frame["movieId"].map(item_map).astype(np.int64),
            "timestamp": frame["timestamp"].astype(np.int64),
        }
    )
    # Sort by time first: every downstream temporal operation assumes it.
    out = out.sort_values(["timestamp", "user_index"]).reset_index(drop=True)

    return InteractionData(frame=out, user_ids=user_ids, item_ids=item_ids)
