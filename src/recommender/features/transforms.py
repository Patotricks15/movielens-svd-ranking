"""Feature transforms.

Every transform in this module takes the *training partition only* and, where
the value depends on when it is computed, an explicit ``as_of`` timestamp.
That signature is the point: a transform cannot accidentally read the future
because it is never handed the future.

MovieLens carries real Unix event times, so time-aware features here are
expressed in calendar days rather than in interaction-count proxies.
"""

from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
import pandas as pd

SECONDS_PER_DAY = 86_400.0
_YEAR_PATTERN = re.compile(r"\((\d{4})\)\s*$")


def _empty(dtype: str = "float64") -> pd.Series:
    return pd.Series(dtype=dtype)


def user_activity_count(train: pd.DataFrame, log_transform: bool = True) -> pd.Series:
    """Number of training interactions per user.

    Args:
        train: Training partition.
        log_transform: Apply ``log1p``. Activity counts are heavy-tailed; the
            raw scale would let a handful of power users dominate any model
            that consumes this feature linearly.

    Returns:
        Series indexed by ``user_index``.
    """
    if train.empty:
        return _empty()
    counts = train.groupby("user_index").size().astype("float64")
    return np.log1p(counts) if log_transform else counts


def user_mean_rating(train: pd.DataFrame) -> pd.Series:
    """Mean explicit rating per user over the training partition.

    Only positives survive implicit-feedback filtering, so this measures how
    generously a user rates the things they liked, not their overall bias.
    """
    if train.empty:
        return _empty()
    return train.groupby("user_index")["rating"].mean().astype("float64")


def user_tenure_days(train: pd.DataFrame) -> pd.Series:
    """Calendar days between a user's first and last training interaction.

    Separates a user with 50 events over three years from a user with 50
    events in one weekend; the two have very different taste stability.
    """
    if train.empty:
        return _empty()
    grouped = train.groupby("user_index")["timestamp"]
    return ((grouped.max() - grouped.min()) / SECONDS_PER_DAY).astype("float64")


def user_recency_days(train: pd.DataFrame, as_of: float) -> pd.Series:
    """Calendar days from a user's last training event to ``as_of``.

    Args:
        train: Training partition.
        as_of: Point-in-time cut-off, in Unix seconds.

    Returns:
        Series indexed by ``user_index``; larger means staler.

    Raises:
        ValueError: If any training event is at or after ``as_of``, which
            would mean the caller passed a cut-off inside the training window.
    """
    if train.empty:
        return _empty()
    last = train.groupby("user_index")["timestamp"].max()
    if (last >= as_of).any():
        raise ValueError(
            "user_recency_days received training events at or after the "
            f"as_of cut-off {as_of}; the partition is not point-in-time safe."
        )
    return ((as_of - last) / SECONDS_PER_DAY).astype("float64")


def item_popularity_rank(
    train: pd.DataFrame, n_items: int, normalize: bool = True
) -> pd.Series:
    """Dense popularity rank over all items, 0 being the most popular.

    Items unseen in training share the last rank rather than being dropped,
    so the series is total over the catalogue and the serving path never has
    to handle a missing key.
    """
    counts = (
        train.groupby("item_index").size()
        if not train.empty
        else pd.Series(dtype="int64")
    )
    counts = counts.reindex(range(n_items), fill_value=0)
    ranks = counts.rank(method="dense", ascending=False).astype("float64") - 1.0
    if normalize and n_items > 1:
        ranks = ranks / float(n_items - 1)
    return ranks


def item_trailing_popularity(
    train: pd.DataFrame,
    n_items: int,
    as_of: float,
    window_days: float = 90.0,
    log_transform: bool = True,
) -> pd.Series:
    """Interaction count inside the ``window_days`` ending at ``as_of``.

    This is the feature the temporal split exists to justify. Lifetime
    popularity is dominated by whatever was popular years ago; a trailing
    window tracks what is popular *now*, which is what a recommender serving
    traffic at ``as_of`` actually needs.

    Args:
        train: Training partition.
        n_items: Catalogue size, so the result covers every item.
        as_of: Point-in-time cut-off, in Unix seconds.
        window_days: Width of the trailing window.
        log_transform: Apply ``log1p`` to the counts.

    Returns:
        Series indexed by ``item_index`` of length ``n_items``.

    Raises:
        ValueError: If ``window_days`` is not positive.
    """
    if window_days <= 0:
        raise ValueError(f"window_days must be positive, got {window_days}.")

    if train.empty:
        return pd.Series(np.zeros(n_items), index=range(n_items), dtype="float64")

    lower = as_of - window_days * SECONDS_PER_DAY
    window = train[(train["timestamp"] >= lower) & (train["timestamp"] < as_of)]
    counts = (
        window.groupby("item_index").size()
        if not window.empty
        else pd.Series(dtype="int64")
    )
    counts = counts.reindex(range(n_items), fill_value=0).astype("float64")
    return np.log1p(counts) if log_transform else counts


def item_genre_tokens(
    catalogue: pd.DataFrame, item_ids: np.ndarray, separator: str = "|"
) -> pd.Series:
    """Normalized genre token string per item index.

    Args:
        catalogue: Raw ``movies.csv`` frame.
        item_ids: Original movie ids ordered by internal item index.
        separator: Token separator used by MovieLens.

    Returns:
        Series indexed by ``item_index``; items absent from the catalogue map
        to the empty string rather than to NaN.
    """
    lookup = catalogue.set_index("movieId")["genres"].to_dict()
    tokens = []
    for movie_id in item_ids:
        raw = lookup.get(int(movie_id), "")
        if not isinstance(raw, str) or raw in {"", "(no genres listed)"}:
            tokens.append("")
            continue
        parts = sorted({p.strip() for p in raw.split(separator) if p.strip()})
        tokens.append("|".join(parts))
    return pd.Series(tokens, index=range(len(item_ids)), dtype="object")


def item_release_year(
    catalogue: pd.DataFrame, item_ids: np.ndarray, fallback: float = 0.0
) -> pd.Series:
    """Release year parsed from the MovieLens title suffix.

    MovieLens has no release-year column; the year is embedded in the title
    as a trailing ``(YYYY)``. Titles that do not carry one get ``fallback``,
    which downstream code must treat as "unknown", not as year zero.
    """
    lookup = catalogue.set_index("movieId")["title"].to_dict()
    years = []
    for movie_id in item_ids:
        title = lookup.get(int(movie_id), "")
        match = _YEAR_PATTERN.search(title) if isinstance(title, str) else None
        years.append(float(match.group(1)) if match else fallback)
    return pd.Series(years, index=range(len(item_ids)), dtype="float64")


def user_genre_affinity(
    train: pd.DataFrame,
    genre_tokens: pd.Series,
    as_of: float,
    half_life_days: float = 180.0,
    top_k: int = 5,
) -> pd.Series:
    """Time-decayed genre preferences, as a ranked token string.

    Each training interaction contributes weight :math:`2^{-\\Delta t / h}`
    where :math:`\\Delta t` is the age of the event in days at ``as_of`` and
    :math:`h` is the half-life. A genre a user abandoned two years ago is
    therefore discounted relative to one they watched last month, which a
    plain count cannot express.

    Args:
        train: Training partition.
        genre_tokens: Output of :func:`item_genre_tokens`.
        as_of: Point-in-time cut-off, in Unix seconds.
        half_life_days: Days at which an event's weight halves.
        top_k: Number of genres retained per user.

    Returns:
        Series indexed by ``user_index`` of ``"|"``-joined genres, strongest
        first.

    Raises:
        ValueError: If ``half_life_days`` or ``top_k`` is not positive.
    """
    if half_life_days <= 0:
        raise ValueError(f"half_life_days must be positive, got {half_life_days}.")
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}.")
    if train.empty:
        return pd.Series(dtype="object")

    token_map = genre_tokens.to_dict()
    scores: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    users = train["user_index"].to_numpy()
    items = train["item_index"].to_numpy()
    ages = (as_of - train["timestamp"].to_numpy()) / SECONDS_PER_DAY
    weights = np.power(2.0, -np.maximum(ages, 0.0) / half_life_days)

    for user, item, weight in zip(users, items, weights, strict=True):
        tokens = token_map.get(int(item), "")
        if not tokens:
            continue
        bucket = scores[int(user)]
        for token in tokens.split("|"):
            bucket[token] += float(weight)

    affinity = {
        user: "|".join(
            token
            for token, _ in sorted(bucket.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
        )
        for user, bucket in scores.items()
    }
    return pd.Series(affinity, dtype="object")
