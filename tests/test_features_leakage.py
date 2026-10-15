"""Leakage controls.

These tests are adversarial by design. Each one constructs a situation in
which a leaking implementation would produce a *different* answer from a
correct one, and asserts the correct answer. A test that merely checks
``build_features`` returns something would pass against a pipeline that reads
the test set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recommender.data.splitting import Split, temporal_split
from recommender.features import build_features, resolve_as_of
from recommender.features.transforms import (
    SECONDS_PER_DAY,
    item_trailing_popularity,
    user_genre_affinity,
    user_recency_days,
)


def test_future_event_injection_does_not_move_features(
    split, interactions, catalogue
):
    """Control 1: forging held-out events must not change fitted features.

    The evaluation partitions are overwritten with a fabricated blockbuster
    that every user consumes. If any transform reads beyond the training
    partition, its output will shift. Nothing may move.
    """
    baseline = build_features(split, interactions, catalogue)

    forged_item = 0
    users = np.arange(split.n_users)
    future_ts = int(split.train["timestamp"].max()) + 10 * int(SECONDS_PER_DAY)
    forged = pd.DataFrame(
        {
            "user_index": users,
            "item_index": np.full(split.n_users, forged_item, dtype=np.int64),
            "rating": np.full(split.n_users, 5.0),
            "timestamp": np.full(split.n_users, future_ts, dtype=np.int64),
        }
    )

    poisoned = Split(
        train=split.train,
        validation=forged,
        test=forged,
        n_users=split.n_users,
        n_items=split.n_items,
        protocol="loo",  # forces as_of off the validation frame
    )
    after = build_features(poisoned, interactions, catalogue)

    for name in ("user_activity_count", "user_mean_rating", "user_tenure_days"):
        pd.testing.assert_series_equal(
            getattr(baseline, name), getattr(after, name), check_names=False
        )
    np.testing.assert_allclose(
        baseline.item_popularity_rank.to_numpy(),
        after.item_popularity_rank.to_numpy(),
    )


def test_fit_partition_isolation(interactions, catalogue):
    """Control 2: a later cut-off must be a strict superset in history.

    Features fitted at an earlier cut-off cannot depend on events after it,
    so every user's activity count at the early cut-off must be no greater
    than at the late one. A pipeline reading the whole frame would produce
    identical counts for both, which this test rejects.
    """
    early = build_features(
        temporal_split(interactions, 0.50, 0.60), interactions, catalogue
    )
    late = build_features(
        temporal_split(interactions, 0.80, 0.90), interactions, catalogue
    )

    shared = early.user_activity_count.index.intersection(
        late.user_activity_count.index
    )
    assert len(shared) > 0

    early_counts = early.user_activity_count.loc[shared]
    late_counts = late.user_activity_count.loc[shared]
    assert (early_counts <= late_counts + 1e-9).all()
    # And the cut-offs must actually differ, or the check above is vacuous.
    assert (early_counts < late_counts - 1e-9).any()
    assert early.as_of < late.as_of


def test_label_destruction_collapses_signal(split, interactions, catalogue):
    """Control 3: destroying the training signal must destroy the features.

    Item indices in the training partition are shuffled. Genuine
    collaborative features are computed from that association, so the
    per-user genre affinity must change. If it survives, the feature is
    reading something other than the partition it was handed.
    """
    rng = np.random.default_rng(0)
    scrambled_train = split.train.copy()
    scrambled_train["item_index"] = rng.permutation(
        scrambled_train["item_index"].to_numpy()
    )

    scrambled = Split(
        train=scrambled_train,
        validation=split.validation,
        test=split.test,
        n_users=split.n_users,
        n_items=split.n_items,
        protocol=split.protocol,
    )

    baseline = build_features(split, interactions, catalogue)
    after = build_features(scrambled, interactions, catalogue)

    shared = baseline.user_genre_affinity.index.intersection(
        after.user_genre_affinity.index
    )
    differing = (
        baseline.user_genre_affinity.loc[shared] != after.user_genre_affinity.loc[shared]
    ).mean()
    assert differing > 0.5, (
        "Scrambling the user-item association left genre affinity nearly "
        f"unchanged ({differing:.1%} differ); the feature is not reading the "
        "training partition."
    )


def test_recency_rejects_cutoff_inside_training_window(split):
    """A cut-off inside the training window is a programming error, not a NaN."""
    inside = float(split.train["timestamp"].median())
    with pytest.raises(ValueError, match="point-in-time safe"):
        user_recency_days(split.train, as_of=inside)


def test_trailing_popularity_excludes_the_cutoff_instant(split):
    """The window is half-open: an event exactly at ``as_of`` is future."""
    train = split.train
    as_of = float(train["timestamp"].max()) + 1.0
    boundary = train.copy()
    boundary.loc[boundary.index[0], "timestamp"] = int(as_of)

    without = item_trailing_popularity(train, split.n_items, as_of, window_days=1e6)
    with_boundary = item_trailing_popularity(
        boundary, split.n_items, as_of, window_days=1e6
    )
    item = int(train.loc[train.index[0], "item_index"])
    assert with_boundary.iloc[item] < without.iloc[item]


def test_genre_affinity_decay_prefers_recent_events(store, split, interactions, catalogue):
    """A short half-life must weight recent genres more than a long one.

    Equal weighting would make the two configurations identical; that they
    differ is what proves the decay is applied rather than declared.
    """
    as_of = resolve_as_of(split)
    short = user_genre_affinity(
        split.train, store.item_genre_tokens, as_of=as_of, half_life_days=7.0
    )
    long = user_genre_affinity(
        split.train, store.item_genre_tokens, as_of=as_of, half_life_days=5000.0
    )
    shared = short.index.intersection(long.index)
    assert (short.loc[shared] != long.loc[shared]).any()


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_invalid_decay_is_rejected(store, split, bad):
    with pytest.raises(ValueError, match="half_life_days must be positive"):
        user_genre_affinity(
            split.train, store.item_genre_tokens, as_of=resolve_as_of(split), half_life_days=bad
        )
