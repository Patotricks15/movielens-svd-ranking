"""Behavioral tests for the classical recommender models."""

from __future__ import annotations

import numpy as np
import pytest

from recommender.evaluation import evaluate, rating_rmse
from recommender.models import (
    ItemKNNRecommender,
    PopularityRecommender,
    RandomRecommender,
    SVDRecommender,
)


def test_svd_training_reduces_rmse(split):
    model = SVDRecommender(n_factors=12, epochs=12, random_state=7).fit(split)
    assert model.train_rmse_ is not None
    assert model.train_rmse_[-1] < model.train_rmse_[0]


def test_svd_is_reproducible(split):
    first = SVDRecommender(n_factors=8, epochs=4, random_state=3).fit(split)
    second = SVDRecommender(n_factors=8, epochs=4, random_state=3).fit(split)
    np.testing.assert_allclose(first.score_user(1), second.score_user(1))


def test_recommendations_exclude_history(split):
    user, history = next(iter(split.user_positives().items()))
    model = SVDRecommender(n_factors=8, epochs=5).fit(split)
    ranked = model.recommend(user, k=10, exclude=history)
    assert len(set(ranked) & history) == 0
    assert len(ranked) == 10


def test_cold_user_falls_back_to_item_bias(split):
    model = SVDRecommender(n_factors=8, epochs=3).fit(split)
    scores = model.score_user(split.n_users + 1)
    assert scores.shape == (split.n_items,)
    assert np.isfinite(scores).all()


@pytest.mark.parametrize(
    "model",
    [
        PopularityRecommender(),
        ItemKNNRecommender(top_k_neighbours=20),
        RandomRecommender(random_state=1),
    ],
)
def test_baselines_follow_contract(model, split):
    model.fit(split)
    user, history = next(iter(split.user_positives().items()))
    ranked = model.recommend(user, k=7, exclude=history)
    assert ranked.shape == (7,)
    assert len(np.unique(ranked)) == 7


def test_evaluation_reports_ranking_and_rating_metrics(split):
    model = SVDRecommender(n_factors=8, epochs=5).fit(split)
    report = evaluate(model, split, k=10, model_name="SVD")
    assert report.model == "SVD"
    assert report.protocol == "temporal"
    assert 0.0 <= report.ndcg <= 1.0
    assert 0.0 <= report.coverage <= 1.0
    assert np.isfinite(rating_rmse(model, split))


def test_unknown_partition_is_explicit(split):
    model = PopularityRecommender().fit(split)
    with pytest.raises(ValueError, match="partition"):
        evaluate(model, split, partition="future")
