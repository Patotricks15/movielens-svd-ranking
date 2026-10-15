"""Offline/online ranking parity and request-contract tests."""

from __future__ import annotations

import json

import numpy as np

from recommender.models import SVDRecommender
from recommender.serving import build_topn_payloads, parse_request


def test_published_topn_matches_offline_model(split, interactions):
    model = SVDRecommender(n_factors=8, random_state=2).fit(split)
    user = int(split.train["user_index"].iloc[0])
    payload = build_topn_payloads(
        model,
        split,
        interactions.item_ids,
        interactions.user_ids,
        k=5,
        users=[user],
    )[0]
    online = [int(row["movie_id"]) for row in payload["recommendations"]]
    offline_indices = model.recommend(
        user, k=5, exclude=split.user_positives().get(user, set())
    )
    offline = [int(interactions.item_ids[index]) for index in offline_indices]
    assert online == offline
    json.dumps(payload)


def test_request_parser_validates_inputs():
    assert parse_request({"user_id": "12", "k": "5"}) == ("12", 5)


def test_topn_rejects_misaligned_ids(split, interactions):
    model = SVDRecommender(n_factors=4).fit(split)
    with np.testing.assert_raises_regex(ValueError, "item_ids"):
        build_topn_payloads(
            model,
            split,
            interactions.item_ids[:-1],
            interactions.user_ids,
        )
