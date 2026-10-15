"""Schema, registry and split-integrity tests."""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from recommender.data.splitting import cold_user_rate, temporal_split
from recommender.features import load_feature_specs, validate_registry
from recommender.features.registry import TRANSFORM_REGISTRY
from recommender.features.spec import FeatureSpec, default_spec_path


def test_spec_file_parses_and_is_complete():
    specs = load_feature_specs()
    assert len(specs) == 9
    validate_registry(specs)
    assert {s.transform for s in specs} == set(TRANSFORM_REGISTRY)


def test_every_feature_documents_a_rationale():
    for spec in load_feature_specs():
        assert len(spec.rationale.strip()) > 40, spec.name


def test_declared_dtypes_match_materialized_series(store):
    for spec in load_feature_specs():
        series = getattr(store, spec.name)
        assert str(series.dtype) == spec.dtype, (
            f"{spec.name} declares {spec.dtype} but materialized as {series.dtype}"
        )


def test_point_in_time_features_are_declared_consistently():
    pit = {s.name for s in load_feature_specs() if s.point_in_time}
    assert pit == {
        "user_recency_days",
        "user_genre_affinity",
        "item_trailing_popularity",
    }


def test_unknown_family_is_rejected():
    with pytest.raises(ValueError, match="unknown family"):
        FeatureSpec("x", "nonsense", "float64", "t", {}, "none", False, {}, "why")


def test_point_in_time_feature_must_fit_on_train_only():
    with pytest.raises(ValueError, match="point-in-time but fits on"):
        FeatureSpec("x", "user", "float64", "t", {}, "catalogue", True, {}, "a reason")


def test_missing_rationale_is_rejected():
    with pytest.raises(ValueError, match="must document a rationale"):
        FeatureSpec("x", "user", "float64", "t", {}, "none", False, {}, "   ")


def test_duplicate_declaration_is_rejected(tmp_path):
    entry = {
        "name": "dup", "family": "user", "dtype": "float64",
        "transform": "user_activity_count", "rationale": "duplicated on purpose",
    }
    path = tmp_path / "features.yaml"
    path.write_text(yaml.safe_dump({"features": [entry, entry]}))
    with pytest.raises(ValueError, match="Duplicate feature declaration"):
        load_feature_specs(path)


def test_registry_rejects_undeclared_transform():
    specs = [s for s in load_feature_specs() if s.transform != "user_tenure_days"]
    with pytest.raises(ValueError, match="implemented but not declared"):
        validate_registry(specs)


def test_spec_path_resolves_from_package():
    assert default_spec_path().exists()


def test_item_features_cover_the_whole_catalogue(store):
    for name in ("item_popularity_rank", "item_trailing_popularity",
                 "item_genre_tokens", "item_release_year"):
        assert len(getattr(store, name)) == store.n_items, name


def test_exported_frames_have_no_nulls(store):
    for name, frame in store.to_frames().items():
        assert not frame.isna().any().any(), name


def test_unknown_entities_get_declared_fills(store):
    user = store.user_vector(10**9)
    assert user["user_activity_count"] == 0.0
    assert user["user_genre_affinity"] == ""
    assert user["user_mean_rating"] == pytest.approx(store.global_rating_mean)

    item = store.item_vector(10**9)
    assert item["item_popularity_rank"] == 1.0
    assert item["item_trailing_popularity"] == 0.0


def test_release_year_is_parsed_from_title(store):
    years = store.item_release_year.to_numpy()
    parsed = years[years > 0]
    assert len(parsed) > 0
    assert parsed.min() >= 1870 and parsed.max() <= 2100


def test_partitions_are_disjoint_and_ordered(interactions):
    split = temporal_split(interactions)
    assert split.train["timestamp"].max() < split.validation["timestamp"].min()
    assert split.validation["timestamp"].max() < split.test["timestamp"].min()
    assert len(split.validation) > 0 and len(split.test) > 0


def test_loo_holds_out_exactly_two_events_per_evaluated_user(loo_split):
    assert (loo_split.validation.groupby("user_index").size() == 1).all()
    assert (loo_split.test.groupby("user_index").size() == 1).all()
    assert set(loo_split.test["user_index"]) <= set(loo_split.train["user_index"])


def test_temporal_split_excludes_cold_users_and_reports_the_rate(interactions):
    split = temporal_split(interactions)
    known = set(split.train["user_index"])
    assert set(split.test["user_index"]) <= known
    rate = cold_user_rate(interactions, split)
    assert 0.0 < rate < 1.0


@pytest.mark.parametrize(
    "train_q,val_q", [(0.9, 0.8), (0.0, 0.5), (0.5, 1.0), (0.5, 0.5)]
)
def test_invalid_quantiles_are_rejected(interactions, train_q, val_q):
    with pytest.raises(ValueError, match="train_quantile"):
        temporal_split(interactions, train_q, val_q)


def test_train_matrix_entries_are_aligned(split):
    users, items = split.train_matrix_entries()
    assert users.shape == items.shape == (len(split.train),)
    assert users.dtype == np.int64 and items.dtype == np.int64
    assert users.max() < split.n_users and items.max() < split.n_items
