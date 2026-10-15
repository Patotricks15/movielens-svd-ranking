"""Declarative, point-in-time feature engineering."""

from recommender.features.pipeline import FeatureStore, build_features, resolve_as_of
from recommender.features.registry import TRANSFORM_REGISTRY, validate_registry
from recommender.features.spec import FeatureSpec, load_feature_specs

__all__ = [
    "TRANSFORM_REGISTRY",
    "FeatureSpec",
    "FeatureStore",
    "build_features",
    "load_feature_specs",
    "resolve_as_of",
    "validate_registry",
]
