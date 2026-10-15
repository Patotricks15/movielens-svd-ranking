"""Classical collaborative-filtering models."""

from recommender.models.baselines import (
    ItemKNNRecommender,
    PopularityRecommender,
    RandomRecommender,
)
from recommender.models.svd import SVDRecommender

__all__ = [
    "ItemKNNRecommender",
    "PopularityRecommender",
    "RandomRecommender",
    "SVDRecommender",
]
