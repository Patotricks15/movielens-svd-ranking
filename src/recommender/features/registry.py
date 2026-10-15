"""Registry binding declared feature names to their implementations."""

from __future__ import annotations

from collections.abc import Callable

from recommender.features import transforms
from recommender.features.spec import FeatureSpec

TRANSFORM_REGISTRY: dict[str, Callable[..., object]] = {
    "user_activity_count": transforms.user_activity_count,
    "user_mean_rating": transforms.user_mean_rating,
    "user_tenure_days": transforms.user_tenure_days,
    "user_recency_days": transforms.user_recency_days,
    "user_genre_affinity": transforms.user_genre_affinity,
    "item_popularity_rank": transforms.item_popularity_rank,
    "item_trailing_popularity": transforms.item_trailing_popularity,
    "item_genre_tokens": transforms.item_genre_tokens,
    "item_release_year": transforms.item_release_year,
}

#: Transforms that must receive an explicit ``as_of`` cut-off.
POINT_IN_TIME_TRANSFORMS = frozenset(
    {"user_recency_days", "user_genre_affinity", "item_trailing_popularity"}
)


def validate_registry(specs: list[FeatureSpec]) -> None:
    """Assert the declaration and the implementation agree in both directions.

    A one-directional check would let dead transforms accumulate, or let a
    feature be computed without ever being declared and therefore without
    ever being covered by the leakage tests.

    Args:
        specs: Parsed feature specifications.

    Raises:
        ValueError: If a declared feature has no implementation, an
            implemented transform is never declared, or a transform that
            reads event times is not declared ``point_in_time``.
    """
    declared = {spec.transform for spec in specs}
    implemented = set(TRANSFORM_REGISTRY)

    problems: list[str] = []
    if unimplemented := declared - implemented:
        problems.append(f"declared but not implemented: {sorted(unimplemented)}")
    if undeclared := implemented - declared:
        problems.append(f"implemented but not declared: {sorted(undeclared)}")

    mislabelled = sorted(
        spec.name
        for spec in specs
        if spec.transform in POINT_IN_TIME_TRANSFORMS and not spec.point_in_time
    )
    if mislabelled:
        problems.append(f"time-dependent but not declared point_in_time: {mislabelled}")

    if problems:
        raise ValueError("Feature registry mismatch -- " + "; ".join(problems))
