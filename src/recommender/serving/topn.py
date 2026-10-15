"""Build the top-N payloads that get published to DynamoDB.

This is the offline half of the serving path. Keeping it here, rather than
inside the publishing script, is what lets `tests/test_features_parity.py`
assert that the served ranking is byte-for-byte the ranking the offline
evaluation scored -- the classic training/serving skew bug.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from recommender.data.splitting import Split


def build_topn_payloads(
    model: Any,
    split: Split,
    item_ids: np.ndarray,
    user_ids: np.ndarray,
    k: int = 20,
    users: Iterable[int] | None = None,
    model_name: str = "svd",
) -> list[dict[str, Any]]:
    """Materialize per-user top-``k`` lists in DynamoDB item shape.

    Args:
        model: A fitted recommender exposing ``recommend``.
        split: Split whose training partition defines each user's history.
        item_ids: Original item ids aligned to internal item indices.
        user_ids: Original user ids aligned to internal user indices.
        k: List length to store.
        users: Internal user indices; defaults to every user in the split.
        model_name: Stored alongside the list for A/B attribution.

    Returns:
        One dict per user, ready for ``table.put_item``.

    Raises:
        ValueError: If ``k`` is not positive or the id arrays are misaligned.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}.")
    if user_ids.shape[0] != split.n_users:
        raise ValueError(
            f"user_ids has {user_ids.shape[0]} entries but the split has {split.n_users} users."
        )
    if item_ids.shape[0] != split.n_items:
        raise ValueError(
            f"item_ids has {item_ids.shape[0]} entries but the split has {split.n_items} items."
        )

    history = {
        int(user): set(map(int, items))
        for user, items in split.train.groupby("user_index")["item_index"]
    }
    targets = range(split.n_users) if users is None else users

    payloads = []
    for user_index in targets:
        ranked = model.recommend(int(user_index), k=k, exclude=history.get(int(user_index), set()))
        payloads.append(
            {
                "user_id": str(int(user_ids[int(user_index)])),
                "model": model_name,
                "recommendations": [
                    {"rank": rank, "movie_id": str(int(item_ids[int(item)]))}
                    for rank, item in enumerate(ranked, start=1)
                ],
            }
        )
    return payloads
