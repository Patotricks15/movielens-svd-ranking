"""Fetch MovieLens, or generate a synthetic fixture with the same schema.

MovieLens is distributed by GroupLens under its own terms and is not
redistributed with this repository. The synthetic generator exists so the
pipeline and its tests run with no network access and no credentials.
"""

from __future__ import annotations

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/{name}.zip"

GENRES = [
    "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
    "Drama", "Fantasy", "Horror", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western",
]


def generate_synthetic(
    n_users: int = 1200,
    n_items: int = 600,
    seed: int = 20260115,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a fixture with taste clusters, popularity skew and real time drift.

    Three properties matter for this project specifically:

    * **Taste clusters** give a collaborative model recoverable structure.
      Without them every model would score at the random baseline.
    * **Pareto popularity** reproduces the long tail that makes coverage a
      meaningful metric.
    * **Timestamp drift** makes item popularity change over the time axis, so
      a global temporal split is genuinely harder than a random one and the
      point-in-time machinery has something real to protect against.

    Args:
        n_users: Number of synthetic users.
        n_items: Number of synthetic movies.
        seed: Seed controlling the whole generation.

    Returns:
        ``(ratings, movies)`` frames matching the MovieLens schema.
    """
    rng = np.random.default_rng(seed)

    n_clusters = 8
    item_cluster = rng.integers(0, n_clusters, size=n_items)
    user_cluster = rng.integers(0, n_clusters, size=n_users)

    base_popularity = rng.pareto(1.5, size=n_items) + 1.0
    base_popularity /= base_popularity.sum()

    # Each item has an era: popularity peaks at a point on the time axis.
    item_era = rng.uniform(0.0, 1.0, size=n_items)

    movies = pd.DataFrame(
        {
            "movieId": np.arange(1, n_items + 1),
            "title": [f"Synthetic Movie {i:04d} ({1990 + i % 30})" for i in range(1, n_items + 1)],
            "genres": [
                "|".join(sorted(rng.choice(GENRES, size=rng.integers(1, 4), replace=False)))
                for _ in range(n_items)
            ],
        }
    )

    start_ts = 1_100_000_000
    span = 200_000_000

    rows = []
    for user in range(1, n_users + 1):
        cluster = user_cluster[user - 1]
        n_interactions = int(np.clip(rng.lognormal(3.0, 0.8), 5, 220))
        n_interactions = min(n_interactions, n_items)

        # Users are active in a window, not across the whole span.
        user_centre = rng.uniform(0.1, 0.95)
        window = rng.uniform(0.05, 0.25)

        affinity = np.where(item_cluster == cluster, 5.0, 1.0)
        # Items are likelier to be picked near their era, and near the
        # user's active window.
        era_fit = np.exp(-((item_era - user_centre) ** 2) / (2 * window**2))
        weights = base_popularity * affinity * (0.3 + era_fit)
        weights /= weights.sum()

        chosen = rng.choice(n_items, size=n_interactions, replace=False, p=weights)

        for item in chosen:
            offset = np.clip(rng.normal(user_centre, window / 2), 0.0, 1.0)
            timestamp = int(start_ts + offset * span)
            liked = rng.random() < (0.75 if item_cluster[item] == cluster else 0.35)
            scale = [4.0, 4.5, 5.0] if liked else [0.5, 1.0, 2.0, 3.0]
            rating = float(rng.choice(scale))
            rows.append((user, int(item) + 1, rating, timestamp))

    ratings = pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])
    ratings = ratings.sort_values("timestamp").reset_index(drop=True)
    return ratings, movies


def download_movielens(name: str = "ml-latest-small") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download and unpack a MovieLens release.

    Args:
        name: Release name, e.g. ``ml-latest-small`` or ``ml-25m``.

    Returns:
        ``(ratings, movies)``.

    Raises:
        RuntimeError: If the archive cannot be fetched or lacks the files.
    """
    url = MOVIELENS_URL.format(name=name)
    print(f"downloading {url} ...")
    try:
        with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310
            payload = response.read()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Could not download {url}: {exc}") from exc

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        try:
            ratings_name = next(n for n in archive.namelist() if n.endswith("ratings.csv"))
            movies_name = next(n for n in archive.namelist() if n.endswith("movies.csv"))
        except StopIteration as exc:
            raise RuntimeError(f"{name}.zip does not contain ratings.csv and movies.csv") from exc

        with archive.open(ratings_name) as handle:
            ratings = pd.read_csv(handle)
        with archive.open(movies_name) as handle:
            movies = pd.read_csv(handle)

    return ratings, movies


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--synthetic", action="store_true", help="Generate a local fixture.")
    source.add_argument("--movielens", metavar="RELEASE", help="Download a MovieLens release.")
    parser.add_argument("--users", type=int, default=1200)
    parser.add_argument("--items", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260115)
    args = parser.parse_args()

    if args.synthetic:
        ratings, movies = generate_synthetic(args.users, args.items, args.seed)
    else:
        ratings, movies = download_movielens(args.movielens)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ratings.to_csv(DATA_DIR / "ratings.csv", index=False)
    movies.to_csv(DATA_DIR / "movies.csv", index=False)

    span = ratings["timestamp"].max() - ratings["timestamp"].min()
    print(f"ratings: {len(ratings):,} rows -> {DATA_DIR / 'ratings.csv'}")
    print(f"movies:  {len(movies):,} rows -> {DATA_DIR / 'movies.csv'}")
    print(f"users:   {ratings['userId'].nunique():,}")
    print(f"span:    {span / 86400:.0f} days")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
