"""Materialize the declared features and write them to Parquet."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recommender.data import (  # noqa: E402
    build_implicit_feedback,
    cold_user_rate,
    leave_last_one_out,
    load_movie_metadata,
    load_ratings,
    temporal_split,
)
from recommender.features import build_features  # noqa: E402

DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", type=Path, default=DATA_DIR / "ratings.csv")
    parser.add_argument("--movies", type=Path, default=DATA_DIR / "movies.csv")
    parser.add_argument("--out", type=Path, default=ARTIFACT_DIR)
    parser.add_argument(
        "--protocol", choices=["temporal", "loo"], default="temporal",
        help="Evaluation protocol the features are fitted for.",
    )
    args = parser.parse_args()

    ratings = load_ratings(args.ratings)
    catalogue = load_movie_metadata(args.movies)
    data = build_implicit_feedback(ratings)

    split = (
        temporal_split(data) if args.protocol == "temporal" else leave_last_one_out(data)
    )
    store = build_features(split, data, catalogue)

    args.out.mkdir(parents=True, exist_ok=True)
    frames = store.to_frames()
    for name, frame in frames.items():
        target = args.out / f"features_{name}_{args.protocol}.parquet"
        frame.to_parquet(target, index=False)
        print(f"{name:6s} {frame.shape[0]:>6,} rows x {frame.shape[1]} cols -> {target}")

    print()
    print(f"protocol        {split.protocol}")
    print(f"as_of           {store.as_of:.0f}")
    print(f"train/val/test  {len(split.train):,} / {len(split.validation):,} / {len(split.test):,}")
    if split.protocol == "temporal":
        print(f"cold-user rate  {cold_user_rate(data, split):.1%}")
    print(f"features        {len(store.specs)} declared, all materialized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
