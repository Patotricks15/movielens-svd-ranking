"""Command-line entry points for training, evaluation, and recommendation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from recommender.data import (
    build_implicit_feedback,
    load_movie_metadata,
    load_ratings,
    temporal_split,
)
from recommender.evaluation import evaluate
from recommender.models import ItemKNNRecommender, PopularityRecommender, SVDRecommender

ROOT = Path(__file__).resolve().parents[2]


def _load():
    ratings = load_ratings(ROOT / "data" / "ratings.csv")
    movies = load_movie_metadata(ROOT / "data" / "movies.csv")
    data = build_implicit_feedback(ratings)
    return data, movies, temporal_split(data)


def _fit_models(split):
    return {
        "PureSVD": SVDRecommender(n_factors=24).fit(split),
        "ItemKNN": ItemKNNRecommender(top_k_neighbours=50).fit(split),
        "Popularity": PopularityRecommender().fit(split),
    }


def train_command(_: argparse.Namespace) -> None:
    _, _, split = _load()
    model = SVDRecommender(n_factors=24).fit(split)
    print(
        json.dumps(
            {"n_factors": model.n_factors, "reconstruction_rmse": model.reconstruction_rmse_}
        )
    )


def evaluate_command(args: argparse.Namespace) -> None:
    _, _, split = _load()
    rows = []
    for name, model in _fit_models(split).items():
        row = evaluate(model, split, k=args.k, partition=args.partition, model_name=name).as_row()
        rows.append(row)
    print(json.dumps(rows, indent=2))


def recommend_command(args: argparse.Namespace) -> None:
    data, movies, split = _load()
    matches = (data.user_ids == args.user_id).nonzero()[0]
    if matches.size == 0:
        raise SystemExit(f"Unknown user id {args.user_id}.")
    user = int(matches[0])
    model = SVDRecommender(n_factors=24).fit(split)
    ranked = model.recommend(user, k=args.k, exclude=split.user_positives().get(user, set()))
    lookup = movies.set_index("movieId")
    payload = [
        {
            "rank": rank,
            "movie_id": int(data.item_ids[item]),
            "title": str(lookup.loc[data.item_ids[item], "title"]),
        }
        for rank, item in enumerate(ranked, start=1)
    ]
    print(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.set_defaults(func=train_command)
    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("--partition", choices=["validation", "test"], default="test")
    evaluate_parser.add_argument("--k", type=int, default=10)
    evaluate_parser.set_defaults(func=evaluate_command)
    recommend = commands.add_parser("recommend")
    recommend.add_argument("--user-id", type=int, required=True)
    recommend.add_argument("--k", type=int, default=10)
    recommend.set_defaults(func=recommend_command)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
