"""Run the classical benchmark and export Markdown and LaTeX tables."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recommender.data import build_implicit_feedback, load_ratings, temporal_split  # noqa: E402
from recommender.evaluation import evaluate  # noqa: E402
from recommender.models import (  # noqa: E402
    ItemKNNRecommender,
    PopularityRecommender,
    SVDRecommender,
)


def main() -> None:
    data = build_implicit_feedback(load_ratings(ROOT / "data" / "ratings.csv"))
    split = temporal_split(data)
    models = {
        "PureSVD": SVDRecommender(n_factors=24).fit(split),
        "ItemKNN": ItemKNNRecommender(top_k_neighbours=50).fit(split),
        "Popularity": PopularityRecommender().fit(split),
    }
    reports = [
        evaluate(model, split, model_name=name).as_row() for name, model in models.items()
    ]
    markdown = [
        "| Model | NDCG@10 | Recall@10 | MRR@10 | Coverage |",
        "|---|---:|---:|---:|---:|",
    ]
    latex = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Model & NDCG@10 & Recall@10 & MRR@10 & Coverage \\",
        r"\midrule",
    ]
    for row in reports:
        markdown.append(
            f"| {row['model']} | {row['ndcg']:.4f} | {row['recall']:.4f} | "
            f"{row['mrr']:.4f} | {row['coverage']:.4f} |"
        )
        latex.append(
            f"{row['model']} & {row['ndcg']:.4f} & {row['recall']:.4f} & "
            f"{row['mrr']:.4f} & {row['coverage']:.4f} \\\\"
        )
    latex.extend([r"\bottomrule", r"\end{tabular}"])
    table_dir = ROOT / "paper" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    (table_dir / "results.tex").write_text("\n".join(latex) + "\n")
    readme = ROOT / "README.md"
    content = readme.read_text()
    replacement = "<!-- RESULTS:START -->\n" + "\n".join(markdown) + "\n<!-- RESULTS:END -->"
    content = re.sub(
        r"<!-- RESULTS:START -->.*?<!-- RESULTS:END -->",
        replacement,
        content,
        flags=re.DOTALL,
    )
    readme.write_text(content)


if __name__ == "__main__":
    main()
