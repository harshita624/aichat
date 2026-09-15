from __future__ import annotations

from src.intents import discover as discover_module
from src.intents import taxonomy as taxonomy_module
from src.baselines import tfidf as tfidf_module
from src.retrieval import index as index_module


def main() -> None:
    for label, fn in [
        ("Discovering intent clusters", discover_module.main),
        ("Building intent taxonomy", taxonomy_module.main),
        ("Building TF-IDF baseline", tfidf_module.main),
        ("Building retrieval index", index_module.build_index),
    ]:
        print(
            f"\n{'=' * 60}\n"
            f"STEP: {label}\n"
            f"{'=' * 60}"
        )
        fn()

    print(
        "\nPipeline complete. Next: annotate "
        "data/golden/golden_set_template.csv, "
        "then run python -m scripts.run_evaluation"
    )


if __name__ == "__main__":
    main()