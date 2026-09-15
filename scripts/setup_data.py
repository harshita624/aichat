from __future__ import annotations

from src.config import get_logger
from src.data import inspect as inspect_module
from src.data import explore as explore_module
from src.data import clean as clean_module
from src.data import conversations as conversations_module
from src.data import sample as sample_module

logger = get_logger(__name__)


def main() -> None:
    steps = [
        ("Inspecting raw dataset", inspect_module.main),
        ("Exploring brands and selecting one", explore_module.main),
        ("Cleaning text", clean_module.main),
        ("Reconstructing conversations", conversations_module.main),
        ("Building local dev sample", sample_module.main),
    ]

    for label, fn in steps:
        print(
            f"\n{'=' * 60}\n"
            f"STEP: {label}\n"
            f"{'=' * 60}"
        )
        fn()

    print(
        "\nData pipeline complete. "
        "Next: run python -m src.intents.discover"
    )


if __name__ == "__main__":
    main()