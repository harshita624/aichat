from __future__ import annotations

from src.evaluation import evaluate as evaluate_module
from src.evaluation import judge as judge_module
from src.evaluation import judge_agreement as judge_agreement_module
from src.evaluation import failure_analysis as failure_analysis_module


def main() -> None:
    for label, fn in [
        (
            "Running evaluation against golden set",
            evaluate_module.main,
        ),
        (
            "Running LLM-as-judge",
            judge_module.main,
        ),
        (
            "Computing judge/human agreement",
            judge_agreement_module.main,
        ),
        (
            "Running failure analysis",
            failure_analysis_module.main,
        ),
    ]:
        print(
            f"\n{'=' * 60}\n"
            f"STEP: {label}\n"
            f"{'=' * 60}"
        )
        fn()


if __name__ == "__main__":
    main()