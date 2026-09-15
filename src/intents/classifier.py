from __future__ import annotations

import json

import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.config import DATA_PROCESSED, get_logger
from src.intents.lexical_signals import (
    base_taxonomy_category,
    best_category,
    confidence_from_score,
)

logger = get_logger(__name__)

INTENT_MODEL_PATH = DATA_PROCESSED / "intent_model.joblib"
TAXONOMY_PATH = DATA_PROCESSED / "taxonomy.json"

OVERRIDE_CONFIDENCE_CAP = 0.35


class IntentClassifier:
    def __init__(self) -> None:
        if not INTENT_MODEL_PATH.exists() or not TAXONOMY_PATH.exists():
            raise FileNotFoundError(
                "Run python -m src.intents.discover "
                "then python -m src.intents.taxonomy first."
            )

        artifacts = joblib.load(INTENT_MODEL_PATH)

        self.vectorizer = artifacts["vectorizer"]
        self.kmeans = artifacts["kmeans"]

        taxonomy = json.loads(
            TAXONOMY_PATH.read_text(
                encoding="utf-8"
            )
        )

        self.cluster_to_name = {
            item["cluster_id"]: item["name"]
            for item in taxonomy["intents"]
        }

        self.intent_metadata = {
            item["name"]: item
            for item in taxonomy["intents"]
        }

    def _semantic_predict(self, text: str) -> dict:
        vec = self.vectorizer.transform([text])

        if vec.nnz == 0:
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "raw_similarity": 0.0,
                "cluster_id": -1,
            }

        sims = cosine_similarity(
            vec,
            self.kmeans.cluster_centers_,
        )[0]

        ranked = np.argsort(sims)[::-1]
        best_cluster = int(ranked[0])

        top1 = float(sims[best_cluster])
        top2 = (
            float(sims[ranked[1]])
            if len(ranked) > 1
            else 0.0
        )

        margin = max(
            0.0,
            min(1.0, top1 - top2),
        )

        intent_name = self.cluster_to_name.get(
            best_cluster,
            f"cluster_{best_cluster}",
        )

        return {
            "intent": intent_name,
            "confidence": round(margin, 4),
            "raw_similarity": round(top1, 4),
            "cluster_id": best_cluster,
        }

    def _resolve_taxonomy_name(
        self,
        category: str,
    ) -> str:
        if category in self.intent_metadata:
            return category

        candidates = [
            name
            for name in self.intent_metadata
            if base_taxonomy_category(name) == category
        ]

        if candidates:
            return max(
                candidates,
                key=lambda name: self.intent_metadata[name].get(
                    "frequency_pct",
                    0,
                ),
            )

        return category

    def predict(self, text: str) -> dict:
        if not text or not text.strip():
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "cluster_id": -1,
                "lexical_category": None,
                "lexical_score": 0.0,
                "method": "empty_input",
            }

        semantic = self._semantic_predict(text)

        lexical_category, lexical_score, lexical_scores = best_category(
            text
        )

        if semantic["cluster_id"] == -1:
            if lexical_category is not None:
                resolved_name = self._resolve_taxonomy_name(
                    lexical_category
                )

                lexical_confidence = min(
                    confidence_from_score(lexical_score),
                    OVERRIDE_CONFIDENCE_CAP,
                )

                return {
                    **semantic,
                    "intent": resolved_name,
                    "confidence": round(
                        lexical_confidence,
                        4,
                    ),
                    "lexical_category": lexical_category,
                    "lexical_score": round(
                        lexical_score,
                        2,
                    ),
                    "method": "lexical_only",
                }

            return {
                **semantic,
                "lexical_category": None,
                "lexical_score": 0.0,
                "method": "no_vocabulary_overlap",
            }

        result = {
            **semantic,
            "lexical_category": lexical_category,
            "lexical_score": round(
                lexical_score,
                2,
            ),
            "method": "semantic",
        }

        if lexical_category is None:
            return result

        lexical_confidence = confidence_from_score(
            lexical_score
        )

        resolved_name = self._resolve_taxonomy_name(
            lexical_category
        )

        semantic_name = semantic["intent"]

        if resolved_name == semantic_name:
            result["confidence"] = round(
                max(
                    semantic["confidence"],
                    lexical_confidence,
                ),
                4,
            )
            result["method"] = "semantic_lexical_agree"
            return result

        semantic_category = base_taxonomy_category(
            semantic_name
        )

        meta = self.intent_metadata.get(
            semantic_name
        )

        semantic_is_low_trust = (
            meta is None
            or meta.get("matched_candidate_label") is False
        )

        semantic_lexical_score = lexical_scores.get(
            semantic_category,
            0.0,
        )

        semantic_is_unsupported_magnet = (
            semantic_category is not None
            and semantic_category != lexical_category
            and semantic_lexical_score == 0.0
        )

        semantic_has_strong_lexical_disadvantage = (
            semantic_category is not None
            and semantic_category != lexical_category
            and lexical_score >= 2.0
            and lexical_score >= 2.0 * semantic_lexical_score
        )

        if (
            semantic_is_low_trust
            or semantic_is_unsupported_magnet
            or semantic_has_strong_lexical_disadvantage
        ):
            result["intent"] = resolved_name
            result["confidence"] = round(
                min(
                    max(
                        semantic["confidence"],
                        lexical_confidence,
                    ),
                    OVERRIDE_CONFIDENCE_CAP,
                ),
                4,
            )
            result["method"] = "lexical_override"
            return result

        return result

    def is_auto_eligible(
        self,
        intent_name: str,
    ) -> bool:
        if intent_name == "uncategorized__dm":
            return True

        meta = self.intent_metadata.get(
            intent_name
        )

        return bool(
            meta
            and meta.get(
                "auto_handle_eligible",
                False,
            )
        )


if __name__ == "__main__":
    clf = IntentClassifier()

    examples = [
        "I never received my refund for the order I cancelled",
        "My package is really late and the tracking hasn't updated in days",
        "I was charged twice on my card for the same order",
        "The clubcard discount didn't apply at checkout",
        "Moldy on the inside",
        "Bought this today and it's nearly a month out of date",
        "my pakage hasnt arived",
    ]

    for msg in examples:
        print(
            f"Message: {msg}\n"
            f"Prediction: {clf.predict(msg)}\n"
        )