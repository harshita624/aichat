from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

from src.config import (
    DATA_PROCESSED,
    SELECTED_BRAND_PATH,
    conversations_path_for,
    get_logger,
)

logger = get_logger(__name__)

TFIDF_MODEL_PATH = DATA_PROCESSED / "tfidf_baseline_model.joblib"
INTENT_MODEL_PATH = DATA_PROCESSED / "intent_model.joblib"
TAXONOMY_PATH = DATA_PROCESSED / "taxonomy.json"


class TfidfBaseline:
    def __init__(self) -> None:
        self.intent_vectorizer = self.intent_clf = self.retrieval_vectorizer = None
        self.retrieval_matrix = self.corpus_df = None

    def fit(self, conversations_df: pd.DataFrame) -> None:
        if not INTENT_MODEL_PATH.exists() or not TAXONOMY_PATH.exists():
            raise FileNotFoundError(
                "Run src.intents.discover and src.intents.taxonomy first."
            )

        artifacts = joblib.load(INTENT_MODEL_PATH)
        cluster_vectorizer = artifacts["vectorizer"]
        kmeans = artifacts["kmeans"]

        taxonomy = json.loads(
            TAXONOMY_PATH.read_text()
        )

        cluster_to_name = {
            i["cluster_id"]: i["name"]
            for i in taxonomy["intents"]
        }

        texts = (
            conversations_df["customer_message"]
            .fillna("")
            .astype(str)
        )

        pseudo_labels = kmeans.predict(
            cluster_vectorizer.transform(texts)
        )

        pseudo_label_names = np.array(
            [
                cluster_to_name.get(c, "unknown")
                for c in pseudo_labels
            ]
        )

        logger.info(
            f"Training TF-IDF+LogReg on "
            f"{len(texts)} pseudo-labeled examples..."
        )

        self.intent_vectorizer = TfidfVectorizer(
            max_df=0.5,
            min_df=2,
            stop_words="english",
            max_features=5000,
        )

        X_intent = self.intent_vectorizer.fit_transform(
            texts
        )

        self.intent_clf = LogisticRegression(
            max_iter=1000
        )

        self.intent_clf.fit(
            X_intent,
            pseudo_label_names,
        )

        self.retrieval_vectorizer = TfidfVectorizer(
            max_df=0.5,
            min_df=1,
            stop_words="english",
            max_features=5000,
        )

        self.retrieval_matrix = (
            self.retrieval_vectorizer.fit_transform(texts)
        )

        self.corpus_df = conversations_df.reset_index(
            drop=True
        )

    def predict_intent(self, message: str) -> dict:
        vec = self.intent_vectorizer.transform(
            [message]
        )

        pred = self.intent_clf.predict(vec)[0]

        confidence = float(
            np.max(
                self.intent_clf.predict_proba(vec)[0]
            )
        )

        return {
            "intent": pred,
            "confidence": round(
                confidence,
                4,
            ),
        }

    def retrieve(
        self,
        message: str,
        top_k: int = 1,
        exclude_conversation_id: str | None = None,
    ) -> list[dict]:
        vec = self.retrieval_vectorizer.transform(
            [message]
        )

        sims = cosine_similarity(
            vec,
            self.retrieval_matrix,
        )[0]

        results = []

        for idx in np.argsort(sims)[::-1]:
            row = self.corpus_df.iloc[idx]

            if (
                exclude_conversation_id
                and row["conversation_id"]
                == exclude_conversation_id
            ):
                continue

            results.append(
                {
                    "conversation_id": row["conversation_id"],
                    "historical_customer_message": row[
                        "customer_message"
                    ],
                    "historical_agent_response": row[
                        "agent_response"
                    ],
                    "similarity": round(
                        float(sims[idx]),
                        4,
                    ),
                }
            )

            if len(results) >= top_k:
                break

        return results

    def predict(
        self,
        message: str,
        exclude_conversation_id: str | None = None,
    ) -> dict:
        intent_result = self.predict_intent(
            message
        )

        retrieved = self.retrieve(
            message,
            top_k=1,
            exclude_conversation_id=exclude_conversation_id,
        )

        reply = (
            retrieved[0]["historical_agent_response"]
            if retrieved
            else "No similar historical response found."
        )

        return {
            "intent": intent_result["intent"],
            "intent_confidence": intent_result["confidence"],
            "reply": reply,
            "retrieved_evidence": retrieved,
        }

    def save(self) -> None:
        joblib.dump(
            {
                "intent_vectorizer": self.intent_vectorizer,
                "intent_clf": self.intent_clf,
                "retrieval_vectorizer": self.retrieval_vectorizer,
                "retrieval_matrix": self.retrieval_matrix,
                "corpus_df": self.corpus_df,
            },
            TFIDF_MODEL_PATH,
        )

    @classmethod
    def load(cls) -> "TfidfBaseline":
        artifacts = joblib.load(
            TFIDF_MODEL_PATH
        )

        instance = cls()

        instance.intent_vectorizer = artifacts[
            "intent_vectorizer"
        ]
        instance.intent_clf = artifacts[
            "intent_clf"
        ]
        instance.retrieval_vectorizer = artifacts[
            "retrieval_vectorizer"
        ]
        instance.retrieval_matrix = artifacts[
            "retrieval_matrix"
        ]
        instance.corpus_df = artifacts[
            "corpus_df"
        ]

        return instance


def main() -> None:
    if not SELECTED_BRAND_PATH.exists():
        print(
            "No selected brand found. "
            "Run the data pipeline first."
        )
        return

    brand = json.loads(
        SELECTED_BRAND_PATH.read_text()
    )["brand"]

    conv_path = conversations_path_for(
        brand
    )

    if not conv_path.exists():
        print(
            f"No conversations file at {conv_path}."
        )
        return

    df = pd.read_csv(
        conv_path
    )

    baseline = TfidfBaseline()

    baseline.fit(
        df
    )

    baseline.save()

    print(
        f"TF-IDF baseline saved to "
        f"{TFIDF_MODEL_PATH}"
    )

    print(
        json.dumps(
            baseline.predict(
                "My refund has not arrived yet"
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()