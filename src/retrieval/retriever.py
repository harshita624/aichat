from __future__ import annotations

import json
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import faiss
from sentence_transformers import SentenceTransformer

from src.config import SETTINGS, get_logger
from src.evaluation.leakage import exclude_self_from_results
from src.retrieval.index import FAISS_INDEX_PATH, CORPUS_RECORDS_PATH

logger = get_logger(__name__)


class Retriever:
    def __init__(self) -> None:
        if not FAISS_INDEX_PATH.exists() or not CORPUS_RECORDS_PATH.exists():
            raise FileNotFoundError(
                "Run python -m src.retrieval.index first."
            )

        self.index = faiss.read_index(
            str(FAISS_INDEX_PATH)
        )
        self.records = json.loads(
            CORPUS_RECORDS_PATH.read_text()
        )

        try:
            self.model = SentenceTransformer(
                SETTINGS.embedding_model,
                local_files_only=True,
            )
        except TypeError:
            self.model = SentenceTransformer(
                SETTINGS.embedding_model
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not load the embedding model from the local cache. "
                "Run scripts/setup_data.py once with network access, or set "
                "EMBEDDING_MODEL to a local sentence-transformers model path."
            ) from exc

    def search(
        self,
        query: str,
        top_k: int = 5,
        exclude_conversation_id: str | None = None,
        exclude_message_text: str | None = None,
    ) -> list[dict]:
        if not query or not query.strip():
            return []

        any_exclusion_active = bool(
            exclude_conversation_id or exclude_message_text
        )
        fetch_k = top_k + 3 if any_exclusion_active else top_k

        query_vec = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        scores, indices = self.index.search(
            query_vec,
            min(fetch_k, self.index.ntotal),
        )

        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.records):
                continue

            record = self.records[idx]

            results.append(
                {
                    "conversation_id": record["conversation_id"],
                    "historical_customer_message": record[
                        "customer_message"
                    ],
                    "historical_agent_response": record[
                        "agent_response"
                    ],
                    "full_context": record.get("full_context", ""),
                    "similarity": round(float(score), 4),
                }
            )

        return exclude_self_from_results(
            results,
            exclude_conversation_id,
            exclude_message_text,
        )[:top_k]


if __name__ == "__main__":
    for result in Retriever().search(
        "my refund has not arrived",
        top_k=3,
    ):
        print(
            f"[{result['similarity']}] "
            f"{result['historical_customer_message'][:80]} -> "
            f"{result['historical_agent_response'][:80]}"
        )