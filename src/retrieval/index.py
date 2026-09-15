from __future__ import annotations

import json

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.config import (
    DATA_PROCESSED,
    SELECTED_BRAND_PATH,
    SETTINGS,
    conversations_path_for,
    get_logger,
)

logger = get_logger(__name__)

FAISS_INDEX_PATH = DATA_PROCESSED / "retrieval.faiss"
CORPUS_RECORDS_PATH = DATA_PROCESSED / "retrieval_corpus.json"


def build_index() -> None:
    if not SELECTED_BRAND_PATH.exists():
        raise FileNotFoundError(
            "No selected brand found. Run the data pipeline first."
        )

    brand = json.loads(
        SELECTED_BRAND_PATH.read_text()
    )["brand"]

    conv_path = conversations_path_for(brand)

    if not conv_path.exists():
        raise FileNotFoundError(
            f"No conversations file at {conv_path}."
        )

    df = pd.read_csv(conv_path).dropna(
        subset=["customer_message", "agent_response"]
    )

    logger.info(
        f"Loading embedding model: {SETTINGS.embedding_model} "
        "(first run downloads it)..."
    )

    model = SentenceTransformer(
        SETTINGS.embedding_model
    )

    texts = df["customer_message"].astype(str).tolist()

    logger.info(
        f"Embedding {len(texts)} customer messages..."
    )

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(
        index,
        str(FAISS_INDEX_PATH),
    )

    records = df[
        [
            "conversation_id",
            "customer_message",
            "agent_response",
            "full_context",
        ]
    ].to_dict(orient="records")

    CORPUS_RECORDS_PATH.write_text(
        json.dumps(records)
    )

    logger.info(
        f"Saved FAISS index ({index.ntotal} vectors) "
        f"and {len(records)} corpus records."
    )


if __name__ == "__main__":
    build_index()