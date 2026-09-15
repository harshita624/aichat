from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import (
    DATA_PROCESSED,
    SELECTED_BRAND_PATH,
    conversations_path_for,
    get_logger,
)
from src.text_normalize import TFIDF_STOP_WORDS, tfidf_preprocess

logger = get_logger(__name__)

N_CLUSTERS = 12
TOP_KEYWORDS_PER_CLUSTER = 8
EXAMPLES_PER_CLUSTER = 5

INTENT_MODEL_PATH = DATA_PROCESSED / "intent_model.joblib"
INTENT_CLUSTERS_PATH = DATA_PROCESSED / "intent_clusters.json"


def discover_clusters(
    texts: pd.Series,
) -> tuple[dict, TfidfVectorizer, KMeans]:
    vectorizer = TfidfVectorizer(
        preprocessor=tfidf_preprocess,
        max_df=0.5,
        min_df=3,
        stop_words=list(TFIDF_STOP_WORDS),
        ngram_range=(1, 2),
        max_features=5000,
    )

    X = vectorizer.fit_transform(texts)

    n_clusters = min(
        N_CLUSTERS,
        max(2, len(texts) // 20),
    )

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
    )

    labels = kmeans.fit_predict(X)
    feature_names = np.array(
        vectorizer.get_feature_names_out()
    )

    clusters = []

    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        size = int(mask.sum())

        if size == 0:
            continue

        centroid = kmeans.cluster_centers_[cluster_id]

        top_idx = centroid.argsort()[::-1][
            :TOP_KEYWORDS_PER_CLUSTER
        ]

        top_keywords = feature_names[
            top_idx
        ].tolist()

        cluster_texts = texts[mask]

        examples = cluster_texts.sample(
            min(
                EXAMPLES_PER_CLUSTER,
                len(cluster_texts),
            ),
            random_state=42,
        ).tolist()

        clusters.append(
            {
                "cluster_id": cluster_id,
                "size": size,
                "frequency_pct": round(
                    size / len(texts) * 100,
                    2,
                ),
                "top_keywords": top_keywords,
                "example_messages": examples,
            }
        )

    clusters.sort(
        key=lambda cluster: cluster["size"],
        reverse=True,
    )

    return (
        {
            "clusters": clusters,
            "n_clusters": n_clusters,
            "total_messages_used": len(texts),
        },
        vectorizer,
        kmeans,
    )


def main() -> None:
    if not SELECTED_BRAND_PATH.exists():
        print(
            "No selected brand found. "
            "Run python -m scripts.setup_data first."
        )
        return

    brand = json.loads(
        SELECTED_BRAND_PATH.read_text(
            encoding="utf-8"
        )
    )["brand"]

    conv_path = conversations_path_for(brand)

    if not conv_path.exists():
        print(
            f"No conversations file at {conv_path}. "
            "Run python -m src.data.conversations first."
        )
        return

    df = pd.read_csv(conv_path)

    texts = df["customer_message"].dropna().astype(str)
    texts = texts[texts.str.len() >= 3]

    if len(texts) < 40:
        print(
            f"Only {len(texts)} usable messages — "
            "too few for meaningful clustering."
        )
        return

    logger.info(
        f"Discovering intent clusters from "
        f"{len(texts)} customer messages..."
    )

    summary, vectorizer, kmeans = discover_clusters(
        texts
    )

    INTENT_CLUSTERS_PATH.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    joblib.dump(
        {
            "vectorizer": vectorizer,
            "kmeans": kmeans,
        },
        INTENT_MODEL_PATH,
    )

    print(
        f"\nDiscovered {summary['n_clusters']} clusters "
        f"from {summary['total_messages_used']} messages.\n"
    )

    for cluster in summary["clusters"]:
        print(
            f"Cluster {cluster['cluster_id']} "
            f"(size={cluster['size']}, "
            f"{cluster['frequency_pct']}%): "
            f"{cluster['top_keywords']}"
        )

    print(
        f"\nSaved to {INTENT_CLUSTERS_PATH} "
        f"and {INTENT_MODEL_PATH}"
    )
    print(
        "Next: python -m src.intents.taxonomy"
    )


if __name__ == "__main__":
    main()