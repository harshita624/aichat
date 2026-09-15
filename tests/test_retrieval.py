import numpy as np

import faiss

from src.evaluation.leakage import exclude_self_from_results


def test_exclude_self_from_results_removes_matching_id():
    results = [
        {"conversation_id": "brand_1", "similarity": 0.99},
        {"conversation_id": "brand_2", "similarity": 0.80},
        {"conversation_id": "brand_3", "similarity": 0.75},
    ]

    filtered = exclude_self_from_results(
        results,
        exclude_conversation_id="brand_2",
    )

    assert "brand_2" not in [r["conversation_id"] for r in filtered]
    assert len(filtered) == 2


def test_exclude_self_from_results_no_op_when_id_is_none():
    results = [
        {"conversation_id": "brand_1", "similarity": 0.99}
    ]

    assert exclude_self_from_results(
        results,
        exclude_conversation_id=None,
    ) == results


def test_faiss_inner_product_retrieval_returns_nearest_match():
    rng = np.random.default_rng(42)

    base = rng.normal(
        size=(10, 16)
    ).astype("float32")

    base /= np.linalg.norm(
        base,
        axis=1,
        keepdims=True,
    )

    index = faiss.IndexFlatIP(16)
    index.add(base)

    query = base[3:4].copy()

    scores, indices = index.search(query, 1)

    assert indices[0][0] == 3
    assert scores[0][0] > 0.99