from rag_evaluation import evaluate_retrieval, lexical_groundedness


def test_retrieval_metrics():
    metrics = evaluate_retrieval([
        {"retrieved": ["a", "b"], "relevant": ["b"]},
        {"retrieved": ["c", "d"], "relevant": ["c"]},
    ], k=2)
    assert metrics["recall_at_k"] == 1.0
    assert metrics["mrr_at_k"] == 0.75


def test_groundedness_proxy():
    assert lexical_groundedness("claim accepted", ["claim accepted today"]) == 1.0
