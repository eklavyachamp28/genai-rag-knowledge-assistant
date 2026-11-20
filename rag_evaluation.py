"""Simple retrieval and groundedness evaluation on labeled examples."""
from __future__ import annotations
import re
import math


def evaluate_retrieval(cases: list[dict], k: int = 3) -> dict[str, float]:
    """Each case: {'retrieved': [source ids], 'relevant': [source ids]}."""
    if not cases or k < 1:
        raise ValueError("cases must be nonempty and k >= 1")
    recalls, reciprocal_ranks, precisions, ndcgs = [], [], [], []
    for case in cases:
        relevant = set(case["relevant"])
        ranked = list(dict.fromkeys(case["retrieved"]))[:k]
        precisions.append(len(relevant.intersection(ranked)) / k)
        dcg = sum(1.0 / math.log2(i + 2) for i, source in enumerate(ranked) if source in relevant)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(relevant))))
        ndcgs.append(dcg / idcg if idcg else 0.0)
        recalls.append(len(relevant.intersection(ranked)) / len(relevant) if relevant else 1.0)
        reciprocal_ranks.append(next((1.0 / (i + 1) for i, source in enumerate(ranked) if source in relevant), 0.0))
    return {"recall_at_k": sum(recalls) / len(recalls), "mrr_at_k": sum(reciprocal_ranks) / len(reciprocal_ranks),
            "precision_at_k": sum(precisions) / len(precisions),
            "ndcg_at_k": sum(ndcgs) / len(ndcgs)}


def lexical_groundedness(answer: str, contexts: list[str]) -> float:
    """Transparent token-overlap proxy, NOT an LLM judge or factual faithfulness guarantee."""
    tokens = set(re.findall(r"\b\w+\b", answer.lower()))
    evidence = set(re.findall(r"\b\w+\b", " ".join(contexts).lower()))
    return len(tokens & evidence) / len(tokens) if tokens else 1.0
