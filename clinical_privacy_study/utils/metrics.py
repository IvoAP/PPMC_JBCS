"""Tie-aware importance ranking and paired comparison utilities."""

import numpy as np
from scipy.stats import kendalltau, rankdata, spearmanr


def ranking(scores: dict[str, float]) -> dict[str, float]:
    """Rank high scores first, assigning average ranks to ties."""
    ranks = rankdata([-score for score in scores.values()], method="average")
    return dict(zip(scores, ranks, strict=True))


def compare_scores(original: dict[str, float], other: dict[str, float], k: int) -> dict:
    """Compare the same features; ties at the top-k boundary break by name."""
    if original.keys() != other.keys() or not original or k < 1:
        raise ValueError("Rank comparisons require identical features and positive k.")
    k = min(k, len(original))
    a, b = [np.array([scores[key] for key in original]) for scores in (original, other)]
    defined = np.ptp(a) > 0 and np.ptp(b) > 0
    left, right = [
        set(sorted(scores, key=lambda key: (-scores[key], key))[:k])
        for scores in (original, other)
    ]
    return {
        "k": k, "overlap_count": len(left & right), "top_k_overlap": len(left & right) / k,
        "jaccard": len(left & right) / len(left | right),
        "spearman": float(spearmanr(a, b).statistic) if defined else None,
        "kendall": float(kendalltau(a, b).statistic) if defined else None,
        "importance_difference": {key: other[key] - original[key] for key in original},
    }
