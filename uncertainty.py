"""Deterministic, conditional request-level percentile bootstrap intervals."""

import random
import statistics


def interval(values, seed=20260905):
    if len(values) < 5:
        return {"status": "insufficient_samples", "n": len(values)}
    rng = random.Random(seed)
    estimates = sorted(statistics.median(rng.choices(values, k=len(values)))
                       for _ in range(10000))
    return {"status": "estimated", "method": "percentile-bootstrap-median",
            "confidence": 0.95, "resamples": 10000, "seed": seed,
            "n": len(values), "lower": round(estimates[249], 6),
            "upper": round(estimates[9749], 6)}


def paired(left, right):
    """Pair by retained request nonce, refusing missing/duplicate observations."""
    def indexed(rows):
        ids = [row.get("request_nonce") for row in rows]
        if not all(isinstance(key, str) and key for key in ids) or len(set(ids)) != len(ids):
            raise ValueError("paired comparison needs unique retained request nonces")
        if any(not row.get("completion_validation", {}).get("valid") for row in rows):
            raise ValueError("paired comparison requires all basic output gates to pass")
        return dict(zip(ids, rows))
    lhs, rhs = indexed(left), indexed(right)
    if lhs.keys() != rhs.keys():
        raise ValueError("paired comparison requires identical request nonces")
    if any(lhs[key]["decode_tokens_per_second"] <= 0 for key in lhs):
        raise ValueError("paired comparison requires positive baseline rates")
    differences = [100 * (rhs[key]["decode_tokens_per_second"] /
                         lhs[key]["decode_tokens_per_second"] - 1) for key in lhs]
    result = interval(differences)
    result["median_percent_change"] = statistics.median(differences)
    result["resolvable"] = (result["lower"] > 0 or result["upper"] < 0
                            if result["status"] == "estimated" else None)
    return result
