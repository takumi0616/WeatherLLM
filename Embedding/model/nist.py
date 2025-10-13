# -*- coding: utf-8 -*-
"""
NIST score as a 'model' interface for Embedding pipeline.

Exposes:
- embed_and_score(text_a: str, text_b: str, ...) -> float

Where:
- text_a: hypothesis (generated comment)
- text_b: reference (original comment)
Returns sentence-level NIST (unbounded, >= 0). If NLTK is unavailable or fails, returns 0.0.
"""

from __future__ import annotations

from typing import Optional

from .text_metrics_common import prepare_tokens, calc_nist


def embed_and_score(
    text_a: str,
    text_b: str,
    device: str = "cpu",
    task: Optional[str] = None,
    truncate_dim: Optional[int] = None,
    n: int = 5,
) -> float:
    """
    Compute sentence-level NIST score between hypothesis (text_a) and reference (text_b).
    Normalize to [0,1] by dividing by self-score (ref vs ref). Exact match returns 1.0.
    """
    # Exact-match guard (robust to library/fallback differences)
    if (text_a or "").strip() == (text_b or "").strip():
        return 1.0

    ref_tokens, hyp_tokens = prepare_tokens(text_a, text_b)

    # If token sequences are identical and non-empty, treat as perfect match
    if ref_tokens == hyp_tokens and len(ref_tokens) > 0:
        return 1.0

    raw = float(calc_nist(ref_tokens, hyp_tokens, n=n))
    self_max = float(calc_nist(ref_tokens, ref_tokens, n=n))

    if self_max > 0.0:
        val = raw / self_max
        if val < 0.0:
            val = 0.0
        if val > 1.0:
            val = 1.0
        return val

    # If we cannot compute a meaningful self-score, return 1.0 for any positive raw, else 0.0
    return 1.0 if raw > 0.0 else 0.0
