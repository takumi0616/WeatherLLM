# -*- coding: utf-8 -*-
"""
BLEU score as a 'model' interface for Embedding pipeline.

Exposes:
- embed_and_score(text_a: str, text_b: str, ...) -> float

Where:
- text_a: hypothesis (generated comment)
- text_b: reference (original comment)
Returns BLEU in [0, 1]. Dependencies are optional; falls back to 0.0 if unavailable.
"""

from __future__ import annotations

from typing import Optional

from .text_metrics_common import prepare_tokens, calc_bleu


def embed_and_score(
    text_a: str,
    text_b: str,
    device: str = "cpu",
    task: Optional[str] = None,
    truncate_dim: Optional[int] = None,
) -> float:
    """
    Compute sentence-level BLEU score between hypothesis (text_a) and reference (text_b).
    Value is normalized to [0, 1].
    """
    # Exact-match guard to ensure identical texts yield 1.0 even without external libs
    if (text_a or "").strip() == (text_b or "").strip():
        return 1.0

    ref_tokens, hyp_tokens = prepare_tokens(text_a, text_b)

    # If token sequences are identical and non-empty, treat as perfect match
    if ref_tokens == hyp_tokens and len(ref_tokens) > 0:
        return 1.0

    return float(calc_bleu(ref_tokens, hyp_tokens))
