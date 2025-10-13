# -*- coding: utf-8 -*-
"""
ROUGE-1 F1 as a 'model' interface for Embedding pipeline.

Exposes:
- embed_and_score(text_a: str, text_b: str, ...) -> float

Where:
- text_a: hypothesis (generated comment)
- text_b: reference (original comment)
Returns ROUGE-1 F1 in [0, 1]. Falls back to char-level F1 if rouge_score is unavailable.
"""

from __future__ import annotations

from typing import Optional

from .text_metrics_common import prepare_tokens, calc_rouge1_f1


def embed_and_score(
    text_a: str,
    text_b: str,
    device: str = "cpu",
    task: Optional[str] = None,
    truncate_dim: Optional[int] = None,
) -> float:
    """
    Compute ROUGE-1 F1 between hypothesis (text_a) and reference (text_b).
    Value is in [0, 1].
    """
    # Exact-match guard (robust to library/fallback differences)
    if (text_a or "").strip() == (text_b or "").strip():
        return 1.0

    # prepare_tokens returns (ref_tokens, hyp_tokens)
    ref_tokens, hyp_tokens = prepare_tokens(text_a, text_b)

    # If token sequences are identical and non-empty, treat as perfect overlap
    if ref_tokens == hyp_tokens and len(ref_tokens) > 0:
        return 1.0

    return float(calc_rouge1_f1(ref_tokens, hyp_tokens, ref_text=text_b, hyp_text=text_a))
