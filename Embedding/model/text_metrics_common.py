# -*- coding: utf-8 -*-
"""
Common helpers for text-based metrics (BLEU, NIST, ROUGE-1 F1) used under Embedding/model.
This module avoids external dependencies when possible and provides graceful fallbacks.
"""

from __future__ import annotations

from typing import List, Tuple

# Optional deps
try:
    import sacrebleu  # type: ignore
except Exception:
    sacrebleu = None

try:
    from rouge_score import rouge_scorer  # type: ignore
except Exception:
    rouge_scorer = None

try:
    from nltk.translate import nist_score as _nist  # type: ignore
except Exception:
    _nist = None


def tokenize_ja(text: str) -> List[str]:
    """
    Tokenize Japanese text. Prefer MeCab (fugashi), then Janome; fallback to char-level tokens.
    """
    try:
        from fugashi import Tagger  # type: ignore
        tagger = Tagger()
        return [w.surface for w in tagger(text)]
    except Exception:
        try:
            from janome.tokenizer import Tokenizer  # type: ignore
            return [t.surface for t in Tokenizer().tokenize(text)]
        except Exception:
            return list(text)  # fallback: char-level


def normalize_tokens(tokens: List[str]) -> List[str]:
    """
    Simple normalization: strip, remove punctuation-like marks, lowercase, remove empties.
    """
    import re as _re
    cleaned: List[str] = []
    for t in tokens:
        t = t.strip()
        t = _re.sub(r"[、。．，・：；！!？\?「」『』（）\(\)\[\]\{\}”“\"'`’\-_/\\…･･･~＾^＋+＝=＊*＜＞<>＆&％%＄$#＠@|]", "", t)
        t = _re.sub(r"\s+", "", t)
        t = t.lower()
        if t:
            cleaned.append(t)
    return cleaned


def char_level_f1(a: str, b: str) -> float:
    """
    Character-level F1 as a robust fallback for ROUGE-1 when tokenization or libraries are unavailable.
    """
    import re as _re
    from collections import Counter
    def norm(s: str) -> str:
        s = s.strip().lower()
        s = _re.sub(r"[、。．，・：；！!？\?「」『』（）\(\)\[\]\{\}”“\"'`’\-_/\\…･･･~＾^＋+＝=＊*＜＞<>＆&％%＄$#＠@|]", "", s)
        s = _re.sub(r"\s+", "", s)
        return s
    a_norm = norm(a)
    b_norm = norm(b)
    if not a_norm or not b_norm:
        return 0.0
    ca = Counter(list(a_norm))
    cb = Counter(list(b_norm))
    overlap = sum((ca & cb).values())
    if overlap == 0:
        return 0.0
    prec = overlap / sum(cb.values())
    rec = overlap / sum(ca.values())
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def calc_bleu(ref_tokens: List[str], hyp_tokens: List[str]) -> float:
    """
    BLEU using sacrebleu if available. Returns a score in [0, 1].
    """
    if sacrebleu is None:
        return 0.0
    try:
        score = sacrebleu.sentence_bleu(" ".join(hyp_tokens), [" ".join(ref_tokens)], tokenize="none").score
        return float(score) / 100.0  # normalize to 0..1
    except Exception:
        return 0.0


def calc_nist(ref_tokens: List[str], hyp_tokens: List[str], n: int = 5) -> float:
    """
    NIST sentence score using NLTK if available. Returns raw NIST (unbounded >= 0).
    If NLTK not available or fails, returns 0.0.
    """
    if _nist is None:
        return 0.0
    try:
        return float(_nist.sentence_nist([ref_tokens], hyp_tokens, n))
    except Exception:
        return 0.0


def calc_rouge1_f1(ref_tokens: List[str], hyp_tokens: List[str], ref_text: str, hyp_text: str) -> float:
    """
    ROUGE-1 F1 using rouge_score if available; fallback to char-level F1.
    Returns a score in [0, 1].
    """
    if rouge_scorer is not None:
        try:
            scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=False)
            scores = scorer.score(" ".join(ref_tokens), " ".join(hyp_tokens))
            return float(scores['rouge1'].fmeasure)
        except Exception:
            pass
    # Fallback
    return char_level_f1(ref_text, hyp_text)


def prepare_tokens(text_a: str, text_b: str) -> Tuple[List[str], List[str]]:
    """
    Convenience function: tokenize and normalize both reference and hypothesis texts.
    Returns (ref_tokens, hyp_tokens).
    """
    ref_tokens = normalize_tokens(tokenize_ja(text_b))
    hyp_tokens = normalize_tokens(tokenize_ja(text_a))
    return ref_tokens, hyp_tokens
