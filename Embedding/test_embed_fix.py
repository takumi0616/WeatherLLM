# -*- coding: utf-8 -*-
"""
BGE-M3 と embeddinggemma-300m の修正動作確認用スクリプト

実行例:
python src/WeatherLLM/Embedding/test_embed_fix.py
"""

import os
import sys
from typing import Optional

# 画像ユーティリティ読み込み回避（transformers が torchvision を要求しないように）
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

# import パス調整（llm_utils を解決するために src/WeatherLLM を追加）
_THIS_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))  # src/WeatherLLM
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch  # noqa: E402
from model import BGE_M3 as bge_m3_mod  # noqa: E402
from model import embeddinggemma_300m as gemma_mod  # noqa: E402


def _run_bge_m3(text_a: str, text_b: str, device: str) -> Optional[float]:
    try:
        sim = bge_m3_mod.embed_and_score(
            text_a, text_b, device=device, task="sts", max_length=512
        )
        print(f"[TEST] BGE-M3 similarity(STS)={sim:.6f}")
        return sim
    except Exception as e:
        print(f"[TEST] BGE-M3 FAILED: {e.__class__.__name__}: {e}")
        return None


def _run_gemma(text_a: str, text_b: str, device: str) -> Optional[float]:
    try:
        sim = gemma_mod.embed_and_score(
            text_a, text_b, device=device, task="sts", truncate_dim=256, max_length=512
        )
        print(f"[TEST] embeddinggemma-300m similarity(STS, truncate_dim=256)={sim:.6f}")
        return sim
    except Exception as e:
        print(f"[TEST] embeddinggemma-300m FAILED: {e.__class__.__name__}: {e}")
        return None


def main() -> None:
    qa = "今日は雨です。傘を持って行きましょう。"
    qb = "外は雨模様なので、出かける時は傘が必要です。"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[TEST] device={device}, torch={torch.__version__}")

    sim_bge = _run_bge_m3(qa, qb, device=device)
    sim_gemma = _run_gemma(qa, qb, device=device)

    # どちらかが成功していれば 0 で終了、両方失敗は 1
    if (sim_bge is None) and (sim_gemma is None):
        sys.exit(1)


if __name__ == "__main__":
    main()
