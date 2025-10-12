# -*- coding: utf-8 -*-
"""
static-embedding-japanese を使って2つの文章の類似度（コサイン類似度）を算出するスクリプト。

要件:
- ラッパーモジュール: src/WeatherLLM/Embedding/model/static-embedding-japanese.py を用いる
  （ファイル名にハイフンが含まれるため importlib でロード）
- 比較対象の2文章は以下のファイルから読み込む（本スクリプトファイル基準の相対パス）
  ./data/generate_comment/2022_01_01_gpt_generate_v4.txt
  ./data/original_comment/2022_01_01_original.txt
- 出力: 類似度スコア（-1..1, コサイン類似度）を標準出力へ

実行例:
  python src/WeatherLLM/Embedding/main_v1.py
  python src/WeatherLLM/Embedding/main_v1.py --truncate-dim 256
  python src/WeatherLLM/Embedding/main_v1.py --device cuda --truncate-dim 128
  python src/WeatherLLM/Embedding/main_v1.py --use-first-line
"""

import os
import sys
import argparse
import importlib.util
from typing import Optional, Tuple

# 親ディレクトリ（src/WeatherLLM）を import path に追加して、llm_utils 等の相対外部 import を解決
_THIS_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))  # src/WeatherLLM
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from llm_utils import setup_logger  # noqa: E402

LOGGER = setup_logger("embedding_main")


def _load_static_embedding_module():
    """
    ファイル名にハイフンが含まれるため、importlib でモジュールをロードする。
    戻り値はモジュールオブジェクト。
    """
    module_path = os.path.join(_THIS_DIR, "model", "static-embedding-japanese.py")
    if not os.path.isfile(module_path):
        raise FileNotFoundError(f"ラッパーモジュールが見つかりません: {module_path}")
    spec = importlib.util.spec_from_file_location("static_embedding_japanese", module_path)
    if spec is None or spec.loader is None:
        raise ImportError("importlib でラッパーモジュールの spec を作成できませんでした。")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _read_text(path: str, use_first_line: bool = False) -> str:
    """UTF-8 でファイル読み込み。オプションで最初の非空行のみを使用可能。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"テキストファイルが見つかりません: {path}")
    with open(path, "r", encoding="utf-8") as f:
        if use_first_line:
            for line in f:
                s = line.strip()
                if s:
                    return s
            return ""  # すべて空行だった場合
        return f.read().strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two Japanese texts using static-embedding-japanese.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="推論デバイス（CPU推奨）")
    parser.add_argument(
        "--truncate-dim",
        type=int,
        default=None,
        choices=[32, 64, 128, 256, 512, 1024],
        help="出力次元をマトリョーシカ表現に基づき切り詰め（未指定ならデフォルト1024）",
    )
    parser.add_argument("--use-first-line", action="store_true", help="各ファイルの最初の非空行のみを文章として使用")
    return parser.parse_args()


def resolve_input_paths() -> Tuple[str, str]:
    """課題で指定された2つのテキストファイルの絶対パスを返す。"""
    a = os.path.join(_THIS_DIR, "data", "generate_comment", "2022_01_01_gpt_generate_v4.txt")
    b = os.path.join(_THIS_DIR, "data", "original_comment", "2022_01_01_original.txt")
    return a, b


def main():
    args = parse_args()

    path_a, path_b = resolve_input_paths()
    LOGGER.info(f"Text A: {path_a}")
    LOGGER.info(f"Text B: {path_b}")

    text_a = _read_text(path_a, use_first_line=args.use_first_line)
    text_b = _read_text(path_b, use_first_line=args.use_first_line)

    if not text_a or not text_b:
        raise RuntimeError("入力文章が空です。--use-first-line を外す/付けるなど調整してください。")

    mod = _load_static_embedding_module()

    # モデルをロードしてエンコード（正規化済みベクトル）→ 類似度
    model = mod.get_model(device=args.device, truncate_dim=args.truncate_dim)
    vecs = mod.embed_texts(model, [text_a, text_b], convert_to_tensor=True, normalize=True)

    # similarity API があれば使用、なければコサイン類似度を自前計算
    if hasattr(model, "similarity"):
        sim = float(model.similarity(vecs[0], vecs[1]).item())
    else:
        import torch
        sim = float(torch.nn.functional.cosine_similarity(vecs[0].unsqueeze(0), vecs[1].unsqueeze(0), dim=-1).item())

    dim = int(vecs.shape[-1])
    LOGGER.info(f"Embedding dim: {dim}")
    print(f"Cosine similarity: {sim:.6f}")


if __name__ == "__main__":
    main()
