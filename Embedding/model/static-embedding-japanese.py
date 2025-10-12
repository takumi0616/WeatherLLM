# -*- coding: utf-8 -*-
"""
hotchpotch/static-embedding-japanese を Hugging Face から取得して使用可能にするラッパ。
- 参考実装: src/WeatherLLM/llava.py のダウンロード/ローカル保存方針
- SentenceTransformer >= 3.3.1 を想定（model.similarity 対応）

提供関数:
- get_model(device: str = "cpu", truncate_dim: Optional[int] = None, local_dir: Optional[str] = None)
    -> SentenceTransformer
- embed_texts(model, texts, convert_to_tensor: bool = True, normalize: bool = True)
    -> torch.Tensor | np.ndarray
- embed_and_score(text_a: str, text_b: str, device: str = "cpu", truncate_dim: Optional[int] = None, local_dir: Optional[str] = None)
    -> float  # コサイン類似度（-1..1）

注意:
- オフラインでも使えるように、存在しなければ huggingface_hub.snapshot_download でローカルに保存します
- HF_TOKEN 環境変数が設定されていればそれを使用
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Union

import torch

from llm_utils import (
    setup_logger,
    get_hf_token,
    safe_model_dirname,
)

LOGGER = setup_logger("static-embedding-japanese")

# 依存: sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    raise RuntimeError(
        "sentence-transformers が見つかりません。`pip install -U sentence-transformers>=3.3.1` を実行してください。"
    ) from e

# 依存: huggingface_hub（ローカルスナップショット保存用）
try:
    from huggingface_hub import snapshot_download  # type: ignore
    HAS_HF_HUB = True
except Exception:
    HAS_HF_HUB = False


MODEL_ID = "hotchpotch/static-embedding-japanese"
# StaticEmbedding は 32, 64, 128, 256, 512, 1024 を truncate_dim に指定可能（モデルカード記載）
VALID_TRUNCATE_DIMS = {32, 64, 128, 256, 512, 1024}


def _has_model_files(local_dir: str) -> bool:
    """ローカルにモデルらしきファイル群があるか簡易判定"""
    if not os.path.isdir(local_dir):
        return False
    files = set(os.listdir(local_dir))
    # SentenceTransformer 形式の目印になりやすいファイル群
    keys = {
        "config.json",
        "modules.json",
    }
    if not keys.issubset(files):
        return False
    # 重みファイルがあるか（.safetensors / .bin）
    has_weight = any(fn.endswith((".safetensors", ".bin")) for fn in files)
    return has_weight


def _ensure_local_model_dir(
    model_id: str = MODEL_ID,
    local_dir: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> str:
    """
    モデルをローカルディレクトリへ保存（存在すればスキップ）。
    - huggingface_hub が無い場合は、Hub からの自動ダウンロードを行わず、model_id をそのまま返す
      （SentenceTransformer 側でオンライン読み込みされる想定）
    """
    if local_dir is None:
        safe_id = safe_model_dirname(model_id)
        local_dir = os.path.join("models", safe_id)

    if _has_model_files(local_dir):
        LOGGER.info(f"[static-embedding] local snapshot found. skip download: {local_dir}")
        return local_dir

    if not HAS_HF_HUB:
        LOGGER.warning(
            "[static-embedding] huggingface_hub が見つからないため、ローカル保存はスキップします。"
            "オンラインで SentenceTransformer から直接読み込みます。"
        )
        return model_id

    os.makedirs(local_dir, exist_ok=True)
    LOGGER.info(f"[static-embedding] downloading snapshot to: {local_dir}")
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        token=hf_token,
        local_dir_use_symlinks=False,
    )
    return local_dir


def get_model(
    device: str = "cpu",
    truncate_dim: Optional[int] = None,
    local_dir: Optional[str] = None,
) -> "SentenceTransformer":
    """
    StaticEmbedding (Japanese) モデルをロードして返す。
    - device: "cpu" / "cuda"（自動で 'cuda:0' を使います。複数GPU対応は本関数では考慮外）
    - truncate_dim: {32, 64, 128, 256, 512, 1024} から選択（None なら 1024 次元のまま）
    - local_dir: 任意。指定がなければ models/{safe_model_dirname(MODEL_ID)} に保存

    戻り値: SentenceTransformer インスタンス
    """
    if truncate_dim is not None and truncate_dim not in VALID_TRUNCATE_DIMS:
        raise ValueError(f"truncate_dim は {sorted(VALID_TRUNCATE_DIMS)} から選んでください: got {truncate_dim}")

    hf_token = get_hf_token(enable_transfer=True)
    path_or_id = _ensure_local_model_dir(MODEL_ID, local_dir, hf_token)

    # SentenceTransformer 3.3.1 以降を想定
    model = SentenceTransformer(
        path_or_id,
        device=("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu") if device in ("cuda", "cpu") else device,
        truncate_dim=truncate_dim,
    )
    LOGGER.info(f"[static-embedding] model loaded from: {path_or_id} (device={model.device}, truncate_dim={truncate_dim})")
    return model


def embed_texts(
    model: "SentenceTransformer",
    texts: Union[str, Sequence[str]],
    convert_to_tensor: bool = True,
    normalize: bool = True,
):
    """
    テキスト群をエンコードしてベクトルを返す。
    - convert_to_tensor: True なら torch.Tensor、False なら np.ndarray
    - normalize: True で正規化（コサイン類似度用途向け）
    """
    if isinstance(texts, str):
        texts = [texts]
    return model.encode(
        list(texts),
        convert_to_tensor=convert_to_tensor,
        normalize_embeddings=normalize,
        show_progress_bar=False,
    )


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """torch.nn.functional.cosine_similarity の薄いラッパ。1D 同士のとき float を返す。"""
    if a.dim() == 1:
        a = a.unsqueeze(0)
    if b.dim() == 1:
        b = b.unsqueeze(0)
    sim = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    return float(sim.squeeze().item())


def embed_and_score(
    text_a: str,
    text_b: str,
    device: str = "cpu",
    truncate_dim: Optional[int] = None,
    local_dir: Optional[str] = None,
) -> float:
    """
    2つの文章のコサイン類似度を返すユーティリティ。
    - モデルのロード → エンコード → 類似度算出 までを一括実行
    """
    model = get_model(device=device, truncate_dim=truncate_dim, local_dir=local_dir)

    # SentenceTransformer v3 では model.similarity も利用可能（戻り torch.Tensor）
    try:
        vecs = model.encode([text_a, text_b], convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
        if hasattr(model, "similarity"):
            sims = model.similarity(vecs[0], vecs[1])  # shape: (1,1)
            return float(sims.item())
        # フォールバックで自前計算
        return _cosine_sim(vecs[0], vecs[1])
    except Exception:
        # さらに堅牢化: 逐次 encode
        a = embed_texts(model, text_a, convert_to_tensor=True, normalize=True)
        b = embed_texts(model, text_b, convert_to_tensor=True, normalize=True)
        return _cosine_sim(a[0], b[0])


if __name__ == "__main__":
    # 簡易テスト
    qa = "美味しいラーメン屋に行きたい"
    qb = "隠れた豚骨の名店がある。スープが最高で、麺の硬さも選べる。"

    model = get_model(device="cpu", truncate_dim=128)
    emb = embed_texts(model, [qa, qb], convert_to_tensor=True, normalize=True)
    if hasattr(model, "similarity"):
        sim = float(model.similarity(emb[0], emb[1]).item())
    else:
        sim = _cosine_sim(emb[0], emb[1])

    print(f"[self-test] similarity={sim:.4f} (truncate_dim=128)")
