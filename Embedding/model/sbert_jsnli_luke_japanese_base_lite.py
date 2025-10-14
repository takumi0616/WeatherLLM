# -*- coding: utf-8 -*-
"""
oshizo/sbert-jsnli-luke-japanese-base-lite を Hugging Face から取得して使用可能にするラッパ。

要件:
- Sentence-Transformers 5.1.1
- Transformers 4.57.0
- 可能なら huggingface_hub によりローカルへスナップショット保存（存在すれば再DLしない）

提供関数:
- get_model(device: str = "cpu", local_dir: Optional[str] = None) -> SentenceTransformer
- embed_texts(model, texts, convert_to_tensor: bool = True, normalize: bool = True)
- embed_and_score(text_a: str, text_b: str, device: str = "cpu", task: str = "sts",
                  local_dir: Optional[str] = None) -> float
  2文のコサイン類似度（-1..1）を返す（対称エンコード、task は互換のため受け取るが未使用）

備考:
- 既存の他モデル実装（e5, ruri, static, simcse 等）と同じインターフェース/方針で実装
"""

from __future__ import annotations

import os
from typing import Optional, Sequence, Union

import torch

from llm_utils import (
    setup_logger,
    get_hf_token,
    safe_model_dirname,
)

LOGGER = setup_logger("sbert-jsnli-luke-ja-base-lite")

# 依存: sentence-transformers
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception as e:
    raise RuntimeError(
        "sentence-transformers が見つかりません。`pip install -U sentence-transformers` を実行してください。"
    ) from e

# 依存: huggingface_hub（ローカルスナップショット保存用）
try:
    from huggingface_hub import snapshot_download  # type: ignore
    HAS_HF_HUB = True
except Exception:
    HAS_HF_HUB = False

# モデルID（必要なら環境変数で上書き可能）
DEFAULT_MODEL_ID = "oshizo/sbert-jsnli-luke-japanese-base-lite"
ALT_MODEL_IDS = ["hotchpotch/sbert-jsnli-luke-japanese-base-lite"]


def _resolve_model_id() -> str:
    mid = os.environ.get("SBERT_JSNLI_LUKE_JA_BASE_LITE_ID", "").strip()
    return mid or DEFAULT_MODEL_ID


def _has_model_files(local_dir: str) -> bool:
    """
    ローカルに SentenceTransformer 形式の主要ファイルがあるか簡易判定。
    ST 形式の典型: config.json, modules.json (+ 重み .safetensors / .bin)
    """
    if not os.path.isdir(local_dir):
        return False
    files = set(os.listdir(local_dir))
    keys = {"config.json", "modules.json"}
    if not keys.issubset(files):
        return False
    has_weight = any(fn.endswith((".safetensors", ".bin")) for fn in files)
    return has_weight


def _ensure_local_model_dir(
    model_id: Optional[str] = None,
    local_dir: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> str:
    """
    モデルをローカルディレクトリへ保存（存在すればスキップ）。
    - huggingface_hub が無い場合は、Hub からの自動ダウンロードを行わず、model_id をそのまま返す
      （SentenceTransformer 側でオンライン読み込みにフォールバック）
    環境変数:
      - SBERT_JSNLI_LUKE_JA_BASE_LITE_LOCAL_DIR: 既にダウンロード済みのローカルディレクトリを優先使用
      - SBERT_JSNLI_LUKE_JA_BASE_LITE_ID      : 使用する HF モデルIDを上書き
    """
    model_id = model_id or _resolve_model_id()

    # 明示ローカル指定があれば優先
    env_local = os.environ.get("SBERT_JSNLI_LUKE_JA_BASE_LITE_LOCAL_DIR", "").strip()
    if env_local:
        if _has_model_files(env_local):
            LOGGER.info(f"[sbert-luke-lite] using local dir from env: {env_local}")
            return env_local
        else:
            LOGGER.warning(f"[sbert-luke-lite] env local dir is not a valid ST dir: {env_local}")

    if local_dir is None:
        safe_id = safe_model_dirname(model_id)
        local_dir = os.path.join("models", safe_id)

    if _has_model_files(local_dir):
        LOGGER.info(f"[sbert-luke-lite] local snapshot found. skip download: {local_dir}")
        return local_dir

    if not HAS_HF_HUB:
        LOGGER.warning(
            "[sbert-luke-lite] huggingface_hub が見つからないため、ローカル保存はスキップします。"
            "オンラインで SentenceTransformer から直接読み込みます。"
        )
        return model_id

    os.makedirs(local_dir, exist_ok=True)
    # 候補リスト（環境指定 -> 既定 -> 代替ID）の順に試行
    candidates = [model_id] + [mid for mid in (ALT_MODEL_IDS if 'ALT_MODEL_IDS' in globals() else []) if mid != model_id]
    last_err = None
    for rid in candidates:
        try:
            LOGGER.info(f"[sbert-luke-lite] downloading snapshot to: {local_dir} (repo_id={rid})")
            snapshot_download(
                repo_id=rid,
                local_dir=local_dir,
                token=hf_token,
                local_dir_use_symlinks=False,
            )
            return local_dir
        except Exception as e:
            last_err = e
            LOGGER.warning(f"[sbert-luke-lite] snapshot_download failed for {rid}: {e}")
    # すべて失敗時は最後の例外を送出
    raise last_err


def get_model(
    device: str = "cpu",
    local_dir: Optional[str] = None,
) -> "SentenceTransformer":
    """
    sbert-jsnli-luke-japanese-base-lite モデルをロードして返す。
    - device: "cpu" / "cuda"（自動で 'cuda:0' を選択）
    - local_dir: 任意（未指定なら models/{safe_model_dirname(MODEL_ID)}）
    """
    model_id = _resolve_model_id()
    hf_token = get_hf_token(enable_transfer=True)
    path_or_id = _ensure_local_model_dir(model_id, local_dir, hf_token)

    st_device = ("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu") if device in ("cuda", "cpu") else device
    model = SentenceTransformer(
        path_or_id,
        device=st_device,
    )
    LOGGER.info(f"[sbert-luke-lite] model loaded from: {path_or_id} (device={model.device})")
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
    - normalize: True で L2 正規化（コサイン類似度用途向け）
    """
    if isinstance(texts, str):
        texts = [texts]
    # sentence-transformers 5.x では normalize_embeddings が利用可能
    try:
        return model.encode(
            list(texts),
            convert_to_tensor=convert_to_tensor,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
    except TypeError:
        # normalize_embeddings 非対応の古いバージョン向けフォールバック
        vecs = model.encode(
            list(texts),
            convert_to_tensor=convert_to_tensor,
            show_progress_bar=False,
        )
        if convert_to_tensor and normalize:
            vecs = torch.nn.functional.normalize(vecs, p=2, dim=-1)
        return vecs


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
    task: str = "sts",
    local_dir: Optional[str] = None,
) -> float:
    """
    2つの文章のコサイン類似度を返すユーティリティ。
    - このモデルは特別なプレフィックス不要。対称エンコードで比較。
    - task 引数はインターフェース整合のために受け取るが未使用。
    """
    _ = task  # 未使用（インターフェース整合）
    model = get_model(device=device, local_dir=local_dir)

    # まとめてエンコード（正規化有効）
    vecs = embed_texts(model, [text_a, text_b], convert_to_tensor=True, normalize=True)

    # similarity API があれば使用、なければ自前コサイン
    if hasattr(model, "similarity"):
        try:
            sims = model.similarity(vecs[0], vecs[1])  # shape: (1,1) または (1,)
            return float(sims.item())
        except Exception:
            pass
    return _cosine_sim(vecs[0], vecs[1])


if __name__ == "__main__":
    # 簡易自己テスト（CPU）
    qa = "今日は雨です。傘を持って行きましょう。"
    qb = "外は雨模様なので、出かける時は傘が必要です。"
    sim = embed_and_score(qa, qb, device="cpu")
    print(f"[self-test] sbert-jsnli-luke-japanese-base-lite similarity={sim:.4f}")
