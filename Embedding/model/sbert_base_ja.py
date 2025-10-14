# -*- coding: utf-8 -*-
"""
sbert-base-ja を Hugging Face から取得して使用可能にするラッパ。

要件:
- Sentence-Transformers: 5.1.1
- Transformers: 4.57.0
- 可能なら huggingface_hub によりローカルへスナップショット保存（存在すれば再DLしない）

提供関数:
- get_model(device: str = "cpu", local_dir: Optional[str] = None) -> SentenceTransformer
- embed_texts(model, texts, convert_to_tensor: bool = True, normalize: bool = True)
- embed_and_score(text_a: str, text_b: str, device: str = "cpu", task: str = "sts",
                  local_dir: Optional[str] = None) -> float
  2文のコサイン類似度（-1..1）を返す（対称エンコード、task は互換のため受け取るが未使用）

環境変数（任意）:
- SBERT_BASE_JA_ID         : 使用したい HF モデルIDを明示（例: cl-nagoya/sbert-base-ja）
- SBERT_BASE_JA_LOCAL_DIR  : 既にダウンロード済みのローカル ST 形式ディレクトリを直接指定
"""

from __future__ import annotations

import os
from typing import Optional, Sequence, Union, List

import torch

from llm_utils import (
    setup_logger,
    get_hf_token,
    safe_model_dirname,
)

LOGGER = setup_logger("sbert-base-ja")

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


def _resolve_model_ids() -> List[str]:
    """
    使用候補のモデルIDリストを返す。環境変数 SBERT_BASE_JA_ID があればそれを最優先で使用。
    既知の公開モデル（SBERT系/代替）を優先候補として列挙。
    """
    env_id = os.getenv("SBERT_BASE_JA_ID", "").strip()
    if env_id:
        return [env_id]
    # 代表的な候補。上から順に試す（公開確認済みのIDを優先）。
    candidates = [
        "sonoisa/sentence-bert-base-ja-mean-tokens-v2",
        "sonoisa/sentence-bert-base-ja-mean-tokens",
        "oshizo/sbert-jsnli-luke-japanese-base-lite",
    ]
    return candidates


def _has_model_files(local_dir: str) -> bool:
    """
    ローカルに SentenceTransformer 形式の主要ファイルがあるか簡易判定。
    典型: config.json, modules.json (+ 重み .safetensors / .bin)
    """
    if not os.path.isdir(local_dir):
        return False
    files = set(os.listdir(local_dir))
    keys = {"config.json", "modules.json"}
    if not keys.issubset(files):
        return False
    has_weight = any(fn.endswith((".safetensors", ".bin")) for fn in files)
    return has_weight


def _ensure_local_any(
    model_ids: Sequence[str],
    local_dir: Optional[str],
    hf_token: Optional[str],
) -> Union[str, os.PathLike]:
    """
    複数候補の任意モデルIDについて、ローカルディレクトリへ保存（存在すればスキップ）。
    - SBERT_BASE_JA_LOCAL_DIR が ST 形式で有効ならそれを優先採用
    - local_dir が明示された場合はそこに保存（既存ならそれを使用）
    - local_dir が未指定の場合は候補IDごとに models/{safe_model_dirname(repo)} へ保存
    - huggingface_hub が無い場合は最初の候補IDを返し、SentenceTransformer 側のオンライン読み込みに委譲
    """
    # 1) 事前ダウンロード済みのローカル指定を優先
    env_local = os.getenv("SBERT_BASE_JA_LOCAL_DIR", "").strip()
    if env_local:
        if _has_model_files(env_local):
            LOGGER.info(f"[sbert-base-ja] using local dir from env: {env_local}")
            return env_local
        else:
            LOGGER.warning(f"[sbert-base-ja] SBERT_BASE_JA_LOCAL_DIR は ST 形式として無効です: {env_local}")

    # 2) local_dir が指定されている場合、まずその有効性を確認
    if local_dir:
        if _has_model_files(local_dir):
            LOGGER.info(f"[sbert-base-ja] local snapshot found. skip download: {local_dir}")
            return local_dir
        # 指定はあるが中身が無ければ、このディレクトリに対して候補IDからのダウンロードを試す
        target_dirs = [(rid, local_dir) for rid in model_ids]
    else:
        # 候補IDごとに独立した保存先を用意
        target_dirs = [(rid, os.path.join("models", safe_model_dirname(rid))) for rid in model_ids]

    # 3) huggingface_hub が無い場合は、最初の候補IDを返してオンライン読み込みに委譲
    if not HAS_HF_HUB:
        primary_id = model_ids[0] if model_ids else "sonoisa/sentence-bert-base-ja-mean-tokens-v2"
        LOGGER.warning(
            "[sbert-base-ja] huggingface_hub が見つからないため、ローカル保存はスキップします。"
            f"オンラインで SentenceTransformer から直接読み込みます: {primary_id}"
        )
        return primary_id

    # 4) 各候補IDについて、既存ローカルを優先し、無ければ取得を試みる
    last_err: Optional[Exception] = None
    for rid, tdir in target_dirs:
        try:
            if _has_model_files(tdir):
                LOGGER.info(f"[sbert-base-ja] local snapshot found. skip download: {tdir}")
                return tdir
            os.makedirs(tdir, exist_ok=True)
            LOGGER.info(f"[sbert-base-ja] downloading snapshot to: {tdir} (repo_id={rid})")
            snapshot_download(
                repo_id=rid,
                local_dir=tdir,
                token=hf_token,
                local_dir_use_symlinks=False,
            )
            return tdir
        except Exception as e:
            last_err = e
            LOGGER.warning(f"[sbert-base-ja] snapshot_download failed for {rid}: {e}")

    # 5) すべて失敗 -> オンライン読み込みに委譲（最初の候補IDにする）
    primary_id = model_ids[0] if model_ids else "sonoisa/sentence-bert-base-ja-mean-tokens-v2"
    if last_err is not None:
        LOGGER.warning(f"[sbert-base-ja] all candidates failed to download. fallback to online load: {primary_id}")
    return primary_id


def get_model(
    device: str = "cpu",
    local_dir: Optional[str] = None,
) -> "SentenceTransformer":
    """
    sbert-base-ja モデルをロードして返す。
    - device: "cpu" / "cuda"（自動で 'cuda:0' を選択）
    - local_dir: 任意（未指定なら models/{safe_model_dirname(primary_id)}）
    """
    candidates = _resolve_model_ids()
    hf_token = get_hf_token(enable_transfer=True)
    path_or_id = _ensure_local_any(candidates, local_dir, hf_token)

    st_device = ("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu") if device in ("cuda", "cpu") else device
    model = SentenceTransformer(
        path_or_id,
        device=st_device,
    )
    LOGGER.info(f"[sbert-base-ja] model loaded from: {path_or_id} (device={model.device})")
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
    - 本モデルは特別なプレフィックス不要の対称エンコード
    - task 引数はインターフェース整合のため受け取るが未使用
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
    print(f"[self-test] sbert-base-ja similarity={sim:.4f}")
