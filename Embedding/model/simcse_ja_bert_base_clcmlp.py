# -*- coding: utf-8 -*-
"""
cl-nagoya/simcse-ja-bert-base-clcmlp を Hugging Face から取得して使用可能にするラッパ。

提供関数:
- get_model(device: str = "cpu", local_dir: Optional[str] = None) -> SentenceTransformer
- embed_texts(model, texts, convert_to_tensor: bool = True, normalize: bool = True)
- embed_and_score(text_a: str, text_b: str, device: str = "cpu",
                  task: str = "sts", local_dir: Optional[str] = None) -> float
  2文のコサイン類似度（-1..1）を返す

仕様:
- 既存の各モデル実装（ruri/e5/static/plamo 等）と同じ流儀で、huggingface_hub によりローカルスナップショットを確保
- すでにローカルに保存済みであれば再ダウンロードせずスキップ
- Sentence-Transformers 5.1.1 / Transformers 4.57.0 環境で動作
- SimCSE は特段のプロンプト/プレフィックス不要の対称エンコード（全タスク同様に比較）

環境変数:
- SIMCSE_JA_BERT_BASE_CLCMLP_ID          : 使用する HF モデルIDを上書き（デフォルト: pkshatech/simcse-ja-bert-base-clcmlp）
- SIMCSE_JA_BERT_BASE_CLCMLP_LOCAL_DIR   : 既にダウンロード済みのローカルディレクトリ（config.json / modules.json 等が存在するST形式のディレクトリ）
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

LOGGER = setup_logger("simcse-ja-bert-base-clcmlp")

# 依存: sentence-transformers
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception as e:
    raise RuntimeError(
        "sentence-transformers が見つかりません。`pip install -U sentence-transformers` を実行してください。"
    ) from e

# 依存: huggingface_hub（スナップショット保存）
try:
    from huggingface_hub import snapshot_download  # type: ignore
    HAS_HF_HUB = True
except Exception:
    HAS_HF_HUB = False

# 404 の詳細判定用（あれば使用、無ければ汎用 Exception を使用）
try:
    from huggingface_hub.errors import RepositoryNotFoundError  # type: ignore
except Exception:
    class RepositoryNotFoundError(Exception):  # type: ignore
        pass


DEFAULT_MODEL_ID = "pkshatech/simcse-ja-bert-base-clcmlp"


def _resolve_model_id() -> str:
    """
    既定モデルIDの解決。環境変数で上書き可。
    例:
      export SIMCSE_JA_BERT_BASE_CLCMLP_ID='owner/repo'
    """
    mid = os.getenv("SIMCSE_JA_BERT_BASE_CLCMLP_ID", "").strip()
    return mid or DEFAULT_MODEL_ID


def _has_model_files(local_dir: str) -> bool:
    """ローカルに SentenceTransformer 形式の主要ファイルがあるか簡易判定"""
    if not os.path.isdir(local_dir):
        return False
    files = set(os.listdir(local_dir))
    # SentenceTransformer の目印になりやすい
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

    環境変数:
      - SIMCSE_JA_BERT_BASE_CLCMLP_LOCAL_DIR: 既にダウンロード済みのローカルディレクトリを指定するとそれを優先採用
      - SIMCSE_JA_BERT_BASE_CLCMLP_ID      : 使用する HF モデルIDを上書き
    """
    model_id = model_id or _resolve_model_id()

    # 1) ローカルディレクトリの環境変数が指定され、かつ実体がある場合はそれを優先
    env_local = os.getenv("SIMCSE_JA_BERT_BASE_CLCMLP_LOCAL_DIR", "").strip()
    if env_local:
        if _has_model_files(env_local):
            LOGGER.info(f"[simcse-bert] using local dir from env: {env_local}")
            return env_local
        else:
            LOGGER.warning(f"[simcse-bert] env SIMCSE_JA_BERT_BASE_CLCMLP_LOCAL_DIR is set but not a valid ST dir: {env_local}")

    # 2) 既定の保存先を決定
    if local_dir is None:
        safe_id = safe_model_dirname(model_id)
        local_dir = os.path.join("models", safe_id)

    # 3) 既にローカルにあればそのまま使用
    if _has_model_files(local_dir):
        LOGGER.info(f"[simcse-bert] local snapshot found. skip download: {local_dir}")
        return local_dir

    # 4) huggingface_hub が無い場合はダウンロードを諦め、model_id を返してオンライン読み込みに委譲
    if not HAS_HF_HUB:
        LOGGER.warning(
            "[simcse-bert] huggingface_hub が見つからないため、ローカル保存はスキップします。"
            "オンラインで SentenceTransformer から直接読み込みます。"
        )
        return model_id

    # 5) Hub から取得（404 なら分かりやすいエラーに差し替え）
    os.makedirs(local_dir, exist_ok=True)
    LOGGER.info(f"[simcse-bert] downloading snapshot to: {local_dir} (repo_id={model_id})")
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            token=hf_token,
            local_dir_use_symlinks=False,
        )
        return local_dir
    except RepositoryNotFoundError as e:
        raise RuntimeError(
            "simcse-ja-bert-base-clcmlp のモデルリポジトリが見つかりませんでした。\n"
            f"- 指定された repo_id: '{model_id}' は存在しない可能性があります。\n"
            "- 対応策:\n"
            "  1) 正しい Hugging Face のモデルIDを環境変数で指定してください:\n"
            "     export SIMCSE_JA_BERT_BASE_CLCMLP_ID='owner/repo_name'\n"
            "  2) 既にダウンロード済みのローカルパスを指定してください:\n"
            "     export SIMCSE_JA_BERT_BASE_CLCMLP_LOCAL_DIR='/path/to/local/model_dir'\n"
            "     (config.json / modules.json と重みファイルが含まれているディレクトリ)\n"
            "  3) private/gated の場合は認証トークン(HF_TOKEN 等)を設定してください。\n"
        ) from e
    except Exception as e:
        # 通信・権限・その他の失敗はオンライン読み込みへフォールバック（ST 側で詳細が出る）
        LOGGER.warning(f"[simcse-bert] snapshot_download failed ({type(e).__name__}): {e}. Fallback to online load by SentenceTransformer.")
        return model_id


def get_model(
    device: str = "cpu",
    local_dir: Optional[str] = None,
) -> "SentenceTransformer":
    """
    simcse-ja-bert-base-clcmlp をロードして返す。
    - device: "cpu" / "cuda"（自動で 'cuda:0' を使用）
    """
    model_id = _resolve_model_id()
    hf_token = get_hf_token(enable_transfer=True)
    path_or_id = _ensure_local_model_dir(model_id, local_dir, hf_token)

    model = SentenceTransformer(
        path_or_id,
        device=("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu") if device in ("cuda", "cpu") else device,
    )
    LOGGER.info(f"[simcse-bert] model loaded from: {path_or_id} (device={model.device})")
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
        # normalize_embeddings 非対応の場合のフォールバック
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
    - SimCSE は対称エンコード（タスクに関わらず特別な prefix 等は付与しない）
    """
    _ = task  # インターフェース整合のために受け取るが未使用（SimCSEは対称）
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


# ------------- Convenience self-test -------------

if __name__ == "__main__":
    qa = "今日は雨です。傘を持って行きましょう。"
    qb = "外は雨模様なので、出かける時は傘が必要です。"
    sim = embed_and_score(qa, qb, device="cpu")
    print(f"[self-test] simcse-ja-bert-base-clcmlp similarity={sim:.4f}")
