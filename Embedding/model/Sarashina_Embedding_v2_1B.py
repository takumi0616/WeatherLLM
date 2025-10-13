# -*- coding: utf-8 -*-
"""
Sarashina-Embedding-v2-1B を Hugging Face から取得して使用可能にするラッパ。

参考:
- src/WeatherLLM/llava.py のローカル保存方針
- モデルカード: https://huggingface.co/sbintuitions/sarashina-embedding-v2-1b

提供関数:
- get_model(device: str = "cpu", local_dir: Optional[str] = None)
    -> SentenceTransformer
- add_prefix_for_task(text: str, side: str, task: str = "sts") -> str
    タスクに応じた推奨の prefix/instruction を付与した文字列を返す
- embed_texts(model, texts, convert_to_tensor: bool = True, normalize: bool = True)
    -> torch.Tensor | np.ndarray
- embed_and_score(text_a: str, text_b: str, device: str = "cpu",
                  task: str = "sts", local_dir: Optional[str] = None) -> float
    2文の類似度（コサイン類似度）を返す。STS タスクでは両文に query 形式の指示+プレフィックスを付与。

注意:
- オフラインでも使えるように、存在しなければ huggingface_hub.snapshot_download でローカルに保存します
- HF_TOKEN 環境変数が設定されていればそれを使用
- Sentence Transformers は 3.3+ 以上を想定（model.similarity があればそれを使用）
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

LOGGER = setup_logger("sarashina-embedding-v2-1b")

# 依存: sentence-transformers
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ModuleNotFoundError as e:
    raise RuntimeError(
        "sentence-transformers が見つかりません。`pip install -U sentence-transformers` を実行してください。"
    ) from e

# transformers のメジャーバージョン互換チェック（v5以降は未対応）
try:
    import transformers  # type: ignore
    _tf_major = int(str(getattr(transformers, "__version__", "0")).split(".")[0])
    if _tf_major >= 5:
        raise RuntimeError(
            f"Transformers {transformers.__version__} は未対応です。"
            "環境を 'transformers<5' に固定してから実行してください。"
        )
except Exception:
    # transformers が未インストール等のケースではここでは落とさない（環境側で解決される想定）
    pass

# 依存: huggingface_hub（ローカルスナップショット保存用）
try:
    from huggingface_hub import snapshot_download  # type: ignore
    HAS_HF_HUB = True
except Exception:
    HAS_HF_HUB = False


MODEL_ID = "sbintuitions/sarashina-embedding-v2-1b"

# モデルカード推奨のプレフィックス（主なタスク）
INSTRUCTION_STS = "task: クエリを与えるので，もっともクエリに意味が似ている一節を探してください。"
INSTRUCTION_RETRIEVAL = "task: 質問を与えるので、その質問に答えるのに役立つ関連文書を検索してください。"
INSTRUCTION_RERANK = "task: 与えられたクエリに対して、候補文書を関連度順に並べ替えてください。"
INSTRUCTION_CLUSTER = "task: 与えられたドキュメントのトピックまたはテーマを特定してください。"
INSTRUCTION_CLASSIFY = "task: 与えられたレビューを適切な評価カテゴリに分類してください。"

VALID_TASKS = {"sts", "retrieval", "reranking", "clustering", "classification"}


def _has_model_files(local_dir: str) -> bool:
    """ローカルに SentenceTransformer の主要ファイルがあるか簡易判定"""
    if not os.path.isdir(local_dir):
        return False
    files = set(os.listdir(local_dir))
    keys = {"config.json", "modules.json"}
    if not keys.issubset(files):
        return False
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
      （SentenceTransformer 側でオンライン読み込みにフォールバック）
    """
    if local_dir is None:
        safe_id = safe_model_dirname(model_id)
        local_dir = os.path.join("models", safe_id)

    if _has_model_files(local_dir):
        LOGGER.info(f"[sarashina] local snapshot found. skip download: {local_dir}")
        return local_dir

    if not HAS_HF_HUB:
        LOGGER.warning(
            "[sarashina] huggingface_hub が見つからないため、ローカル保存はスキップします。"
            "オンラインで SentenceTransformer から直接読み込みます。"
        )
        return model_id

    os.makedirs(local_dir, exist_ok=True)
    LOGGER.info(f"[sarashina] downloading snapshot to: {local_dir}")
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        token=hf_token,
        local_dir_use_symlinks=False,
    )
    return local_dir


def get_model(
    device: str = "cpu",
    local_dir: Optional[str] = None,
) -> "SentenceTransformer":
    """
    Sarashina-Embedding-v2-1B モデルをロードして返す。
    - device: "cpu" / "cuda"（自動で 'cuda:0' を使います）
    - local_dir: 任意。指定がなければ models/{safe_model_dirname(MODEL_ID)} に保存
    """
    hf_token = get_hf_token(enable_transfer=True)
    path_or_id = _ensure_local_model_dir(MODEL_ID, local_dir, hf_token)

    model = SentenceTransformer(
        path_or_id,
        device=("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu") if device in ("cuda", "cpu") else device,
    )
    # 1792 次元 / 最終トークンプーリング（モデルカードより）
    LOGGER.info(f"[sarashina] model loaded from: {path_or_id} (device={model.device})")
    return model


def add_prefix_for_task(text: str, side: str, task: str = "sts") -> str:
    """
    タスク別に推奨の prefix/instruction を付与する。
    - side: "query" / "doc" を想定（STS では両方 query 形式）
    - task: {"sts","retrieval","reranking","clustering","classification"}
    """
    if task not in VALID_TASKS:
        raise ValueError(f"task は {sorted(VALID_TASKS)} から選んでください: got {task}")

    if task == "sts":
        # STS は両方とも query 形式（モデルカード推奨）
        return f"{INSTRUCTION_STS}\nquery: {text}"

    if task == "retrieval":
        if side == "query":
            return f"{INSTRUCTION_RETRIEVAL}\nquery: {text}"
        return f"text: {text}"

    if task == "reranking":
        if side == "query":
            return f"{INSTRUCTION_RERANK}\nquery: {text}"
        return f"text: {text}"

    if task == "clustering":
        if side == "query":
            return f"{INSTRUCTION_CLUSTER}\nquery: {text}"
        # Document 側のテンプレートは明示なしのため、そのまま or text: を付けても良い
        return text

    if task == "classification":
        if side == "query":
            return f"{INSTRUCTION_CLASSIFY}\nquery: {text}"
        return text

    # 到達しない想定
    return text


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
    task: str = "sts",
    local_dir: Optional[str] = None,
) -> float:
    """
    2つの文章のコサイン類似度を返すユーティリティ。
    - STS タスクでは、両文に同じ instruction + query プレフィックスを付与（モデルカード準拠）
    - それ以外のタスクでは、query/doc でフォーマットを分ける
    """
    model = get_model(device=device, local_dir=local_dir)

    if task == "sts":
        a_fmt = add_prefix_for_task(text_a, side="query", task="sts")
        b_fmt = add_prefix_for_task(text_b, side="query", task="sts")
    else:
        a_fmt = add_prefix_for_task(text_a, side="query", task=task)
        b_fmt = add_prefix_for_task(text_b, side="doc", task=task)

    # まとめてエンコード（正規化有効）
    vecs = model.encode([a_fmt, b_fmt], convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)

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
    qa = "Sarashinaは日本語に強いLLMです。"
    qb = "サラシナは日本語LLMとして公開されています。"

    sim = embed_and_score(qa, qb, device="cpu", task="sts")
    print(f"[self-test] similarity(STS)={sim:.4f}")
