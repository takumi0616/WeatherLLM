# -*- coding: utf-8 -*-
"""
pkshatech/GLuCoSE-base-ja-v2 を Hugging Face から取得して使用可能にするラッパ。

参考:
- src/WeatherLLM/llava.py / 既存モデル群の方針（snapshot_download によるローカル保存、存在時は再DLしない）
- モデルカード要点（GLuCoSE v2）:
  - SentenceTransformer 形式 / 出力次元 768 / 最大長 512
  - 重要: 推論時は入力先頭に "query: " または "passage: " プレフィックスが必要
    - STS のような対称タスクでは、両方とも "query: " を付与してよい

提供関数:
- get_model(device: str = "cpu", local_dir: Optional[str] = None)
    -> SentenceTransformer
- add_prefix_for_task(text: str, side: str, task: str = "sts") -> str
    GLuCoSE 推奨のフォーマットに従い、クエリ/ドキュメント側へプレフィックスを付与
- embed_texts(model, texts, convert_to_tensor: bool = True, normalize: bool = True)
    -> torch.Tensor | np.ndarray
- embed_and_score(text_a: str, text_b: str, device: str = "cpu",
                  task: str = "sts", local_dir: Optional[str] = None) -> float
    2文の類似度（コサイン類似度）を返す。STS タスクでは両文を query 形式にする。

注意:
- オフラインでも使えるように、存在しなければ huggingface_hub.snapshot_download でローカルに保存します
- HF_TOKEN 環境変数が設定されていればそれを使用
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

LOGGER = setup_logger("glucose-ja-v2")

# 依存: sentence-transformers
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception as e:
    raise RuntimeError(
        "sentence-transformers が見つかりません。`pip install -U sentence-transformers fugashi sentencepiece unidic-lite` を実行してください。"
    ) from e

# 依存: huggingface_hub（ローカルスナップショット保存用）
try:
    from huggingface_hub import snapshot_download  # type: ignore
    HAS_HF_HUB = True
except Exception:
    HAS_HF_HUB = False


MODEL_ID = "pkshatech/GLuCoSE-base-ja-v2"

# GLuCoSE の推奨プレフィックス（英語）
QUERY_PREFIX = "query: "
DOC_PREFIX = "passage: "

# main_v1.py と整合のために同じタスク名を許容
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
        LOGGER.info(f"[glucose] local snapshot found. skip download: {local_dir}")
        return local_dir

    if not HAS_HF_HUB:
        LOGGER.warning(
            "[glucose] huggingface_hub が見つからないため、ローカル保存はスキップします。"
            "オンラインで SentenceTransformer から直接読み込みます。"
        )
        return model_id

    os.makedirs(local_dir, exist_ok=True)
    LOGGER.info(f"[glucose] downloading snapshot to: {local_dir}")
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
    pkshatech/GLuCoSE-base-ja-v2 モデルをロードして返す。
    - device: "cpu" / "cuda"（自動で 'cuda:0' を使います）
    - local_dir: 任意。指定がなければ models/{safe_model_dirname(MODEL_ID)} に保存
    """
    hf_token = get_hf_token(enable_transfer=True)
    path_or_id = _ensure_local_model_dir(MODEL_ID, local_dir, hf_token)

    model = SentenceTransformer(
        path_or_id,
        device=("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu") if device in ("cuda", "cpu") else device,
    )
    # 768 次元 / 最大長 512（モデルカードより）
    LOGGER.info(f"[glucose] model loaded from: {path_or_id} (device={model.device})")
    return model


def add_prefix_for_task(text: str, side: str, task: str = "sts") -> str:
    """
    タスク別に GLuCoSE 推奨のプレフィックスを付与する。
    - side: "query" / "doc" を想定（STS では両方 query 形式）
    - task: {"sts","retrieval","reranking","clustering","classification"}
    """
    if task not in VALID_TASKS:
        raise ValueError(f"task は {sorted(VALID_TASKS)} から選んでください: got {task}")

    if task == "sts":
        # STS は両方とも query 形式で比較
        return f"{QUERY_PREFIX}{text}"

    if task in {"retrieval", "reranking"}:
        if side == "query":
            return f"{QUERY_PREFIX}{text}"
        return f"{DOC_PREFIX}{text}"

    # その他のタスクではテンプレ未定義のため、そのまま返す
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
    # sentence-transformers 5.x でも normalize_embeddings が利用可能
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
    - STS タスクでは、両文を "query: " 形式でフォーマット
    - それ以外の（retrieval/reranking）では query/doc を分ける
    """
    model = get_model(device=device, local_dir=local_dir)

    if task == "sts":
        a_fmt = add_prefix_for_task(text_a, side="query", task="sts")
        b_fmt = add_prefix_for_task(text_b, side="query", task="sts")
    else:
        a_fmt = add_prefix_for_task(text_a, side="query", task=task)
        b_fmt = add_prefix_for_task(text_b, side="doc", task=task)

    # まとめてエンコード（正規化有効）
    vecs = embed_texts(model, [a_fmt, b_fmt], convert_to_tensor=True, normalize=True)

    a = vecs[0] if vecs.dim() == 2 else vecs
    b = vecs[1] if vecs.dim() == 2 else vecs

    # similarity API があれば使用、なければ自前コサイン
    if hasattr(model, "similarity"):
        try:
            sims = model.similarity(a, b)  # shape: (1,1) または (1,)
            return float(sims.item())
        except Exception:
            pass
    return _cosine_sim(a, b)


# ------------- Convenience APIs for external use -------------

# デフォルトの2ファイル（このリポジトリ構成に合わせた絶対パスを計算）
_EMBED_DIR = os.path.dirname(os.path.dirname(__file__))
DEFAULT_FILE_A = os.path.join(_EMBED_DIR, "data", "generate_comment", "2022_01_01_gpt_generate_v4.txt")
DEFAULT_FILE_B = os.path.join(_EMBED_DIR, "data", "original_comment", "2022_01_01_original.txt")


def _read_text(path: str) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"テキストファイルが見つかりません: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def compute_cosine_similarity(
    text_a: str,
    text_b: str,
    device: str = "cpu",
    task: str = "sts",
    local_dir: Optional[str] = None,
) -> float:
    """外部から即利用できる共通API。2文を引数に取り、コサイン類似度を返す。"""
    return embed_and_score(text_a=text_a, text_b=text_b, device=device, task=task, local_dir=local_dir)


def compare_files(
    file_a: Optional[str] = None,
    file_b: Optional[str] = None,
    device: str = "cpu",
    task: str = "sts",
    local_dir: Optional[str] = None,
) -> float:
    """
    2つのファイルパスを受け取り、中身を読み込んで類似度を返す。
    file_a/file_b が None の場合は既定の2ファイルを使用。
    """
    path_a = file_a or DEFAULT_FILE_A
    path_b = file_b or DEFAULT_FILE_B
    text_a = _read_text(path_a)
    text_b = _read_text(path_b)
    return compute_cosine_similarity(text_a, text_b, device=device, task=task, local_dir=local_dir)


def compare_default_files(
    device: str = "cpu",
    task: str = "sts",
    local_dir: Optional[str] = None,
) -> float:
    """既定の2ファイル（Embedding/data/...）を読み、類似度を返す。"""
    return compare_files(None, None, device=device, task=task, local_dir=local_dir)


if __name__ == "__main__":
    # 簡易自己テスト（CPU）
    qa = "PKSHAはどんな会社ですか？"
    qb = "PKSHAはAIソフトウェアを中心に企業の業務を支援する会社です。"

    sim = embed_and_score(qa, qb, device="cpu", task="sts")
    print(f"[self-test] GLuCoSE v2 similarity(STS)={sim:.4f}")
