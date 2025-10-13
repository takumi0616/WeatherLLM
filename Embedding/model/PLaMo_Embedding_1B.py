# -*- coding: utf-8 -*-
"""
pfnet/plamo-embedding-1b を Hugging Face から取得して使用可能にするラッパ。

参考:
- モデルカード: https://huggingface.co/pfnet/plamo-embedding-1b
- src/WeatherLLM/llava.py のローカル保存方針を踏襲（snapshot_download でローカル保存し、存在すれば再DLしない）

提供関数:
- get_model(device: str = "cpu", local_dir: Optional[str] = None)
    -> Tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizerBase]
- embed_texts(model, tokenizer, texts, mode: str = "document", normalize: bool = True)
    -> torch.Tensor  # shape: (N, 2048)
- embed_and_score(text_a: str, text_b: str, device: str = "cpu",
                  task: str = "sts", local_dir: Optional[str] = None) -> float
    2文の類似度（コサイン類似度）を返す。
    - STS/分類/クラスタリング等: 両方 document エンコード
    - 検索/リランキング: 片方は query, もう片方は document エンコード

要件:
- transformers, torch, sentencepiece
- 可能なら huggingface_hub（ローカルキャッシュ作成用）。無ければオンライン読み込みにフォールバック。

注意:
- PLaMo-Embedding-1B は trust_remote_code=True で AutoModel/AutoTokenizer をロードし、
  model.encode_query / model.encode_document を使用します。
- 最大コンテキスト長は 4096（encode_query は内部でプレフィックス付与のため若干短くなる）。
"""

from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from llm_utils import (
    setup_logger,
    get_hf_token,
    safe_model_dirname,
)

LOGGER = setup_logger("plamo-embedding-1b")

# 依存: transformers
try:
    from transformers import AutoModel, AutoTokenizer  # type: ignore
except Exception as e:
    raise RuntimeError(
        "transformers が見つかりません。`pip install -U transformers sentencepiece` を実行してください。"
    ) from e

# 依存: huggingface_hub（ローカルスナップショット保存用）
try:
    from huggingface_hub import snapshot_download  # type: ignore
    HAS_HF_HUB = True
except Exception:
    HAS_HF_HUB = False


MODEL_ID = "pfnet/plamo-embedding-1b"
VALID_TASKS = {"sts", "retrieval", "reranking", "clustering", "classification"}


def _has_model_files(local_dir: str) -> bool:
    """ローカルに Transformers 形式の主要ファイルがあるか簡易判定"""
    if not os.path.isdir(local_dir):
        return False
    files = set(os.listdir(local_dir))
    # Transformers の手掛かりになりやすいファイル群
    keys = {"config.json", "tokenizer.json"}
    if not keys.intersection(files):  # tokenizer.json が無いモデルもあるため緩めに判定
        # 代替チェック
        if "config.json" not in files:
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
      （Transformers 側でオンライン読み込み/キャッシュされる想定）
    """
    if local_dir is None:
        safe_id = safe_model_dirname(model_id)
        local_dir = os.path.join("models", safe_id)

    if _has_model_files(local_dir):
        LOGGER.info(f"[plamo] local snapshot found. skip download: {local_dir}")
        return local_dir

    if not HAS_HF_HUB:
        LOGGER.warning(
            "[plamo] huggingface_hub が見つからないため、ローカル保存はスキップします。"
            "オンラインで Transformers から直接読み込みます。"
        )
        return model_id

    os.makedirs(local_dir, exist_ok=True)
    LOGGER.info(f"[plamo] downloading snapshot to: {local_dir}")
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
) -> Tuple["AutoModel", "AutoTokenizer"]:
    """
    PLaMo-Embedding-1B モデルとトークナイザーをロードして返す。
    - device: "cpu" / "cuda"（自動で 'cuda:0' を使います）
    - local_dir: 任意。指定がなければ models/{safe_model_dirname(MODEL_ID)} に保存
    戻り値: (model, tokenizer)
    """
    hf_token = get_hf_token(enable_transfer=True)
    path_or_id = _ensure_local_model_dir(MODEL_ID, local_dir, hf_token)

    # トークナイザー
    tokenizer = AutoTokenizer.from_pretrained(
        path_or_id,
        trust_remote_code=True,
        # token=hf_token,  # 多くの transformers バージョンで未対応のため、環境変数経由を推奨
    )

    # モデル本体（BF16 重みだが device に応じて自動調整される）
    torch_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(
        path_or_id,
        trust_remote_code=True,
        # token=hf_token,
    ).to(torch_device).eval()

    LOGGER.info(f"[plamo] model loaded from: {path_or_id} (device={torch_device})")
    return model, tokenizer


def _as_list(texts: Union[str, Sequence[str]]) -> Sequence[str]:
    if isinstance(texts, str):
        return [texts]
    return list(texts)


def embed_texts(
    model,
    tokenizer,
    texts: Union[str, Sequence[str]],
    mode: str = "document",
    normalize: bool = True,
) -> torch.Tensor:
    """
    テキスト群をエンコードしてベクトルを返す（shape: (N, 2048)）。
    - mode: "document" | "query"
    - normalize: True で L2 正規化（コサイン類似度用途向け）
    """
    texts_list = _as_list(texts)

    # PLaMo のカスタム API を使用
    emb: Optional[torch.Tensor] = None
    if mode == "query":
        # encode_query はバッチ未対応の可能性に備えて逐次処理（高速化したければ try バッチ→except 逐次に変更）
        vecs = []
        for t in texts_list:
            with torch.inference_mode():
                v = model.encode_query(t, tokenizer)  # shape: (1, 2048) あるいは (2048,)
            v = v if isinstance(v, torch.Tensor) else torch.tensor(v)
            if v.dim() == 1:
                v = v.unsqueeze(0)
            vecs.append(v)
        emb = torch.cat(vecs, dim=0)
    else:
        # document 側はバッチ対応が期待できるためまとめて呼ぶ。失敗時は逐次にフォールバック。
        try:
            with torch.inference_mode():
                v = model.encode_document(texts_list, tokenizer)  # shape: (N, 2048)
            emb = v if isinstance(v, torch.Tensor) else torch.tensor(v)
        except Exception:
            vecs = []
            for t in texts_list:
                with torch.inference_mode():
                    v = model.encode_document(t, tokenizer)
                v = v if isinstance(v, torch.Tensor) else torch.tensor(v)
                if v.dim() == 1:
                    v = v.unsqueeze(0)
                vecs.append(v)
            emb = torch.cat(vecs, dim=0)

    if normalize:
        emb = F.normalize(emb, p=2, dim=-1)
    return emb


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
    - STS/クラスタリング/分類: 両方 document エンコード
    - 検索/リランキング: A=Query, B=Document
    """
    if task not in VALID_TASKS:
        raise ValueError(f"task は {sorted(VALID_TASKS)} から選んでください: got {task}")

    model, tokenizer = get_model(device=device, local_dir=local_dir)

    if task in {"retrieval", "reranking"}:
        a = embed_texts(model, tokenizer, text_a, mode="query", normalize=True)[0]
        b = embed_texts(model, tokenizer, text_b, mode="document", normalize=True)[0]
    else:
        # STS/その他は対称比較として document/document で揃える
        a = embed_texts(model, tokenizer, text_a, mode="document", normalize=True)[0]
        b = embed_texts(model, tokenizer, text_b, mode="document", normalize=True)[0]

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
    """
    外部から即利用できる共通API。2文を引数に取り、コサイン類似度を返す。
    """
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
    """
    既定の2ファイル（Embedding/data/...）を読み、類似度を返す。
    """
    return compare_files(None, None, device=device, task=task, local_dir=local_dir)


if __name__ == "__main__":
    # 簡易自己テスト（CPU）
    qa = "PLaMo-Embedding-1Bとは何ですか？"
    qb = "PLaMo-Embedding-1Bは日本語のテキスト埋め込みモデルです。"
    sim_sts = embed_and_score(qa, qb, device="cpu", task="sts")
    print(f"[self-test] similarity(STS)={sim_sts:.4f}")

    # Retrieval 風（query/document を分ける）
    q = "天気予報の精度を上げるには？"
    d = "アンサンブル予報を活用すると精度が向上する可能性があります。"
    sim_ret = embed_and_score(q, d, device="cpu", task="retrieval")
    print(f"[self-test] similarity(Retrieval)={sim_ret:.4f}")
