# -*- coding: utf-8 -*-
"""
OpenAI の text-embedding-3-large を使用して 2 文のコサイン類似度を算出するラッパ。

要件:
- OpenAI Embeddings API を利用（ローカルの HF モデルは使わない）
- 入力: 2 つの文章（str）
- 出力: コサイン類似度（float, -1..1）
- 環境変数 OPENAI_API_KEY が必要
- 次元圧縮: dimensions 引数で任意に指定可能（例: 256, 512, 1024 など）。未指定時はモデル既定（3072）。

参考:
- src/WeatherLLM/Embedding/main_v1.py の他モデル実装パターン（embed_and_score を提供）
- OpenAI Embeddings ドキュメント（text-embedding-3-large）
"""

from __future__ import annotations

import os
import time
from typing import List, Optional, Sequence, Union

import torch

from llm_utils import setup_logger

LOGGER = setup_logger("text-embedding-3-large")

# program.main_v2 の .env ロードユーティリティを可能なら再利用
try:
    from program.main_v2 import find_env_path as _find_env_path, load_api_key as _load_api_key  # type: ignore
except Exception:
    _find_env_path = None  # type: ignore
    _load_api_key = None  # type: ignore

# OpenAI SDK (>=1.0.0)
try:
    from openai import OpenAI
except Exception as e:
    raise RuntimeError(
        "openai パッケージが見つかりません。`pip install -U openai` (>=1.0.0) を実行してください。"
    ) from e


MODEL_ID = "text-embedding-3-large"


def _newline_to_space(text: str) -> str:
    return text.replace("\n", " ").strip()


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.dim() == 1:
        a = a.unsqueeze(0)
    if b.dim() == 1:
        b = b.unsqueeze(0)
    sim = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    return float(sim.squeeze().item())


def resolve_api_key(explicit: Optional[str] = None) -> str:
    """
    APIキーの解決順序:
      1) 関数引数 explicit
      2) 環境変数: OPENAI_API_KEY / OpenAI_KEY_TOKEN / OPENAI_KEY_TOKEN
      3) program.main_v2 の find_env_path + load_api_key による .env 読み込み
      4) python-dotenv による簡易 .env 読み込み
    """
    if explicit:
        return explicit

    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OpenAI_KEY_TOKEN") or os.environ.get("OPENAI_KEY_TOKEN")
    if key:
        return str(key)

    # Try program's helpers
    if _find_env_path is not None and _load_api_key is not None:
        try:
            env_path = _find_env_path()
            key = _load_api_key(env_path)
            if key:
                return str(key)
        except Exception:
            pass

    # Fallback: dotenv
    try:
        from dotenv import load_dotenv
        load_dotenv()
        key = os.environ.get("OpenAI_KEY_TOKEN") or os.environ.get("OPENAI_KEY_TOKEN") or os.environ.get("OPENAI_API_KEY")
        if key:
            return str(key)
    except Exception:
        pass

    raise RuntimeError("OpenAI APIキーが見つかりません。OpenAI_KEY_TOKEN または OPENAI_API_KEY を .env もしくは環境変数に設定してください。")


def get_client(api_key: Optional[str] = None, base_url: Optional[str] = None, organization: Optional[str] = None) -> OpenAI:
    """
    OpenAI クライアントを返す。環境変数:
      - OPENAI_API_KEY（必須）
      - OPENAI_BASE_URL（任意, 互換プロキシ利用時など）
      - OPENAI_ORG（任意）
    """
    key = resolve_api_key(api_key)
    kwargs = {"api_key": key}
    base = base_url or os.getenv("OPENAI_BASE_URL")
    if base:
        kwargs["base_url"] = base
    org = organization or os.getenv("OPENAI_ORG")
    if org:
        kwargs["organization"] = org
    return OpenAI(**kwargs)  # type: ignore[arg-type]


def _request_embeddings(
    client: OpenAI,
    inputs: Sequence[str],
    model: str = MODEL_ID,
    dimensions: Optional[int] = None,
    encoding_format: str = "float",
    max_retries: int = 5,
    retry_base_wait: float = 1.0,
) -> List[List[float]]:
    """
    Embeddings API を呼び出し、ベクトルのリストを返す。レート制限などに指数バックオフで再試行。
    """
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = client.embeddings.create(
                model=model,
                input=list(inputs),
                encoding_format=encoding_format,
                **({"dimensions": dimensions} if dimensions is not None else {}),
            )
            return [d.embedding for d in resp.data]
        except Exception as e:
            last_err = e
            wait = retry_base_wait * (2 ** attempt)
            LOGGER.warning(f"[OpenAI] embeddings.create 失敗。retry in {wait:.1f}s (attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(wait)
    assert last_err is not None
    raise last_err


def embed_texts(
    texts: Union[str, Sequence[str]],
    dimensions: Optional[int] = None,
    normalize: bool = True,
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    文章群をエンコードして Torch テンソルを返す。
    - texts: str または List[str]
    - dimensions: 出力次元（None の場合はデフォルト 3072）
    - normalize: True で L2 正規化（OpenAI 既定は正規化済だが保険的に再正規化）
    - device: None の場合は 'cpu'
    """
    if isinstance(texts, str):
        texts = [texts]
    # OpenAI 推奨: 改行は空白へ
    proc = [_newline_to_space(t) for t in texts]

    client = get_client()
    vecs = _request_embeddings(client, proc, model=MODEL_ID, dimensions=dimensions)
    emb = torch.tensor(vecs, dtype=torch.float32)  # CPU テンソル
    if normalize:
        emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
    if device and device != "cpu":
        # 類似度は軽量なので CPU で十分だが、引数で希望があれば移動
        try:
            emb = emb.to(device)
        except Exception:
            pass
    return emb


def embed_and_score(
    text_a: str,
    text_b: str,
    *,
    dimensions: Optional[int] = None,
    device: Optional[str] = None,
) -> float:
    """
    2 文のコサイン類似度を返すユーティリティ。
    - dimensions を指定すると OpenAI 側の 'dimensions' パラメータで低次元化（例: 256/512/1024 等）
    - device は埋め込みテンソルの配置先。未指定なら CPU。
    """
    embs = embed_texts([text_a, text_b], dimensions=dimensions, normalize=True, device=device)
    a, b = embs[0], embs[1]
    return _cosine_sim(a, b)


if __name__ == "__main__":
    # 簡易自己テスト
    qa = "今日は雨です。傘を持って行きましょう。"
    qb = "外は雨模様なので、出かける時は傘が必要です。"
    try:
        sim = embed_and_score(qa, qb, dimensions=512)
        print(f"[self-test] text-embedding-3-large similarity(dim=512)={sim:.4f}")
    except Exception as e:
        print(f"[self-test] failed: {e}")
