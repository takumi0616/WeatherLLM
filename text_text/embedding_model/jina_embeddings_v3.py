# -*- coding: utf-8 -*-
"""
jinaai/jina-embeddings-v3 を Hugging Face から取得して使用可能にするラッパ。

参考:
- src/WeatherLLM/llava.py のローカル保存方針（snapshot_download によるローカル保存、存在時は再DLしない）
- モデルカード: https://huggingface.co/jinaai/jina-embeddings-v3
  - 8192 トークンまでの長文対応（RoPE）
  - Task LoRA によりタスク別エンベッディング: 
      'retrieval.query', 'retrieval.passage', 'separation', 'classification', 'text-matching'
  - Matryoshka Embedding（32, 64, 128, 256, 512, 768, 1024 次元への truncate_dim）

提供関数:
- get_model(device: str = "cpu", local_dir: Optional[str] = None)
    -> SentenceTransformer
- embed_texts(model, texts, task: Optional[str] = None, truncate_dim: Optional[int] = None, ...)
    テキストをエンコードしてベクトルを返す（torch.Tensor/np.ndarray）
- embed_and_score(text_a: str, text_b: str, device: str = "cpu",
                  task: str = "sts", local_dir: Optional[str] = None,
                  truncate_dim: Optional[int] = None, max_length: Optional[int] = None) -> float
    2文の類似度（コサイン類似度）を返す。
    - main_v1.py のタスク指定（sts/retrieval/reranking/clustering/classification）を
      Jina の task へ次の通りマッピング:
        sts -> text-matching（両方同一）
        retrieval -> retrieval.query / retrieval.passage
        reranking -> separation（両方同一）
        clustering -> separation（両方同一）
        classification -> classification（両方同一）

注意:
- オフラインでも使えるように、存在しなければ huggingface_hub.snapshot_download でローカルに保存します
- HF_TOKEN 環境変数が設定されていればそれを使用
- SentenceTransformers 3.1 以降 + trust_remote_code=True を想定
"""

from __future__ import annotations

import os
from typing import Optional, Sequence, Union, Dict, Any

import torch

from llm_utils import (
    setup_logger,
    get_hf_token,
    safe_model_dirname,
)

LOGGER = setup_logger("jina-embeddings-v3")

# 依存バージョンチェック: torch>=2.6 必須（Jina v3 の flash 実装に準拠）
try:
    from packaging import version as _v
    if _v.parse(torch.__version__) < _v.parse("2.6.0"):
        raise RuntimeError(
            f"PyTorch {torch.__version__} が検出されました。jina-embeddings-v3 には torch>=2.6 が必要です。"
            " environments_gpu/llm_env.yml で pytorch=2.6.* を指定して再構築してください。"
        )
except Exception:
    # packaging 未導入や torch 未導入等ではここでは落とさない（後続の import で失敗する想定）
    pass

# 任意: flash-attn の有無を警告（未インストールの場合は実装側でフォールバックする想定）
try:
    import flash_attn  # type: ignore
except Exception:
    LOGGER.warning("[jina] flash-attn が見つかりません。実装側でフォールバックしない場合は 'pip install flash-attn' が必要です。")

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


MODEL_ID = "jinaai/jina-embeddings-v3"

# main_v1.py と整合のために同じタスク名を許容
VALID_TASKS = {"sts", "retrieval", "reranking", "clustering", "classification"}

def _patch_xlmroberta_for_jina() -> None:
    """
    Jina の動的モジュールが transformers の一部内部関数
    `create_position_ids_from_input_ids` をインポートできない環境向けの互換パッチ。
    先に transformers 側の XLM-R モジュールに同名関数を注入する。
    """
    try:
        from transformers.models.xlm_roberta import modeling_xlm_roberta as xlm_mod  # type: ignore
        if not hasattr(xlm_mod, "create_position_ids_from_input_ids"):
            def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length: int = 0):
                # 位置IDはパディングを0とする累積から生成（近似的な互換実装）
                mask = (input_ids != padding_idx).int()
                cumsum = mask.cumsum(dim=-1) + int(past_key_values_length)
                position_ids = cumsum * mask + padding_idx
                return position_ids.long()
            xlm_mod.create_position_ids_from_input_ids = create_position_ids_from_input_ids  # type: ignore[attr-defined]
            LOGGER.warning("[jina] applied compatibility patch for XLM-R (create_position_ids_from_input_ids)")
    except Exception as e:
        LOGGER.warning(f"[jina] XLM-R compatibility patch failed: {e}")

# Jina のタスク名へマッピング
def _jina_task_for(task: str, side: str) -> str:
    """
    main_v1 のタスク指定を Jina の task/prompt_name にマップする。
    - side: 'query' または 'doc'
    """
    if task not in VALID_TASKS:
        raise ValueError(f"task は {sorted(VALID_TASKS)} から選んでください: got {task}")

    if task == "sts":
        # STS（対称類似）: text-matching を両方に適用
        return "text-matching"

    if task == "retrieval":
        return "retrieval.query" if side == "query" else "retrieval.passage"

    if task in {"reranking", "clustering"}:
        # separation はクラスタリング/リランキング用途
        return "separation"

    if task == "classification":
        return "classification"

    # 到達しない想定
    return "text-matching"


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
        LOGGER.info(f"[jina] local snapshot found. skip download: {local_dir}")
        return local_dir

    if not HAS_HF_HUB:
        LOGGER.warning(
            "[jina] huggingface_hub が見つからないため、ローカル保存はスキップします。"
            "オンラインで SentenceTransformer から直接読み込みます。"
        )
        return model_id

    os.makedirs(local_dir, exist_ok=True)
    LOGGER.info(f"[jina] downloading snapshot to: {local_dir}")
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
    jina-embeddings-v3 モデルをロードして返す。
    - device: "cpu" / "cuda"（自動で 'cuda:0' を使用）
    - local_dir: 任意。指定がなければ models/{safe_model_dirname(MODEL_ID)} に保存
    """
    hf_token = get_hf_token(enable_transfer=True)
    path_or_id = _ensure_local_model_dir(MODEL_ID, local_dir, hf_token)

    # 動的モジュールの import 前に互換パッチを適用（transformers の関数欠如に対処）
    _patch_xlmroberta_for_jina()

    st_kwargs = {
        "trust_remote_code": True,
        "device": ("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu") if device in ("cuda", "cpu") else device,
        # 互換性確保のためコードリビジョンを固定（モデルカードの告知で推奨のもの）
        "model_kwargs": {"code_revision": "da863dd04a4e5dce6814c6625adfba87b83838aa"},
    }
    try:
        model = SentenceTransformer(path_or_id, **st_kwargs)
    except TypeError:
        # 古い sentence-transformers では model_kwargs 未対応のため除去して再試行
        st_kwargs.pop("model_kwargs", None)
        model = SentenceTransformer(path_or_id, **st_kwargs)

    LOGGER.info(f"[jina] model loaded from: {path_or_id} (device={model.device})")
    return model


def embed_texts(
    model: "SentenceTransformer",
    texts: Union[str, Sequence[str]],
    task: Optional[str] = None,
    truncate_dim: Optional[int] = None,
    max_length: Optional[int] = None,
    convert_to_tensor: bool = True,
    normalize: bool = True,
):
    """
    テキスト群をエンコードしてベクトルを返す。
    - task: Jina の task 名（例: 'text-matching', 'retrieval.query', ...）
    - truncate_dim: Matryoshka Embedding での次元切り詰め（32..1024）
    - max_length: 入力の最大トークン長（8192 まで、短く切る場合に指定）
    - convert_to_tensor: True なら torch.Tensor、False なら np.ndarray
    - normalize: True で正規化（コサイン類似度用途向け）
    """
    if isinstance(texts, str):
        texts = [texts]

    # SentenceTransformer v3.1+ では encode に task/prompt_name を渡せる
    kwargs: Dict[str, Any] = {
        "convert_to_tensor": convert_to_tensor,
        "normalize_embeddings": normalize,
        "show_progress_bar": False,
    }
    if task is not None:
        kwargs["task"] = task
        # モデル実装によっては prompt_name を task と一致させる
        kwargs["prompt_name"] = task
    if truncate_dim is not None:
        kwargs["truncate_dim"] = truncate_dim
    if max_length is not None:
        kwargs["max_length"] = max_length

    try:
        return model.encode(list(texts), **kwargs)
    except TypeError:
        # 古いバージョンの SentenceTransformer では task/prompt_name 非対応
        for k in ("task", "prompt_name", "truncate_dim", "max_length"):
            kwargs.pop(k, None)
        return model.encode(list(texts), **kwargs)


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
    truncate_dim: Optional[int] = None,
    max_length: Optional[int] = None,
) -> float:
    """
    2つの文章のコサイン類似度を返すユーティリティ。
    - STS タスクでは 'text-matching' を両文に適用
    - retrieval では query/doc を 'retrieval.query' / 'retrieval.passage' として別々にエンコード
    - reranking/clustering は 'separation' を両文に適用
    - classification は 'classification' を両文に適用
    """
    model = get_model(device=device, local_dir=local_dir)

    task_a = _jina_task_for(task, side="query")
    task_b = _jina_task_for(task, side="doc")

    # 片方ずつエンコード（retrieval では task が異なるため）
    a_vec = embed_texts(
        model,
        text_a,
        task=task_a,
        truncate_dim=truncate_dim,
        max_length=max_length,
        convert_to_tensor=True,
        normalize=True,
    )
    b_vec = embed_texts(
        model,
        text_b,
        task=task_b,
        truncate_dim=truncate_dim,
        max_length=max_length,
        convert_to_tensor=True,
        normalize=True,
    )

    a = a_vec[0] if a_vec.dim() == 2 else a_vec
    b = b_vec[0] if b_vec.dim() == 2 else b_vec

    # similarity API があれば使用、なければ自前コサイン
    if hasattr(model, "similarity"):
        try:
            sims = model.similarity(a, b)  # shape: (1,1) または (1,)
            return float(sims.item())
        except Exception:
            pass
    return _cosine_sim(a, b)


if __name__ == "__main__":
    # 簡易自己テスト（CPU）
    qa = "今日は雨です。傘を持って行きましょう。"
    qb = "外は雨模様なので、出かける時は傘が必要です。"

    sim_sts = embed_and_score(qa, qb, device="cpu", task="sts", truncate_dim=256)
    print(f"[self-test] similarity(STS, truncate_dim=256)={sim_sts:.4f}")

    # retrieval モードの確認（異なる task を左右に付与）
    q = "明日の東京の天気は？"
    d = "東京は明日は雨で、最高気温は18度の予報です。"
    sim_ret = embed_and_score(q, d, device="cpu", task="retrieval")
    print(f"[self-test] similarity(retrieval)={sim_ret:.4f}")
