# -*- coding: utf-8 -*-
"""
google/embeddinggemma-300m を Hugging Face から取得して使用可能にするラッパ。

参考:
- src/WeatherLLM/llava.py のローカル保存方針（snapshot_download によるローカル保存、存在時は再DLしない）
- モデルカード: https://huggingface.co/google/embeddinggemma-300m
  - Sentence-Transformers 形式（Gemma3 Text をバックボーン）
  - prompt_name により用途最適化（例: STS / Retrieval-query / Retrieval-document / Reranking / Clustering / Classification）
  - 出力次元 768（MRL により 512/256/128 などへ truncate_dim 可能）
  - NOTE: EmbeddingGemma の活性は float16 非対応。float32 or bfloat16 を用いること。

提供関数:
- get_model(device: str = "cpu", local_dir: Optional[str] = None)
    -> SentenceTransformer
- embed_texts(model, texts, prompt_name: Optional[str] = None, truncate_dim: Optional[int] = None, ...)
    テキストをエンコードしてベクトルを返す（torch.Tensor/np.ndarray）
- embed_and_score(text_a: str, text_b: str, device: str = "cpu",
                  task: str = "sts", local_dir: Optional[str] = None,
                  truncate_dim: Optional[int] = None, max_length: Optional[int] = None) -> float
    2文の類似度（コサイン類似度）を返す。
    - main_v1.py のタスク指定（sts/retrieval/reranking/clustering/classification）を
      EmbeddingGemma の prompt_name に次の通りマッピング:
        sts -> STS（両方同一）
        retrieval -> Retrieval-query / Retrieval-document
        reranking -> Reranking（両方同一）
        clustering -> Clustering（両方同一）
        classification -> Classification（両方同一）

注意:
- オフラインでも使えるように、存在しなければ huggingface_hub.snapshot_download でローカルに保存します
- HF_TOKEN 環境変数が設定されていればそれを使用
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

LOGGER = setup_logger("embeddinggemma-300m")

def _patch_masking_utils_for_torch_lt_26() -> None:
    """
    transformers.masking_utils.create_causal_mask / create_sliding_window_causal_mask に渡される
    or_mask_function / and_mask_function が torch<2.6 で例外になる問題を回避するため、
    実行時に該当キーワードを取り除く薄いモンキーパッチを適用する。
    - Transformers 4.57.0 + torch<2.6 の既知問題に対するワークアラウンド
    - 一度だけ適用（再入不可）
    """
    try:
        from transformers import masking_utils as _mu  # type: ignore
        if getattr(_mu, "_patched_drop_mask_fun_for_torch_lt26", False):
            return

        # create_causal_mask のラップ
        if hasattr(_mu, "create_causal_mask"):
            _orig_causal = _mu.create_causal_mask

            def _wrapped_create_causal_mask(*args, **kwargs):
                kwargs.pop("or_mask_function", None)
                kwargs.pop("and_mask_function", None)
                return _orig_causal(*args, **kwargs)

            _mu.create_causal_mask = _wrapped_create_causal_mask  # type: ignore[assignment]

        # create_sliding_window_causal_mask のラップ
        if hasattr(_mu, "create_sliding_window_causal_mask"):
            _orig_sw = _mu.create_sliding_window_causal_mask

            def _wrapped_create_sliding_window_causal_mask(*args, **kwargs):
                kwargs.pop("or_mask_function", None)
                kwargs.pop("and_mask_function", None)
                return _orig_sw(*args, **kwargs)

            _mu.create_sliding_window_causal_mask = _wrapped_create_sliding_window_causal_mask  # type: ignore[assignment]

        setattr(_mu, "_patched_drop_mask_fun_for_torch_lt26", True)
        LOGGER.info("[gemma] patched masking_utils.{create_causal_mask,create_sliding_window_causal_mask} to drop or/and_mask_function for torch<2.6")
    except Exception as e:
        # パッチ失敗時はログのみ（そのままだと ValueError が再現する可能性）
        LOGGER.warning(f"[gemma] patch masking_utils skipped: {e}")

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


MODEL_ID = "google/embeddinggemma-300m"

# main_v1.py と整合のために同じタスク名を許容
VALID_TASKS = {"sts", "retrieval", "reranking", "clustering", "classification"}


def _gemma_prompt_for(task: str, side: str) -> str:
    """
    main_v1 のタスク指定を EmbeddingGemma の prompt_name にマップする。
    - side: 'query' または 'doc'
    """
    if task not in VALID_TASKS:
        raise ValueError(f"task は {sorted(VALID_TASKS)} から選んでください: got {task}")

    if task == "sts":
        return "STS"
    if task == "retrieval":
        return "Retrieval-query" if side == "query" else "Retrieval-document"
    if task == "reranking":
        return "Reranking"
    if task == "clustering":
        return "Clustering"
    if task == "classification":
        return "Classification"
    return "STS"


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
        LOGGER.info(f"[gemma] local snapshot found. skip download: {local_dir}")
        return local_dir

    if not HAS_HF_HUB:
        LOGGER.warning(
            "[gemma] huggingface_hub が見つからないため、ローカル保存はスキップします。"
            "オンラインで SentenceTransformer から直接読み込みます。"
        )
        return model_id

    os.makedirs(local_dir, exist_ok=True)
    LOGGER.info(f"[gemma] downloading snapshot to: {local_dir}")
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
    embeddinggemma-300m モデルをロードして返す。
    - device: "cpu" / "cuda"（自動で 'cuda:0' を使用）
    - local_dir: 任意。指定がなければ models/{safe_model_dirname(MODEL_ID)} に保存
    注意: EmbeddingGemma は float16 非対応。既定で float32（もしくは BF16 サポート時は内部で BF16 を使う実装もある）が使われます。
    """
    hf_token = get_hf_token(enable_transfer=True)
    path_or_id = _ensure_local_model_dir(MODEL_ID, local_dir, hf_token)

    st_device = ("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu") if device in ("cuda", "cpu") else device
    # Transformers 4.57.0 では gemma3_text の公式サポートが不十分なため、
    # trust_remote_code=True を付与してリポジトリアイテム側の実装に委譲し、
    # かつ bin -> torch.load を回避するため safetensors を強制する。
    st_kwargs = {
        "device": st_device,
        "trust_remote_code": True,
        # Transformers 側へ渡す追加引数は model_kwargs に入る（ST 5.1.1 仕様）
        # - use_safetensors: bin の torch.load を避ける
        # - attn_implementation: torch<2.6 互換性の高い 'eager' を強制
        "model_kwargs": {"use_safetensors": True, "attn_implementation": "eager"},
    }
    try:
        # torch<2.6 での Gemma3 + Transformers 4.57.0 のマスク生成互換問題に対するパッチ
        _patch_masking_utils_for_torch_lt_26()
        model = SentenceTransformer(path_or_id, **st_kwargs)
    except TypeError:
        # 古い/将来のAPI差異に備えてフォールバック
        st_kwargs.pop("model_kwargs", None)
        try:
            model = SentenceTransformer(path_or_id, **st_kwargs)
        except TypeError:
            st_kwargs.pop("trust_remote_code", None)
            model = SentenceTransformer(path_or_id, **st_kwargs)
    LOGGER.info(f"[gemma] model loaded from: {path_or_id} (device={model.device})")
    return model


def embed_texts(
    model: "SentenceTransformer",
    texts: Union[str, Sequence[str]],
    prompt_name: Optional[str] = None,
    truncate_dim: Optional[int] = None,
    max_length: Optional[int] = None,
    convert_to_tensor: bool = True,
    normalize: bool = True,
):
    """
    テキスト群をエンコードしてベクトルを返す。
    - prompt_name: EmbeddingGemma の用途別プロンプト名（例: 'STS', 'Retrieval-query', ...）
    - truncate_dim: Matryoshka Embedding での次元切り詰め（128/256/512/768）
    - max_length: 入力の最大トークン長（既定はモデル設定、必要なら短く）
    - convert_to_tensor: True なら torch.Tensor、False なら np.ndarray
    - normalize: True で正規化（コサイン類似度用途向け）
    """
    if isinstance(texts, str):
        texts = [texts]

    kwargs: Dict[str, Any] = {
        "convert_to_tensor": convert_to_tensor,
        "normalize_embeddings": normalize,
        "show_progress_bar": False,
    }
    if prompt_name is not None:
        # 一部バージョンでは 'task' でも可。両方渡して互換度を上げる。
        kwargs["prompt_name"] = prompt_name
        kwargs["task"] = prompt_name
    if truncate_dim is not None:
        kwargs["truncate_dim"] = truncate_dim
    if max_length is not None:
        kwargs["max_length"] = max_length

    try:
        return model.encode(list(texts), **kwargs)
    except TypeError:
        # 古いバージョンの SentenceTransformer では prompt_name/task 非対応
        for k in ("prompt_name", "task", "truncate_dim", "max_length"):
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
    - STS タスクでは 'STS' を両文に適用
    - retrieval では query/doc を 'Retrieval-query' / 'Retrieval-document' として別々にエンコード
    - reranking は 'Reranking' を両文に適用
    - clustering は 'Clustering' を両文に適用
    - classification は 'Classification' を両文に適用
    """
    if task not in VALID_TASKS:
        raise ValueError(f"task は {sorted(VALID_TASKS)} から選んでください: got {task}")

    model = get_model(device=device, local_dir=local_dir)

    prompt_a = _gemma_prompt_for(task, side="query")
    prompt_b = _gemma_prompt_for(task, side="doc")

    # 片方ずつエンコード（retrieval では左右で prompt が異なるため）
    a_vec = embed_texts(
        model,
        text_a,
        prompt_name=prompt_a,
        truncate_dim=truncate_dim,
        max_length=max_length,
        convert_to_tensor=True,
        normalize=True,
    )
    b_vec = embed_texts(
        model,
        text_b,
        prompt_name=prompt_b,
        truncate_dim=truncate_dim,
        max_length=max_length,
        convert_to_tensor=True,
        normalize=True,
    )

    a = a_vec[0] if isinstance(a_vec, torch.Tensor) and a_vec.dim() == 2 else a_vec
    b = b_vec[0] if isinstance(b_vec, torch.Tensor) and b_vec.dim() == 2 else b_vec

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
    truncate_dim: Optional[int] = None,
    max_length: Optional[int] = None,
) -> float:
    """
    外部から即利用できる共通API。2文を引数に取り、コサイン類似度を返す。
    """
    return embed_and_score(
        text_a=text_a,
        text_b=text_b,
        device=device,
        task=task,
        local_dir=local_dir,
        truncate_dim=truncate_dim,
        max_length=max_length,
    )


def compare_files(
    file_a: Optional[str] = None,
    file_b: Optional[str] = None,
    device: str = "cpu",
    task: str = "sts",
    local_dir: Optional[str] = None,
    truncate_dim: Optional[int] = None,
    max_length: Optional[int] = None,
) -> float:
    """
    2つのファイルパスを受け取り、中身を読み込んで類似度を返す。
    file_a/file_b が None の場合は既定の2ファイルを使用。
    """
    path_a = file_a or DEFAULT_FILE_A
    path_b = file_b or DEFAULT_FILE_B
    text_a = _read_text(path_a)
    text_b = _read_text(path_b)
    return compute_cosine_similarity(
        text_a, text_b, device=device, task=task, local_dir=local_dir, truncate_dim=truncate_dim, max_length=max_length
    )


def compare_default_files(
    device: str = "cpu",
    task: str = "sts",
    local_dir: Optional[str] = None,
    truncate_dim: Optional[int] = None,
    max_length: Optional[int] = None,
) -> float:
    """
    既定の2ファイル（Embedding/data/...）を読み、類似度を返す。
    """
    return compare_files(None, None, device=device, task=task, local_dir=local_dir, truncate_dim=truncate_dim, max_length=max_length)


if __name__ == "__main__":
    # 簡易自己テスト（CPU）
    qa = "今日は雨です。傘を持って行きましょう。"
    qb = "外は雨模様なので、出かける時は傘が必要です。"

    sim_sts = embed_and_score(qa, qb, device="cpu", task="sts", truncate_dim=256)
    print(f"[self-test] EmbeddingGemma similarity(STS, truncate_dim=256)={sim_sts:.4f}")

    # retrieval モードの確認（異なる prompt_name を左右に付与）
    q = "明日の東京の天気は？"
    d = "東京は明日は雨で、最高気温は18度の予報です。"
    sim_ret = embed_and_score(q, d, device="cpu", task="retrieval")
    print(f"[self-test] EmbeddingGemma similarity(retrieval)={sim_ret:.4f}")
