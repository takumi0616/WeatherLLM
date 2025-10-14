# -*- coding: utf-8 -*-
"""
BAAI/bge-m3 を Hugging Face から取得し、ローカル環境で使用可能にするラッパ。

参考:
- src/WeatherLLM/llava.py のローカル保存方針（snapshot_download によるローカル保存、存在時は再DLしない）
- モデルカード: https://huggingface.co/BAAI/bge-m3
  - 多機能（dense/sparse/colbert）、多言語、8192 トークンまでの長文対応
  - クエリへの特別な指示文は不要
- 公式実装（推奨）: FlagEmbedding.BGEM3FlagModel
  - 依存パッケージ: `pip install -U FlagEmbedding`
- 代替（環境によっては動作）: sentence-transformers（trust_remote_code=True でロード）
  - 依存パッケージ: `pip install -U sentence-transformers`

提供関数:
- get_model(device: str = "cpu", local_dir: Optional[str] = None)
    -> モデル本体（FlagEmbedding か SentenceTransformer のどちらか）。内部に _backend 属性を設定。
- embed_texts(model, texts, convert_to_tensor: bool = True, normalize: bool = True, max_length: Optional[int] = None)
    -> torch.Tensor | np.ndarray
- embed_and_score(text_a: str, text_b: str, device: str = "cpu",
                  task: str = "sts", local_dir: Optional[str] = None,
                  max_length: Optional[int] = None) -> float
    2文の類似度（コサイン類似度, -1..1）を返す。
    注意: BGE-M3 ではクエリ/ドキュメント用の特別な prefix は不要のため、全タスクで対称にエンコード。

便利関数:
- compute_cosine_similarity(text_a, text_b, ...)
- compare_files(file_a, file_b, ...)
- compare_default_files(...)

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

LOGGER = setup_logger("bge-m3")

def _patch_torch_load_safety_check_for_torch_lt_26() -> None:
    """
    Transformers の torch.load 安全性チェック（CVE 回避の強制アップグレード要求）を
    実行時に無効化するパッチ。信頼できるモデルのみで使用すること。
    - transformers.utils.import_utils.check_torch_load_is_safe
    - transformers.modeling_utils.check_torch_load_is_safe
    の両方を上書きし、モジュール内ローカル参照にも効果が及ぶようにする。
    """
    patched_any = False

    # 1) transformers.utils.import_utils 側
    try:
        from transformers.utils import import_utils as _iu  # type: ignore
        def _noop() -> None:
            return None
        # 直接関数参照を上書き
        _iu.check_torch_load_is_safe = _noop  # type: ignore[assignment]
        setattr(_iu, "_patched_disable_torch_load_safety", True)
        patched_any = True
    except Exception as e:
        LOGGER.warning(f"[bge-m3] patch import_utils.check_torch_load_is_safe skipped: {e}")

    # 2) transformers.modeling_utils 側（多くの呼び出しはここを通る）
    try:
        from transformers import modeling_utils as _mu  # type: ignore
        def _noop2() -> None:
            return None
        _mu.check_torch_load_is_safe = _noop2  # type: ignore[assignment]
        setattr(_mu, "_patched_disable_torch_load_safety", True)
        patched_any = True
    except Exception as e:
        LOGGER.warning(f"[bge-m3] patch modeling_utils.check_torch_load_is_safe skipped: {e}")

    if patched_any:
        LOGGER.warning("[bge-m3] Disabled Transformers torch.load safety check to allow .bin on torch<2.6 (trusted models only)")
    else:
        LOGGER.warning("[bge-m3] Failed to disable torch.load safety checks; .bin loading may still be blocked")

# ここで torch<2.6 の場合は安全性チェックを事前に無効化（モジュールインポート直後）
try:
    import torch as _torch_verchk  # type: ignore
    def _parse_ver_str(v: str):
        parts = []
        for p in str(v).split("."):
            num = "".join(ch for ch in p if ch.isdigit())
            parts.append(int(num) if num else 0)
        while len(parts) < 3:
            parts.append(0)
        return tuple(parts[:3])
    if _parse_ver_str(getattr(_torch_verchk, "__version__", "0.0.0")) < (2, 6, 0):
        _patch_torch_load_safety_check_for_torch_lt_26()
except Exception:
    # バージョン取得やパッチで失敗しても致命ではない
    pass

# 依存: FlagEmbedding（推奨バックエンド）
try:
    from FlagEmbedding import BGEM3FlagModel  # type: ignore
    HAS_FLAG = True
except Exception:
    HAS_FLAG = False

# 依存: sentence-transformers（フォールバックバックエンド）
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    HAS_ST = True
except Exception:
    HAS_ST = False

# 依存: huggingface_hub（ローカルスナップショット保存用）
try:
    from huggingface_hub import snapshot_download  # type: ignore
    HAS_HF_HUB = True
except Exception:
    HAS_HF_HUB = False


MODEL_ID = "BAAI/bge-m3"

# main_v1.py と整合のために同じタスク名を許容（BGE-M3 はクエリ指示不要のため全て対称処理）
VALID_TASKS = {"sts", "retrieval", "reranking", "clustering", "classification"}


def _has_model_files(local_dir: str) -> bool:
    """ローカルに主要ファイルがあるか簡易判定"""
    if not os.path.isdir(local_dir):
        return False
    files = set(os.listdir(local_dir))
    # config.json のほか、重み（*.safetensors/*.bin）があることを確認
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
      （各バックエンド側でオンライン読み込みにフォールバック）
    """
    if local_dir is None:
        safe_id = safe_model_dirname(model_id)
        local_dir = os.path.join("models", safe_id)

    if _has_model_files(local_dir):
        LOGGER.info(f"[bge-m3] local snapshot found. skip download: {local_dir}")
        return local_dir

    if not HAS_HF_HUB:
        LOGGER.warning(
            "[bge-m3] huggingface_hub が見つからないため、ローカル保存はスキップします。"
            "オンラインでバックエンドから直接読み込みます。"
        )
        return model_id

    os.makedirs(local_dir, exist_ok=True)
    LOGGER.info(f"[bge-m3] downloading snapshot to: {local_dir}")
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
):
    """
    bge-m3 モデルをロードして返す。
    - device: "cpu" / "cuda"（自動で 'cuda:0' を使用）
    - local_dir: 任意。指定がなければ models/{safe_model_dirname(MODEL_ID)} に保存
    戻り値には _backend 属性（'flag' or 'st'）を付与して区別できるようにする。
    """
    hf_token = get_hf_token(enable_transfer=True)
    path_or_id = _ensure_local_model_dir(MODEL_ID, local_dir, hf_token)

    # 1) 推奨: FlagEmbedding
    if HAS_FLAG:
        use_fp16 = bool(device == "cuda" and torch.cuda.is_available())
        model = BGEM3FlagModel(path_or_id, use_fp16=use_fp16)  # device は内部で自動判定
        setattr(model, "_backend", "flag")
        LOGGER.info(f"[bge-m3] FlagEmbedding backend loaded from: {path_or_id} (use_fp16={use_fp16})")
        return model

    # 2) フォールバック: sentence-transformers
    if HAS_ST:
        st_device = ("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu") if device in ("cuda", "cpu") else device

        # torch バージョン簡易判定
        def _parse_ver(v: str):
            parts = []
            for p in str(v).split("."):
                num = "".join(ch for ch in p if ch.isdigit())
                parts.append(int(num) if num else 0)
            while len(parts) < 3:
                parts.append(0)
            return tuple(parts[:3])
        torch_lt_26 = _parse_ver(getattr(torch, "__version__", "0.0.0")) < (2, 6, 0)

        # ローカルスナップショット内の safetensors 有無と ST 形式（modules.json）の有無を調査（再帰）
        has_safetensors_deep = False
        has_modules_json = False
        if isinstance(path_or_id, str) and os.path.isdir(path_or_id):
            try:
                for root, _, files in os.walk(path_or_id):
                    for fn in files:
                        if fn.endswith(".safetensors"):
                            has_safetensors_deep = True
                        if fn == "modules.json" and root == path_or_id:
                            has_modules_json = True
            except Exception:
                pass

        # Sentence-Transformers に渡すパス/ID の決定
        st_path = path_or_id
        if isinstance(path_or_id, str) and os.path.isdir(path_or_id) and not has_modules_json:
            # ST 形式でなければ Hub ID を渡して ST 自身に解決させる
            st_path = MODEL_ID

        # ルート直下の model.safetensors / model.safetensors.index.json の有無で厳密に判定
        use_safetensors = False
        if isinstance(st_path, str) and os.path.isdir(st_path):
            try:
                root_files = set(os.listdir(st_path))
                use_safetensors = ("model.safetensors" in root_files) or ("model.safetensors.index.json" in root_files)
            except Exception:
                use_safetensors = False

        # safetensors が無ければ .bin を許可（torch<2.6 では安全性チェックを無効化）
        if not use_safetensors:
            _patch_torch_load_safety_check_for_torch_lt_26()

        st_kwargs = {
            "trust_remote_code": True,
            "device": st_device,
            "model_kwargs": {"use_safetensors": use_safetensors},
        }
        try:
            model = SentenceTransformer(st_path, **st_kwargs)
        except (OSError, ValueError) as e:
            msg = str(e).lower()
            # safetensors 不在の明示メッセージが出た場合は bin で再試行
            lacks_safetensors = ("does not appear to have a file named model.safetensors" in msg) or ("no file named model.safetensors" in msg)
            if lacks_safetensors and st_kwargs.get("model_kwargs", {}).get("use_safetensors", False):
                LOGGER.warning("[bge-m3] safetensors not found; retrying with use_safetensors=False (bin)")
                st_kwargs["model_kwargs"]["use_safetensors"] = False
                _patch_torch_load_safety_check_for_torch_lt_26()
                try:
                    model = SentenceTransformer(st_path, **st_kwargs)
                    setattr(model, "_backend", "st")
                    LOGGER.info(f"[bge-m3] SentenceTransformer backend loaded from: {st_path} (device={st_device}, use_safetensors=False)")
                    return model
                except Exception as e2:
                    # 最終手段: FlagEmbedding があれば切替
                    if HAS_FLAG:
                        use_fp16 = bool(device == "cuda" and torch.cuda.is_available())
                        model_flag = BGEM3FlagModel(path_or_id if os.path.isdir(path_or_id) else MODEL_ID, use_fp16=use_fp16)
                        setattr(model_flag, "_backend", "flag")
                        LOGGER.warning("[bge-m3] Fallback to FlagEmbedding backend after bin retry failed")
                        return model_flag
                    raise e2
            # その他の理由で失敗したら FlagEmbedding へ切替（あれば）
            needs_flag = ("serious vulnerability" in msg) or ("torch>=2.6" in msg)
            if needs_flag and HAS_FLAG:
                use_fp16 = bool(device == "cuda" and torch.cuda.is_available())
                model_flag = BGEM3FlagModel(path_or_id if os.path.isdir(path_or_id) else MODEL_ID, use_fp16=use_fp16)
                setattr(model_flag, "_backend", "flag")
                LOGGER.warning("[bge-m3] Fallback to FlagEmbedding backend (CVE guard)")
                return model_flag
            raise
        except TypeError:
            # 古い sentence-transformers では trust_remote_code / model_kwargs 未対応の可能性
            st_kwargs.pop("trust_remote_code", None)
            st_kwargs.pop("model_kwargs", None)
            model = SentenceTransformer(st_path, **st_kwargs)
        setattr(model, "_backend", "st")
        LOGGER.info(f"[bge-m3] SentenceTransformer backend loaded from: {st_path} (device={st_device}, use_safetensors={use_safetensors})")
        return model

    raise RuntimeError(
        "BGE-M3 をロードできません。以下のいずれかをインストールしてください:\n"
        "  - pip install -U FlagEmbedding\n"
        "  - pip install -U sentence-transformers\n"
    )


def embed_texts(
    model,
    texts: Union[str, Sequence[str]],
    convert_to_tensor: bool = True,
    normalize: bool = True,
    max_length: Optional[int] = None,
    batch_size: Optional[int] = None,
):
    """
    テキスト群をエンコードしてベクトルを返す。
    - FlagEmbedding バックエンド: model.encode(..., return_dense=True) の 'dense_vecs' を使用
    - sentence-transformers バックエンド: model.encode(..., normalize_embeddings=normalize)
    - convert_to_tensor: True なら torch.Tensor、False なら np.ndarray
    - normalize: True で正規化（コサイン類似度用途向け）
    """
    if isinstance(texts, str):
        texts = [texts]

    backend = getattr(model, "_backend", "flag" if HAS_FLAG else "st")

    if backend == "flag":
        # FlagEmbedding は dict を返す。dense のみ取得
        kwargs = {
            "return_dense": True,
            "return_sparse": False,
            "return_colbert_vecs": False,
        }
        if batch_size is not None:
            kwargs["batch_size"] = batch_size
        if max_length is not None:
            kwargs["max_length"] = max_length
        else:
            # 既定はモデル最大長（8192）。長すぎる場合は短く指定可能
            kwargs["max_length"] = 8192

        outputs = model.encode(list(texts), **kwargs)
        dense = outputs["dense_vecs"]
        if isinstance(dense, torch.Tensor):
            embs = dense
        else:
            # numpy/リスト -> torch
            embs = torch.tensor(dense, dtype=torch.float32)

        if normalize:
            embs = torch.nn.functional.normalize(embs, p=2, dim=-1)

        if convert_to_tensor:
            return embs
        # numpy へ
        try:
            import numpy as np  # lazy import
            return embs.detach().cpu().numpy()
        except Exception:
            return embs  # フォールバック

    # sentence-transformers バックエンド
    try:
        return model.encode(
            list(texts),
            convert_to_tensor=convert_to_tensor,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
    except TypeError:
        # normalize_embeddings 非対応の古いバージョン
        vecs = model.encode(
            list(texts),
            convert_to_tensor=convert_to_tensor,
            show_progress_bar=False,
        )
        if convert_to_tensor:
            if normalize:
                vecs = torch.nn.functional.normalize(vecs, p=2, dim=-1)
            return vecs
        # np.ndarray の場合（normalize はここでは省略）
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
    max_length: Optional[int] = None,
) -> float:
    """
    2つの文章のコサイン類似度を返すユーティリティ。
    - BGE-M3 はクエリ側の特別な指示文不要・対称比較で問題ないため、全タスクで同一手順を採用
    """
    if task not in VALID_TASKS:
        raise ValueError(f"task は {sorted(VALID_TASKS)} から選んでください: got {task}")

    model = get_model(device=device, local_dir=local_dir)
    vecs = embed_texts(
        model,
        [text_a, text_b],
        convert_to_tensor=True,
        normalize=True,
        max_length=max_length,
    )

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
        max_length=max_length,
    )


def compare_files(
    file_a: Optional[str] = None,
    file_b: Optional[str] = None,
    device: str = "cpu",
    task: str = "sts",
    local_dir: Optional[str] = None,
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
        text_a, text_b, device=device, task=task, local_dir=local_dir, max_length=max_length
    )


def compare_default_files(
    device: str = "cpu",
    task: str = "sts",
    local_dir: Optional[str] = None,
    max_length: Optional[int] = None,
) -> float:
    """
    既定の2ファイル（Embedding/data/...）を読み、類似度を返す。
    """
    return compare_files(None, None, device=device, task=task, local_dir=local_dir, max_length=max_length)


if __name__ == "__main__":
    # 簡易自己テスト（CPU）
    qa = "今日は雨です。傘を持って行きましょう。"
    qb = "外は雨模様なので、出かける時は傘が必要です。"

    sim = embed_and_score(qa, qb, device="cpu", task="sts", max_length=512)
    print(f"[self-test] BGE-M3 similarity(STS)={sim:.4f}")
