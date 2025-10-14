# -*- coding: utf-8 -*-
"""
cl-nagoya/JaColBERTv2 をローカル環境で使用可能にするラッパ。

提供関数:
- embed_and_score(text_a: str, text_b: str, device: str = "cpu",
                  task: str = "sts", local_dir: Optional[str] = None,
                  max_length: int = 512) -> float
    2つの文章の類似度スコア（-1..1）を返す。
    - Sentence-Transformers 形式でロード可能な場合は通常のコサイン類似度
    - それ以外（Transformers バックエンド）では ColBERT 風の Late Interaction 類似度
      ・STS: 双方向平均 (q→d と d→q)
      ・retrieval/reranking: 片方向 (q→d)
      ・その他: 双方向平均

環境変数（任意）:
- JACOLBERT_V2_ID         : 既定モデルIDを上書き（デフォルト: 'bclavie/JaColBERTv2'）
- JACOLBERT_V2_LOCAL_DIR  : 既にダウンロード済みのローカル ST 形式ディレクトリを明示
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

LOGGER = setup_logger("JaColBERTv2")

# 依存: sentence-transformers（利用可能なら最優先で使う）
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    HAS_ST = True
except Exception:
    HAS_ST = False

# 依存: transformers（フォールバックで使用）
try:
    from transformers import AutoTokenizer, AutoModel  # type: ignore
    HAS_HF = True
except Exception:
    HAS_HF = False

# 依存: huggingface_hub（ローカルスナップショット保存）
try:
    from huggingface_hub import snapshot_download  # type: ignore
    HAS_HF_HUB = True
except Exception:
    HAS_HF_HUB = False


DEFAULT_MODEL_ID = "bclavie/JaColBERTv2"


def _resolve_model_id() -> str:
    mid = os.getenv("JACOLBERT_V2_ID", "").strip()
    return mid or DEFAULT_MODEL_ID


def _has_st_dir(local_dir: str) -> bool:
    """Sentence-Transformers 形式らしきディレクトリか簡易判定"""
    if not os.path.isdir(local_dir):
        return False
    files = set(os.listdir(local_dir))
    if not {"config.json", "modules.json"}.issubset(files):
        return False
    has_weight = any(fn.endswith((".safetensors", ".bin")) for fn in files)
    return has_weight


def _ensure_local_model_dir(
    model_id: str,
    local_dir: Optional[str],
    hf_token: Optional[str],
) -> str:
    """
    ローカル保存を確保（存在すればスキップ）。
    - 環境変数 JACOLBERT_V2_LOCAL_DIR が ST 形式として有効ならそれを優先
    - huggingface_hub が無ければ model_id を返し、ライブラリ側のオンラインロードに委譲
    """
    # 環境変数でローカル指定があれば優先
    env_local = os.getenv("JACOLBERT_V2_LOCAL_DIR", "").strip()
    if env_local and _has_st_dir(env_local):
        LOGGER.info(f"[JaColBERTv2] Use local dir from env: {env_local}")
        return env_local
    elif env_local:
        LOGGER.warning(f"[JaColBERTv2] JACOLBERT_V2_LOCAL_DIR は ST 形式として不正です: {env_local}")

    if local_dir is None:
        safe_id = safe_model_dirname(model_id)
        local_dir = os.path.join("models", safe_id)

    if _has_st_dir(local_dir):
        LOGGER.info(f"[JaColBERTv2] local snapshot found. skip download: {local_dir}")
        return local_dir

    if not HAS_HF_HUB:
        LOGGER.warning(
            "[JaColBERTv2] huggingface_hub が見つからないため、ローカル保存はスキップします。"
            "バックエンド側のオンライン読み込みに委譲します。"
        )
        return model_id

    os.makedirs(local_dir, exist_ok=True)
    LOGGER.info(f"[JaColBERTv2] downloading snapshot to: {local_dir} (repo_id={model_id})")
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            token=hf_token,
            local_dir_use_symlinks=False,
        )
    except Exception as e:
        LOGGER.warning(f"[JaColBERTv2] snapshot_download failed: {e}. Fallback to online load: {model_id}")
        return model_id
    return local_dir


class _ModelBundle:
    """
    内部利用のロード済みモデル束:
    - backend: 'st' | 'hf'
    - st_model: SentenceTransformer | None
    - hf_model: AutoModel | None
    - tokenizer: AutoTokenizer | None
    - device: torch.device 文字列
    """
    def __init__(self, backend: str, device: str, st_model=None, hf_model=None, tokenizer=None):
        self.backend = backend
        self.device = device
        self.st_model = st_model
        self.hf_model = hf_model
        self.tokenizer = tokenizer


def _load_model(
    device: str = "cpu",
    local_dir: Optional[str] = None,
) -> _ModelBundle:
    """
    可能なら ST バックエンド、無理なら Transformers バックエンドでロード。
    """
    model_id = _resolve_model_id()
    hf_token = get_hf_token(enable_transfer=True)
    path_or_id = _ensure_local_model_dir(model_id, local_dir, hf_token)

    torch_device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"

    # 1) Sentence-Transformers バックエンド（ローカル ST 形式が実体として存在する場合のみ）
    if HAS_ST and isinstance(path_or_id, str) and os.path.isdir(path_or_id) and _has_st_dir(path_or_id):
        try:
            st_model = SentenceTransformer(
                path_or_id,
                device=torch_device,
            )
            LOGGER.info(f"[JaColBERTv2] ST backend loaded from: {path_or_id} (device={st_model.device})")
            return _ModelBundle(backend="st", device=torch_device, st_model=st_model)
        except Exception as e:
            LOGGER.warning(f"[JaColBERTv2] ST backend load failed, fallback to HF: {e}")

    # 2) Transformers バックエンド
    if not HAS_HF:
        raise RuntimeError(
            "transformers が見つからないため JaColBERTv2 をロードできません。\n"
            "pip install -U transformers sentencepiece を実行してください。"
        )
    try:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                path_or_id,
                use_fast=True,
                trust_remote_code=True,
            )
        except Exception:
            # tokenizer がリポジトリ内に無い場合のフォールバック（モデルカード記載のベース）
            tokenizer = AutoTokenizer.from_pretrained(
                "tohoku-nlp/bert-base-japanese-v3",
                use_fast=True,
                trust_remote_code=True,
            )
        hf_model = AutoModel.from_pretrained(
            path_or_id,
            trust_remote_code=True,
        ).to(torch_device).eval()
    except Exception as e:
        raise RuntimeError(
            "JaColBERTv2 のモデルリポジトリにアクセスできませんでした。"
            " 環境変数 JACOLBERT_V2_ID（正しい HF モデルID）"
            " または JACOLBERT_V2_LOCAL_DIR（ローカルのモデルディレクトリ）を設定してください。"
        ) from e
    LOGGER.info(f"[JaColBERTv2] HF backend loaded from: {path_or_id} (device={torch_device})")
    return _ModelBundle(backend="hf", device=torch_device, hf_model=hf_model, tokenizer=tokenizer)


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """1D ベクトル同士のコサイン類似度を返す。"""
    if a.dim() == 1:
        a = a.unsqueeze(0)
    if b.dim() == 1:
        b = b.unsqueeze(0)
    sim = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    return float(sim.squeeze().item())


def _encode_st(
    st_model: "SentenceTransformer",
    text: Union[str, Sequence[str]],
    normalize: bool = True,
):
    if isinstance(text, str):
        text = [text]
    try:
        return st_model.encode(
            list(text),
            convert_to_tensor=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
    except TypeError:
        vecs = st_model.encode(list(text), convert_to_tensor=True, show_progress_bar=False)
        if normalize:
            vecs = torch.nn.functional.normalize(vecs, p=2, dim=-1)
        return vecs


def _token_embeddings_hf(
    model,
    tokenizer,
    text: str,
    device: str,
    max_length: int = 512,
) -> torch.Tensor:
    """
    Transformers バックエンドで 1 文からトークン埋め込み行列 (T, D) を取得。
    - 特殊トークンは除外（special_tokens_mask と attention_mask を用いる）
    - L2 正規化はここでは行わない（上位で実施）
    """
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
        return_special_tokens_mask=True,
        add_special_tokens=True,
    )
    for k, v in enc.items():
        enc[k] = v.to(device)

    with torch.inference_mode():
        out = model(**enc)
    if hasattr(out, "last_hidden_state"):
        hidden = out.last_hidden_state  # (1, T, D)
    elif isinstance(out, (tuple, list)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
        hidden = out[0]
    else:
        raise RuntimeError("モデル出力から token embeddings を取得できませんでした。")

    hidden = hidden.squeeze(0)  # (T, D)
    attn = enc["attention_mask"].squeeze(0).bool()  # (T,)
    spm = enc.get("special_tokens_mask", None)
    if spm is not None:
        spm = spm.squeeze(0).bool()
        mask = attn & (~spm)
    else:
        mask = attn

    tok_embs = hidden[mask]  # (T_eff, D)
    if tok_embs.numel() == 0:
        tok_embs = hidden[attn]
        if tok_embs.numel() == 0:
            tok_embs = hidden
    return tok_embs


def _colbert_dir_sim(q_tok: torch.Tensor, d_tok: torch.Tensor) -> float:
    """
    ColBERT 風の片方向スコア: 各 query トークンと document トークンのコサイン類似度行列から、
    query 側で最大類似を取り、平均を返す。q_tok/d_tok は (T, D) の L2 正規化済みテンソル。
    戻り値は [-1, 1] 範囲に収まる。
    """
    sim_mat = q_tok @ d_tok.t()
    max_per_q = sim_mat.max(dim=1).values
    if max_per_q.numel() == 0:
        return 0.0
    return float(max_per_q.mean().item())


def _colbert_bidir_sim(a_tok: torch.Tensor, b_tok: torch.Tensor) -> float:
    """
    双方向平均（a→b と b→a の平均）。
    """
    if a_tok.size(-1) != b_tok.size(-1):
        return 0.0
    a_n = torch.nn.functional.normalize(a_tok, p=2, dim=-1)
    b_n = torch.nn.functional.normalize(b_tok, p=2, dim=-1)
    s_ab = _colbert_dir_sim(a_n, b_n)
    s_ba = _colbert_dir_sim(b_n, a_n)
    return (s_ab + s_ba) * 0.5


VALID_TASKS = {"sts", "retrieval", "reranking", "clustering", "classification"}


def embed_and_score(
    text_a: str,
    text_b: str,
    device: str = "cpu",
    task: str = "sts",
    local_dir: Optional[str] = None,
    max_length: int = 512,
) -> float:
    """
    2 文の類似度（-1..1）を返す。
    - ST バックエンド: 句ベクトルのコサイン類似度
    - HF バックエンド: ColBERT 風の Late Interaction 類似度
      ・STS/その他: 双方向平均
      ・retrieval/reranking: 片方向 (A を query、B を document)
    """
    if task not in VALID_TASKS:
        raise ValueError(f"task は {sorted(VALID_TASKS)} から選択してください: got {task}")

    bundle = _load_model(device=device, local_dir=local_dir)

    # 1) Sentence-Transformers backend
    if bundle.backend == "st" and bundle.st_model is not None:
        vecs = _encode_st(bundle.st_model, [text_a, text_b], normalize=True)
        a, b = vecs[0], vecs[1]
        if hasattr(bundle.st_model, "similarity"):
            try:
                return float(bundle.st_model.similarity(a, b).item())
            except Exception:
                pass
        return _cosine_sim(a, b)

    # 2) Transformers backend: ColBERT 風 late interaction
    if bundle.backend == "hf" and bundle.hf_model is not None and bundle.tokenizer is not None:
        a_tok = _token_embeddings_hf(bundle.hf_model, bundle.tokenizer, text_a, bundle.device, max_length=max_length)
        b_tok = _token_embeddings_hf(bundle.hf_model, bundle.tokenizer, text_b, bundle.device, max_length=max_length)

        if task in {"retrieval", "reranking"}:
            a_n = torch.nn.functional.normalize(a_tok, p=2, dim=-1)
            b_n = torch.nn.functional.normalize(b_tok, p=2, dim=-1)
            return _colbert_dir_sim(a_n, b_n)

        return _colbert_bidir_sim(a_tok, b_tok)

    raise RuntimeError("JaColBERTv2 をロードできませんでした。transformers または sentence-transformers の環境を確認してください。")


# ------------- 便利関数（他実装と整合） -------------

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
    max_length: int = 512,
) -> float:
    """外部から即利用できる共通API。2文を引数に取り、スコア（-1..1）を返す。"""
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
    max_length: int = 512,
) -> float:
    """
    2つのファイルパスを受け取り、中身を読み込んでスコアを返す。
    file_a/file_b が None の場合は既定の2ファイルを使用。
    """
    path_a = file_a or DEFAULT_FILE_A
    path_b = file_b or DEFAULT_FILE_B
    text_a = _read_text(path_a)
    text_b = _read_text(path_b)
    return compute_cosine_similarity(
        text_a=text_a,
        text_b=text_b,
        device=device,
        task=task,
        local_dir=local_dir,
        max_length=max_length,
    )


def compare_default_files(
    device: str = "cpu",
    task: str = "sts",
    local_dir: Optional[str] = None,
    max_length: int = 512,
) -> float:
    """既定の 2 ファイルを読み、スコアを返す。"""
    return compare_files(None, None, device=device, task=task, local_dir=local_dir, max_length=max_length)


if __name__ == "__main__":
    # 簡易自己テスト（CPU）
    qa = "今日は雨です。傘を持って行きましょう。"
    qb = "外は雨模様なので、出かける時は傘が必要です。"
    try:
        s = embed_and_score(qa, qb, device="cpu", task="sts")
        print(f"[self-test] JaColBERTv2 similarity={s:.4f}")
    except Exception as e:
        print(f"[self-test] failed: {e}")
