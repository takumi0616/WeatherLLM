# -*- coding: utf-8 -*-
import os
import sys
import re
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

import torch


def setup_logger(name: str = "llm") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def select_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def resolve_device_map(device_arg: str) -> Any:
    if device_arg == "auto":
        return "auto"
    if device_arg == "cuda":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device_arg == "cpu":
        return "cpu"
    return device_arg


def enable_tf32(logger: Optional[logging.Logger] = None) -> None:
    if not torch.cuda.is_available():
        return
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if logger:
            logger.info("Enabled TF32 acceleration (matmul/cudnn).")
    except Exception:
        pass


def get_hf_token(enable_transfer: bool = True) -> Optional[str]:
    token = os.getenv("HF_TOKEN", None)
    if token and enable_transfer:
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    return token


def ensure_pad_token(tokenizer, model) -> None:
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        model.config.pad_token_id = tokenizer.pad_token_id
    except Exception:
        pass


def build_borea_prompt(messages: List[Dict[str, str]]) -> str:
    prompt = ""
    for message in messages:
        if message["role"] == "system":
            prompt += "<|system|>\n" + message["content"] + "<|end|>\n"
        elif message["role"] == "user":
            prompt += "<|user|>\n" + message["content"] + "<|end|>\n"
        elif message["role"] == "assistant":
            prompt += "<|assistant|>\n" + message["content"] + "<|end|>\n"
    prompt += "<|assistant|>\n"
    return prompt


def try_apply_chat_template(tokenizer, messages: List[Dict[str, Any]], add_generation_prompt: bool = True):
    apply_fn = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_fn):
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            return_tensors="pt",
        )
    # フォールバック（単純連結）
    text = ""
    for m in messages:
        text += f"{m['role']}: {m['content']}\n"
    # 末尾に assistant のプロンプト
    if add_generation_prompt:
        text += "assistant: "
    return tokenizer(text, return_tensors="pt")


def safe_model_dirname(model_id: str) -> str:
    # 例: "meta-llama/Meta-Llama-3-8B-Instruct" -> "meta-llama_Meta-Llama-3-8B-Instruct"
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", model_id)


def make_output_dir(base_dir: str, model_id: str, tag: Optional[str] = None) -> str:
    safe_id = safe_model_dirname(model_id)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [base_dir, safe_id, ts]
    if tag:
        parts.append(tag)
    out_dir = os.path.join(*parts)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def save_text(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ===== Distributed (DDP) helpers =====

def get_env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default


def is_dist_available() -> bool:
    try:
        import torch.distributed as dist  # type: ignore
        return dist.is_available()
    except Exception:
        return False


def is_dist_initialized() -> bool:
    if not is_dist_available():
        return False
    try:
        import torch.distributed as dist  # type: ignore
        return dist.is_initialized()
    except Exception:
        return False


def get_local_rank() -> int:
    return get_env_int("LOCAL_RANK", 0)


def get_rank() -> int:
    return get_env_int("RANK", 0)


def get_world_size() -> int:
    return get_env_int("WORLD_SIZE", 1)


def is_main_process() -> bool:
    return get_rank() == 0


def init_distributed(logger: Optional[logging.Logger] = None, backend: Optional[str] = None) -> bool:
    """
    torchrun 等で LOCAL_RANK/RANK/WORLD_SIZE が設定されている場合に限り初期化。
    それ以外は何もしない（単GPU/単プロセス動作）。
    """
    if not is_dist_available():
        return False

    # 環境変数がなければ DDP 初期化は行わない
    if "LOCAL_RANK" not in os.environ and "RANK" not in os.environ and "WORLD_SIZE" not in os.environ:
        return False

    if is_dist_initialized():
        return True

    try:
        import torch.distributed as dist  # type: ignore
        if backend is None:
            backend = "nccl" if torch.cuda.is_available() else "gloo"

        local_rank = get_local_rank()
        if torch.cuda.is_available():
            try:
                torch.cuda.set_device(local_rank)
            except Exception:
                pass

        dist.init_process_group(backend=backend)
        if logger and is_main_process():
            logger.info(f"Initialized distributed process group (backend={backend}, world_size={get_world_size()}, rank={get_rank()}, local_rank={local_rank})")
        return True
    except Exception as e:
        if logger:
            logger.warning(f"Distributed init skipped/failed: {e}")
        return False


def barrier() -> None:
    if is_dist_initialized():
        try:
            import torch.distributed as dist  # type: ignore
            dist.barrier()
        except Exception:
            pass


def distributed_device_map(device_arg: str) -> Any:
    """
    DDP 初期化済みなら各プロセスを単一GPUに固定。
    それ以外は以下の方針:
      - 複数GPUが見つかり、device_arg が 'auto' または 'cuda' の場合は 'auto'（自動シャーディング）を返す
      - それ以外は従来の device_map 解決を使用
    """
    if is_dist_initialized() and torch.cuda.is_available():
        return f"cuda:{get_local_rank()}"
    if torch.cuda.is_available():
        try:
            if torch.cuda.device_count() > 1 and device_arg in ("auto", "cuda"):
                return "auto"
        except Exception:
            pass
    return resolve_device_map(device_arg)


# ===== Single-process multi-GPU (DataParallel) helpers =====

def is_multi_gpu() -> bool:
    try:
        return torch.cuda.is_available() and torch.cuda.device_count() > 1
    except Exception:
        return False


def maybe_wrap_data_parallel(model: Any) -> Any:
    """
    torchrun / DDP を使わずに python 単発実行でも複数GPUを使いたい場合のフォールバック。
    - DDP 未初期化かつ 複数GPU のときに限り DataParallel でラップする
    - それ以外はそのまま返す
    注意:
      - DataParallel は単一プロセス内のデータ並列。巨大モデルのモデル並列は device_map='auto' を推奨（推論時）。
      - 学習速度 / 挙動はDDPより劣ることがあるため、より厳密な分散学習には torchrun を推奨。
    """
    if not is_dist_initialized() and is_multi_gpu():
        try:
            return torch.nn.DataParallel(model)
        except Exception:
            return model
    return model


def unwrap_model(model: Any) -> Any:
    """DataParallel 等でラップされている場合に元のモジュールを返す"""
    return getattr(model, "module", model)
