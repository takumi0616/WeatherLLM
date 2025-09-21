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
