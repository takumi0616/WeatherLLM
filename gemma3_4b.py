# -*- coding: utf-8 -*-
"""
Gemma 系（Gemma-2 / Gemma-3）をローカルPython環境で実行するための純Pythonスクリプト。
Colab依存（!pip, login など）を排除し、CLI化・ログ整備・出力先整理・HF_TOKEN対応。
モデルIDに応じて Gemma-2 (テキスト専用) と Gemma-3 (マルチモーダル対応) を自動で扱い分けます。
このスクリプトではテキストのみを対象とします（画像は利用しません）。

例:
  Gemma-2（日本語Instruct）で生成:
    python src/WeatherLLM/gemma3_4b.py --model google/gemma-2-2b-jpn-it --prompt "あなたの好きな食べ物は？"

  Gemma-3（4B it）で生成（テキストのみ）:
    python src/WeatherLLM/gemma3_4b.py --model google/gemma-3-4b-it --prompt "あなたの好きな食べ物は？"
"""

import os
import sys
import argparse
from typing import List, Dict, Any, Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

# Gemma-3 用（環境にある場合のみ使用。無ければインポート時に例外→Gemma-2経路にフォールバック）
try:
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration  # type: ignore
    HAS_GEMMA3 = True
except Exception:
    HAS_GEMMA3 = False

from llm_utils import (
    setup_logger,
    select_dtype,
    resolve_device_map,
    enable_tf32,
    get_hf_token,
    try_apply_chat_template,
    make_output_dir,
    save_text,
    save_json,
)

LOGGER = setup_logger("gemma")


def is_gemma3_model_id(model_id: str) -> bool:
    return "gemma-3" in model_id.lower()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Gemma-2 / Gemma-3 locally (text only).")
    # デフォルトは元のノートで使っていた Gemma-2 日本語ITモデル
    parser.add_argument("--model", type=str, default="google/gemma-2-2b-jpn-it", help="HF Hub のモデルID or ローカルパス")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="device_map 指定")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bfloat16", "float16", "float32"], help="内部dtype")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="生成最大トークン数")
    parser.add_argument("--temperature", type=float, default=0.7, help="生成温度")
    parser.add_argument("--top-p", type=float, default=0.95, help="nucleus samplingの確率質量")
    parser.add_argument("--top-k", type=int, default=50, help="top-k サンプリング")
    parser.add_argument("--seed", type=int, default=0, help="乱数シード")
    parser.add_argument("--prompt", type=str, default="あなたの好きな食べ物は何ですか？", help="ユーザプロンプト（テキストのみ）")
    parser.add_argument("--out-base", type=str, default="src/WeatherLLM/outputs", help="出力のベースディレクトリ")
    parser.add_argument("--tag", type=str, default=None, help="出力ディレクトリに付与するタグ")
    return parser.parse_args()


def resolve_torch_dtype(arg: str) -> torch.dtype:
    if arg == "auto":
        return select_dtype()
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping[arg]


def build_messages_for_text(prompt: str) -> List[Dict[str, Any]]:
    # テキストのみの標準的な chat_template 用メッセージ
    return [
        {"role": "system", "content": "あなたは親切なAIアシスタントです。日本語で丁寧に回答してください。"},
        {"role": "user", "content": prompt},
    ]


def build_gemma3_chat_for_text(prompt: str) -> List[Dict[str, Any]]:
    # Gemma-3 はマルチモーダル形式。ここではテキストのみの content を作成
    return [
        {"role": "system", "content": [{"type": "text", "text": "あなたは親切なAIアシスタントです。日本語で丁寧に回答してください。"}]},
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]


def run_gemma2(args, out_dir: str, dtype: torch.dtype, device_map: Any, hf_token: Optional[str]) -> str:
    LOGGER.info(f"[Gemma-2] Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=hf_token, use_fast=True)

    LOGGER.info(f"[Gemma-2] Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map=device_map,
        torch_dtype=dtype,
        trust_remote_code=True,
        token=hf_token,
        low_cpu_mem_usage=True,
    ).eval()

    messages = build_messages_for_text(args.prompt)
    input_ids = try_apply_chat_template(tokenizer, messages, add_generation_prompt=True)
    input_ids = input_ids.to(model.device)

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    with torch.no_grad():
        outputs = model.generate(input_ids, **gen_kwargs)

    # 応答部分のみ抽出（プロンプト分を除外）
    response_ids = outputs[0][input_ids.size(1):]
    text = tokenizer.decode(response_ids, skip_special_tokens=True)

    save_json(os.path.join(out_dir, "messages.json"), messages)
    save_text(os.path.join(out_dir, "generation.txt"), text)
    return text


def run_gemma3(args, out_dir: str, dtype: torch.dtype, device_map: Any, hf_token: Optional[str]) -> str:
    if not HAS_GEMMA3:
        raise RuntimeError("Gemma3ForConditionalGeneration/AutoProcessor が利用できません。transformers が対応版か確認してください。")

    LOGGER.info(f"[Gemma-3] Loading processor: {args.model}")
    processor = AutoProcessor.from_pretrained(args.model, token=hf_token, trust_remote_code=True)

    LOGGER.info(f"[Gemma-3] Loading model: {args.model}")
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model,
        device_map=device_map,
        torch_dtype=dtype,
        trust_remote_code=True,
        token=hf_token,
        low_cpu_mem_usage=True,
    ).eval()

    chat = build_gemma3_chat_for_text(args.prompt)
    # Gemma-3 は processor.apply_chat_template を推奨
    inputs = processor.apply_chat_template(
        chat,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    # dtype を指定（bf16/fp16など）。テキストのみなので to(dtype=) でOK
    inputs = {k: v.to(model.device, dtype=dtype if v.dtype.is_floating_point else None) for k, v in inputs.items()}

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    with torch.inference_mode():
        outputs = model.generate(**inputs, **gen_kwargs)

    # 入力長を元に応答部分だけ取り出す
    input_len = inputs["input_ids"].shape[-1]
    response_ids = outputs[0][input_len:]

    text = processor.decode(response_ids, skip_special_tokens=True)
    save_json(os.path.join(out_dir, "messages.json"), chat)
    save_text(os.path.join(out_dir, "generation.txt"), text)
    return text


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    out_dir = make_output_dir(args.out_base, args.model, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    save_json(os.path.join(out_dir, "run_args.json"), vars(args))

    dtype = resolve_torch_dtype(args.dtype)
    device_map = resolve_device_map(args.device)
    LOGGER.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        LOGGER.info(f"GPU: {torch.cuda.get_device_name(0)}")
    enable_tf32(LOGGER)
    LOGGER.info(f"Selected dtype: {dtype}")
    LOGGER.info(f"device_map: {device_map}")

    hf_token = get_hf_token(enable_transfer=True)

    is_g3 = is_gemma3_model_id(args.model)
    try:
        if is_g3:
            LOGGER.info("Detected Gemma-3 model id. Switching to Gemma-3 pipeline.")
            text = run_gemma3(args, out_dir, dtype, device_map, hf_token)
        else:
            LOGGER.info("Using Gemma-2 pipeline.")
            text = run_gemma2(args, out_dir, dtype, device_map, hf_token)
    except Exception as e:
        LOGGER.error(f"Generation failed: {e}")
        raise

    print("=== Gemma Output ===")
    print(text)
    LOGGER.info(f"All done. Outputs saved under: {out_dir}")


if __name__ == "__main__":
    main()
