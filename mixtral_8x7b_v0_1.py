# -*- coding: utf-8 -*-
"""
Mixtral-8x7B をローカルPython環境で実行するための純Pythonスクリプト。
Colab依存（!pip, huggingface-cli login など）を排除し、CLI化・ログ整備・出力先整理・HF_TOKEN対応・4bit量子化に対応。

例:
  生成（既定: 4bit量子化, bf16, device_map=auto）
    python src/WeatherLLM/mixtral_8x7b_v0_1.py

  モデル/プロンプト/出力指定:
    python src/WeatherLLM/mixtral_8x7b_v0_1.py --model mistralai/Mixtral-8x7B-Instruct-v0.1 \\
      --prompt "Who is the cutest in Madoka Magica?"

  8bit/FP16へ切替:
    python src/WeatherLLM/mixtral_8x7b_v0_1.py --no-4bit --dtype float16
"""

import os
import sys
import argparse
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

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

LOGGER = setup_logger("mixtral")


def default_messages(prompt: str) -> List[Dict[str, str]]:
    if prompt:
        return [{"role": "user", "content": prompt}]
    # デフォルトの英語プロンプト
    return [{"role": "user", "content": "Who is the cutest in Madoka Magica?"}]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Mixtral-8x7B locally (quantized inference).")
    parser.add_argument("--model", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1", help="HF Hub のモデルID or ローカルパス")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="device_map 指定")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bfloat16", "float16", "float32"], help="内部dtype")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="生成最大トークン数")
    parser.add_argument("--temperature", type=float, default=0.5, help="生成温度")
    parser.add_argument("--top-p", type=float, default=0.95, help="nucleus samplingの確率質量")
    parser.add_argument("--top-k", type=int, default=40, help="top-k サンプリング")
    parser.add_argument("--seed", type=int, default=0, help="乱数シード")
    parser.add_argument("--prompt", type=str, default="", help="ユーザプロンプト（messagesに反映）")
    parser.add_argument("--out-base", type=str, default="src/WeatherLLM/outputs", help="出力のベースディレクトリ")
    parser.add_argument("--tag", type=str, default=None, help="出力ディレクトリに付与するタグ")
    # 量子化オプション
    parser.add_argument("--no-4bit", action="store_true", help="4bit量子化を無効化（既定は有効）")
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

    # 量子化設定（既定: 4bit）
    quant_4bit = not args.no_4bit
    quant_cfg = None
    if quant_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if dtype == torch.bfloat16 else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        LOGGER.info("Using 4bit quantization (bitsandbytes).")

    LOGGER.info(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=hf_token, use_fast=True)

    LOGGER.info(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map=device_map,
        torch_dtype=dtype if not quant_4bit else None,
        quantization_config=quant_cfg,
        trust_remote_code=False,
        token=hf_token,
        low_cpu_mem_usage=True,
    ).eval()

    messages = default_messages(args.prompt)
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

    # 応答部分のみを抽出（chat_template利用時は先頭プロンプト分を除外）
    response_ids = outputs[0][input_ids.size(1):]
    text = tokenizer.decode(response_ids, skip_special_tokens=True)

    print("=== Mixtral Output ===")
    print(text)

    save_json(os.path.join(out_dir, "messages.json"), messages)
    save_text(os.path.join(out_dir, "generation.txt"), text)
    LOGGER.info(f"All done. Outputs saved under: {out_dir}")


if __name__ == "__main__":
    main()
