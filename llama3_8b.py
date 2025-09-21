# -*- coding: utf-8 -*-
"""
Llama 3 系（既定 8B Instruct）をローカルPython環境で実行するための純Pythonスクリプト。
Colab依存（!pip, huggingface-cli login など）を排除し、CLI化・ログ整備・出力先整理・HF_TOKEN対応・量子化に対応。

例:
  生成（既定: bf16, device_map=auto, 4bit量子化はデフォルト無効）
    python src/WeatherLLM/llama3_8b.py

  モデル/プロンプト/出力指定:
    python src/WeatherLLM/llama3_8b.py --model meta-llama/Meta-Llama-3-8B-Instruct \\
      --prompt "短い自己紹介を日本語で"

  4bit量子化を使う（大きいモデルでメモリ節約）:
    python src/WeatherLLM/llama3_8b.py --use-4bit
"""

import os
import sys
import argparse
from typing import List, Dict, Optional

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

LOGGER = setup_logger("llama3")


def default_messages(prompt: str) -> List[Dict[str, str]]:
    if prompt:
        return [
            {"role": "system", "content": "あなたは親切で丁寧なAIアシスタントです。日本語で回答してください。"},
            {"role": "user", "content": prompt},
        ]
    return [
        {"role": "system", "content": "あなたは親切で丁寧なAIアシスタントです。日本語で回答してください。"},
        {"role": "user", "content": "クマが海辺に行ってアザラシと友達になり、最後は家に帰る短編小説を書いてください。"},
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Llama 3 locally (optionally quantized).")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="HF Hub のモデルID or ローカルパス")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="device_map 指定")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bfloat16", "float16", "float32"], help="内部dtype")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="生成最大トークン数")
    parser.add_argument("--temperature", type=float, default=0.6, help="生成温度")
    parser.add_argument("--top-p", type=float, default=0.9, help="nucleus samplingの確率質量")
    parser.add_argument("--seed", type=int, default=0, help="乱数シード")
    parser.add_argument("--prompt", type=str, default="", help="ユーザプロンプト（messagesに反映）")
    parser.add_argument("--out-base", type=str, default="src/WeatherLLM/outputs", help="出力のベースディレクトリ")
    parser.add_argument("--tag", type=str, default=None, help="出力ディレクトリに付与するタグ")
    # 量子化オプション
    parser.add_argument("--use-4bit", action="store_true", help="4bit量子化を有効化（bitsandbytes 必須）")
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

    # 量子化設定（オプション: --use-4bit）
    quant_cfg = None
    if args.use_4bit:
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
        torch_dtype=dtype if quant_cfg is None else None,
        quantization_config=quant_cfg,
        trust_remote_code=False,
        token=hf_token,
        low_cpu_mem_usage=True,
    ).eval()

    messages = default_messages(args.prompt)
    input_ids = try_apply_chat_template(tokenizer, messages, add_generation_prompt=True)
    input_ids = input_ids.to(model.device)

    # Llama 3 では <|eot_id|> が終端として使われる場合がある
    terminators: Optional[List[int]] = None
    try:
        eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if isinstance(eot_id, int) and eot_id != tokenizer.unk_token_id:
            terminators = [tokenizer.eos_token_id, eot_id]
    except Exception:
        pass

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token_id=terminators if terminators is not None else tokenizer.eos_token_id,
    )

    with torch.no_grad():
        outputs = model.generate(input_ids, **gen_kwargs)

    # 応答部分のみを抽出（chat_template利用時は先頭プロンプト分を除外）
    response_ids = outputs[0][input_ids.size(1):]
    text = tokenizer.decode(response_ids, skip_special_tokens=True)

    print("=== Llama 3 Output ===")
    print(text)

    save_json(os.path.join(out_dir, "messages.json"), messages)
    save_text(os.path.join(out_dir, "generation.txt"), text)
    LOGGER.info(f"All done. Outputs saved under: {out_dir}")


if __name__ == "__main__":
    main()
