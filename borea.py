# -*- coding: utf-8 -*-
"""
Borea (Phi-3.5 系) をローカルPython環境で実行するための純Pythonスクリプト。
Colab依存（!pip, !nvidia-smi など）を除去し、CLI化・ログ整備・出力先整理・HF_TOKEN対応を実施。

主な機能
- 生成のみ or 最小学習(--train)の実行
- GPU/CPU 自動判定と dtype 最適化（bf16/fp16/float32）
- TF32 有効化（対応GPUのみ）
- HF_TOKEN (.env) を自動利用してモデル/トークナイザーの取得
- pad_token 未定義モデルへの対処
- DynamicCache.get_max_length の暫定パッチ（phi3系のみ・必要時）
- 生成/学習の成果物を outputs/ 以下に時刻付きディレクトリで整理保存

例:
  生成のみ:
    python src/WeatherLLM/borea.py

  生成 + ミニ学習:
    python src/WeatherLLM/borea.py --train --epochs 1 --batch-size 1 --output-dir src/WeatherLLM/fine_tuned-model
"""

import os
import sys
import argparse
import logging
from typing import List, Dict, Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset

# 共通ユーティリティ
from llm_utils import (
    setup_logger,
    select_dtype,
    resolve_device_map,
    enable_tf32,
    get_hf_token,
    ensure_pad_token,
    build_borea_prompt,
    make_output_dir,
    save_text,
    save_json,
)

LOGGER = setup_logger("borea")


def maybe_monkey_patch_dynamic_cache(logger: logging.Logger) -> None:
    """
    一部モデル（phi3系等）で DynamicCache.get_max_length が非推奨になった件への暫定対処。
    既に修正済みな場合はスキップ。失敗しても致命ではない。
    """
    try:
        from transformers.models.phi3.modeling_phi3 import DynamicCache  # type: ignore

        def new_get_max_length(self):
            return getattr(self, "cache_position", None)

        DynamicCache.get_max_length = new_get_max_length  # type: ignore
        logger.info("Applied monkey patch: DynamicCache.get_max_length -> cache_position")
    except Exception as e:
        logger.info(f"DynamicCache monkey patch skipped ({e})")


def build_sample_messages() -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "あなたは日本語能力が高い高度なAIです。特別な指示がない限り日本語で返答してください。",
        },
        {
            "role": "user",
            "content": "「生き物デザイナー」という職業があります。これは、自分が考えたオリジナルの生き物をデザインし、実際にDNAを編集して作り出す仕事です。あなたが生き物デザイナーである場合、どんな生き物を作りたいですか？また、その生き物が持つ特徴や能力について説明してください。",
        },
    ]


def build_tiny_train_dataset() -> Dataset:
    """最小限のサンプルで学習用データセットを作る（デモ目的）"""
    train_data = [
        {
            "instruction": "日本の首都はどこですか？",
            "response": "日本の首都は東京です。",
        },
        {
            "instruction": "富士山の高さはどれくらいですか？",
            "response": "富士山の高さは3776メートルです。",
        },
    ]
    return Dataset.from_list(train_data)


def preprocess_for_causal_lm(examples, tokenizer, max_length: int = 1024):
    """Causal LM用に入力を連結しトークナイズ。学習時のラベルは DataCollator に任せる"""
    inputs = []
    for instruction, response in zip(examples["instruction"], examples["response"]):
        prompt_text = f"<|user|>\n{instruction}<|end|>\n<|assistant|>\n"
        target = response + "<|end|>"
        full_text = prompt_text + target
        inputs.append(full_text)
    model_inputs = tokenizer(
        inputs,
        max_length=max_length,
        truncation=True,
    )
    return model_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Borea LLM locally (generate and optional tiny fine-tuning).")
    parser.add_argument("--model", type=str, default="HODACHI/Borea-Phi-3.5-mini-Instruct-Jp", help="HF Hub のモデルID or ローカルパス")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="device_map 指定（auto推奨）")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="生成する最大トークン数")
    parser.add_argument("--temperature", type=float, default=0.7, help="生成温度")
    parser.add_argument("--top-p", type=float, default=0.9, help="nucleus samplingの確率質量")
    parser.add_argument("--seed", type=int, default=0, help="乱数シード")
    parser.add_argument("--train", action="store_true", help="最小限のデモ用学習を実行")
    parser.add_argument("--epochs", type=int, default=1, help="学習エポック数")
    parser.add_argument("--batch-size", type=int, default=1, help="デバイスごとの学習バッチサイズ")
    parser.add_argument("--output-dir", type=str, default="src/WeatherLLM/fine_tuned-model", help="学習済みモデルの保存先（再現性のため明示保存）")
    parser.add_argument("--max-length-train", type=int, default=1024, help="学習時の最大系列長")
    parser.add_argument("--out-base", type=str, default="src/WeatherLLM/outputs", help="生成/ログのベース出力ディレクトリ")
    parser.add_argument("--tag", type=str, default=None, help="出力ディレクトリ名に付与するタグ")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # 出力先の準備
    out_dir = make_output_dir(args.out_base, args.model, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    save_json(os.path.join(out_dir, "run_args.json"), vars(args))

    # 実行環境情報
    dtype = select_dtype()
    device_map = resolve_device_map(args.device)
    LOGGER.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        LOGGER.info(f"GPU: {torch.cuda.get_device_name(0)}")
    enable_tf32(LOGGER)
    LOGGER.info(f"Selected dtype: {dtype}")
    LOGGER.info(f"device_map: {device_map}")

    # DynamicCacheの暫定パッチ（必要時のみ）
    maybe_monkey_patch_dynamic_cache(LOGGER)

    # モデル/トークナイザーのロード（HF_TOKEN対応）
    LOGGER.info(f"Loading model: {args.model}")
    hf_token = get_hf_token(enable_transfer=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map=device_map,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        token=hf_token,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=hf_token)
    ensure_pad_token(tokenizer, model)

    # 生成前に簡単なテストプロンプト
    messages = build_sample_messages()
    prompt = build_borea_prompt(messages)
    save_json(os.path.join(out_dir, "prompt.json"), messages)

    # パイプラインで生成
    LOGGER.info("Running initial generation...")
    gen_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map=device_map,
    )
    generation_args = dict(
        max_new_tokens=args.max_new_tokens,
        return_full_text=False,
        temperature=args.temperature,
        do_sample=True,
        top_p=args.top_p,
    )
    out = gen_pipe(prompt, **generation_args)
    text0 = out[0]["generated_text"]
    print("=== 生成結果（初回） ===")
    print(text0)
    save_text(os.path.join(out_dir, "generation.txt"), text0)

    # （任意）最小限の学習デモ
    if args.train:
        LOGGER.info("Preparing tiny training dataset...")
        raw_ds = build_tiny_train_dataset()

        def _map_fn(batch):
            return preprocess_for_causal_lm(batch, tokenizer, max_length=args.max_length_train)

        tokenized = raw_ds.map(_map_fn, batched=True, remove_columns=raw_ds.column_names)

        # Collator で動的パディング＋ラベル作成（Causal LM）
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        os.makedirs(args.output_dir, exist_ok=True)

        # 16bit / bf16 設定
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        use_fp16 = torch.cuda.is_available() and not use_bf16

        training_args = TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            save_steps=500,
            save_total_limit=2,
            logging_steps=10,
            logging_dir=os.path.join(args.output_dir, "logs"),
            report_to="none",
            fp16=use_fp16,
            bf16=use_bf16,
            learning_rate=2e-5,
            optim="adamw_torch",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized,
            data_collator=collator,
        )

        LOGGER.info("Starting training...")
        trainer.train()

        LOGGER.info("Saving fine-tuned model...")
        trainer.save_model(args.output_dir)

        # 再ロード（念のため DynamicCache パッチも再適用）
        LOGGER.info("Reloading fine-tuned model for generation check...")
        maybe_monkey_patch_dynamic_cache(LOGGER)
        ft_model = AutoModelForCausalLM.from_pretrained(
            args.output_dir,
            device_map=device_map,
            torch_dtype=dtype,
            trust_remote_code=True,
        )

        ft_pipe = pipeline(
            "text-generation",
            model=ft_model,
            tokenizer=tokenizer,
            device_map=device_map,
        )

        new_messages = [
            {
                "role": "system",
                "content": "あなたは日本語能力が高い高度なAIです。特別な指示がない限り日本語で返答してください。",
            },
            {"role": "user", "content": "富士山の高さはどれくらいですか？"},
        ]
        new_prompt = build_borea_prompt(new_messages)
        save_json(os.path.join(out_dir, "prompt_after_train.json"), new_messages)

        out2 = ft_pipe(new_prompt, **generation_args)
        text1 = out2[0]["generated_text"]
        print("=== 生成結果（学習後） ===")
        print(text1)
        save_text(os.path.join(out_dir, "generation_after_train.txt"), text1)

    LOGGER.info(f"All done. Outputs saved under: {out_dir}")


if __name__ == "__main__":
    main()
