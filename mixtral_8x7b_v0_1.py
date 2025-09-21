# -*- coding: utf-8 -*-
"""
Mixtral-8x7B をローカルPython環境で実行するスクリプト。
「単一コマンドでの一連処理（ダウンロード→軽い推論→LoRA/QLoRA微調整→推論）」を追加。

- 既に保存済みであればスキップ（モデルの再ダウンロード/再学習は行わない）
- すべてローカル保存物でオフライン推論可能
- 既定で 4bit 量子化（bitsandbytes）を利用（推論・学習とも省メモリ）

実行コマンド（カレントが src/WeatherLLM の想定）
```shell
# 1コマンドで一連の処理（GPU, 4bit QLoRA, 自動スキップ対応）
notify-run wsl-ubuntu -- nohup python mixtral_8x7b_v0_1.py --run-all --device cuda --max-new-tokens 64 --epochs 1 --batch-size 1 --tag lora_all > mixtral_8x7b.log 2>&1 &
```
"""

import os
import sys
import argparse
from typing import List, Dict, Any, Optional, Tuple

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

from datasets import Dataset

# 追加: LoRA / Hub
try:
    from peft import LoraConfig, get_peft_model, PeftModel
    HAS_PEFT = True
except Exception:
    HAS_PEFT = False

try:
    from huggingface_hub import snapshot_download
    HAS_HF_HUB = True
except Exception:
    HAS_HF_HUB = False

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
    safe_model_dirname,
)

LOGGER = setup_logger("mixtral")


def default_messages(prompt: str) -> List[Dict[str, str]]:
    if prompt:
        return [{"role": "user", "content": prompt}]
    # デフォルトの英語プロンプト
    return [{"role": "user", "content": "Who is the cutest in Madoka Magica?"}]


def build_tiny_train_dataset() -> Dataset:
    """最小限のサンプルで学習用データセット（デモ目的）"""
    train_data = [
        {"instruction": "日本の首都はどこですか？", "response": "日本の首都は東京です。"},
        {"instruction": "富士山の高さはどれくらいですか？", "response": "富士山の高さは3776メートルです。"},
    ]
    return Dataset.from_list(train_data)


def preprocess_for_causal_lm(examples, tokenizer, max_length: int = 1024):
    """Causal LM 用のシンプルなテキスト結合（LoRA学習対象）"""
    inputs = []
    for instruction, response in zip(examples["instruction"], examples["response"]):
        prompt_text = f"User: {instruction}\nAssistant: "
        target = response
        full_text = prompt_text + target
        inputs.append(full_text)
    model_inputs = tokenizer(inputs, max_length=max_length, truncation=True)
    return model_inputs


def resolve_torch_dtype(arg: str) -> torch.dtype:
    if arg == "auto":
        return select_dtype()
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping[arg]


def has_base_model(local_dir: str) -> bool:
    if not os.path.isdir(local_dir):
        return False
    files = os.listdir(local_dir)
    has_cfg = "config.json" in files
    has_weight = any(fname.endswith((".safetensors", ".bin")) for fname in files)
    return has_cfg and has_weight


def has_lora_adapter(local_dir: str) -> bool:
    if not os.path.isdir(local_dir):
        return False
    files = os.listdir(local_dir)
    return ("adapter_config.json" in files) and any(
        fname.startswith("adapter_model") and fname.endswith(".safetensors") for fname in files
    )


def ensure_local_model_dir(model_id: str, local_dir: str, hf_token: Optional[str]) -> None:
    """
    モデルIDからローカルディレクトリへ保存（存在すればスキップ）。
    snapshot_download があればそれでスナップショット取得、無ければ from_pretrained して save_pretrained。
    """
    if has_base_model(local_dir):
        LOGGER.info(f"[local model] already exists: {local_dir} (skip download)")
        return
    os.makedirs(local_dir, exist_ok=True)
    LOGGER.info(f"[local model] downloading to: {local_dir}")
    if HAS_HF_HUB:
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            token=hf_token,
            local_dir_use_symlinks=False,
        )
    else:
        tmp_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cpu",
            torch_dtype=torch.float32,
            trust_remote_code=False,
            low_cpu_mem_usage=True,
            token=hf_token,
        )
        tmp_tok = AutoTokenizer.from_pretrained(model_id, token=hf_token, use_fast=True)
        tmp_model.save_pretrained(local_dir)
        tmp_tok.save_pretrained(local_dir)
        del tmp_model


def load_model_and_tokenizer_from_local(
    local_dir: str,
    device_map: Any,
    dtype: torch.dtype,
    hf_token: Optional[str],
    use_4bit: bool = True,
    for_training: bool = False,
) -> Tuple[Any, Any]:
    """
    推論/学習用にローカルモデルをロード。既定で 4bit を使用（QLoRA/省メモリ）。
    """
    quant_cfg = None
    torch_dtype_arg = dtype
    if use_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if dtype == torch.bfloat16 else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        torch_dtype_arg = None  # quantization_config と併用時は dtype=None

    model = AutoModelForCausalLM.from_pretrained(
        local_dir,
        device_map=device_map,
        torch_dtype=torch_dtype_arg,
        quantization_config=quant_cfg,
        trust_remote_code=False,
        token=hf_token,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(local_dir, token=hf_token, use_fast=True)
    return model, tokenizer


def run_single_generation(
    model: Any,
    tokenizer: Any,
    device_map: Any,
    messages: List[Dict[str, Any]],
    out_dir: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> str:
    input_ids = try_apply_chat_template(tokenizer, messages, add_generation_prompt=True)
    input_ids = input_ids.to(model.device)
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    with torch.no_grad():
        outputs = model.generate(input_ids, **gen_kwargs)
    response_ids = outputs[0][input_ids.size(1):]
    text = tokenizer.decode(response_ids, skip_special_tokens=True)
    return text


def train_lora_and_save(
    base_model: Any,
    tokenizer: Any,
    train_max_length: int,
    epochs: int,
    per_device_bs: int,
    ft_output_dir: str,
) -> None:
    if not HAS_PEFT:
        raise RuntimeError("peft が見つかりません。LoRA に必要です。conda env に peft をインストールしてください。")

    LOGGER.info("[LoRA] building tiny dataset...")
    raw_ds = build_tiny_train_dataset()

    def _map_fn(batch):
        return preprocess_for_causal_lm(batch, tokenizer, max_length=train_max_length)

    tokenized = raw_ds.map(_map_fn, batched=True, remove_columns=raw_ds.column_names)

    # Mistral/Mixtral の一般的な target_modules
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    LOGGER.info("[LoRA] wrapping base model...")
    peft_model = get_peft_model(base_model, lora_cfg)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    os.makedirs(ft_output_dir, exist_ok=True)

    # 16bit / bf16 設定（GPU 前提）
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    training_args = TrainingArguments(
        output_dir=ft_output_dir,
        per_device_train_batch_size=per_device_bs,
        num_train_epochs=epochs,
        save_steps=500,
        save_total_limit=2,
        logging_steps=10,
        logging_dir=os.path.join(ft_output_dir, "logs"),
        report_to="none",
        fp16=use_fp16,
        bf16=use_bf16,
        learning_rate=2e-5,
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
    )

    LOGGER.info("[LoRA] training start...")
    trainer.train()

    LOGGER.info("[LoRA] saving adapter...")
    peft_model.save_pretrained(ft_output_dir)
    try:
        tokenizer.save_pretrained(ft_output_dir)
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Mixtral-8x7B locally (quantized inference / QLoRA fine-tuning).")
    parser.add_argument("--model", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1", help="HF Hub のモデルID or ローカルパス")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="device_map 指定")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bfloat16", "float16", "float32"], help="内部dtype")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="生成最大トークン数")
    parser.add_argument("--temperature", type=float, default=0.5, help="生成温度")
    parser.add_argument("--top-p", type=float, default=0.95, help="nucleus samplingの確率質量")
    parser.add_argument("--top-k", type=int, default=40, help="top-k サンプリング")
    parser.add_argument("--seed", type=int, default=0, help="乱数シード")
    parser.add_argument("--prompt", type=str, default="", help="ユーザプロンプト（messagesに反映）")

    # 一連処理を 1 コマンドで実施
    parser.add_argument("--run-all", action="store_true", help="(推奨) ダウンロード→軽い推論→QLoRA学習→推論 を一括実行。既存保存物があればスキップ。")

    # ローカル保存先（省略時は model ID から自動決定）
    parser.add_argument("--local-model-dir", type=str, default=None, help="ベースモデルの保存先ディレクトリ（未存在ならダウンロード）")
    parser.add_argument("--ft-output-dir", type=str, default=None, help="LoRA 学習済みアダプタの保存先（未存在なら学習）")

    # LoRA 学習設定
    parser.add_argument("--epochs", type=int, default=1, help="学習エポック数")
    parser.add_argument("--batch-size", type=int, default=1, help="デバイスごとの学習バッチサイズ")
    parser.add_argument("--max-length-train", type=int, default=1024, help="学習時の最大系列長")

    # 出力ディレクトリ
    parser.add_argument("--out-base", type=str, default="outputs", help="生成/ログのベース出力ディレクトリ")
    parser.add_argument("--tag", type=str, default=None, help="出力ディレクトリ名に付与するタグ")
    # 量子化オプション（従来互換）
    parser.add_argument("--no-4bit", action="store_true", help="4bit量子化を無効化（既定は有効）")
    return parser.parse_args()


def run_all_pipeline(args: argparse.Namespace) -> None:
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

    # 保存先決定
    safe_id = safe_model_dirname(args.model)
    local_model_dir = args.local_model_dir or os.path.join("models", safe_id)
    ft_output_dir = args.ft_output_dir or os.path.join("models", f"{safe_id}-lora")

    # モデルのローカル保存を確保
    hf_token = get_hf_token(enable_transfer=True)
    ensure_local_model_dir(args.model, local_model_dir, hf_token)

    # ベースモデル推論
    LOGGER.info(f"[phase] initial generation with local base model: {local_model_dir}")
    base_model, base_tokenizer = load_model_and_tokenizer_from_local(
        local_model_dir, device_map, dtype, hf_token, use_4bit=not args.no_4bit, for_training=False
    )
    messages = default_messages(args.prompt)
    save_json(os.path.join(out_dir, "messages_base.json"), messages)
    text0 = run_single_generation(
        model=base_model,
        tokenizer=base_tokenizer,
        device_map=device_map,
        messages=messages,
        out_dir=out_dir,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    save_text(os.path.join(out_dir, "generation_base.txt"), text0)
    print("=== 生成結果（ベースモデル） ===")
    print(text0)
    del base_model  # 学習のため解放

    # LoRA 学習（存在すればスキップ）
    if has_lora_adapter(ft_output_dir):
        LOGGER.info(f"[LoRA] adapter already exists: {ft_output_dir} (skip training)")
    else:
        LOGGER.info(f"[LoRA] training adapter to: {ft_output_dir}")
        train_model, train_tokenizer = load_model_and_tokenizer_from_local(
            local_model_dir, device_map, dtype, hf_token, use_4bit=True, for_training=True
        )
        train_lora_and_save(
            base_model=train_model,
            tokenizer=train_tokenizer,
            train_max_length=args.max_length_train,
            epochs=args.epochs,
            per_device_bs=args.batch_size,
            ft_output_dir=ft_output_dir,
        )
        del train_model

    # LoRA 適用推論
    LOGGER.info(f"[phase] generation with LoRA-adapted model: {ft_output_dir}")
    base_model2, tok2 = load_model_and_tokenizer_from_local(
        local_model_dir, device_map, dtype, hf_token, use_4bit=not args.no_4bit, for_training=False
    )
    try:
        ft_model = PeftModel.from_pretrained(base_model2, ft_output_dir)
    except Exception as e:
        raise RuntimeError(f"LoRA アダプタのロードに失敗: {e}")

    messages2 = [{"role": "user", "content": "What is the height of Mt. Fuji in meters?"}]
    save_json(os.path.join(out_dir, "messages_lora.json"), messages2)
    text1 = run_single_generation(
        model=ft_model,
        tokenizer=tok2,
        device_map=device_map,
        messages=messages2,
        out_dir=out_dir,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    save_text(os.path.join(out_dir, "generation_lora.txt"), text1)
    print("=== 生成結果（LoRA 学習後） ===")
    print(text1)

    LOGGER.info(f"All done. Outputs saved under: {out_dir}")
    LOGGER.info(f"[local base model] {local_model_dir}")
    LOGGER.info(f"[lora adapter] {ft_output_dir}")


def main():
    args = parse_args()

    if args.run_all:
        run_all_pipeline(args)
        return

    # 従来の単発推論（互換維持）
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

    response_ids = outputs[0][input_ids.size(1):]
    text = tokenizer.decode(response_ids, skip_special_tokens=True)

    print("=== Mixtral Output ===")
    print(text)

    save_json(os.path.join(out_dir, "messages.json"), messages)
    save_text(os.path.join(out_dir, "generation.txt"), text)
    LOGGER.info(f"All done. Outputs saved under: {out_dir}")


if __name__ == "__main__":
    main()
