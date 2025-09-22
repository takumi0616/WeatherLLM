# -*- coding: utf-8 -*-
"""
Gemma 系（Gemma-2 / Gemma-3）をローカルPython環境で実行するスクリプト。
「単一コマンドでの一連処理（ダウンロード→軽い推論→LoRA微調整→推論）」を追加。

- 既に保存済みであればスキップ（モデルの再ダウンロード/再学習は行わない）
- すべてローカル保存物でオフライン推論可能
- 本スクリプトではテキストのみを対象とします（Gemma-3はマルチモーダル対応ですが、ここではテキストのみ）
- LoRA は Gemma-2 経路で対応（Gemma-3 は本スクリプトでは学習をスキップし推論のみ）

実行コマンド（カレントが src/WeatherLLM の想定）
```shell
# 1コマンドで一連の処理（GPU, 自動スキップ対応）
notify-run via-tml2 -- nohup python gemma3_4b.py --run-all --device cuda --max-new-tokens 2048 --epochs 10 --batch-size 2 --tag lora_all > gemma3_4b.log 2>&1 &
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
)

# Gemma-3 用（環境にある場合のみ使用。無ければインポート時に例外→Gemma-2経路にフォールバック）
try:
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration  # type: ignore
    HAS_GEMMA3 = True
except Exception:
    HAS_GEMMA3 = False

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
    # DDP helpers
    init_distributed,
    distributed_device_map,
    get_world_size,
    is_main_process,
    barrier,
    maybe_wrap_data_parallel,
    unwrap_model,
)

LOGGER = setup_logger("gemma")


def is_gemma3_model_id(model_id: str) -> bool:
    return "gemma-3" in model_id.lower()


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
    """
    Gemma-2用のメッセージを構築。
    systemロールをサポートしていないため、userロールのみを使用。
    システムの指示はユーザープロンプトに統合。
    """
    # 方法1: シンプルにuserロールのみ使用
    return [
        {"role": "user", "content": prompt},
    ]
    
    # 方法2: システムの指示をユーザープロンプトに統合する場合
    # system_instruction = "あなたは親切なAIアシスタントです。日本語で丁寧に回答してください。"
    # combined_prompt = f"{system_instruction}\n\n{prompt}"
    # return [
    #     {"role": "user", "content": combined_prompt},
    # ]


def build_gemma3_chat_for_text(prompt: str) -> List[Dict[str, Any]]:
    """
    Gemma-3用のメッセージを構築。
    Gemma-3もsystemロールに対応していない可能性があるため、userロールのみ使用。
    """
    # systemロールを削除し、userロールのみを使用
    return [
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]


def build_tiny_train_dataset() -> "Dataset":
    from datasets import Dataset
    train_data = [
        {"instruction": "日本の首都はどこですか？", "response": "日本の首都は東京です。"},
        {"instruction": "富士山の高さはどれくらいですか？", "response": "富士山の高さは3776メートルです。"},
    ]
    return Dataset.from_list(train_data)


def preprocess_for_causal_lm(examples, tokenizer, max_length: int = 1024):
    """Causal LM用に入力を連結しトークナイズ。学習時のラベルは DataCollator に任せる（LoRA学習の簡易デモ用）"""
    inputs = []
    for instruction, response in zip(examples["instruction"], examples["response"]):
        prompt_text = f"ユーザー: {instruction}\nアシスタント: "
        target = response
        full_text = prompt_text + target
        inputs.append(full_text)
    model_inputs = tokenizer(inputs, max_length=max_length, truncation=True)
    return model_inputs


# ---------- ローカル保存/検出ヘルパ ----------

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
            dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            token=hf_token,
        )
        tmp_tok = AutoTokenizer.from_pretrained(model_id, token=hf_token, use_fast=True)
        tmp_model.save_pretrained(local_dir)
        tmp_tok.save_pretrained(local_dir)
        del tmp_model


# ---------- ロード/生成ヘルパ ----------

def load_gemma2_from_local(local_dir: str, device_map: Any, dtype: torch.dtype, hf_token: Optional[str]):
    tokenizer = AutoTokenizer.from_pretrained(local_dir, token=hf_token, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        local_dir,
        device_map=device_map,
        dtype=dtype,
        trust_remote_code=True,
        token=hf_token,
        low_cpu_mem_usage=True,
    ).eval()
    return model, tokenizer


def load_gemma3_from_local(local_dir: str, device_map: Any, dtype: torch.dtype, hf_token: Optional[str]):
    if not HAS_GEMMA3:
        raise RuntimeError("Gemma3ForConditionalGeneration/AutoProcessor が利用できません。transformers が対応版か確認してください。")
    processor = AutoProcessor.from_pretrained(local_dir, token=hf_token, trust_remote_code=True)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        local_dir,
        device_map=device_map,
        dtype=dtype,
        trust_remote_code=True,
        token=hf_token,
        low_cpu_mem_usage=True,
    ).eval()
    return model, processor


def run_gemma2_generate(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float, top_p: float, top_k: int) -> str:
    messages = build_messages_for_text(prompt)
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
    return tokenizer.decode(response_ids, skip_special_tokens=True)


def run_gemma3_generate(model, processor, prompt: str, dtype: torch.dtype, max_new_tokens: int, temperature: float, top_p: float, top_k: int) -> str:
    chat = build_gemma3_chat_for_text(prompt)
    inputs = processor.apply_chat_template(
        chat,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device, dtype=dtype if v.dtype.is_floating_point else None) for k, v in inputs.items()}
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    with torch.inference_mode():
        outputs = model.generate(**inputs, **gen_kwargs)
    input_len = inputs["input_ids"].shape[-1]
    response_ids = outputs[0][input_len:]
    return processor.decode(response_ids, skip_special_tokens=True)


def train_lora_and_save_gemma2(
    base_model,
    tokenizer,
    train_max_length: int,
    epochs: int,
    per_device_bs: int,
    ft_output_dir: str,
) -> None:
    if not HAS_PEFT:
        raise RuntimeError("peft が見つかりません。LoRA に必要です。conda env に peft をインストールしてください。")

    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
    from datasets import Dataset

    LOGGER.info("[LoRA] building tiny dataset...")
    raw_ds = build_tiny_train_dataset()

    def _map_fn(batch):
        return preprocess_for_causal_lm(batch, tokenizer, max_length=train_max_length)

    tokenized = raw_ds.map(_map_fn, batched=True, remove_columns=raw_ds.column_names)

    # Gemma-2（LLaMA系類似）の一般的な target_modules
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
    # python 単発実行でも複数GPUがある場合は DataParallel で自動並列
    peft_model = maybe_wrap_data_parallel(peft_model)

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
        ddp_find_unused_parameters=(get_world_size() > 1),
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
    unwrap_model(peft_model).save_pretrained(ft_output_dir)
    try:
        tokenizer.save_pretrained(ft_output_dir)
    except Exception:
        pass


# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Gemma-2 / Gemma-3 locally (text only, with optional LoRA on Gemma-2).")
    # 既定は Gemma-2 日本語ITモデル
    parser.add_argument("--model", type=str, default="google/gemma-2-2b-jpn-it", help="HF Hub のモデルID or ローカルパス")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="device_map 指定")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bfloat16", "float16", "float32"], help="内部dtype")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="生成最大トークン数")
    parser.add_argument("--temperature", type=float, default=0.7, help="生成温度")
    parser.add_argument("--top-p", type=float, default=0.95, help="nucleus samplingの確率質量")
    parser.add_argument("--top-k", type=int, default=50, help="top-k サンプリング")
    parser.add_argument("--seed", type=int, default=0, help="乱数シード")
    parser.add_argument("--prompt", type=str, default="あなたの好きな食べ物は何ですか？", help="ユーザプロンプト（テキストのみ）")

    # 一連処理を 1 コマンドで実施
    parser.add_argument("--run-all", action="store_true", help="(推奨) ダウンロード→軽い推論→(Gemma-2のみ)LoRA学習→推論 を一括実行。既存保存物があればスキップ。")

    # ローカル保存先（省略時は model ID から自動決定）
    parser.add_argument("--local-model-dir", type=str, default=None, help="ベースモデルの保存先ディレクトリ（未存在ならダウンロード）")
    parser.add_argument("--ft-output-dir", type=str, default=None, help="LoRA 学習済みアダプタの保存先（未存在なら学習）")

    # LoRA 学習設定（Gemma-2 のみ）
    parser.add_argument("--epochs", type=int, default=1, help="学習エポック数")
    parser.add_argument("--batch-size", type=int, default=1, help="デバイスごとの学習バッチサイズ")
    parser.add_argument("--max-length-train", type=int, default=1024, help="学習時の最大系列長")

    # 出力ディレクトリ
    parser.add_argument("--out-base", type=str, default="outputs", help="生成/ログのベース出力ディレクトリ")
    parser.add_argument("--tag", type=str, default=None, help="出力ディレクトリ名に付与するタグ")
    return parser.parse_args()


def run_all_pipeline(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)

    out_dir = make_output_dir(args.out_base, args.model, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    save_json(os.path.join(out_dir, "run_args.json"), vars(args))

    dtype = resolve_torch_dtype(args.dtype)
    init_distributed(LOGGER)
    device_map = distributed_device_map(args.device)
    LOGGER.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        LOGGER.info(f"GPU: {torch.cuda.get_device_name(0)}")
    enable_tf32(LOGGER)
    LOGGER.info(f"Selected dtype: {dtype}")
    LOGGER.info(f"device_map: {device_map}")

    # 保存先決定
    hf_token = get_hf_token(enable_transfer=True)
    safe_id = safe_model_dirname(args.model)
    local_model_dir = args.local_model_dir or os.path.join("models", safe_id)
    ft_output_dir = args.ft_output_dir or os.path.join("models", f"{safe_id}-lora")

    # モデルのローカル保存を確保（DDP: rank0のみダウンロードし同期）
    if is_main_process():
        ensure_local_model_dir(args.model, local_model_dir, hf_token)
    barrier()

    if is_gemma3_model_id(args.model):
        LOGGER.info("Detected Gemma-3 model id. LoRA学習はスキップし、推論のみ実施します（テキストのみ）。")
        model, processor = load_gemma3_from_local(local_model_dir, device_map, dtype, hf_token)
        text0 = run_gemma3_generate(model, processor, args.prompt, dtype, args.max_new_tokens, args.temperature, args.top_p, args.top_k)
        print("=== 生成結果（Gemma-3 ベースモデル） ===")
        print(text0)
        save_text(os.path.join(out_dir, "generation_base.txt"), text0)
        save_json(os.path.join(out_dir, "chat_base.json"), build_gemma3_chat_for_text(args.prompt))
        LOGGER.info(f"All done. Outputs saved under: {out_dir}")
        LOGGER.info(f"[local base model] {local_model_dir}")
        return

    # Gemma-2: ベースモデル推論
    LOGGER.info(f"[phase] initial generation with local base model: {local_model_dir}")
    model2, tok2 = load_gemma2_from_local(local_model_dir, device_map, dtype, hf_token)
    text_base = run_gemma2_generate(model2, tok2, args.prompt, args.max_new_tokens, args.temperature, args.top_p, args.top_k)
    print("=== 生成結果（Gemma-2 ベースモデル） ===")
    print(text_base)
    save_text(os.path.join(out_dir, "generation_base.txt"), text_base)
    save_json(os.path.join(out_dir, "messages_base.json"), build_messages_for_text(args.prompt))
    del model2  # 学習のため解放

    # LoRA 学習（存在すればスキップ）
    if has_lora_adapter(ft_output_dir):
        LOGGER.info(f"[LoRA] adapter already exists: {ft_output_dir} (skip training)")
    else:
        LOGGER.info(f"[LoRA] training adapter to: {ft_output_dir}")
        train_model, train_tok = load_gemma2_from_local(local_model_dir, device_map, dtype, hf_token)
        train_lora_and_save_gemma2(
            base_model=train_model,
            tokenizer=train_tok,
            train_max_length=args.max_length_train,
            epochs=args.epochs,
            per_device_bs=args.batch_size,
            ft_output_dir=ft_output_dir,
        )
        del train_model

    # LoRA 適用推論
    LOGGER.info(f"[phase] generation with LoRA-adapted model: {ft_output_dir}")
    base_model3, tok3 = load_gemma2_from_local(local_model_dir, device_map, dtype, hf_token)
    try:
        ft_model = PeftModel.from_pretrained(base_model3, ft_output_dir)
    except Exception as e:
        raise RuntimeError(f"LoRA アダプタのロードに失敗しました: {e}")

    text_lora = run_gemma2_generate(ft_model, tok3, args.prompt, args.max_new_tokens, args.temperature, args.top_p, args.top_k)
    print("=== 生成結果（Gemma-2 LoRA 学習後） ===")
    print(text_lora)
    save_text(os.path.join(out_dir, "generation_lora.txt"), text_lora)
    save_json(os.path.join(out_dir, "messages_lora.json"), build_messages_for_text(args.prompt))

    LOGGER.info(f"All done. Outputs saved under: {out_dir}")
    LOGGER.info(f"[local base model] {local_model_dir}")
    LOGGER.info(f"[lora adapter] {ft_output_dir}")


def main():
    args = parse_args()

    if args.run_all:
        run_all_pipeline(args)
        return

    # 従来の単発実行（互換維持）
    torch.manual_seed(args.seed)

    out_dir = make_output_dir(args.out_base, args.model, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    save_json(os.path.join(out_dir, "run_args.json"), vars(args))

    dtype = resolve_torch_dtype(args.dtype)
    init_distributed(LOGGER)
    device_map = distributed_device_map(args.device)
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
            LOGGER.info("Detected Gemma-3 model id. Switching to Gemma-3 pipeline (text only).")
            # Gemma-3
            if not HAS_GEMMA3:
                raise RuntimeError("Gemma3ForConditionalGeneration/AutoProcessor が利用できません。transformers が対応版か確認してください。")
            processor = AutoProcessor.from_pretrained(args.model, token=hf_token, trust_remote_code=True)
            model = Gemma3ForConditionalGeneration.from_pretrained(
                args.model,
                device_map=device_map,
                dtype=dtype,
                trust_remote_code=True,
                token=hf_token,
                low_cpu_mem_usage=True,
            ).eval()
            text = run_gemma3_generate(model, processor, args.prompt, dtype, args.max_new_tokens, args.temperature, args.top_p, args.top_k)
        else:
            # Gemma-2
            tokenizer = AutoTokenizer.from_pretrained(args.model, token=hf_token, use_fast=True)
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                device_map=device_map,
                dtype=dtype,
                trust_remote_code=True,
                token=hf_token,
                low_cpu_mem_usage=True,
            ).eval()
            text = run_gemma2_generate(model, tokenizer, args.prompt, args.max_new_tokens, args.temperature, args.top_p, args.top_k)
    except Exception as e:
        LOGGER.error(f"Generation failed: {e}")
        raise

    print("=== Gemma Output ===")
    print(text)
    LOGGER.info(f"All done. Outputs saved under: {out_dir}")


if __name__ == "__main__":
    main()
