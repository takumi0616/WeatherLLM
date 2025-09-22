# -*- coding: utf-8 -*-
"""
OpenAI gpt-oss 系 (20B / 120B) をローカル Python 環境で実行するスクリプト。
「単一コマンドでの一連処理（ダウンロード→軽い推論→(20Bのみ)QLoRA微調整→推論）」に対応。

- 既に保存済みであればスキップ（モデルの再ダウンロード/再学習は行わない）
- すべてローカル保存物でオフライン推論可能
- 20B は省メモリ学習のため QLoRA（4bit）に対応（任意）
- 120B は本スクリプトでは推論のみ（学習はスキップ）

実行例（カレントが src/WeatherLLM の想定）
```shell
# 20B: ダウンロード→軽い推論→QLoRA→推論（1コマンド）
notify-run via-tml2 -- nohup python gpt_oss.py --run-all --model openai/gpt-oss-20b --device cuda --max-new-tokens 512 --epochs 10 --batch-size 2 --tag lora_all > gptoss_20b.log 2>&1 &

# 20B: メモリ厳しいときは推論も4bit量子化
notify-run via-tml2 -- nohup python gpt_oss.py --run-all --model openai/gpt-oss-20b --device cuda --use-4bit-inference --max-new-tokens 256 --epochs 10 --batch-size 1 --tag lora_all > gptoss_20b.log 2>&1 &

# 120B: ダウンロード→軽い推論（学習はスキップ）
notify-run via-tml2 -- nohup python gpt_oss.py --run-all --model openai/gpt-oss-120b --device cuda --max-new-tokens 256 --tag gptoss_120b > gptoss_120b.log 2>&1 &
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
    AutoConfig,
)

# 量子化/学習
try:
    from transformers import BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling
    HAS_TRAIN = True
except Exception:
    HAS_TRAIN = False

try:
    from datasets import Dataset
    HAS_DATASETS = True
except Exception:
    HAS_DATASETS = False

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
    ensure_pad_token,
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

LOGGER = setup_logger("gptoss")


def is_120b(model_id: str) -> bool:
    return "120b" in model_id.lower()


def is_20b(model_id: str) -> bool:
    return "20b" in model_id.lower()


def default_messages(prompt: str) -> List[Dict[str, Any]]:
    """
    gpt-oss は Harmony 応答形式で学習されており、Transformers の chat_template を使うのが推奨。
    tokenizer.apply_chat_template が存在すれば try_apply_chat_template で自動適用される。
    """
    if not prompt:
        prompt = "量子力学を高校生にも分かるように、やさしく簡潔に説明してください。"
    # system ロールも使えるが、モデルカードでは system に "Reasoning: high" 等を入れてもよいと記載あり。
    return [
        {"role": "system", "content": "You are a helpful Japanese AI assistant. Reasoning: medium. Please reply in Japanese."},
        {"role": "user", "content": prompt},
    ]


def build_tiny_train_dataset() -> "Dataset":
    if not HAS_DATASETS:
        raise RuntimeError("datasets が見つかりません。Conda 環境に datasets をインストールしてください。")
    train_data = [
        {"instruction": "日本の首都はどこですか？", "response": "日本の首都は東京です。"},
        {"instruction": "富士山の高さはどれくらいですか？", "response": "富士山の高さは3776メートルです。"},
    ]
    return Dataset.from_list(train_data)


def preprocess_for_causal_lm(examples, tokenizer, max_length: int = 1024):
    """
    学習用の極小データプリプロセス。
    chat_template を用いず、簡易に "ユーザー: ...\\nアシスタント: ..." を作る（デモ目的）。
    実務では tokenizer.apply_chat_template を用いた整形式が推奨。
    """
    inputs = []
    for instruction, response in zip(examples["instruction"], examples["response"]):
        full_text = f"ユーザー: {instruction}\nアシスタント: {response}"
        inputs.append(full_text)
    return tokenizer(inputs, max_length=max_length, truncation=True)


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
    snapshot_download があればスナップショット取得、無ければ from_pretrained→save_pretrained。
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
def has_mxfp4_quant(local_dir: str, hf_token: Optional[str]) -> bool:
    """
    モデル設定に MXFP4 量子化（Mxfp4Config）が埋め込まれているかを検出。
    """
    try:
        cfg = AutoConfig.from_pretrained(local_dir, token=hf_token, trust_remote_code=True)
        qc = getattr(cfg, "quantization_config", None)
        if qc is None:
            return False
        if isinstance(qc, dict):
            method = str(qc.get("quant_method") or qc.get("quant_type") or qc.get("quantizer_type") or "").lower()
            if "mxfp4" in method:
                return True
            qclass = str(qc.get("quant_class") or "").lower()
            return "mxfp4" in qclass
        else:
            # オブジェクト: クラス名で判定
            return "mxfp4" in qc.__class__.__name__.lower()
    except Exception:
        return False

def load_model_and_tokenizer_from_local(
    local_dir: str,
    device_map: Any,
    dtype: torch.dtype,
    hf_token: Optional[str],
    for_training: bool = False,
    use_qlora: bool = True,
    use_4bit_inference: bool = False,
) -> Tuple[Any, Any]:
    """
    モデルとトークナイザーをロード。
    - 学習時: 既定で QLoRA(4bit)
    - 推論時: --use-4bit-inference 指定で 4bit 量子化も可能（メモリ不足対策）
    """
    quant_cfg = None
    torch_dtype_arg = dtype

    # MXFP4 量子化モデルの場合、BitsAndBytes の 4bit 指定は競合するため無視する
    mxfp4_detected = has_mxfp4_quant(local_dir, hf_token)
    if mxfp4_detected and ((for_training and use_qlora) or use_4bit_inference):
        LOGGER.warning("[WARNING] MXFP4 量子化モデルを検出。BitsAndBytes の 4bit 量子化指定は無視します（競合回避）。")
    elif (for_training and use_qlora) or use_4bit_inference:
        if "BitsAndBytesConfig" not in globals():
            raise RuntimeError("bitsandbytes/transformers の 4bit 量子化が利用できません。Conda 環境を確認してください。")
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if dtype == torch.bfloat16 else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        torch_dtype_arg = None  # 量子化設定と dtype は併用しない

        if use_4bit_inference:
            LOGGER.info("[INFO] Using 4-bit quantization for inference to save memory")

    tokenizer = AutoTokenizer.from_pretrained(local_dir, token=hf_token, use_fast=True)
    load_kwargs = dict(
        device_map=device_map,
        dtype=torch_dtype_arg,
        attn_implementation="eager",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        token=hf_token,
    )
    if quant_cfg is not None:
        load_kwargs["quantization_config"] = quant_cfg

    model = AutoModelForCausalLM.from_pretrained(
        local_dir,
        **load_kwargs,
    )
    try:
        tokenizer.padding_side = "right"
    except Exception:
        pass
    try:
        ensure_pad_token(tokenizer, model)
    except Exception:
        pass
    return model, tokenizer


def run_single_generation(
    model: Any,
    tokenizer: Any,
    prompt_messages: List[Dict[str, Any]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> str:
    """
    chat_template があれば try_apply_chat_template 経由で Harmony 形式が自動適用される。
    """
    input_ids = try_apply_chat_template(tokenizer, prompt_messages, add_generation_prompt=True)
    input_ids = input_ids.to(model.device)
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    with torch.no_grad():
        attention_mask = torch.ones_like(input_ids)
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)
    response_ids = outputs[0][input_ids.size(1):]
    text = tokenizer.decode(response_ids, skip_special_tokens=True)
    return text


def train_lora_and_save_20b(
    base_model: Any,
    tokenizer: Any,
    train_max_length: int,
    epochs: int,
    per_device_bs: int,
    ft_output_dir: str,
) -> None:
    if not HAS_PEFT:
        raise RuntimeError("peft が見つかりません。LoRA/QLoRA に必要です。conda env を確認してください。")
    if not HAS_TRAIN or not HAS_DATASETS:
        raise RuntimeError("transformers[training]/datasets が見つかりません。Conda 環境に導入してください。")

    LOGGER.info("[LoRA/QLoRA] building tiny dataset...")
    raw_ds = build_tiny_train_dataset()

    def _map_fn(batch):
        return preprocess_for_causal_lm(batch, tokenizer, max_length=train_max_length)

    tokenized = raw_ds.map(_map_fn, batched=True, remove_columns=raw_ds.column_names)

    # LLaMA/Mixtral 等で一般的な target_modules（大抵のアーキテクチャで通る）
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    LOGGER.info("[LoRA/QLoRA] wrapping base model...")
    peft_model = get_peft_model(base_model, lora_cfg)
    # torchrun を使わず python 単発実行でも複数GPUがある場合は DataParallel で並列化
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

    LOGGER.info("[LoRA/QLoRA] training start...")
    trainer.train()

    LOGGER.info("[LoRA/QLoRA] saving adapter...")
    unwrap_model(peft_model).save_pretrained(ft_output_dir)
    try:
        tokenizer.save_pretrained(ft_output_dir)
    except Exception:
        pass


# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OpenAI gpt-oss (20B/120B) locally (20B optional QLoRA).")
    parser.add_argument("--model", type=str, default="openai/gpt-oss-20b", help="HF Hub のモデルID or ローカルパス")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="device_map 指定")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="生成最大トークン数")
    parser.add_argument("--temperature", type=float, default=0.6, help="生成温度")
    parser.add_argument("--top-p", type=float, default=0.9, help="nucleus sampling の確率質量")
    parser.add_argument("--top-k", type=int, default=50, help="top-k サンプリング")
    parser.add_argument("--seed", type=int, default=0, help="乱数シード")
    parser.add_argument("--prompt", type=str, default="", help="ユーザプロンプト（messagesに反映）")

    # 一連処理を 1 コマンドで実施
    parser.add_argument("--run-all", action="store_true", help="(推奨) ダウンロード→軽い推論→(20Bのみ)QLoRA学習→推論 を一括実行。既存保存物があればスキップ。")

    # メモリ不足対策（推論も4bitで）
    parser.add_argument("--use-4bit-inference", action="store_true", help="推論時にも4bit量子化を使用（GPU メモリ不足対策）")

    # ローカル保存先（省略時は model ID から自動決定）
    parser.add_argument("--local-model-dir", type=str, default=None, help="ベースモデルの保存先ディレクトリ（未存在ならダウンロード）")
    parser.add_argument("--ft-output-dir", type=str, default=None, help="LoRA 学習済みアダプタの保存先（未存在なら学習）")

    # LoRA 学習設定（20B のみ）
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

    dtype = select_dtype()
    init_distributed(LOGGER)
    device_map = distributed_device_map(args.device)
    LOGGER.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        LOGGER.info(f"GPU: {torch.cuda.get_device_name(0)}")
        # GPU メモリ情報を表示（目安）
        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            LOGGER.info(f"GPU Total Memory: {total_memory:.2f} GiB")
            if total_memory < 24 and not args.use_4bit_inference:
                LOGGER.warning(f"Limited GPU memory detected ({total_memory:.2f} GiB). Consider using --use-4bit-inference.")
        except Exception:
            pass
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

    # ベースモデル推論
    LOGGER.info(f"[phase] initial generation with local base model: {local_model_dir}")
    base_model, base_tokenizer = load_model_and_tokenizer_from_local(
        local_model_dir, device_map, dtype, hf_token,
        for_training=False,
        use_4bit_inference=args.use_4bit_inference
    )
    messages = default_messages(args.prompt)
    save_json(os.path.join(out_dir, "messages_base.json"), messages)
    text0 = run_single_generation(
        model=base_model,
        tokenizer=base_tokenizer,
        prompt_messages=messages,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    save_text(os.path.join(out_dir, "generation_base.txt"), text0)
    print("=== 生成結果（ベースモデル） ===")
    print(text0)

    # 120B は学習スキップ
    if is_120b(args.model):
        LOGGER.info("Detected 120B model. Skipping LoRA/QLoRA training and finishing after base inference.")
        LOGGER.info(f"All done. Outputs saved under: {out_dir}")
        LOGGER.info(f"[local base model] {local_model_dir}")
        return

    # 20B: LoRA/QLoRA 学習（存在すればスキップ）
    if not is_20b(args.model):
        LOGGER.warning("Model id does not include '20b' nor '120b'. Proceeding without training.")
        LOGGER.info(f"All done. Outputs saved under: {out_dir}")
        LOGGER.info(f"[local base model] {local_model_dir}")
        return

    # MXFP4 量子化モデルでは BitsAndBytes の QLoRA と競合するため学習をスキップ
    if has_mxfp4_quant(local_model_dir, hf_token):
        LOGGER.info("[LoRA/QLoRA] MXFP4 量子化モデルを検出。本スクリプトでは LoRA 学習をスキップし、推論のみ実施します。")
        LOGGER.info(f"All done. Outputs saved under: {out_dir}")
        LOGGER.info(f"[local base model] {local_model_dir}")
        return

    del base_model  # 学習のため解放
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if has_lora_adapter(ft_output_dir):
        LOGGER.info(f"[LoRA/QLoRA] adapter already exists: {ft_output_dir} (skip training)")
    else:
        LOGGER.info(f"[LoRA/QLoRA] training adapter to: {ft_output_dir}")
        train_model, train_tokenizer = load_model_and_tokenizer_from_local(
            local_model_dir, device_map, dtype, hf_token, for_training=True, use_qlora=True
        )
        train_lora_and_save_20b(
            base_model=train_model,
            tokenizer=train_tokenizer,
            train_max_length=args.max_length_train,
            epochs=args.epochs,
            per_device_bs=args.batch_size,
            ft_output_dir=ft_output_dir,
        )
        del train_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # LoRA 適用推論
    LOGGER.info(f"[phase] generation with LoRA-adapted model: {ft_output_dir}")
    base_model2, tok2 = load_model_and_tokenizer_from_local(
        local_model_dir, device_map, dtype, hf_token,
        for_training=False,
        use_4bit_inference=args.use_4bit_inference
    )
    try:
        ft_model = PeftModel.from_pretrained(base_model2, ft_output_dir)
    except Exception as e:
        raise RuntimeError(f"LoRA アダプタのロードに失敗しました: {e}")

    messages2 = [
        {"role": "system", "content": "日本語で丁寧に回答してください。Reasoning: medium."},
        {"role": "user", "content": "富士山の高さは？"},
    ]
    save_json(os.path.join(out_dir, "messages_lora.json"), messages2)
    text1 = run_single_generation(
        model=ft_model,
        tokenizer=tok2,
        prompt_messages=messages2,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    save_text(os.path.join(out_dir, "generation_lora.txt"), text1)
    print("=== 生成結果（LoRA/QLoRA 学習後） ===")
    print(text1)

    LOGGER.info(f"All done. Outputs saved under: {out_dir}")
    LOGGER.info(f"[local base model] {local_model_dir}")
    LOGGER.info(f"[lora adapter] {ft_output_dir}")


def main():
    args = parse_args()

    if args.run_all:
        run_all_pipeline(args)
        return

    # 単発モード（互換維持）
    torch.manual_seed(args.seed)
    out_dir = make_output_dir(args.out_base, args.model, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    save_json(os.path.join(out_dir, "run_args.json"), vars(args))

    dtype = select_dtype()
    init_distributed(LOGGER)
    device_map = distributed_device_map(args.device)
    LOGGER.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        LOGGER.info(f"GPU: {torch.cuda.get_device_name(0)}")
    enable_tf32(LOGGER)
    LOGGER.info(f"Selected dtype: {dtype}")
    LOGGER.info(f"device_map: {device_map}")

    hf_token = get_hf_token(enable_transfer=True)

    model, tokenizer = load_model_and_tokenizer_from_local(
        args.model, device_map, dtype, hf_token,
        for_training=False,
        use_4bit_inference=args.use_4bit_inference
    )
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
        attention_mask = torch.ones_like(input_ids)
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)
    response_ids = outputs[0][input_ids.size(1):]
    text = tokenizer.decode(response_ids, skip_special_tokens=True)
    print("=== gpt-oss Output ===")
    print(text)

    save_json(os.path.join(out_dir, "messages.json"), messages)
    save_text(os.path.join(out_dir, "generation.txt"), text)
    LOGGER.info(f"All done. Outputs saved under: {out_dir}")


if __name__ == "__main__":
    main()
