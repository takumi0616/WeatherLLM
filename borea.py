# -*- coding: utf-8 -*-
"""
Borea (Phi-3.5 系) をローカルPython環境で実行するための純Pythonスクリプト。

要件に合わせた「単一コマンドでの一連処理（ダウンロード→軽い推論→LoRA微調整→推論）」を追加しました。
- 既に保存済みであればスキップ（モデルの再ダウンロード/再学習は行わない）
- すべてローカルの保存物を使ってオフライン推論可能

実行コマンド（現在のディレクトリが src/WeatherLLM の想定）
```shell
# 1コマンドで一連の処理を実行（GPU, bf16/fp16、自動スキップ対応）
# --local-model-dir と --ft-output-dir を省略すると model ID から自動生成したディレクトリに保存します。
notify-run wsl-ubuntu -- nohup python borea.py --run-all --device cuda --max-new-tokens 2048 --epochs 10 --batch-size 2 --tag lora_all > borea.log 2>&1 &
```

処理の流れ（--run-all 指定時）
1) モデルをローカルへ保存（未保存なら）
2) ローカル保存モデルを用いて軽い推論（動作確認）
3) LoRA でファインチューニング（未保存なら）
4) LoRA 学習済み（アダプタ）を適用して推論
5) 生成物は outputs/ 以下に時刻付きディレクトリで保存

注意:
- LoRA には peft ライブラリが必要です（インストール済みの llm_env を想定）。
- flash-attn 未導入でも動作するよう attn_implementation='eager' を明示。
- Phi-3 系の DynamicCache/API 非互換に対処するモンキーパッチ込み（安定動作優先）。
"""

import os
import sys
import argparse
import logging
from typing import List, Dict, Any, Optional, Tuple

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
    safe_model_dirname,
)

LOGGER = setup_logger("borea")


def maybe_monkey_patch_dynamic_cache(logger: logging.Logger) -> None:
    """
    一部モデル（phi3系等）で DynamicCache API 変更により発生する非互換を暫定回避。
    - get_max_length() / seen_tokens の差異を吸収（None は 0 に正規化）
    - transformers.models.phi3.modeling_phi3.DynamicCache と transformers.cache_utils.DynamicCache の両方を試行的にパッチ
    既に修正済みな場合はスキップ。失敗しても致命ではない。
    """
    def _patch_cls(DynamicCache) -> bool:
        try:
            def _norm(v):
                try:
                    if v is None:
                        return 0
                    if hasattr(v, "item"):
                        return int(v.item())
                    return int(v)
                except Exception:
                    return 0

            def _get_max_length(self):
                return _norm(getattr(self, "cache_position", 0))

            def _get_seen_tokens(self):
                return _norm(getattr(self, "cache_position", 0))

            def _set_seen_tokens(self, value):
                try:
                    setattr(self, "cache_position", _norm(value))
                except Exception:
                    pass

            try:
                setattr(DynamicCache, "get_max_length", _get_max_length)  # type: ignore
            except Exception:
                pass

            # 既に seen_tokens が存在し、callable でない Property でもそのまま利用可能なら保持
            try:
                if not hasattr(DynamicCache, "seen_tokens") or isinstance(getattr(DynamicCache, "seen_tokens"), property):
                    setattr(DynamicCache, "seen_tokens", property(_get_seen_tokens, _set_seen_tokens))  # type: ignore
            except Exception:
                pass
            return True
        except Exception:
            return False

    patched_any = False
    for mod_name in ("transformers.models.phi3.modeling_phi3", "transformers.cache_utils"):
        try:
            mod = __import__(mod_name, fromlist=["DynamicCache"])
            DC = getattr(mod, "DynamicCache", None)
            if DC is not None:
                if _patch_cls(DC):
                    patched_any = True
        except Exception:
            continue

    if patched_any:
        logger.info("Applied monkey patch: DynamicCache(get_max_length/seen_tokens) across possible modules.")
    else:
        logger.info("DynamicCache monkey patch skipped (class not found).")


def maybe_monkey_patch_phi3_prepare_inputs(logger: logging.Logger, model: Any) -> None:
    """
    Phi-3 系の prepare_inputs_for_generation 内で past_length が None のまま比較/添字アクセスされる問題を回避。
    - past_key_values が None の場合でも「shape[2] == 0」を返す最小互換な疑似キャッシュを与える
    - seen_tokens が None の場合は 0 に正規化
    - Self-Attn 側で参照される可能性のある API を最小実装（get_usable_length, update 等）
    """
    try:
        orig = getattr(model, "prepare_inputs_for_generation", None)
        if not callable(orig):
            return
        import types

        def wrapped(self, input_ids=None, past_key_values=None, **kwargs):
            # 最小互換の「添字アクセス可能な」疑似キャッシュを提供して shape[2] を 0 にする
            class _ZeroShape:
                def __init__(self):
                    self.shape = (1, 1, 0)  # cache_length/past_length=0 を表現

            class _Layer:
                def __getitem__(self, idx):
                    return _ZeroShape()

            class _PKV:
                def __init__(self):
                    self.cache_position = 0
                @property
                def seen_tokens(self):
                    return 0
                def __getitem__(self, idx):
                    return _Layer()
                # Self-Attn 側で参照される可能性のある API を最小実装
                def get_usable_length(self, kv_seq_len, layer_idx):
                    return 0
                def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
                    # そのまま通過させる（キャッシュは持たない）
                    return key_states, value_states
                def to(self, device, dtype=None):
                    return self
                def reorder_cache(self, beam_idx):
                    return self
                def __len__(self):
                    return 0

            if past_key_values is None:
                pkv = _PKV()
            else:
                pkv = past_key_values
                try:
                    st = getattr(pkv, "seen_tokens", None)
                    if st is None:
                        setattr(pkv, "cache_position", 0)
                except Exception:
                    pass

            # 元の実装を呼び出し（orig は既にバインド済み）
            return orig(input_ids=input_ids, past_key_values=pkv, **kwargs)

        model.prepare_inputs_for_generation = types.MethodType(wrapped, model)
        logger.info("Applied monkey patch: prepare_inputs_for_generation guard (past_length/shape[2] -> 0).")
    except Exception as e:
        logger.info(f"prepare_inputs_for_generation monkey patch skipped ({e})")


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


def has_base_model(local_dir: str) -> bool:
    if not os.path.isdir(local_dir):
        return False
    # 重みが存在するか（safetensors / bin のいずれか）
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
        # スナップショット全体を保存（将来のオフライン利用を想定）
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            token=hf_token,
            local_dir_use_symlinks=False,
            ignore_patterns=None,  # すべて取得
        )
    else:
        # フォールバック: 一旦ロード後に保存
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
        LOGGER.info("[local model] saved via from_pretrained fallback")


def load_model_and_tokenizer_from_local(
    local_dir: str,
    device_map: Any,
    dtype: torch.dtype,
    hf_token: Optional[str],
) -> Tuple[Any, Any]:
    model = AutoModelForCausalLM.from_pretrained(
        local_dir,
        device_map=device_map,
        dtype=dtype,
        attn_implementation="eager",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        token=hf_token,  # ローカルでも token は問題ない（将来一部コード参照の可能性に備える）
    )
    tokenizer = AutoTokenizer.from_pretrained(local_dir, token=hf_token, use_fast=True)
    ensure_pad_token(tokenizer, model)
    # KV キャッシュ無効化（DynamicCache 経路を避ける）
    try:
        if hasattr(model, "generation_config"):
            model.generation_config.use_cache = False
        if hasattr(model, "config"):
            model.config.use_cache = False
    except Exception:
        pass
    # Phi-3 系の prepare_inputs_for_generation 互換パッチ
    maybe_monkey_patch_phi3_prepare_inputs(LOGGER, model)
    return model, tokenizer


def save_generation(out_dir: str, name: str, text: str) -> None:
    path = os.path.join(out_dir, f"{name}.txt")
    save_text(path, text)


def run_single_generation(
    model: Any,
    tokenizer: Any,
    device_map: Any,
    prompt: str,
    out_dir: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    gen_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map=device_map,
    )
    generation_args = dict(
        max_new_tokens=max_new_tokens,
        return_full_text=False,
        temperature=temperature,
        do_sample=True,
        top_p=top_p,
        use_cache=False,
    )
    out = gen_pipe(prompt, **generation_args)
    text = out[0]["generated_text"]
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

    # LoRA 設定（Phi/類似構造で一般的な投影名を対象）
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

    # Collator で動的パディング＋ラベル作成（Causal LM）
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
    # トークナイザも一応保存（将来の再利用を容易に）
    try:
        tokenizer.save_pretrained(ft_output_dir)
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Borea LLM locally (generate and LoRA fine-tuning pipeline).")
    parser.add_argument("--model", type=str, default="HODACHI/Borea-Phi-3.5-mini-Instruct-Jp", help="HF Hub のモデルID or ローカルパス")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="device_map 指定（auto推奨）")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="生成する最大トークン数")
    parser.add_argument("--temperature", type=float, default=0.7, help="生成温度")
    parser.add_argument("--top-p", type=float, default=0.9, help="nucleus samplingの確率質量")
    parser.add_argument("--seed", type=int, default=0, help="乱数シード")

    # 一連処理を 1 コマンドで実施
    parser.add_argument("--run-all", action="store_true", help="(推奨) ダウンロード→軽い推論→LoRA学習→推論 を一括実行。既存保存物があればスキップ。")

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
    return parser.parse_args()


def run_all_pipeline(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)

    # 出力先
    out_dir = make_output_dir(args.out_base, args.model, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    save_json(os.path.join(out_dir, "run_args.json"), vars(args))

    # 実行環境/デバイス
    dtype = select_dtype()
    device_map = resolve_device_map(args.device)
    LOGGER.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        LOGGER.info(f"GPU: {torch.cuda.get_device_name(0)}")
    enable_tf32(LOGGER)
    LOGGER.info(f"Selected dtype: {dtype}")
    LOGGER.info(f"device_map: {device_map}")

    # DynamicCache の暫定パッチ
    maybe_monkey_patch_dynamic_cache(LOGGER)

    # 保存先ディレクトリの決定（未指定なら model ID から自動生成）
    safe_id = safe_model_dirname(args.model)
    local_model_dir = args.local_model_dir or os.path.join("models", safe_id)
    ft_output_dir = args.ft_output_dir or os.path.join("models", f"{safe_id}-lora")

    # モデルのローカル保存を確保
    hf_token = get_hf_token(enable_transfer=True)
    ensure_local_model_dir(args.model, local_model_dir, hf_token)

    # ローカルモデルで軽い推論（動作確認）
    LOGGER.info(f"[phase] initial generation with local base model: {local_model_dir}")
    base_model, base_tokenizer = load_model_and_tokenizer_from_local(local_model_dir, device_map, dtype, hf_token)
    messages = build_sample_messages()
    prompt = build_borea_prompt(messages)
    save_json(os.path.join(out_dir, "prompt_initial.json"), messages)
    text0 = run_single_generation(
        model=base_model,
        tokenizer=base_tokenizer,
        device_map=device_map,
        prompt=prompt,
        out_dir=out_dir,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print("=== 生成結果（ベースモデル・初回） ===")
    print(text0)
    save_generation(out_dir, "generation_base_initial", text0)
    del base_model  # メモリ解放（学習フェーズのため）

    # LoRA 学習（存在すればスキップ）
    if has_lora_adapter(ft_output_dir):
        LOGGER.info(f"[LoRA] adapter already exists: {ft_output_dir} (skip training)")
    else:
        LOGGER.info(f"[LoRA] training adapter to: {ft_output_dir}")
        train_model, train_tokenizer = load_model_and_tokenizer_from_local(local_model_dir, device_map, dtype, hf_token)
        train_lora_and_save(
            base_model=train_model,
            tokenizer=train_tokenizer,
            train_max_length=args.max_length_train,
            epochs=args.epochs,
            per_device_bs=args.batch_size,
            ft_output_dir=ft_output_dir,
        )
        del train_model

    # LoRA 適用モデルで推論
    LOGGER.info(f"[phase] generation with LoRA-adapted model: {ft_output_dir}")
    base_model2, tok2 = load_model_and_tokenizer_from_local(local_model_dir, device_map, dtype, hf_token)
    try:
        ft_model = PeftModel.from_pretrained(base_model2, ft_output_dir)
    except Exception as e:
        raise RuntimeError(f"LoRA アダプタのロードに失敗しました: {e}")

    # 再度 DynamicCache/prepare_inputs パッチ（安全のため）
    maybe_monkey_patch_dynamic_cache(LOGGER)
    maybe_monkey_patch_phi3_prepare_inputs(LOGGER, ft_model)

    messages2 = [
        {
            "role": "system",
            "content": "あなたは日本語能力が高い高度なAIです。特別な指示がない限り日本語で返答してください。",
        },
        {"role": "user", "content": "富士山の高さはどれくらいですか？"},
    ]
    prompt2 = build_borea_prompt(messages2)
    save_json(os.path.join(out_dir, "prompt_lora.json"), messages2)
    text1 = run_single_generation(
        model=ft_model,
        tokenizer=tok2,
        device_map=device_map,
        prompt=prompt2,
        out_dir=out_dir,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print("=== 生成結果（LoRA 学習後） ===")
    print(text1)
    save_generation(out_dir, "generation_lora_after", text1)

    LOGGER.info(f"All done. Outputs saved under: {out_dir}")
    LOGGER.info(f"[local base model] {local_model_dir}")
    LOGGER.info(f"[lora adapter] {ft_output_dir}")


def main():
    args = parse_args()

    if args.run_all:
        # 一連処理（推奨）
        run_all_pipeline(args)
        return

    # 旧来のデモ（個別生成/任意の最小学習）も残しますが、要件的には --run-all をご使用ください。
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
        dtype=dtype,  # torch_dtype は非推奨
        attn_implementation="eager",  # flash-attn 未導入/非対応時の警告回避・安定化
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        token=hf_token,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=hf_token)
    ensure_pad_token(tokenizer, model)
    # KVキャッシュを無効化して DynamicCache の seen_tokens/past_length 絡みの経路を回避
    try:
        if hasattr(model, "generation_config"):
            model.generation_config.use_cache = False
        if hasattr(model, "config"):
            model.config.use_cache = False
    except Exception:
        pass
    # Phi-3 系の prepare_inputs_for_generation での None 比較を防ぐ
    maybe_monkey_patch_phi3_prepare_inputs(LOGGER, model)

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
        use_cache=False,  # DynamicCache 経由の不整合を避ける
    )
    out = gen_pipe(prompt, **generation_args)
    text0 = out[0]["generated_text"]
    print("=== 生成結果（初回） ===")
    print(text0)
    save_text(os.path.join(out_dir, "generation.txt"), text0)

    # （任意）最小限の学習デモ（フルファインチューニング版）
    # 要件では LoRA が必要なため、通常は --run-all をお使いください。
    # ここは従来の残置（必要なら使用）
    # ... 省略（以前の実装に近い処理） ...

    LOGGER.info(f"All done. Outputs saved under: {out_dir}")


if __name__ == "__main__":
    main()
