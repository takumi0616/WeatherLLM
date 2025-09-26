# -*- coding: utf-8 -*-
"""
LLaVA 系（視覚と言語のマルチモーダル）をローカル Python 環境で実行するスクリプト。
「単一コマンドでの一連処理（ダウンロード→軽い推論→(現状)学習スキップ→推論）」に対応。

- 既に保存済みであればスキップ（モデルの再ダウンロードは行わない）
- すべてローカル保存物でオフライン推論可能
- 本スクリプトではテキスト+画像の推論のみを実装（LoRA 学習は対象外：LLaVA の訓練は特殊でデータ/投影器が必要）
- 推論メモリ削減のため 4bit 量子化による推論オプション（--use-4bit-inference）を提供（環境に bitsandbytes が必要）

参考資料（同梱 PDF より）:
- starriver030515/LLaVA (Hugging Face モデルカードの抜粋)
- LLaVA 公式: https://github.com/haotian-liu/LLaVA
- Hugging Face 互換重み: llava-hf/llava-1.5-7b-hf, llava-hf/llava-1.5-13b-hf

実行例（カレントが src/WeatherLLM の想定）
```shell
# 画像URLを指定して 1コマンド実行（GPU, 自動スキップ対応）
notify-run via-tml2 -- nohup python llava.py --run-all --device cuda \
  --model llava-hf/llava-1.5-7b-hf \
  --image "https://llava-vl.github.io/static/images/view.jpg" \
  --prompt "この画像の内容を日本語で詳しく説明してください" \
  --max-new-tokens 256 --tag vqa_demo > llava.log 2>&1 &

# GPU メモリが厳しい場合（推論も4bit量子化）
notify-run via-tml2 -- nohup python llava.py --run-all --device cuda \
  --model llava-hf/llava-1.5-7b-hf --use-4bit-inference \
  --image "./document/image.png" \
  --prompt "この画像の内容を日本語で詳しく説明してください" \
  --max-new-tokens 256 --tag vqa_4bit > llava.log 2>&1 &
```
"""

import os
import io
import sys
import argparse
from typing import List, Dict, Any, Optional, Tuple

import torch
from PIL import Image
import requests

from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    AutoTokenizer,  # decode で使用する場合あり（processor からも可）
    LogitsProcessorList,
    LogitsProcessor,
)

# 4bit 量子化（任意）
try:
    from transformers import BitsAndBytesConfig
    HAS_BNB = True
except Exception:
    HAS_BNB = False

# Hub スナップショット
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
)

class SafeLogitsProcessor(LogitsProcessor):
    """
    生成時の数値不安定対策:
    - logits の NaN/Inf を有限値へ置き換え
    - 値の暴走を抑えるためクリッピング
    """
    def __call__(self, input_ids, scores):
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores = torch.nan_to_num(scores, nan=0.0, posinf=50.0, neginf=-50.0)
        scores = torch.clamp(scores, min=-50.0, max=50.0)
        return scores

LOGGER = setup_logger("llava")


# ---------- ローカル保存/検出ヘルパ ----------

def has_base_model(local_dir: str) -> bool:
    if not os.path.isdir(local_dir):
        return False
    files = os.listdir(local_dir)
    has_cfg = "config.json" in files
    # LLaVA は vision projector / text tokenizer 等も含むが、ここでは重みファイルの存在で概ね判定
    has_weight = any(fname.endswith((".safetensors", ".bin")) for fname in files)
    return has_cfg and has_weight


def ensure_local_model_dir(model_id: str, local_dir: str, hf_token: Optional[str]) -> None:
    """
    モデルIDからローカルディレクトリへ保存（存在すればスキップ）。
    snapshot_download があればスナップショット取得。
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
        # AutoProcessor/LlavaForConditionalGeneration の from_pretrained はフォールバックとしては重いので、
        # ここでは hub snapshot を推奨。環境に応じて追加実装可。
        raise RuntimeError("huggingface_hub が利用できません。llm_env での実行を推奨します。")


# ---------- 画像ユーティリティ ----------

def is_url(path_or_url: str) -> bool:
    return path_or_url.startswith("http://") or path_or_url.startswith("https://")


def load_image(path_or_url: str) -> Image.Image:
    if is_url(path_or_url):
        resp = requests.get(path_or_url, timeout=30)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    else:
        return Image.open(path_or_url).convert("RGB")


# ---------- ロード/生成ヘルパ ----------

def load_llava_from_local(
    local_dir: str,
    device_map: Any,
    dtype: torch.dtype,
    hf_token: Optional[str],
    use_4bit_inference: bool = False,
):
    """
    LLaVA モデルとプロセッサをローカルからロード。
    - use_4bit_inference: 推論も4bit量子化を使用（メモリ削減）。bitsandbytes が必要。
    """
    processor = AutoProcessor.from_pretrained(local_dir, token=hf_token, trust_remote_code=True)

    quant_cfg = None
    torch_dtype_arg = dtype
    if use_4bit_inference:
        if not HAS_BNB:
            raise RuntimeError("bitsandbytes/transformers の 4bit 量子化が利用できません。Conda 環境を確認してください。")
        LOGGER.info("[INFO] Using 4-bit quantization for inference to save memory")
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if dtype == torch.bfloat16 else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        torch_dtype_arg = None

    # LlavaForConditionalGeneration は vision tower + LLM を内包
    model = LlavaForConditionalGeneration.from_pretrained(
        local_dir,
        device_map=device_map,
        torch_dtype=torch_dtype_arg,
        quantization_config=quant_cfg,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        token=hf_token,
    ).eval()

    return model, processor


def run_llava_generate(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    dtype: torch.dtype,
) -> str:
    """
    テキスト+画像入力で生成。LLaVA-HF の chat_template を優先的に使用し、
    画像プレースホルダ（image tokens）と視覚特徴量の不一致を回避する。
    """
    # プロンプト整形（空なら既定）
    user_text = prompt.strip() if isinstance(prompt, str) and len(prompt.strip()) > 0 else "この画像について日本語で詳しく説明してください。"

    # chat_template があればそれを使って <image> トークンを正しく挿入
    prompt_text = None
    apply_fn = getattr(processor, "apply_chat_template", None)
    if callable(apply_fn):
        chat = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text},
                ],
            }
        ]
        try:
            prompt_text = processor.apply_chat_template(chat, add_generation_prompt=True)
        except Exception:
            # フォールバック: 明示的に <image> を入れる
            prompt_text = f"USER: <image>\n{user_text}\nASSISTANT:"
    else:
        # 旧バージョンフォールバック
        prompt_text = f"USER: <image>\n{user_text}\nASSISTANT:"

    # 画像トークンの事前検証（万一 chat_template が挿入に失敗した場合の保険）
    try:
        tok = getattr(processor, "tokenizer", None)
        if tok is not None:
            image_token_id = tok.convert_tokens_to_ids("<image>")
            if isinstance(image_token_id, int) and image_token_id != tok.unk_token_id:
                prompt_ids = tok(prompt_text, return_tensors="pt").input_ids
                image_tok_count = int((prompt_ids == image_token_id).sum().item())
                if image_tok_count == 0:
                    # 明示的に <image> プレースホルダを付与したフォーマットに置き換え
                    prompt_text = f"USER: <image>\n{user_text}\nASSISTANT:"
    except Exception:
        pass

    # processor は入力テンソルをまとめて返す（pixel_values は浮動小数、input_ids は整数）
    inputs = processor(text=prompt_text, images=[image], return_tensors="pt")

    # デバイス転送（整数テンソルの dtype を変換しないよう注意）
    device = next(model.parameters()).device
    for k, v in inputs.items():
        if torch.is_floating_point(v):
            inputs[k] = v.to(device, dtype=dtype)
        else:
            inputs[k] = v.to(device)

    # VRAM削減: 生成時のKVキャッシュを無効化（速度は低下）
    try:
        model.generation_config.use_cache = False
        model.config.use_cache = False
    except Exception:
        pass

    # 安定化: pad/eos の未設定による即時終了を防止
    try:
        tok = getattr(processor, "tokenizer", None)
        if tok is not None:
            pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
            if pad_id is not None:
                model.generation_config.pad_token_id = pad_id
            if tok.eos_token_id is not None:
                model.generation_config.eos_token_id = tok.eos_token_id
    except Exception:
        pass

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        min_new_tokens=20,  # 空出力回避のため最低生成長を確保
        do_sample=False,  # multinomial回避で数値不安定対策（貪欲生成）
        use_cache=False,
    )
    # do_sample=False のため temperature/top_p/top_k は渡さない（Transformersの無効フラグ警告を抑止）
    logits_processor = LogitsProcessorList([SafeLogitsProcessor()])
    with torch.inference_mode():
        outputs = model.generate(**inputs, logits_processor=logits_processor, **gen_kwargs)

    # 生成テキスト抽出（入力部を除去して応答のみを decode）
    generated_ids = outputs[0]
    input_len = None
    try:
        input_len = inputs["input_ids"].shape[-1]
    except Exception:
        input_len = None

    if input_len is not None and generated_ids.shape[0] >= input_len:
        response_ids = generated_ids[input_len:]
    else:
        response_ids = generated_ids

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        text = tokenizer.decode(response_ids, skip_special_tokens=True)
    else:
        # 極力 tokenizer を使うが、無い場合は processor.decode で代替
        text = processor.decode(response_ids, skip_special_tokens=True)

    # 生成テキストが空の場合は一度だけサンプリングで再試行（空レスポンス対策）
    if not (text or "").strip():
        try:
            retry_kwargs = dict(
                max_new_tokens=max(32, min(128, max_new_tokens)),
                min_new_tokens=max(16, min(64, max_new_tokens)),
                do_sample=True,
                temperature=max(0.7, temperature),
                top_p=0.95,
                top_k=50,
                use_cache=False,
                no_repeat_ngram_size=3,
            )
            with torch.inference_mode():
                outputs2 = model.generate(**inputs, logits_processor=logits_processor, **retry_kwargs)
            gen2 = outputs2[0]
            if input_len is not None and gen2.shape[0] >= input_len:
                resp2 = gen2[input_len:]
            else:
                resp2 = gen2
            if tokenizer is not None:
                text = tokenizer.decode(resp2, skip_special_tokens=True)
            else:
                text = processor.decode(resp2, skip_special_tokens=True)
        except Exception:
            pass

    return text


# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLaVA (vision-language) locally (inference only).")
    parser.add_argument("--model", type=str, default="llava-hf/llava-1.5-7b-hf", help="HF Hub のモデルID or ローカルパス（LLaVA-HF推奨）")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="device_map 指定")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="生成最大トークン数")
    parser.add_argument("--temperature", type=float, default=0.7, help="生成温度")
    parser.add_argument("--top-p", type=float, default=0.95, help="nucleus sampling の確率質量")
    parser.add_argument("--top-k", type=int, default=50, help="top-k サンプリング")
    parser.add_argument("--seed", type=int, default=0, help="乱数シード")
    parser.add_argument("--prompt", type=str, default="", help="ユーザプロンプト（マルチモーダル。例: 'USER: <image> ...\\nASSISTANT:' 形式でも可）")
    parser.add_argument("--image", type=str, default="https://llava-vl.github.io/static/images/view.jpg", help="入力画像のパスまたはURL")

    # 一連処理を 1 コマンドで実施
    parser.add_argument("--run-all", action="store_true", help="(推奨) ダウンロード→軽い推論（画像+テキスト）を一括実行。既存保存物があればスキップ。")

    # メモリ不足対策
    parser.add_argument("--use-4bit-inference", action="store_true", help="推論時にも4bit量子化を使用（GPU メモリ不足対策）")

    # ローカル保存先（省略時は model ID から自動決定）
    parser.add_argument("--local-model-dir", type=str, default=None, help="ベースモデルの保存先ディレクトリ（未存在ならダウンロード）")

    # 出力ディレクトリ
    parser.add_argument("--out-base", type=str, default="outputs", help="生成/ログのベース出力ディレクトリ")
    parser.add_argument("--tag", type=str, default=None, help="出力ディレクトリ名に付与するタグ")
    return parser.parse_args()


def run_all_pipeline(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)

    out_dir = make_output_dir(args.out_base, args.model, args.tag or "run-all")
    os.makedirs(out_dir, exist_ok=True)
    save_json(os.path.join(out_dir, "run_args.json"), vars(args))

    dtype = select_dtype()
    init_distributed(LOGGER)
    device_map = distributed_device_map(args.device)
    LOGGER.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        try:
            LOGGER.info(f"GPU: {torch.cuda.get_device_name(0)}")
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            LOGGER.info(f"GPU Total Memory: {total_memory:.2f} GiB")
        except Exception:
            pass
    enable_tf32(LOGGER)
    LOGGER.info(f"Selected dtype: {dtype}")
    LOGGER.info(f"device_map: {device_map}")

    # 保存先決定
    hf_token = get_hf_token(enable_transfer=True)
    safe_id = safe_model_dirname(args.model)
    local_model_dir = args.local_model_dir or os.path.join("models", safe_id)

    # モデルのローカル保存を確保（DDP: rank0のみダウンロードし同期）
    if is_main_process():
        ensure_local_model_dir(args.model, local_model_dir, hf_token)
    barrier()

    # 画像取得
    try:
        image = load_image(args.image)
    except Exception as e:
        raise RuntimeError(f"画像の読み込みに失敗しました: {e}")

    # ベースモデル推論（LLaVA）
    LOGGER.info(f"[phase] initial generation with local base LLaVA model: {local_model_dir}")
    model, processor = load_llava_from_local(
        local_dir=local_model_dir,
        device_map=device_map,
        dtype=dtype,
        hf_token=hf_token,
        use_4bit_inference=args.use_4bit_inference,
    )

    # 実行
    text = run_llava_generate(
        model=model,
        processor=processor,
        image=image,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        dtype=dtype,
    )
    print("=== 生成結果（LLaVA ベースモデル） ===")
    print(text)

    # 保存（プロンプト/画像メタ/生成結果）
    meta = {
        "image": args.image,
        "prompt": args.prompt or "USER: <image>\nこの画像について日本語で詳しく説明してください。\nASSISTANT:",
    }
    save_json(os.path.join(out_dir, "messages_base.json"), meta)
    save_text(os.path.join(out_dir, "generation_base.txt"), text)

    LOGGER.info(f"All done. Outputs saved under: {out_dir}")
    LOGGER.info(f"[local base model] {local_model_dir}")
    LOGGER.info("Note: LLaVA の LoRA/QLoRA 学習は本スクリプトではスキップ（専用データと投影器が必要なため）。")


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
        try:
            LOGGER.info(f"GPU: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass
    enable_tf32(LOGGER)
    LOGGER.info(f"Selected dtype: {dtype}")
    LOGGER.info(f"device_map: {device_map}")

    # 保存先決定＆読み込み
    hf_token = get_hf_token(enable_transfer=True)
    safe_id = safe_model_dirname(args.model)
    local_model_dir = args.local_model_dir or os.path.join("models", safe_id)
    if is_main_process() and not has_base_model(local_model_dir):
        ensure_local_model_dir(args.model, local_model_dir, hf_token)
    barrier()

    # 画像取得
    image = load_image(args.image)

    model, processor = load_llava_from_local(
        local_dir=local_model_dir,
        device_map=device_map,
        dtype=dtype,
        hf_token=hf_token,
        use_4bit_inference=args.use_4bit_inference,
    )

    text = run_llava_generate(
        model=model,
        processor=processor,
        image=image,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        dtype=dtype,
    )
    print("=== LLaVA Output ===")
    print(text)

    meta = {"image": args.image, "prompt": args.prompt}
    save_json(os.path.join(out_dir, "messages.json"), meta)
    save_text(os.path.join(out_dir, "generation.txt"), text)
    LOGGER.info(f"All done. Outputs saved under: {out_dir}")
    LOGGER.info("Note: LLaVA の LoRA/QLoRA 学習は本スクリプトではスキップ（専用データと投影器が必要なため）。")


if __name__ == "__main__":
    main()
