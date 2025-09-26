# -*- coding: utf-8 -*-
"""
llm.py

WeatherLLM のモデル関連実装を集約:
- OpenAI Responses API (multimodal)
- Local LLaVA backend
- Hugging Face (hf) backend: Qwen2.5-VL-3B/7B をVRAM節約設計でサポート

main_v2.py からモデル依存の処理を切り出し、本モジュールの generate() を呼び出す。
"""

from __future__ import annotations

import os
import json
import base64
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple

# ランタイム依存は遅延 import で扱う（環境差/オプション依存のため）
try:
    import torch  # type: ignore
except Exception:
    torch = None  # CPU-onlyやOpenAIのみの場合はNoneでも良い

# 同一ディレクトリの config を利用
try:
    from . import config as CFG
except Exception:
    import importlib
    CFG = importlib.import_module("config")

# src/WeatherLLM/llava.py を import できるようにパス調整
PROGRAM_DIR = Path(__file__).resolve().parent


def _encode_image_to_base64(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _torch_dtype_from_cfg(dtype_opt: str):
    """
    CFG の文字列指定から torch.dtype を返す。未知なら 'auto' を示すため None を返す。
    """
    if torch is None:
        return None
    m = (dtype_opt or "").lower()
    if m in ("bf16", "bfloat16"):
        return getattr(torch, "bfloat16", None)
    if m in ("fp16", "float16", "half"):
        return getattr(torch, "float16", None)
    if m in ("fp32", "float32"):
        return getattr(torch, "float32", None)
    return None  # auto


def _is_cuda_available() -> bool:
    return (torch is not None) and getattr(torch, "cuda", None) and torch.cuda.is_available()


def _print_slim_json_info(info: Dict[str, object]):
    try:
        print("\nレスポンス情報(json, 抜粋):")
        print(json.dumps(info, ensure_ascii=False, indent=2))
    except Exception:
        pass


def _save_symlink_or_metadata(src_dir: Path | str, dst_dir: Path):
    """
    外部（program フォルダ外）のファイル/ディレクトリへアクセスしないため、何もしません。
    モデル情報は main_v2.py 側で model_info.txt に出力されます。
    """
    return


def _generate_openai(text: str, image_path: Path, run_dir: Path, api_key: Optional[str]) -> Tuple[str, Dict[str, object]]:
    """
    OpenAI Responses API での image+text→text 生成
    """
    base64_image = _encode_image_to_base64(image_path)
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError(f"openai ライブラリのインポートに失敗: {e}")

    client = OpenAI(api_key=api_key)
    try:
        response = client.responses.create(
            model=getattr(CFG, "OPENAI_MODEL", "gpt-4.1"),
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": text},
                        {"type": "input_image", "image_url": f"data:image/png;base64,{base64_image}"},
                    ],
                }
            ],
        )
    except Exception as e:
        print(f"OpenAI API呼び出しで例外: {e}")
        raise

    # Format response
    try:
        data = response.model_dump()
    except Exception:
        try:
            data = json.loads(response.model_dump_json())
        except Exception:
            data = {"raw": str(response)}

    slim = {k: data.get(k) for k in ("id", "model", "usage") if k in data}
    _print_slim_json_info(slim)

    result = getattr(response, "output_text", None)
    if result is None:
        try:
            result = "".join(part.get("text", "") for part in data.get("output", []) if isinstance(part, dict))
        except Exception:
            result = ""

    # メタ情報
    info = {
        "backend": "openai",
        "model": getattr(CFG, "OPENAI_MODEL", "gpt-4.1"),
        "run_dir": str(run_dir),
        "raw": slim,
    }
    return result or "", info


def _generate_llava(text: str, image_path: Path, run_dir: Path) -> Tuple[str, Dict[str, object]]:
    """
    LLaVA (HF) 推論: Transformers の AutoProcessor / AutoModelForCausalLM を用いた簡易実装。
    program フォルダ外のモジュール（src/WeatherLLM/llava.py）には依存しません。
    """
    print("\n[backend] LLaVA (HF) モデルで実行します")

    model_id = getattr(CFG, "LLAVA_LOCAL_DIR", None) or getattr(CFG, "LLAVA_MODEL_ID", "llava-hf/llava-1.5-7b-hf")
    use_4bit = bool(getattr(CFG, "LLAVA_USE_4BIT_INFERENCE", True))

    # CUDA断片化対策
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # dtype 選択
    if _is_cuda_available():
        try:
            bf16_ok = torch.cuda.is_bf16_supported()
        except Exception:
            bf16_ok = False
        dtype = torch.bfloat16 if bf16_ok else torch.float16
    else:
        dtype = torch.float32

    # 量子化(BnB)
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
    }
    if dtype is not None:
        model_kwargs["dtype"] = dtype

    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig  # type: ignore
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=(dtype if dtype is not None else None) or (getattr(torch, "bfloat16", None) if torch else None),
            )
        except Exception as e:
            print(f"[warn] bitsandbytes/4bit設定に失敗しました（非インストール/非対応）。FP系で継続します: {e}")

    # ロード
    try:
        from transformers import AutoProcessor, LlavaForConditionalGeneration  # type: ignore
    except Exception as e:
        raise RuntimeError(f"transformers が見つかりません。インストールしてください: {e}")

    # Processor
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        raise RuntimeError(f"LLaVA Processorのロードに失敗しました: {e}")

    # Model（attn_implementation はデフォルトでOK）
    try:
        model = LlavaForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
    except Exception as e:
        raise RuntimeError(f"LLaVA モデルのロードに失敗しました: {e}")

    # 入力の用意
    try:
        from PIL import Image  # type: ignore
        pil_image = Image.open(str(image_path)).convert("RGB")
    except Exception as e_img:
        raise RuntimeError(f"PILで画像を開けませんでした: {e_img}")

    # LLaVA: 可能なら chat template を使って正規の会話形式にエンコード（プロンプトの混入を防止）
    # フォールバックとして USER/ASSISTANT 形式の手書きテンプレートも用意
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text},
            ],
        }
    ]
    try:
        text_for_model = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text_for_model],
            images=[pil_image],
            padding=True,
            return_tensors="pt",
        )
    except Exception:
        prompt = f"USER: <image>\n{text}\nASSISTANT:"
        inputs = processor(images=pil_image, text=prompt, return_tensors="pt")
    # 可能ならモデルのデバイスへ移動
    try:
        inputs = inputs.to(getattr(model, "device", "cuda" if _is_cuda_available() else "cpu"))
    except Exception:
        try:
            if _is_cuda_available():
                inputs = {k: (v.cuda() if hasattr(v, "cuda") else v) for k, v in inputs.items()}
        except Exception:
            pass

    # 生成
    max_new_tokens = int(getattr(CFG, "LLAVA_MAX_NEW_TOKENS", 256))
    temperature = float(getattr(CFG, "LLAVA_TEMPERATURE", 0.7))
    top_p = float(getattr(CFG, "LLAVA_TOP_P", 0.95))
    top_k = int(getattr(CFG, "LLAVA_TOP_K", 50))

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": True if temperature and temperature > 0 else False,
    }
    try:
        generated_ids = model.generate(**inputs, **gen_kwargs)
    except Exception:
        # フォールバック
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.0, top_p=1.0, do_sample=False)

    # 入力分(input_ids)を取り除いてからデコード（プロンプト混入を防止）
    try:
        in_ids = inputs.input_ids
        generated_ids_trimmed = [out_ids[len(in_ids_i):] for in_ids_i, out_ids in zip(in_ids, generated_ids)]
    except Exception:
        generated_ids_trimmed = generated_ids

    try:
        result = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    except Exception:
        try:
            first_ids = generated_ids_trimmed[0] if isinstance(generated_ids_trimmed, (list, tuple)) else generated_ids[0]
            result = processor.decode(first_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        except Exception:
            result = ""

    # provenance
    _save_symlink_or_metadata(Path(model_id), run_dir)

    info = {
        "backend": "llava",
        "model": model_id,
        "params": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "use_4bit": use_4bit,
            "device_map": getattr(model, "hf_device_map", None) or "auto",
            "dtype": str(dtype),
        },
        "run_dir": str(run_dir),
    }
    _print_slim_json_info({"backend": "llava", "model": model_id})
    return str(result or ""), info


def _generate_hf_qwen(text: str, image_path: Path, run_dir: Path, model_id_override: Optional[str] = None, backend_label: str = "qwen") -> Tuple[str, Dict[str, object]]:
    """
    HF backend: Qwen2.5-VL-3B/7B を VRAM節約設計で実行
    - device_map='auto' により自動デバイス割当
    - 4bit量子化（bitsandbytes）が有効なら load_in_4bit
    - 画像解像度は min/max pixels で制御
    - flash_attention_2 が有効設定なら優先（失敗時は sdpa にフォールバック）
    """
    # バックエンド（qwen3b/qwen7b）に応じて個別設定を選択
    pref = "QWEN7B" if backend_label == "qwen7b" else "QWEN3B"

    def _cfg(name: str, default):
        try:
            return getattr(CFG, f"{pref}_{name}")
        except Exception:
            return default

    model_id = (model_id_override or _cfg("MODEL_ID", "Qwen/Qwen2.5-VL-3B-Instruct")) or "Qwen/Qwen2.5-VL-3B-Instruct"
    dtype_opt = _cfg("DTYPE", "auto") or "auto"
    dtype = _torch_dtype_from_cfg(dtype_opt)
    use_4bit = bool(_cfg("USE_4BIT_INFERENCE", True))
    enable_fa2 = bool(_cfg("ENABLE_FLASH_ATTN", False))

    min_pixels = _cfg("MIN_PIXELS", 256 * 28 * 28)
    max_pixels = _cfg("MAX_PIXELS", 896 * 28 * 28)

    max_new_tokens = int(_cfg("MAX_NEW_TOKENS", 256))
    temperature = float(_cfg("TEMPERATURE", 0.7))
    top_p = float(_cfg("TOP_P", 0.95))
    top_k = int(_cfg("TOP_K", 50))

    # lazy imports
    try:
        from transformers import AutoProcessor  # type: ignore
    except Exception as e:
        raise RuntimeError(f"transformers が見つかりません。インストールしてください: {e}")

    # Qwen専用クラス（最新Transformersが必要）
    QWEN_MODEL = None
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as QWEN_MODEL  # type: ignore
    except Exception:
        # 古いTransformersだとエイリアスクラス名が異なる場合があるためフォールバック
        try:
            from transformers import AutoModelForVision2Seq as QWEN_MODEL  # type: ignore
        except Exception as e2:
            raise RuntimeError("Qwen2.5-VL 用のモデルクラスが見つかりません。Transformers を最新に更新してください。") from e2

    # 量子化(BnB)
    quant_config = None
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",  # VRAM節約: 自動割当
    }
    if dtype is not None:
        model_kwargs["dtype"] = dtype

    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig  # type: ignore
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=(dtype if dtype is not None else None) or (getattr(torch, "bfloat16", None) if torch else None),
            )
            model_kwargs["quantization_config"] = quant_config
        except Exception as e:
            print(f"[warn] bitsandbytes/4bit設定に失敗しました（非インストール/非対応）。FP系で継続します: {e}")

    # flash-attention 2
    if enable_fa2:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    # ロード（flash_attnで失敗したらsdpaで再ロード）
    try:
        model = QWEN_MODEL.from_pretrained(model_id, **model_kwargs)
    except Exception as e_fa2:
        if enable_fa2:
            print(f"[warn] flash_attention_2 でのロードに失敗。sdpaで再試行します: {e_fa2}")
            model_kwargs.pop("attn_implementation", None)
            model = QWEN_MODEL.from_pretrained(model_id, **model_kwargs)
        else:
            raise

    # Processor（min/max pixels を設定してVRAM制御）
    try:
        processor = AutoProcessor.from_pretrained(
            model_id,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            trust_remote_code=True,
        )
    except TypeError:
        # 古いTransformersだと min/max_pixels を受け付けない可能性
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    # メッセージをQwenチャットテンプレートに適用
    # 画像は file:// スキームで渡す（qwen_vl_utils が無くても動くようにフォールバック）
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{str(image_path)}"},
                {"type": "text", "text": text},
            ],
        }
    ]
    try:
        # あると便利な補助（無くても動くように分岐）
        from qwen_vl_utils import process_vision_info  # type: ignore
        text_for_model = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text_for_model],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
    except Exception:
        # フォールバック: 画像はPILに読み込んで直接与える
        try:
            from PIL import Image  # type: ignore
            pil_image = Image.open(str(image_path)).convert("RGB")
        except Exception as e_img:
            raise RuntimeError(f"PILで画像を開けませんでした: {e_img}")
        text_for_model = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text_for_model],
            images=[pil_image],
            padding=True,
            return_tensors="pt",
        )

    # 入力テンソルを可能ならGPUへ（device_map=autoの場合でも to("cuda") は一般に可）
    try:
        if _is_cuda_available():
            inputs = inputs.to("cuda")
    except Exception:
        pass

    # 生成
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": True if temperature and temperature > 0 else False,
    }
    try:
        generated_ids = model.generate(**inputs, **gen_kwargs)
    except Exception as e_gen:
        # 生成で落ちる場合、温和な設定でリトライ
        gen_kwargs_fallback = {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "do_sample": False,
        }
        generated_ids = model.generate(**inputs, **gen_kwargs_fallback)

    # 入力分を取り除いてデコード
    try:
        in_ids = inputs.input_ids
        generated_ids_trimmed = [out_ids[len(in_ids_i):] for in_ids_i, out_ids in zip(in_ids, generated_ids)]
    except Exception:
        generated_ids_trimmed = generated_ids

    try:
        output_texts = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        result = output_texts[0] if output_texts else ""
    except Exception:
        # だめならデフォルトデコード
        try:
            result = processor.decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        except Exception:
            result = ""

    # provenance
    _save_symlink_or_metadata(Path(model_id), run_dir)

    # device_map/dtype などのメタ
    device_map_meta = None
    try:
        device_map_meta = getattr(model, "hf_device_map", None)
    except Exception:
        device_map_meta = None

    info = {
        "backend": backend_label,
        "family": "qwen",
        "model": model_id,
        "params": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "use_4bit": use_4bit,
            "dtype": str(dtype) if dtype is not None else "auto",
            "device_map": device_map_meta if device_map_meta is not None else "auto",
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
            "enable_flash_attn": enable_fa2,
        },
        "run_dir": str(run_dir),
    }
    _print_slim_json_info({"backend": backend_label, "family": "qwen", "model": model_id})
    return result or "", info


def _generate_hf(text: str, image_path: Path, run_dir: Path) -> Tuple[str, Dict[str, object]]:
    """
    HFバックエンドのディスパッチャ。
    今回は Qwen2.5-VL のみサポート（要求仕様）。
    """
    family = (getattr(CFG, "HF_FAMILY", None) or "qwen").lower()
    model_id = getattr(CFG, "HF_MODEL_ID", "") or ""
    # 簡易自動判定
    if "qwen2.5-vl" in model_id.lower() or family == "qwen":
        return _generate_hf_qwen(text, image_path, run_dir)
    # 将来拡張: MiniCPM / Ovis / Idefics / etc.
    raise ValueError(f"未対応のHFモデルです: family={family}, model_id={model_id}")


def generate(
    backend: str,
    text: str,
    image_path: Path,
    run_dir: Path,
    api_key: Optional[str] = None,
) -> Tuple[str, Dict[str, object]]:
    """
    backend に応じて生成を実行し、(result_text, meta_data) を返す。
    meta_data は model_info.txt 等の保存/ログ出力に利用可能な最小情報を含む。
    """
    b = (backend or "").lower()
    if b == "openai":
        return _generate_openai(text, image_path, run_dir, api_key)
    if b == "llava":
        return _generate_llava(text, image_path, run_dir)
    if b in ("qwen", "hf", "qwen3b", "qwen7b"):
        # backend 表示名（ログ/保存用）
        backend_label = "qwen" if b in ("qwen", "hf") else b
        # モデルIDの決定（明示タイプは固定ID、genericは設定から）
        if b == "qwen3b":
            mid = "Qwen/Qwen2.5-VL-3B-Instruct"
        elif b == "qwen7b":
            mid = "Qwen/Qwen2.5-VL-7B-Instruct"
        else:
            mid = None  # use CFG.HF_MODEL_ID
        return _generate_hf_qwen(text, image_path, run_dir, model_id_override=mid, backend_label=backend_label)
    raise ValueError(f"Unknown MODEL_BACKEND: {backend}")
