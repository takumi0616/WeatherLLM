#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified runner for WeatherLLM pipelines (v1, v2, v3, v4) integrated into a single script.

Essence:
- Input: weather map image (PNG) + instruction text
- Output: generated text (weather commentary)

Goals:
- Eliminate v1.py, v2.py, v3.py, v4.py by consolidating all logic here
- Support selectable generation backend via config.py:
    * OpenAI Responses API (GPT) for multimodal
    * Local LLaVA (vision-language) model in src/WeatherLLM/llava.py
- Keep behavior consistent: data paths, .env discovery, evaluation (Embedding cosine, BLEU, ROUGE-1 with fallback),
  results saving, console output

Usage examples (from src/WeatherLLM/program):
  python -u main_v1.py --pipeline v4 --date 20220106 --env-file /home/s233319/docker_miniconda/.env
  python -u main_v1.py --pipeline all --auto --limit 3 --env-file /home/s233319/docker_miniconda/.env
  python -u main_v1.py --pipeline v1 --date 20220101 --api-key sk-xxxx

Notes:
- Configure backend and parameters in program/config.py (MODEL_BACKEND="openai"|"llava" and related settings)
- Results are saved to program/results/ as vN_YYYYMMDD_result.txt and vN_YYYYMMDD_response.json
- .env discovery is robust; you can override with --env-file or --api-key if using OpenAI or OpenAI embeddings for evaluation
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import json
import base64
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
try:
    from rouge_score import rouge_scorer
except Exception:
    rouge_scorer = None
try:
    import sacrebleu
except Exception:
    sacrebleu = None

from openai import OpenAI

# Load program config
try:
    from . import config as CFG  # when imported as a module
except Exception:
    import importlib
    CFG = importlib.import_module("config")

# Line-buffer stdout for nohup visibility
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# Directories
PROGRAM_DIR = Path(__file__).resolve().parent
PNG_DIR = PROGRAM_DIR / "data" / "png"
PROMPT_DIR = PROGRAM_DIR / "data" / "prompt_gpt"
NUMERIC_DIR = PROGRAM_DIR / "data" / "Numerical_weather_data"
ORIGINAL_DIR = PROGRAM_DIR / "data" / "original_comment"
RESULTS_DIR = PROGRAM_DIR / "results"

# For importing src/WeatherLLM/llava.py when needed
WEATHERLLM_DIR = PROGRAM_DIR.parent
if str(WEATHERLLM_DIR) not in sys.path:
    sys.path.insert(0, str(WEATHERLLM_DIR))


def find_env_path() -> Path:
    # 1) Explicit path via env var (prefer OPENAI_ENV_FILE; fallback ENV_PATH)
    var = os.environ.get("OPENAI_ENV_FILE") or os.environ.get("ENV_PATH")
    if var:
        p = Path(var)
        if p.exists():
            return p

    # 2) Search upwards from program dir for a .env
    for p in [PROGRAM_DIR, *PROGRAM_DIR.parents]:
        cand = p / ".env"
        if cand.exists():
            return cand

    # 3) If compose/.docker markers exist on any parent, look for sibling .env there
    for p in [PROGRAM_DIR, *PROGRAM_DIR.parents]:
        if (p / "compose.yml").exists() or (p / "compose.gpu.yml").exists() or (p / ".docker" / "Dockerfile").exists() or (p / ".git").exists():
            cand = p / ".env"
            if cand.exists():
                return cand

    # 4) Common workspace/environment variables that may point to repo roots
    for var_name in ("WORKSPACE", "WORKSPACE_FOLDER", "GITHUB_WORKSPACE", "PROJECT_ROOT", "APP_HOME"):
        vv = os.environ.get(var_name)
        if vv:
            cand = Path(vv) / ".env"
            if cand.exists():
                return cand

    # 5) Common host mounts: /home/*/docker_miniconda/.env
    try:
        for cand in Path("/home").glob("*/docker_miniconda/.env"):
            if cand.exists():
                return cand
    except Exception:
        pass

    # 6) $HOME/docker_miniconda/.env
    home_env = Path.home() / "docker_miniconda" / ".env"
    if home_env.exists():
        return home_env

    # 7) /docker_miniconda/.env (root-level)
    root_env = Path("/docker_miniconda/.env")
    if root_env.exists():
        return root_env

    # 8) Direct /app/.env (docker-compose default workdir)
    app_env = Path("/app/.env")
    if app_env.exists():
        return app_env

    # 9) Devcontainer/VS Code Remote common mounts
    for base in (Path("/workspace"), Path("/workspaces")):
        cand = base / "docker_miniconda" / ".env"
        if cand.exists():
            return cand

    # 10) Explicit known host path (provided by user)
    explicit = Path("/home/s233319/docker_miniconda/.env")
    if explicit.exists():
        return explicit

    # 11) Final fallback: program/.env
    return PROGRAM_DIR / ".env"


def load_api_key(env_path: Path) -> str:
    """
    Load OpenAI API key from environment or .env file.
    Expects variable name: OpenAI_KEY_TOKEN
    """
    key = os.environ.get("OpenAI_KEY_TOKEN")
    if key:
        return key

    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=env_path)
        key = os.environ.get("OpenAI_KEY_TOKEN")
        if key:
            return key
    except Exception:
        pass

    if env_path.exists():
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    if k.strip() == "OpenAI_KEY_TOKEN":
                        return v.strip().strip("\"'")
        except Exception:
            pass

    raise RuntimeError(f"OpenAI API key not found. Ensure OpenAI_KEY_TOKEN is set in environment or {env_path}")


def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        raise RuntimeError(f"テキストファイルを読み込めませんでした: {path} ({e})")


def encode_image_to_base64(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def list_available_png_dates() -> List[str]:
    dates: List[str] = []
    if not PNG_DIR.exists():
        return dates
    for p in sorted(PNG_DIR.glob("*.png")):
        m = re.match(r"^(\d{8})\.png$", p.name)
        if m:
            dates.append(m.group(1))
    return dates


def parse_date_yyyymmdd(s: str) -> Tuple[int, int, int]:
    if not re.fullmatch(r"\d{8}", s):
        raise ValueError("日付はYYYYMMDD形式で指定してください (例: 20220101)")
    y = int(s[0:4])
    m = int(s[4:6])
    d = int(s[6:8])
    # Basic validation
    from datetime import datetime
    datetime(year=y, month=m, day=d)
    return y, m, d


def build_japanese_date(y: int, m: int, d: int) -> str:
    return f"{y}年{m}月{d}日"


def find_weather_data_file(y: int, m: int, d: int) -> Path:
    candidates = [
        NUMERIC_DIR / f"{y}-{m}-{d}.txt",
        NUMERIC_DIR / f"{y}-{m:02d}-{d:02d}.txt",
        NUMERIC_DIR / f"{y}-{m:02d}-{d}.txt",
        NUMERIC_DIR / f"{y}-{m}-{d:02d}.txt",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "気象数値データファイルが見つかりませんでした。試行したパス:\n" + "\n".join(str(c) for c in candidates)
    )


# Tokenization and metrics helpers

def tokenize_ja(text: str):
    try:
        from fugashi import Tagger
        tagger = Tagger()
        return [w.surface for w in tagger(text)]
    except Exception:
        try:
            from janome.tokenizer import Tokenizer
            return [t.surface for t in Tokenizer().tokenize(text)]
        except Exception:
            return list(text)  # fallback: char-level


def normalize_tokens(tokens):
    import re as _re
    cleaned = []
    for t in tokens:
        t = t.strip()
        t = _re.sub(r"[、。．，・：；！!？\?「」『』（）\(\)\[\]\{\}”“\"'`’\-_/\\…･･･~＾^＋+＝=＊*＜＞<>＆&％%＄$#＠@|]", "", t)
        t = _re.sub(r"\s+", "", t)
        t = t.lower()
        if t:
            cleaned.append(t)
    return cleaned


def calc_bleu(ref_tokens, hyp_tokens) -> float:
    if sacrebleu is None:
        return 0.0
    return float(sacrebleu.sentence_bleu(" ".join(hyp_tokens), [" ".join(ref_tokens)], tokenize="none").score)


def calc_rouge1_f1(ref_tokens, hyp_tokens) -> float:
    if rouge_scorer is not None:
        scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=False)
        scores = scorer.score(" ".join(ref_tokens), " ".join(hyp_tokens))
        return float(scores['rouge1'].fmeasure)
    # Fallback: unigram F1
    from collections import Counter
    ref_counts = Counter(ref_tokens)
    hyp_counts = Counter(hyp_tokens)
    overlap = sum((ref_counts & hyp_counts).values())
    total_ref = sum(ref_counts.values())
    total_hyp = sum(hyp_counts.values())
    if total_ref == 0 or total_hyp == 0:
        return 0.0
    precision = overlap / total_hyp
    recall = overlap / total_ref
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def char_level_f1(a: str, b: str) -> float:
    import re as _re
    from collections import Counter
    def norm(s: str) -> str:
        s = s.strip().lower()
        s = _re.sub(r"[、。．，・：；！!？\?「」『』（）\(\)\[\]\{\}”“\"'`’\-_/\\…･･･~＾^＋+＝=＊*＜＞<>＆&％%＄$#＠@|]", "", s)
        s = _re.sub(r"\s+", "", s)
        return s
    a_norm = norm(a)
    b_norm = norm(b)
    if not a_norm or not b_norm:
        return 0.0
    ca = Counter(list(a_norm))
    cb = Counter(list(b_norm))
    overlap = sum((ca & cb).values())
    if overlap == 0:
        return 0.0
    prec = overlap / sum(cb.values())
    rec = overlap / sum(ca.values())
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def cosine_similarity(vec1, vec2) -> float:
    v1 = np.array(vec1, dtype=float)
    v2 = np.array(vec2, dtype=float)
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) or 1.0
    return float(np.dot(v1, v2) / denom)


# Prompt builders

def build_text_for_v1(y: int, m: int, d: int) -> str:
    instr = read_text_file(PROMPT_DIR / "v1_instruction.txt")
    weather_date_jp = build_japanese_date(y, m, d)
    return f"入力された画像は{weather_date_jp}9時の天気図です。\n\n{instr}".strip()


def build_text_for_v2(y: int, m: int, d: int) -> str:
    instr = read_text_file(PROMPT_DIR / "v2_instruction.txt")
    weather_date_jp = build_japanese_date(y, m, d)
    return f"入力された画像は{weather_date_jp}9時の天気図です。\n\n\n\n{instr}".strip()


def build_text_for_v3(y: int, m: int, d: int) -> str:
    instr = read_text_file(PROMPT_DIR / "v3_instruction.txt")
    weather_date_jp = build_japanese_date(y, m, d)
    weather_data_path = find_weather_data_file(y, m, d)
    weather_data = read_text_file(weather_data_path)
    return f"""```
以下は{weather_date_jp}の気象データです。
{weather_data}
```

入力された画像は{weather_date_jp}9時の天気図です。

{instr}""".strip(), weather_data_path


def build_text_for_v4(y: int, m: int, d: int) -> str:
    instr = read_text_file(PROMPT_DIR / "v4_instruction.txt")
    weather_date_jp = build_japanese_date(y, m, d)
    weather_data_path = find_weather_data_file(y, m, d)
    weather_data = read_text_file(weather_data_path)
    return f"""```
以下は{weather_date_jp}の気象データです。
{weather_data}
```

入力された画像は{weather_date_jp}9時の天気図です。

{instr}""".strip(), weather_data_path


def run_one(pipeline: str, date: str, env_file: Optional[str], api_key_arg: Optional[str]) -> int:
    # Resolve date
    y, m, d = parse_date_yyyymmdd(date)
    yyyymmdd = f"{y:04d}{m:02d}{d:02d}"
    base_name = f"{pipeline}_{yyyymmdd}"

    # Determine whether OpenAI key is needed (backend=openai OR evaluation embeddings via openai)
    need_openai = (getattr(CFG, "MODEL_BACKEND", "openai") == "openai") or (
        getattr(CFG, "ENABLE_EVALUATION", True) and getattr(CFG, "EVAL_EMBEDDINGS", "openai") == "openai"
    )

    # .env and API key (CLI takes precedence over config.py; else config used)
    api_key: Optional[str] = None
    if env_file:
        os.environ["OPENAI_ENV_FILE"] = env_file
    elif getattr(CFG, "ENV_FILE", None):
        os.environ["OPENAI_ENV_FILE"] = str(CFG.ENV_FILE)

    # Accept API key from CLI or config and set both env var casings for compatibility.
    if api_key_arg:
        os.environ["OPENAI_KEY_TOKEN"] = api_key_arg
        os.environ["OpenAI_KEY_TOKEN"] = api_key_arg  # for load_api_key()
    elif getattr(CFG, "API_KEY", None):
        os.environ["OPENAI_KEY_TOKEN"] = str(CFG.API_KEY)
        os.environ["OpenAI_KEY_TOKEN"] = str(CFG.API_KEY)  # for load_api_key()

    if need_openai:
        env_path = find_env_path()
        api_key = load_api_key(env_path)

    # Prepare image
    image_path = PNG_DIR / f"{yyyymmdd}.png"
    if not image_path.exists():
        raise FileNotFoundError(f"画像ファイルが存在しません: {image_path}")

    # Build prompt text per pipeline
    weather_data_path: Optional[Path] = None
    if pipeline == "v1":
        text = build_text_for_v1(y, m, d)
    elif pipeline == "v2":
        text = build_text_for_v2(y, m, d)
    elif pipeline == "v3":
        text, weather_data_path = build_text_for_v3(y, m, d)
    elif pipeline == "v4":
        text, weather_data_path = build_text_for_v4(y, m, d)
    else:
        raise ValueError(f"Unknown pipeline: {pipeline}")

    # Ensure results dir and run folder
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = RESULTS_DIR / base_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Print request info
    print("リクエスト本文:")
    print(text)
    print(f"\n送信画像: {image_path.name}")
    if weather_data_path:
        print(f"気象データ: {weather_data_path.name}")

    backend = getattr(CFG, "MODEL_BACKEND", "openai")
    data: Dict[str, object] = {}
    result: Optional[str] = None

    if backend == "openai":
        # OpenAI Responses API (multimodal)
        base64_image = encode_image_to_base64(image_path)
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

        print("\nレスポンス情報(json, 抜粋):")
        slim = {k: data.get(k) for k in ("id", "model", "usage") if k in data}
        print(json.dumps(slim, ensure_ascii=False, indent=2))

        result = getattr(response, "output_text", None)
        if result is None:
            try:
                result = "".join(part.get("text", "") for part in data.get("output", []) if isinstance(part, dict))
            except Exception:
                result = ""
    elif backend == "llava":
        # Local LLaVA backend
        print("\n[backend] LLaVA ローカルモデルで実行します")
        try:
            import torch  # lazy import
            from llava import load_llava_from_local, run_llava_generate, load_image as llava_load_image
        except Exception as e:
            raise RuntimeError(f"LLaVA モジュールのインポートに失敗しました: {e}")

        # Tweak CUDA allocator to reduce fragmentation
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        # dtype / device_map (prefer bf16 on GPUs that support it for better numerical stability; else fp16)
        if torch.cuda.is_available():
            try:
                bf16_ok = torch.cuda.is_bf16_supported()
            except Exception:
                bf16_ok = False
            dtype = torch.bfloat16 if bf16_ok else torch.float16
        else:
            dtype = torch.float32
        device_opt = getattr(CFG, "LLAVA_DEVICE", "auto")
        device_map = "auto" if device_opt == "auto" else device_opt

        local_dir = getattr(CFG, "LLAVA_LOCAL_DIR", None) or getattr(CFG, "LLAVA_MODEL_ID", "llava-hf/llava-1.5-7b-hf")
        use_4bit = bool(getattr(CFG, "LLAVA_USE_4BIT_INFERENCE", False))

        # Load image (PIL)
        image = llava_load_image(str(image_path))

        # Helper: load+generate once (for retries)
        def _load_and_generate(use4: bool, devmap, dt):
            model, processor = load_llava_from_local(
                local_dir=local_dir,
                device_map=devmap,
                dtype=dt,
                hf_token=None,
                use_4bit_inference=use4,
            )
            return run_llava_generate(
                model=model,
                processor=processor,
                image=image,
                prompt=text,
                max_new_tokens=int(getattr(CFG, "LLAVA_MAX_NEW_TOKENS", 256)),
                temperature=float(getattr(CFG, "LLAVA_TEMPERATURE", 0.7)),
                top_p=float(getattr(CFG, "LLAVA_TOP_P", 0.95)),
                top_k=int(getattr(CFG, "LLAVA_TOP_K", 50)),
                dtype=dt,
            )

        # Try normal, then 4-bit on GPU, then CPU fallback (+ runtime error fallback)
        try:
            result = _load_and_generate(use_4bit, device_map, dtype)
        except torch.cuda.OutOfMemoryError as _oom1:
            print("[backend] OOM detected -> retry with 4-bit quantization on GPU")
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                result = _load_and_generate(True, "auto", torch.float16)
                use_4bit = True
                device_map = "auto"
                dtype = torch.float16
            except Exception as _e2:
                print("[backend] Retry on GPU failed -> fallback to CPU")
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    result = _load_and_generate(True, "cpu", torch.float32)
                    use_4bit = True
                    device_map = "cpu"
                    dtype = torch.float32
                except Exception as _e3:
                    raise _e3
        except RuntimeError as _rt:
            msg = str(_rt)
            if ("device-side assert" in msg) or ("probability tensor contains" in msg) or ("CUDA error" in msg):
                print("[backend] CUDA runtime error detected -> fallback to CPU (safe mode, no 4bit)")
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    # CPU安全モード: 4bitは使わずfloat32で再実行（数値安定性最優先）
                    result = _load_and_generate(False, "cpu", torch.float32)
                    use_4bit = False
                    device_map = "cpu"
                    dtype = torch.float32
                except Exception as _e4:
                    raise _e4
            else:
                raise

        # Save/Link model into run_dir to keep provenance
        try:
            model_dst = run_dir / "model"
            # Prefer symlink to avoid GBs of copy
            if model_dst.exists():
                if model_dst.is_symlink():
                    pass
                else:
                    shutil.rmtree(model_dst)
            if not model_dst.exists():
                try:
                    os.symlink(os.path.abspath(local_dir), model_dst, target_is_directory=True)
                except Exception:
                    # Fallback: copy lightweight metadata only
                    model_dst.mkdir(parents=True, exist_ok=True)
                    for fname in ("config.json", "generation_config.json", "tokenizer_config.json", "special_tokens_map.json", "processor_config.json"):
                        srcf = Path(local_dir) / fname
                        if srcf.exists():
                            shutil.copy2(srcf, model_dst / fname)
        except Exception as e:
            print(f"[warn] モデル保存/リンクに失敗: {e}")

        # Minimal metadata for JSON output
        data = {
            "backend": "llava",
            "model": local_dir,
            "params": {
                "max_new_tokens": getattr(CFG, "LLAVA_MAX_NEW_TOKENS", 256),
                "temperature": getattr(CFG, "LLAVA_TEMPERATURE", 0.7),
                "top_p": getattr(CFG, "LLAVA_TOP_P", 0.95),
                "top_k": getattr(CFG, "LLAVA_TOP_K", 50),
                "use_4bit": use_4bit,
                "device_map": device_map,
                "dtype": str(dtype),
            },
            "run_dir": str(run_dir),
        }

        print("\nレスポンス情報(json, 抜粋):")
        print(json.dumps({"backend": "llava", "model": local_dir}, ensure_ascii=False, indent=2))
    else:
        raise ValueError(f"Unknown MODEL_BACKEND in config.py: {backend}")

    print("\n----- 生成結果 -----\n")
    print(result or "")

    # Evaluate versus original_comment if available
    metrics: Dict[str, object] = {}
    if bool(getattr(CFG, "ENABLE_EVALUATION", True)):
        orig_path = ORIGINAL_DIR / f"{y:04d}_{m:02d}_{d:02d}_original.txt"
        if orig_path.exists():
            try:
                original_text = read_text_file(orig_path).strip()
                ref_tokens = normalize_tokens(tokenize_ja(original_text))
                hyp_tokens = normalize_tokens(tokenize_ja(result or ""))

                # Embedding cosine similarity (optional)
                if getattr(CFG, "EVAL_EMBEDDINGS", "openai") == "openai":
                    try:
                        client_emb = OpenAI(api_key=api_key) if backend != "openai" else OpenAI(api_key=api_key)
                        emb_model = getattr(CFG, "OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
                        emb_ref = client_emb.embeddings.create(model=emb_model, input=original_text).data[0].embedding
                        emb_hyp = client_emb.embeddings.create(model=emb_model, input=(result or "")).data[0].embedding
                        metrics["embedding_cosine"] = cosine_similarity(emb_ref, emb_hyp)
                    except Exception as e:
                        metrics["embedding_cosine_error"] = str(e)

                # BLEU and ROUGE-1 (F1) with char-level fallback
                metrics["bleu"] = calc_bleu(ref_tokens, hyp_tokens)
                r1 = calc_rouge1_f1(ref_tokens, hyp_tokens)
                if r1 == 0.0:
                    r1 = char_level_f1(original_text, result or "")
                metrics["rouge1_f1"] = r1

                print("\n--- 比較対象 ---")
                print("[original_comment]\n" + original_text)
                print("\n[generated]\n" + (result or ""))
                print("\n--- 評価結果 ---")
                if "embedding_cosine" in metrics:
                    print(f"Embedding cosine: {metrics['embedding_cosine']:.6f}")
                elif "embedding_cosine_error" in metrics:
                    print(f"Embedding cosine: エラー ({metrics.get('embedding_cosine_error')})")
                print(f"BLEU: {metrics.get('bleu', 0.0):.6f}")
                print(f"ROUGE-1 F1: {metrics.get('rouge1_f1', 0.0):.6f}")
            except Exception as e:
                metrics["error"] = f"評価に失敗: {e}"
                print(metrics["error"])
        else:
            print(f"original_comment が見つかりません: {orig_path}")
    else:
        print("評価をスキップしました（config.py ENABLE_EVALUATION=False）")

    # Save results (directory-based + legacy compatibility)
    out_text = run_dir / "result.txt"
    out_json = run_dir / "response.json"
    try:
        # Write outputs
        out_text.write_text(result or "", encoding="utf-8")
        out_json.write_text(json.dumps({"response": data, "metrics": metrics}, ensure_ascii=False, indent=2), encoding="utf-8")

        # Save model info (backend + model)
        try:
            model_info_path = run_dir / "model_info.txt"
            if backend == "openai":
                model_info_path.write_text(
                    f"backend=openai\nmodel={getattr(CFG, 'OPENAI_MODEL', 'gpt-4.1')}\n",
                    encoding="utf-8",
                )
            else:
                model_info_path.write_text(
                    f"backend=llava\nmodel_id={local_dir}\ndevice_map={device_map}\nuse_4bit={use_4bit}\n",
                    encoding="utf-8",
                )
        except Exception as e:
            print(f"[warn] モデル情報の保存に失敗: {e}")


        print(f"\n保存先: {out_text}\nJSON: {out_json}")
    except Exception as e:
        print(f"結果保存に失敗: {e}")

    return 0


def main():
    parser = argparse.ArgumentParser(description="Integrated WeatherLLM runner (v1..v4) with evaluation")
    parser.add_argument("--pipeline", choices=["v1", "v2", "v3", "v4", "all"], default=None,
                        help="実行パイプライン (既定: config.py の PIPELINE)")
    parser.add_argument("--date", action="append", help="対象日 (YYYYMMDD)。繰り返し指定可")
    parser.add_argument("--auto", action="store_true", help="--date未指定時に data/png から自動発見")
    parser.add_argument("--limit", type=int, help="自動発見する日付の上限数")
    parser.add_argument("--env-file", help="環境ファイル(.env)のパスを明示指定")
    parser.add_argument("--api-key", help="OpenAI APIキーを明示指定")
    args = parser.parse_args()

    # Show configuration source and backend summary (Python config)
    try:
        cfg_py = getattr(CFG, "__file__", None)
        if cfg_py:
            from pathlib import Path as _Path
            print(f"[config] Python: {_Path(cfg_py).resolve()}")
        backend = getattr(CFG, "MODEL_BACKEND", "openai")
        print(f"[config] Backend: {backend}")
        if backend == "openai":
            print(f"[config] OpenAI model: {getattr(CFG, 'OPENAI_MODEL', 'gpt-4.1')}")
            if getattr(CFG, "ENABLE_EVALUATION", True) and getattr(CFG, "EVAL_EMBEDDINGS", "openai") == "openai":
                print(f"[config] Embedding model: {getattr(CFG, 'OPENAI_EMBEDDING_MODEL', 'text-embedding-3-large')}")
        else:
            print(f"[config] LLaVA model_id/local_dir: {getattr(CFG, 'LLAVA_MODEL_ID', None)} / {getattr(CFG, 'LLAVA_LOCAL_DIR', None)}")
            print(
                "[config] "
                f"device={getattr(CFG, 'LLAVA_DEVICE', 'auto')} "
                f"4bit={getattr(CFG, 'LLAVA_USE_4BIT_INFERENCE', False)} "
                f"max_new_tokens={getattr(CFG, 'LLAVA_MAX_NEW_TOKENS', 256)} "
                f"T={getattr(CFG, 'LLAVA_TEMPERATURE', 0.7)} "
                f"top_p={getattr(CFG, 'LLAVA_TOP_P', 0.95)} "
                f"top_k={getattr(CFG, 'LLAVA_TOP_K', 50)}"
            )
    except Exception:
        pass

    # Resolve dates
    dates: List[str] = []
    if args.date:
        dates = [str(d) for d in args.date]
    elif args.auto:
        dates = list_available_png_dates()
        if args.limit and args.limit > 0:
            dates = dates[: args.limit]
    else:
        # fallback: one earliest date from png if available
        tmp = list_available_png_dates()
        if tmp:
            dates = [tmp[0]]

    if not dates:
        sys.stderr.write("[main] 対象日が見つかりません。--date または --auto を指定してください。\n")
        sys.exit(2)

    # Pipelines
    selected_pipeline = args.pipeline or getattr(CFG, "PIPELINE", "v4")
    pipelines = ["v1", "v2", "v3", "v4"] if selected_pipeline == "all" else [selected_pipeline]

    # Run
    overall_codes: List[int] = []
    for pipe in pipelines:
        print(f"[main] Pipeline: {pipe} | Dates: {', '.join(dates)}")
        for d in dates:
            try:
                rc = run_one(pipe, d, args.env_file, args.api_key)
            except Exception as e:
                print(f"[main] 例外: {e}")
                rc = 1
            print(f"[main] Completed {pipe} {d} with code {rc}")
            overall_codes.append(rc)

    sys.exit(0 if all(rc == 0 for rc in overall_codes) else 1)


if __name__ == "__main__":
    main()
