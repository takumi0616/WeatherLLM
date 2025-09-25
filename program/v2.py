#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import base64
import argparse
from pathlib import Path
import requests
from datetime import datetime
import sys
from openai import OpenAI
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
from rouge_score import rouge_scorer
import sacrebleu

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
            # フォールバック: 文字単位
            return list(text)

def normalize_tokens(tokens):
    import re
    cleaned = []
    for t in tokens:
        t = t.strip()
        t = re.sub(r"[、。．，・：；！!？\?「」『』（）\(\)\[\]\{\}”“\"'`’\-_/\\…･･･~＾^＋+＝=＊*＜＞<>＆&％%＄$#＠@|]", "", t)
        t = re.sub(r"\s+", "", t)
        t = t.lower()
        if t:
            cleaned.append(t)
    return cleaned

def char_level_f1(a: str, b: str):
    import re
    from collections import Counter
    def norm(s: str) -> str:
        s = s.strip().lower()
        s = re.sub(r"[、。．，・：；！!？\?「」『』（）\(\)\[\]\{\}”“\"'`’\-_/\\…･･･~＾^＋+＝=＊*＜＞<>＆&％%＄$#＠@|]", "", s)
        s = re.sub(r"\s+", "", s)
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

def calc_bleu(ref_tokens, hyp_tokens) -> float:
    if sacrebleu is None:
        return 0.0
    # 事前にトークン化済みなので tokenize="none" で渡す
    return float(sacrebleu.sentence_bleu(" ".join(hyp_tokens), [" ".join(ref_tokens)], tokenize="none").score)

def calc_rouge1_f1(ref_tokens, hyp_tokens) -> float:
    # Prefer library scorer when available
    if rouge_scorer is not None:
        scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=False)
        scores = scorer.score(" ".join(ref_tokens), " ".join(hyp_tokens))
        return float(scores['rouge1'].fmeasure)
    # Fallback: custom unigram F1
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

def cosine_similarity(vec1, vec2) -> float:
    v1 = np.array(vec1, dtype=float)
    v2 = np.array(vec2, dtype=float)
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) or 1.0
    return float(np.dot(v1, v2) / denom)

# Portable paths (no absolute home or fixed root)
BASE_DIR = Path(__file__).resolve().parent
PROGRAM_DIR = BASE_DIR
PNG_DIR = PROGRAM_DIR / "data" / "png"
PROMPT_DIR = PROGRAM_DIR / "data" / "prompt_gpt"
RESULTS_DIR = PROGRAM_DIR / "results"
INSTRUCTION_PATH = PROMPT_DIR / "v2_instruction.txt"
ORIGINAL_DIR = PROGRAM_DIR / "data" / "original_comment"

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

ENV_PATH = find_env_path()

MODEL_NAME = "gpt-4.1"  # Vision-capable model (Responses API)


def load_api_key() -> str:
    """
    Load OpenAI API key from environment or .env file.
    Expects variable name: OpenAI_KEY_TOKEN
    """
    # 1) Try env var directly
    key = os.environ.get("OpenAI_KEY_TOKEN")
    if key:
        return key

    # 2) Try python-dotenv if available
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=ENV_PATH)
        key = os.environ.get("OpenAI_KEY_TOKEN")
        if key:
            return key
    except Exception:
        pass

    # 3) Fallback: parse .env manually
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                if k.strip() == "OpenAI_KEY_TOKEN":
                    return v.strip().strip("\"'")

    raise RuntimeError(f"OpenAI API key not found. Ensure OpenAI_KEY_TOKEN is set in environment or {ENV_PATH}")


def encode_image_to_base64(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        raise RuntimeError(f"テキストファイルを読み込めませんでした: {path} ({e})")


def list_available_png_dates():
    """
    Return a sorted list of dates (YYYYMMDD as int) available in PNG_DIR.
    """
    dates = []
    for p in PNG_DIR.glob("*.png"):
        m = re.match(r"^(\d{8})\.png$", p.name)
        if m:
            try:
                dates.append(int(m.group(1)))
            except ValueError:
                pass
    return sorted(dates)


def parse_date_yyyymmdd(s: str) -> tuple[int, int, int]:
    if not re.fullmatch(r"\d{8}", s):
        raise ValueError("日付はYYYYMMDD形式で指定してください (例: 20220101)")
    y = int(s[0:4])
    m = int(s[4:6])
    d = int(s[6:8])
    # Basic validation
    datetime(year=y, month=m, day=d)
    return y, m, d


def build_japanese_date(y: int, m: int, d: int) -> str:
    return f"{y}年{m}月{d}日"


def main():
    parser = argparse.ArgumentParser(description="v2: 画像(天気図) + 指示文でコメント生成 (Vision)")
    parser.add_argument("--date", help="対象日 (YYYYMMDD)。未指定時はdata/png内の最初の画像を使用します。")
    parser.add_argument("--env-file", help="環境ファイル(.env)のパスを明示指定")
    parser.add_argument("--api-key", help="OpenAI APIキーを明示指定")
    args = parser.parse_args()

    # Resolve env and API key overrides
    if args.env_file:
        os.environ["OPENAI_ENV_FILE"] = args.env_file
    if args.api_key:
        os.environ["OPENAI_KEY_TOKEN"] = args.api_key
    global ENV_PATH
    ENV_PATH = find_env_path()

    # Decide target date
    if args.date:
        y, m, d = parse_date_yyyymmdd(args.date)
        yyyymmdd = f"{y:04d}{m:02d}{d:02d}"
    else:
        available = list_available_png_dates()
        if not available:
            raise RuntimeError(f"画像が見つかりませんでした: {PNG_DIR}")
        yyyymmdd = f"{available[0]}"
        y, m, d = parse_date_yyyymmdd(yyyymmdd)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    image_path = PNG_DIR / f"{yyyymmdd}.png"
    if not image_path.exists():
        raise FileNotFoundError(f"画像ファイルが存在しません: {image_path}")

    # Instruction
    if not INSTRUCTION_PATH.exists():
        raise FileNotFoundError(f"インストラクションファイルが存在しません: {INSTRUCTION_PATH}")
    instruction = read_text_file(INSTRUCTION_PATH)

    # Encode image
    base64_image = encode_image_to_base64(image_path)

    # Build message text
    weather_date_jp = build_japanese_date(y, m, d)
    text = f"""
入力された画像は{weather_date_jp}9時の天気図です。



{instruction}
""".strip()

    api_key = load_api_key()

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "high",
                        },
                    },
                ],
            }
        ],
        "max_completion_tokens": 3500,
    }

    print("リクエスト本文:")
    print(text)
    print(f"\n送信画像: {image_path.name}")

    # Use OpenAI Python SDK (Responses API)
    client = OpenAI(api_key=api_key)
    try:
        response = client.responses.create(
            model=MODEL_NAME,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": text},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{base64_image}",
                        },
                    ],
                }
            ],
        )
    except Exception as e:
        print(f"OpenAI API呼び出しで例外: {e}")
        raise

    # Format and save response
    try:
        data = response.model_dump()
    except Exception:
        try:
            data = json.loads(response.model_dump_json())
        except Exception:
            data = {"raw": str(response)}

    print("\nAPIレスポンス(json, 抜粋):")
    slim = {k: data.get(k) for k in ("id", "model", "usage") if k in data}
    print(json.dumps(slim, ensure_ascii=False, indent=2))

    result = getattr(response, "output_text", None)
    if result is None:
        # Fallback: try to extract text from content if available
        try:
            result = "".join(part.get("text", "") for part in data.get("output", []) if isinstance(part, dict))
        except Exception:
            result = ""

    print("\n----- 生成結果 -----\n")
    print(result)

    # Evaluate against original_comment if available
    metrics = {}
    orig_path = ORIGINAL_DIR / f"{y:04d}_{m:02d}_{d:02d}_original.txt"
    if orig_path.exists():
        try:
            original_text = read_text_file(orig_path).strip()
            ref_tokens = normalize_tokens(tokenize_ja(original_text))
            hyp_tokens = normalize_tokens(tokenize_ja(result or ""))

            # Embedding cosine similarity (text-embedding-3-large)
            try:
                emb_ref = client.embeddings.create(model="text-embedding-3-large", input=original_text).data[0].embedding
                emb_hyp = client.embeddings.create(model="text-embedding-3-large", input=(result or "")).data[0].embedding
                metrics["embedding_cosine"] = cosine_similarity(emb_ref, emb_hyp)
            except Exception as e:
                metrics["embedding_cosine_error"] = str(e)

            # BLEU and ROUGE-1 (F1)
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
            else:
                print(f"Embedding cosine: エラー ({metrics.get('embedding_cosine_error')})")
            print(f"BLEU: {metrics['bleu']:.6f}")
            print(f"ROUGE-1 F1: {metrics['rouge1_f1']:.6f}")
        except Exception as e:
            metrics["error"] = f"評価に失敗: {e}"
            print(metrics["error"])
    else:
        print(f"original_comment が見つかりません: {orig_path}")

    # Save results under program/results
    base_name = f"v2_{yyyymmdd}"
    out_text = RESULTS_DIR / f"{base_name}_result.txt"
    out_json = RESULTS_DIR / f"{base_name}_response.json"
    out_text.write_text(result or "", encoding="utf-8")
    out_json.write_text(json.dumps({"response": data, "metrics": metrics}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n保存先: {out_text}\nJSON: {out_json}")


if __name__ == "__main__":
    main()
