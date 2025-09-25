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

# Portable paths (no absolute home or fixed root)
BASE_DIR = Path(__file__).resolve().parent
PROGRAM_DIR = BASE_DIR
PNG_DIR = PROGRAM_DIR / "data" / "png"
NUMERIC_DIR = PROGRAM_DIR / "data" / "Numerical_weather_data"
PROMPT_DIR = PROGRAM_DIR / "data" / "prompt_gpt"
RESULTS_DIR = PROGRAM_DIR / "results"
INSTRUCTION_PATH = PROMPT_DIR / "v3_instruction.txt"

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

MODEL_NAME = "gpt-4o"  # Vision-capable model


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


def find_weather_data_file(y: int, m: int, d: int) -> Path:
    """
    Try to find a numerical weather data file in NUMERIC_DIR.
    Filenames may or may not be zero-padded based on the dataset provided.
    Will try multiple naming patterns, e.g.:
      - 2022-1-6.txt
      - 2022-01-06.txt
      - 2022-01-6.txt
      - 2022-1-06.txt
    """
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


def main():
    parser = argparse.ArgumentParser(description="v3: 画像(天気図) + 気象数値データ + 指示文でコメント生成 (Vision)")
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

    # Weather numerical data (prefer non-padded, also try padded)
    weather_data_path = find_weather_data_file(y, m, d)
    weather_data = read_text_file(weather_data_path)

    # Encode image
    base64_image = encode_image_to_base64(image_path)

    # Build message text
    weather_date_jp = build_japanese_date(y, m, d)
    text = f"""
```
以下は{weather_date_jp}の気象データです。
{weather_data}
```

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
        "max_tokens": 4000,
    }

    print("リクエスト本文:")
    print(text)
    print(f"\n送信画像: {image_path.name}")
    print(f"気象データ: {weather_data_path.name}")

    resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=120)

    if resp.status_code == 200:
        try:
            data = resp.json()
        except Exception:
            print(resp.text)
            raise
        print("\nAPIレスポンス(json, 抜粋):")
        print(json.dumps({k: data.get(k) for k in ("id", "model", "usage")}, ensure_ascii=False, indent=2))
        result = data["choices"][0]["message"]["content"]
        print("\n----- 生成結果 -----\n")
        print(result)

        # Save results under program/results
        base_name = f"v3_{yyyymmdd}"
        out_text = RESULTS_DIR / f"{base_name}_result.txt"
        out_json = RESULTS_DIR / f"{base_name}_response.json"
        out_text.write_text(result, encoding="utf-8")
        out_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n保存先: {out_text}\nJSON: {out_json}")
    else:
        print(f"APIリクエストに失敗しました: ステータスコード {resp.status_code}")
        try:
            print(resp.text)
        except Exception:
            pass
        resp.raise_for_status()


if __name__ == "__main__":
    main()
