# -*- coding: utf-8 -*-
"""
WeatherLLM 統合ランナー用 設定ファイル（Python 版・見やすい形）

このファイルの CFG を編集して設定します（参考: FrontLine/main_v3_config.py のスタイル）。
CLI の一部オプション（--pipeline, --date, --env-file, --api-key）は従来通り上書き可能です。

バックエンド切替:
- CFG["BACKEND"]["TYPE"] を "openai" | "llava" に設定
- OpenAI 用の詳細は CFG["BACKEND"]["OPENAI"]、LLaVA 用の詳細は CFG["BACKEND"]["LLAVA"] にまとめています

評価:
- CFG["EVALUATION"]["ENABLE"] を True/False
- CFG["EVALUATION"]["EMBEDDINGS"] を "openai" または None
"""

from pathlib import Path

# ベースディレクトリ（program 配下）
BASE_DIR = Path(__file__).resolve().parent
PNG_DIR = BASE_DIR / "data" / "png"

# ======================================
# 中央設定（見やすい階層構造）
# ======================================
CFG: dict = {
    # 実行パイプライン/対象日
    "PIPELINE": "v4",      # "v1" | "v2" | "v3" | "v4" | "all"
    "DATES": [],           # 例: ["20220101", "20220106"]（空なら AUTO_FROM_PNG が有効なら自動発見）
    "AUTO_FROM_PNG": True, # data/png/*.png から自動発見
    "AUTO_LIMIT": None,    # 先頭 N 件に制限（None なら制限なし）

    # 認証/環境
    "ENV_FILE": None,      # 例: "/home/s233319/docker_miniconda/.env"
    "API_KEY": None,       # 例: "sk-xxxx"（指定時は .env 不要）

    # ラン実行系
    "RUN": {
        "PYTHON_BIN": "python",   # "python3" など
        "LOG_TO_FILES": True,     # 各実行のログをファイルに保存（program/配下）
        "ECHO_TO_STDOUT": True,   # 標準出力にも出す
        "TIMEOUT": None,          # 各実行のタイムアウト（秒）Noneで無効
        "PARALLEL": False,        # 複数日付の並列実行（API制限に注意）
    },

    # 生成バックエンド（GPT API or LLaVA）
    "BACKEND": {
        "TYPE": "llava",  # "openai" | "llava"

        "OPENAI": {
            "MODEL": "gpt-4.1",
            "EMBEDDING_MODEL": "text-embedding-3-large",  # 評価用
        },

        "LLAVA": {
            "MODEL_ID": "llava-hf/llava-1.5-7b-hf",  # HF モデル ID or ローカルパス
            "LOCAL_DIR": None,                       # 既にローカルへ保存済みならパス指定可
            "DEVICE": "auto",                        # "auto" | "cpu" | "cuda"
            "USE_4BIT_INFERENCE": True,             # 4bit 量子化（bitsandbytes 必須）
            "MAX_NEW_TOKENS": 256,
            "TEMPERATURE": 0.7,
            "TOP_P": 0.95,
            "TOP_K": 50,
            "SEED": 0,
        },
    },

    # 評価関連
    "EVALUATION": {
        "ENABLE": True,       # True で original_comment との比較を実施
        "EMBEDDINGS": "openai"  # "openai" | None（LLaVAのみで完結させたい場合は None）
    },
}

# ======================================
# 既存コードとの互換用（module 変数へ展開）
# ======================================

# パイプライン/対象日
PIPELINE = CFG["PIPELINE"]
DATES = CFG["DATES"]
AUTO_FROM_PNG = CFG["AUTO_FROM_PNG"]
AUTO_LIMIT = CFG["AUTO_LIMIT"]

# 認証/環境
ENV_FILE = CFG["ENV_FILE"]
API_KEY = CFG["API_KEY"]

# ラン実行系
PYTHON_BIN = CFG["RUN"]["PYTHON_BIN"]
LOG_TO_FILES = CFG["RUN"]["LOG_TO_FILES"]
ECHO_TO_STDOUT = CFG["RUN"]["ECHO_TO_STDOUT"]
TIMEOUT = CFG["RUN"]["TIMEOUT"]
PARALLEL = CFG["RUN"]["PARALLEL"]

# バックエンド
MODEL_BACKEND = CFG["BACKEND"]["TYPE"]

# OpenAI
OPENAI_MODEL = CFG["BACKEND"]["OPENAI"]["MODEL"]
OPENAI_EMBEDDING_MODEL = CFG["BACKEND"]["OPENAI"]["EMBEDDING_MODEL"]

# LLaVA
LLAVA_MODEL_ID = CFG["BACKEND"]["LLAVA"]["MODEL_ID"]
LLAVA_LOCAL_DIR = CFG["BACKEND"]["LLAVA"]["LOCAL_DIR"]
LLAVA_DEVICE = CFG["BACKEND"]["LLAVA"]["DEVICE"]
LLAVA_USE_4BIT_INFERENCE = CFG["BACKEND"]["LLAVA"]["USE_4BIT_INFERENCE"]
LLAVA_MAX_NEW_TOKENS = CFG["BACKEND"]["LLAVA"]["MAX_NEW_TOKENS"]
LLAVA_TEMPERATURE = CFG["BACKEND"]["LLAVA"]["TEMPERATURE"]
LLAVA_TOP_P = CFG["BACKEND"]["LLAVA"]["TOP_P"]
LLAVA_TOP_K = CFG["BACKEND"]["LLAVA"]["TOP_K"]
LLAVA_SEED = CFG["BACKEND"]["LLAVA"]["SEED"]

# 評価
ENABLE_EVALUATION = CFG["EVALUATION"]["ENABLE"]
EVAL_EMBEDDINGS = CFG["EVALUATION"]["EMBEDDINGS"]

def as_dict() -> dict:
    """main_v1.py 等から参照される互換ヘルパ。既存キーを維持して返す。"""
    return {
        "PIPELINE": PIPELINE,
        "DATES": DATES[:],
        "AUTO_FROM_PNG": AUTO_FROM_PNG,
        "AUTO_LIMIT": AUTO_LIMIT,
        "ENV_FILE": ENV_FILE,
        "API_KEY": API_KEY,
        "PYTHON_BIN": PYTHON_BIN,
        "LOG_TO_FILES": LOG_TO_FILES,
        "ECHO_TO_STDOUT": ECHO_TO_STDOUT,
        "TIMEOUT": TIMEOUT,
        "PARALLEL": PARALLEL,
        "BASE_DIR": str(BASE_DIR),
        "PNG_DIR": str(PNG_DIR),
        # Backend/model config
        "MODEL_BACKEND": MODEL_BACKEND,
        "OPENAI_MODEL": OPENAI_MODEL,
        "OPENAI_EMBEDDING_MODEL": OPENAI_EMBEDDING_MODEL,
        "LLAVA_MODEL_ID": LLAVA_MODEL_ID,
        "LLAVA_LOCAL_DIR": LLAVA_LOCAL_DIR,
        "LLAVA_DEVICE": LLAVA_DEVICE,
        "LLAVA_USE_4BIT_INFERENCE": LLAVA_USE_4BIT_INFERENCE,
        "LLAVA_MAX_NEW_TOKENS": LLAVA_MAX_NEW_TOKENS,
        "LLAVA_TEMPERATURE": LLAVA_TEMPERATURE,
        "LLAVA_TOP_P": LLAVA_TOP_P,
        "LLAVA_TOP_K": LLAVA_TOP_K,
        "LLAVA_SEED": LLAVA_SEED,
        "ENABLE_EVALUATION": ENABLE_EVALUATION,
        "EVAL_EMBEDDINGS": EVAL_EMBEDDINGS,
    }

__all__ = [
    "CFG",
    # base
    "BASE_DIR", "PNG_DIR",
    # top-level
    "PIPELINE", "DATES", "AUTO_FROM_PNG", "AUTO_LIMIT",
    "ENV_FILE", "API_KEY",
    # run
    "PYTHON_BIN", "LOG_TO_FILES", "ECHO_TO_STDOUT", "TIMEOUT", "PARALLEL",
    # backend
    "MODEL_BACKEND",
    "OPENAI_MODEL", "OPENAI_EMBEDDING_MODEL",
    "LLAVA_MODEL_ID", "LLAVA_LOCAL_DIR", "LLAVA_DEVICE", "LLAVA_USE_4BIT_INFERENCE",
    "LLAVA_MAX_NEW_TOKENS", "LLAVA_TEMPERATURE", "LLAVA_TOP_P", "LLAVA_TOP_K", "LLAVA_SEED",
    # evaluation
    "ENABLE_EVALUATION", "EVAL_EMBEDDINGS",
    # helper
    "as_dict",
]
