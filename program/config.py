# -*- coding: utf-8 -*-
"""
WeatherLLM 統合ランナー用 設定ファイル（Python 版・見やすい形）

このファイルの CFG を編集して設定します（参考: FrontLine/main_v3_config.py のスタイル）。
CLI の一部オプション（--pipeline, --date, --env-file, --api-key）は従来通り上書き可能です。

バックエンド切替:
- CFG["BACKEND"]["TYPE"] を ["openai", "llava", "qwen3b", "qwen7b", "ovis", "r4b"] の中から1つ以上選択
- 詳細設定はそれぞれ CFG["BACKEND"]["OPENAI"] / CFG["BACKEND"]["LLAVA"] /
  CFG["BACKEND"]["QWEN3B"] / CFG["BACKEND"]["QWEN7B"] /
  CFG["BACKEND"]["OVIS25"] / CFG["BACKEND"]["R4B"] に記載

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
    "PIPELINE": "v1",      # "v1" | "v2" | "v3" | "v4" | "all"
    "DATES": ["20220101"],           # 例: ["20220101", "20220106"]（空なら AUTO_FROM_PNG が有効なら自動発見）
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
        # 複数選択可:
        # - 選択肢: ["openai", "llava", "qwen3b", "qwen7b", "ovis", "r4b"]
        # - 例1: ["qwen3b"]（既定）
        # - 例2: ["qwen7b", "llava"]（Qwen7B と LLaVA を順に実行）
        # - 例3: ["ovis", "r4b"]（Ovis→R-4B の順に実行）
        # - 例4: ["openai"]（OpenAI Responses API を利用）
        "TYPE": ["llava", "qwen3b", "qwen7b", "ovis", "r4b"],

        # OpenAI GPT (multimodal via Responses API)
        "OPENAI": {
            "MODEL": "gpt-4.1",
            "EMBEDDING_MODEL": "text-embedding-3-large",  # 評価用
        },

        # LLaVA (HF)
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

        # Qwen2.5-VL 3B
        "QWEN3B": {
            "MODEL_ID": "Qwen/Qwen2.5-VL-3B-Instruct",
            "DTYPE": "auto",                  # "auto" | "bf16" | "fp16" | "fp32"
            "USE_4BIT_INFERENCE": True,       # bitsandbytes が利用可能なら4bitを使用
            "ENABLE_FLASH_ATTN": False,       # flash_attention_2（不可なら自動フォールバック）
            "MIN_PIXELS": 256 * 28 * 28,      # VRAM節約のための入力画像サイズ下限
            "MAX_PIXELS": 896 * 28 * 28,      # 上限（性能/VRAMのバランス）
            "MAX_NEW_TOKENS": 256,
            "TEMPERATURE": 0.7,
            "TOP_P": 0.95,
            "TOP_K": 50,
        },

        # Qwen2.5-VL 7B
        "QWEN7B": {
            "MODEL_ID": "Qwen/Qwen2.5-VL-7B-Instruct",
            "DTYPE": "auto",
            "USE_4BIT_INFERENCE": True,
            "ENABLE_FLASH_ATTN": False,
            "MIN_PIXELS": 256 * 28 * 28,
            "MAX_PIXELS": 896 * 28 * 28,
            "MAX_NEW_TOKENS": 256,
            "TEMPERATURE": 0.7,
            "TOP_P": 0.95,
            "TOP_K": 50,
        },


        # Ovis2.5-9B
        "OVIS25": {
            "MODEL_ID": "AIDC-AI/Ovis2.5-9B",
            "DTYPE": "auto",
            "USE_4BIT_INFERENCE": True,
            "ENABLE_FLASH_ATTN": False,
            "ENABLE_THINKING": False,
            "ENABLE_THINKING_BUDGET": False,
            "THINKING_BUDGET": 2048,  # 有効時のみ使用
            "MAX_NEW_TOKENS": 1024,
            "TEMPERATURE": 0.7,
            "TOP_P": 0.95,
            "TOP_K": 50,
        },

        # YannQi/R-4B
        "R4B": {
            "MODEL_ID": "YannQi/R-4B",
            "DTYPE": "auto",
            "USE_4BIT_INFERENCE": True,
            "THINKING_MODE": "auto",     # "auto" | "long"(thinking) | "short"(non-thinking)
            "MAX_NEW_TOKENS": 2048,
            "TEMPERATURE": 0.7,
            "TOP_P": 0.95,
            "TOP_K": 50,
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
MODEL_BACKENDS = CFG["BACKEND"]["TYPE"] if isinstance(CFG["BACKEND"]["TYPE"], list) else [CFG["BACKEND"]["TYPE"]]
MODEL_BACKEND = MODEL_BACKENDS[0] if MODEL_BACKENDS else "openai"

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

# Qwen2.5-VL 3B
QWEN3B_MODEL_ID = CFG["BACKEND"]["QWEN3B"]["MODEL_ID"]
QWEN3B_DTYPE = CFG["BACKEND"]["QWEN3B"]["DTYPE"]
QWEN3B_USE_4BIT_INFERENCE = CFG["BACKEND"]["QWEN3B"]["USE_4BIT_INFERENCE"]
QWEN3B_ENABLE_FLASH_ATTN = CFG["BACKEND"]["QWEN3B"]["ENABLE_FLASH_ATTN"]
QWEN3B_MIN_PIXELS = CFG["BACKEND"]["QWEN3B"]["MIN_PIXELS"]
QWEN3B_MAX_PIXELS = CFG["BACKEND"]["QWEN3B"]["MAX_PIXELS"]
QWEN3B_MAX_NEW_TOKENS = CFG["BACKEND"]["QWEN3B"]["MAX_NEW_TOKENS"]
QWEN3B_TEMPERATURE = CFG["BACKEND"]["QWEN3B"]["TEMPERATURE"]
QWEN3B_TOP_P = CFG["BACKEND"]["QWEN3B"]["TOP_P"]
QWEN3B_TOP_K = CFG["BACKEND"]["QWEN3B"]["TOP_K"]

# Qwen2.5-VL 7B
QWEN7B_MODEL_ID = CFG["BACKEND"]["QWEN7B"]["MODEL_ID"]
QWEN7B_DTYPE = CFG["BACKEND"]["QWEN7B"]["DTYPE"]
QWEN7B_USE_4BIT_INFERENCE = CFG["BACKEND"]["QWEN7B"]["USE_4BIT_INFERENCE"]
QWEN7B_ENABLE_FLASH_ATTN = CFG["BACKEND"]["QWEN7B"]["ENABLE_FLASH_ATTN"]
QWEN7B_MIN_PIXELS = CFG["BACKEND"]["QWEN7B"]["MIN_PIXELS"]
QWEN7B_MAX_PIXELS = CFG["BACKEND"]["QWEN7B"]["MAX_PIXELS"]
QWEN7B_MAX_NEW_TOKENS = CFG["BACKEND"]["QWEN7B"]["MAX_NEW_TOKENS"]
QWEN7B_TEMPERATURE = CFG["BACKEND"]["QWEN7B"]["TEMPERATURE"]
QWEN7B_TOP_P = CFG["BACKEND"]["QWEN7B"]["TOP_P"]
QWEN7B_TOP_K = CFG["BACKEND"]["QWEN7B"]["TOP_K"]


# Ovis2.5-9B
OVIS25_MODEL_ID = CFG["BACKEND"]["OVIS25"]["MODEL_ID"]
OVIS25_DTYPE = CFG["BACKEND"]["OVIS25"]["DTYPE"]
OVIS25_USE_4BIT_INFERENCE = CFG["BACKEND"]["OVIS25"]["USE_4BIT_INFERENCE"]
OVIS25_ENABLE_FLASH_ATTN = CFG["BACKEND"]["OVIS25"]["ENABLE_FLASH_ATTN"]
OVIS25_ENABLE_THINKING = CFG["BACKEND"]["OVIS25"]["ENABLE_THINKING"]
OVIS25_ENABLE_THINKING_BUDGET = CFG["BACKEND"]["OVIS25"]["ENABLE_THINKING_BUDGET"]
OVIS25_THINKING_BUDGET = CFG["BACKEND"]["OVIS25"]["THINKING_BUDGET"]
OVIS25_MAX_NEW_TOKENS = CFG["BACKEND"]["OVIS25"]["MAX_NEW_TOKENS"]
OVIS25_TEMPERATURE = CFG["BACKEND"]["OVIS25"]["TEMPERATURE"]
OVIS25_TOP_P = CFG["BACKEND"]["OVIS25"]["TOP_P"]
OVIS25_TOP_K = CFG["BACKEND"]["OVIS25"]["TOP_K"]

# R-4B
R4B_MODEL_ID = CFG["BACKEND"]["R4B"]["MODEL_ID"]
R4B_DTYPE = CFG["BACKEND"]["R4B"]["DTYPE"]
R4B_USE_4BIT_INFERENCE = CFG["BACKEND"]["R4B"]["USE_4BIT_INFERENCE"]
R4B_THINKING_MODE = CFG["BACKEND"]["R4B"]["THINKING_MODE"]
R4B_MAX_NEW_TOKENS = CFG["BACKEND"]["R4B"]["MAX_NEW_TOKENS"]
R4B_TEMPERATURE = CFG["BACKEND"]["R4B"]["TEMPERATURE"]
R4B_TOP_P = CFG["BACKEND"]["R4B"]["TOP_P"]
R4B_TOP_K = CFG["BACKEND"]["R4B"]["TOP_K"]

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
        "MODEL_BACKENDS": MODEL_BACKENDS,
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
        "QWEN3B_MODEL_ID": QWEN3B_MODEL_ID,
        "QWEN3B_DTYPE": QWEN3B_DTYPE,
        "QWEN3B_USE_4BIT_INFERENCE": QWEN3B_USE_4BIT_INFERENCE,
        "QWEN3B_ENABLE_FLASH_ATTN": QWEN3B_ENABLE_FLASH_ATTN,
        "QWEN3B_MIN_PIXELS": QWEN3B_MIN_PIXELS,
        "QWEN3B_MAX_PIXELS": QWEN3B_MAX_PIXELS,
        "QWEN3B_MAX_NEW_TOKENS": QWEN3B_MAX_NEW_TOKENS,
        "QWEN3B_TEMPERATURE": QWEN3B_TEMPERATURE,
        "QWEN3B_TOP_P": QWEN3B_TOP_P,
        "QWEN3B_TOP_K": QWEN3B_TOP_K,
        "QWEN7B_MODEL_ID": QWEN7B_MODEL_ID,
        "QWEN7B_DTYPE": QWEN7B_DTYPE,
        "QWEN7B_USE_4BIT_INFERENCE": QWEN7B_USE_4BIT_INFERENCE,
        "QWEN7B_ENABLE_FLASH_ATTN": QWEN7B_ENABLE_FLASH_ATTN,
        "QWEN7B_MIN_PIXELS": QWEN7B_MIN_PIXELS,
        "QWEN7B_MAX_PIXELS": QWEN7B_MAX_PIXELS,
        "QWEN7B_MAX_NEW_TOKENS": QWEN7B_MAX_NEW_TOKENS,
        "QWEN7B_TEMPERATURE": QWEN7B_TEMPERATURE,
        "QWEN7B_TOP_P": QWEN7B_TOP_P,
        "QWEN7B_TOP_K": QWEN7B_TOP_K,
        "OVIS25_MODEL_ID": OVIS25_MODEL_ID,
        "OVIS25_DTYPE": OVIS25_DTYPE,
        "OVIS25_USE_4BIT_INFERENCE": OVIS25_USE_4BIT_INFERENCE,
        "OVIS25_ENABLE_FLASH_ATTN": OVIS25_ENABLE_FLASH_ATTN,
        "OVIS25_ENABLE_THINKING": OVIS25_ENABLE_THINKING,
        "OVIS25_ENABLE_THINKING_BUDGET": OVIS25_ENABLE_THINKING_BUDGET,
        "OVIS25_THINKING_BUDGET": OVIS25_THINKING_BUDGET,
        "OVIS25_MAX_NEW_TOKENS": OVIS25_MAX_NEW_TOKENS,
        "OVIS25_TEMPERATURE": OVIS25_TEMPERATURE,
        "OVIS25_TOP_P": OVIS25_TOP_P,
        "OVIS25_TOP_K": OVIS25_TOP_K,
        "R4B_MODEL_ID": R4B_MODEL_ID,
        "R4B_DTYPE": R4B_DTYPE,
        "R4B_USE_4BIT_INFERENCE": R4B_USE_4BIT_INFERENCE,
        "R4B_THINKING_MODE": R4B_THINKING_MODE,
        "R4B_MAX_NEW_TOKENS": R4B_MAX_NEW_TOKENS,
        "R4B_TEMPERATURE": R4B_TEMPERATURE,
        "R4B_TOP_P": R4B_TOP_P,
        "R4B_TOP_K": R4B_TOP_K,
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
    "MODEL_BACKEND", "MODEL_BACKENDS",
    "OPENAI_MODEL", "OPENAI_EMBEDDING_MODEL",
    "LLAVA_MODEL_ID", "LLAVA_LOCAL_DIR", "LLAVA_DEVICE", "LLAVA_USE_4BIT_INFERENCE",
    "LLAVA_MAX_NEW_TOKENS", "LLAVA_TEMPERATURE", "LLAVA_TOP_P", "LLAVA_TOP_K", "LLAVA_SEED",
    "QWEN3B_MODEL_ID", "QWEN3B_DTYPE", "QWEN3B_USE_4BIT_INFERENCE", "QWEN3B_ENABLE_FLASH_ATTN",
    "QWEN3B_MIN_PIXELS", "QWEN3B_MAX_PIXELS", "QWEN3B_MAX_NEW_TOKENS", "QWEN3B_TEMPERATURE", "QWEN3B_TOP_P", "QWEN3B_TOP_K",
    "QWEN7B_MODEL_ID", "QWEN7B_DTYPE", "QWEN7B_USE_4BIT_INFERENCE", "QWEN7B_ENABLE_FLASH_ATTN",
    "QWEN7B_MIN_PIXELS", "QWEN7B_MAX_PIXELS", "QWEN7B_MAX_NEW_TOKENS", "QWEN7B_TEMPERATURE", "QWEN7B_TOP_P", "QWEN7B_TOP_K",
    "OVIS25_MODEL_ID", "OVIS25_DTYPE", "OVIS25_USE_4BIT_INFERENCE", "OVIS25_ENABLE_FLASH_ATTN",
    "OVIS25_ENABLE_THINKING", "OVIS25_ENABLE_THINKING_BUDGET", "OVIS25_THINKING_BUDGET", "OVIS25_MAX_NEW_TOKENS",
    "OVIS25_TEMPERATURE", "OVIS25_TOP_P", "OVIS25_TOP_K",
    "R4B_MODEL_ID", "R4B_DTYPE", "R4B_USE_4BIT_INFERENCE", "R4B_THINKING_MODE", "R4B_MAX_NEW_TOKENS",
    "R4B_TEMPERATURE", "R4B_TOP_P", "R4B_TOP_K",
    # evaluation
    "ENABLE_EVALUATION", "EVAL_EMBEDDINGS",
    # helper
    "as_dict",
]
