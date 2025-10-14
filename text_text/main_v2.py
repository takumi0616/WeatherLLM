# -*- coding: utf-8 -*-
"""
main_v2.py

目的:
- main_v1.py の評価ロジックを全て包含
- 追加で、WebLab 10B Instruction SFT（llm_model/weblab_10b_instruction_sft.py）を ./models に自動ダウンロード（既存あれば再利用）して、（13BはGPU要件が高いためデフォルト無効。必要なら import を切替）
  data/Numerical_weather_data の各日付ファイルと data/prompt_gpt/v4_instruction.txt を「そのまま連結」した
  プロンプトで生成を実行
- 生成した LLM 出力を以下で評価（main_v1 と同じ埋め込み評価系を利用）
  1) LLM出力 vs original_comment
  2) LLM出力 vs data/generate_comment（gpt v4）
- 生成文は data/llm_outputs/weblab10b/{YYYY_MM_DD}_weblab10b.txt に保存
- main_v1 相当の比較（generate_comment vs original_comment）も従来通り実行
- ログ:
  - main_v1 互換の結果: result_v1.log（main_v1 と同名互換）
  - LLM 生成・評価の結果: result_v2.log

実行例:
nohup python main_v2.py --run-llm > main_v2.log 2>&1 &
pkill -f "main_v2.py"

注:
- 大規模モデルのため、GPU 環境推奨。bitsandbytes が無ければ FP16/BF16/FP32 でロードします。
- HF_TOKEN 環境変数が設定されていれば Hugging Face のダウンロードに使用します。
"""

import os
import sys
import argparse
import re
import csv
import math
import torch
from typing import Optional, Tuple, List, Dict

# 親ディレクトリ（src/WeatherLLM）を import path に追加して、llm_utils 等の相対外部 import を解決
_THIS_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))  # src/WeatherLLM
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from llm_utils import setup_logger  # noqa: E402
# torchvision の import に伴う不整合回避（transformers が画像ユーティリティを読み込まないようにする）
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

# Embedding/Metric モジュール（main_v1 と同一）
from embedding_model import Sarashina_Embedding_v2_1B as sarashina
from embedding_model import sarashina_embedding_v1_1B as sarashina_v1_mod
from embedding_model import static_embedding_japanese as static_mod
from embedding_model import ruri_base as ruri_mod
from embedding_model import ruri_v3_310m as ruri3_mod
from embedding_model import PLaMo_Embedding_1B as plamo_mod
from embedding_model import Multilingual_E5_base as e5_mod
from embedding_model import multilingual_e5_large as e5_large_mod
from embedding_model import jina_embeddings_v3 as jina_mod
from embedding_model import BGE_M3 as bge_m3_mod
from embedding_model import embeddinggemma_300m as gemma_mod
from embedding_model import GLuCoSE_base_ja_v2 as glucose_mod
from embedding_model import text_embedding_3_large as oai_emb
from embedding_model import bleu as bleu_mod
from embedding_model import nist as nist_mod
from embedding_model import rouge1_f1 as rouge1_mod
from embedding_model import unsup_simcse_ja_large as simcse_mod
from embedding_model import simcse_ja_bert_base_clcmlp as simcse_bert_mod
from embedding_model import sentence_bert_base_ja_mean_tokens_v2 as sbert_v2_mod
from embedding_model import sbert_jsnli_luke_japanese_base_lite as sbert_luke_lite_mod
from embedding_model import sbert_base_ja as sbert_base_mod
from embedding_model import JaColBERTv2_5 as jacolbert25_mod
from embedding_model import JaColBERTv2 as jacolbert2_mod

# 追加: WebLab 10B Instruction SFT ラッパ（既定）
from llm_model import weblab_10b_instruction_sft as llmjp
# 旧: LLM-JP 3.x 13B ラッパ（GPUメモリ要件が高いためデフォルト無効）
# from llm_model import llm_jp_3_1_13b as llmjp

LOGGER = setup_logger("embedding_main_v2")


# ===== main_v1.py 相当の関数群（そのまま収録） =====

def _parse_ver(s: str):
    """'2.7.0' のような文字列を (2,7,0) のタプルに変換（簡易）。非数は 0 扱い。"""
    parts = []
    for p in str(s).split("."):
        num = "".join(ch for ch in p if ch.isdigit())
        parts.append(int(num) if num else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])

def _detect_caps():
    """ライブラリのバージョンと flash-attn の利用可否を検出（失敗は保守的に無効扱い）。"""
    try:
        import sentence_transformers as _st  # type: ignore
        st_ver = _parse_ver(getattr(_st, "__version__", "0.0.0"))
    except Exception:
        st_ver = (0, 0, 0)
    try:
        import transformers as _hf  # type: ignore
        tf_ver = _parse_ver(getattr(_hf, "__version__", "0.0.0"))
    except Exception:
        tf_ver = (0, 0, 0)
    # flash-attn のロード可否（未インストールや不正なビルドは False）
    has_usable_flash_attn = False
    try:
        import importlib  # noqa: F401
        import flash_attn  # type: ignore  # noqa: F401
        has_usable_flash_attn = True
    except Exception:
        has_usable_flash_attn = False
    return st_ver, tf_ver, has_usable_flash_attn

def _read_text(path: str, use_first_line: bool = False) -> str:
    """UTF-8 でファイル読み込み。オプションで最初の非空行のみを使用可能。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"テキストファイルが見つかりません: {path}")
    with open(path, "r", encoding="utf-8") as f:
        if use_first_line:
            for line in f:
                s = line.strip()
                if s:
                    return s
            return ""  # すべて空行だった場合
        return f.read().strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare texts and run an LLM with weather prompts.")
    parser.add_argument(
        "--task",
        type=str,
        default="sts",
        choices=["sts", "retrieval", "reranking", "clustering", "classification"],
        help="Sarashina 推奨のタスク指定（既定: sts は両文を query 形式でエンコード）",
    )
    parser.add_argument(
        "--truncate-dim",
        type=int,
        default=None,
        choices=[32, 64, 128, 256, 512, 1024],
        help="static-embedding-japanese の出力次元を切り詰め（未指定なら1024）",
    )
    parser.add_argument("--use-first-line", action="store_true", help="各ファイルの最初の非空行のみを文章として使用")

    # v2 追加
    parser.add_argument("--run-llm", action="store_true", help="LLM（既定: WebLab 10B）による生成と評価を実行")
    parser.add_argument("--llm-max-new-tokens", type=int, default=320, help="LLM 生成トークン数")
    parser.add_argument("--llm-temperature", type=float, default=0.2, help="LLM 温度")
    parser.add_argument("--skip-v1", action="store_true", help="main_v1 相当の評価（generate vs original）をスキップ")
    return parser.parse_args()


def resolve_pair_paths() -> List[Tuple[str, str, str]]:
    """
    比較対象のペア一覧を返す。
    先頭に自己確認用（同一文同士）のペアを置いたうえで、
    generate_comment と original_comment の日付一致ペアを列挙する。
    各要素は (label, path_a, path_b)。
    """
    gen_dir = os.path.join(_THIS_DIR, "data", "generate_comment")
    org_dir = os.path.join(_THIS_DIR, "data", "original_comment")

    # 自己チェック: 2022_01_01_original.txt を同一文同士で比較
    self_check_path = os.path.join(org_dir, "2022_01_01_original.txt")
    pairs: List[Tuple[str, str, str]] = [("SELF_CHECK_2022_01_01_ORIGINAL", self_check_path, self_check_path)]

    # ディレクトリ走査して日付ごとのファイルを収集
    gen_pat = re.compile(r"^(\d{4}_\d{2}_\d{2})_gpt_generate_v4\.txt$")
    org_pat = re.compile(r"^(\d{4}_\d{2}_\d{2})_original\.txt$")

    gen_map: Dict[str, str] = {}
    org_map: Dict[str, str] = {}

    gen_dir_exists = os.path.isdir(gen_dir)
    org_dir_exists = os.path.isdir(org_dir)

    if gen_dir_exists:
        for fname in os.listdir(gen_dir):
            m = gen_pat.match(fname)
            if m:
                date = m.group(1)
                gen_map[date] = os.path.join(gen_dir, fname)

    if org_dir_exists:
        for fname in os.listdir(org_dir):
            m = org_pat.match(fname)
            if m:
                date = m.group(1)
                org_map[date] = os.path.join(org_dir, fname)

    # 日付の一致を取り、昇順に処理
    for date in sorted(set(gen_map.keys()) & set(org_map.keys())):
        pairs.append((date, gen_map[date], org_map[date]))

    return pairs


def load_human_eval(csv_path: str) -> Dict[Tuple[str, str], float]:
    """
    human_eval.csv を読み込み、(date, model) -> normal_ratio を返す。
    例: key = ("2022_01_01", "gpt v4 Comment:")
    """
    mapping: Dict[Tuple[str, str], float] = {}
    if not os.path.isfile(csv_path):
        return mapping
    try:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                date = (row.get("date") or "").strip()
                model = (row.get("model") or "").strip()
                val_raw = (row.get("normal_ratio") or "").strip()
                if not date or not model or not val_raw:
                    continue
                try:
                    mapping[(date, model)] = float(val_raw)
                except Exception:
                    continue
    except Exception:
        # CSV の読み込みに失敗した場合は空マップを返す
        return mapping
    return mapping


# ===== v1 比較（generate vs original）を実行する関数 =====

def run_v1_evaluation(args: argparse.Namespace) -> None:
    st_ver, tf_ver, has_flash_attn = _detect_caps()
    can_static = st_ver >= (3, 3, 1)          # StaticEmbedding には ST>=3.3.1 が必要
    can_jina = st_ver >= (3, 0, 0)            # Jina v3 は ST>=3 系を前提
    # Embedding Gemma は ST 形式のため、ST>=3.3.1 で trust_remote_code による読込が可能。
    # それ未満の場合のみ Transformers>=5.1 を要求（gemma3_text ネイティブ対応）。
    can_gemma = (st_ver >= (3, 3, 1)) or (tf_ver >= (5, 1, 0))
    # Sarashina(LLaMA系) は TF 4.43 系では flash-attn の import 経路で落ちることがあるため、
    # Transformers>=4.45.0 かつ flash-attn 未検出時のみ許可（保守的）
    can_sarashina = (tf_ver >= (4, 45, 0)) and (not has_flash_attn)

    # デバイス自動選択（GPUがあればCUDA、なければCPU）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOGGER.info(f"[v1] Selected device: {device}")

    result_lines: List[str] = []
    def _print(name: str, val: Optional[float], human_ratio: Optional[float]) -> None:
        if val is None:
            msg = f"[{name}] FAILED"
        else:
            msg = f"[{name}] Score: {val:.6f}"
        if human_ratio is None:
            msg += " | human_eval: N/A"
        else:
            msg += f" | human_eval normal_ratio: {human_ratio:.6f}"
        print(msg)
        result_lines.append(msg)

    # 人手評価CSVを読み込み
    human_csv_path = os.path.join(_THIS_DIR, "data", "human_eval.csv")
    human_map = load_human_eval(human_csv_path)
    CSV_MODEL_KEY = "gpt v4 Comment:"

    # 集計用: 各モデル -> [(human_ratio, model_score), ...]
    compare_data: Dict[str, List[Tuple[float, float]]] = {
        "Sarashina-Embedding-v2-1B": [],
        "Sarashina-Embedding-v1-1B": [],
        "static-embedding-japanese": [],
        "ruri-base": [],
        "ruri-v3-310m": [],
        "PLaMo-Embedding-1B": [],
        "multilingual-e5-base": [],
        "multilingual-e5-large": [],
        "GLuCoSE-base-ja-v2": [],
        "jina-embeddings-v3": [],
        "BGE-M3": [],
        "embeddinggemma-300m": [],
        "text-embedding-3-large": [],
        "unsup-simcse-ja-large": [],
        "simcse-ja-bert-base-clcmlp": [],
        "sentence-bert-base-ja-mean-tokens-v2": [],
        "sbert-base-ja": [],
        "sbert-jsnli-luke-japanese-base-lite": [],
        "BLEU": [],
        "NIST": [],
        "ROUGE-1 F1": [],
    }
    compare_data["JaColBERTv2.5"] = []
    compare_data["JaColBERTv2"] = []

    pairs = resolve_pair_paths()
    if not pairs:
        raise RuntimeError("比較対象のペアが見つかりません。ディレクトリ構成とファイル名を確認してください。")

    for label, path_a, path_b in pairs:
        LOGGER.info("==================================================")
        LOGGER.info(f"[v1] Pair: {label}")
        LOGGER.info(f"[v1] Text A: {path_a}")
        LOGGER.info(f"[v1] Text B: {path_b}")

        # human_eval 正解率（normal_ratio）取得（自己チェックは N/A）
        m_date = re.fullmatch(r"\d{4}_\d{2}_\d{2}", label)
        date_key = m_date.group(0) if m_date else None
        human_ratio = human_map.get((date_key, CSV_MODEL_KEY)) if date_key else None

        try:
            text_a = _read_text(path_a, use_first_line=args.use_first_line)
            text_b = _read_text(path_b, use_first_line=args.use_first_line)
        except Exception:
            LOGGER.exception("[v1] テキスト読み込みに失敗しました")
            print(f"[{label}] FAILED to read texts")
            continue

        if not text_a or not text_b:
            print(f"[{label}] Skipped: 入力文章が空です。--use-first-line を外す/付けるなど調整してください。")
            continue

        sim_sar_v1 = sim_sar = sim_sta = sim_ruri = sim_ruri_v3 = sim_plamo = sim_e5 = sim_e5_large = sim_jina = sim_bge = sim_gemma = sim_glucose = sim_openai = sim_simcse = sim_simcse_bert = None
        sim_sbert = sim_sbert_base = None
        sim_sbert_luke = None
        sim_bleu = sim_nist = sim_rouge = None

        sim_jacolbert25 = sim_jacolbert2 = None

        if can_sarashina:
            try:
                sim_sar = sarashina.embed_and_score(text_a, text_b, device=device, task=args.task)
            except Exception:
                LOGGER.exception("[v1] Sarashina-Embedding-v2-1B failed")
            try:
                sim_sar_v1 = sarashina_v1_mod.embed_and_score(text_a, text_b, device=device, task=args.task)
            except Exception:
                LOGGER.exception("[v1] Sarashina-Embedding-v1-1B failed")
        else:
            LOGGER.warning("[v1] Sarashina-Embedding-v2-1B skipped: flash-attn と非互換の可能性のためスキップ")
            print("[Sarashina-Embedding-v2-1B] SKIPPED")
            print("[Sarashina-Embedding-v1-1B] SKIPPED")

        if (st_ver := _detect_caps()[0]) >= (3, 3, 1):
            try:
                sim_sta = static_mod.embed_and_score(text_a, text_b, device=device, truncate_dim=args.truncate_dim)
            except Exception:
                LOGGER.exception("[v1] static-embedding-japanese failed")
        else:
            LOGGER.warning("[v1] static-embedding-japanese skipped: sentence-transformers>=3.3.1 が必要です。")
            print("[static-embedding-japanese] SKIPPED (requires sentence-transformers>=3.3.1)")

        try:
            sim_ruri = ruri_mod.embed_and_score(text_a, text_b, device=device, task=args.task)
        except Exception:
            LOGGER.exception("[v1] ruri-base failed")

        try:
            sim_ruri_v3 = ruri3_mod.embed_and_score(text_a, text_b, device=device, task=args.task)
        except Exception:
            LOGGER.exception("[v1] ruri-v3-310m failed")

        try:
            sim_plamo = plamo_mod.embed_and_score(text_a, text_b, device=device, task=args.task)
        except Exception:
            LOGGER.exception("[v1] PLaMo-Embedding-1B failed")

        try:
            sim_e5 = e5_mod.embed_and_score(text_a, text_b, device=device, task=args.task)
        except Exception:
            LOGGER.exception("[v1] multilingual-e5-base failed")
        try:
            sim_e5_large = e5_large_mod.embed_and_score(text_a, text_b, device=device, task=args.task)
        except Exception:
            LOGGER.exception("[v1] multilingual-e5-large failed")

        try:
            sim_glucose = glucose_mod.embed_and_score(text_a, text_b, device=device, task=args.task)
        except Exception:
            LOGGER.exception("[v1] GLuCoSE-base-ja-v2 failed")

        try:
            sim_openai = oai_emb.embed_and_score(text_a, text_b, dimensions=args.truncate_dim, device=device)
        except Exception:
            LOGGER.exception("[v1] text-embedding-3-large failed")

        if (st_ver := _detect_caps()[0]) >= (3, 0, 0):
            try:
                sim_jina = jina_mod.embed_and_score(text_a, text_b, device=device, task=args.task, truncate_dim=args.truncate_dim)
            except Exception:
                LOGGER.exception("[v1] jina-embeddings-v3 failed")
        else:
            LOGGER.warning("[v1] jina-embeddings-v3 skipped: sentence-transformers>=3.0 が必要です。")
            print("[jina-embeddings-v3] SKIPPED (requires sentence-transformers>=3.0)")

        try:
            sim_bge = bge_m3_mod.embed_and_score(text_a, text_b, device=device, task=args.task)
        except Exception:
            LOGGER.exception("[v1] BGE-M3 failed")

        try:
            sim_jacolbert25 = jacolbert25_mod.embed_and_score(text_a, text_b, device=device, task=args.task)
        except Exception as e:
            LOGGER.warning(f"[v1] JaColBERTv2.5 skipped: {e}")
        try:
            sim_jacolbert2 = jacolbert2_mod.embed_and_score(text_a, text_b, device=device, task=args.task)
        except Exception as e:
            LOGGER.warning(f"[v1] JaColBERTv2 skipped: {e}")

        if (caps := _detect_caps())[0] >= (3, 3, 1) or caps[1] >= (5, 1, 0):
            try:
                sim_gemma = gemma_mod.embed_and_score(
                    text_a, text_b, device=device, task=args.task, truncate_dim=args.truncate_dim
                )
            except Exception:
                LOGGER.exception("[v1] embeddinggemma-300m failed")
        else:
            LOGGER.warning("[v1] embeddinggemma-300m skipped: ST>=3.3.1 または TF>=5.1 が必要です。")
            print("[embeddinggemma-300m] SKIPPED (requires Sentence-Transformers>=3.3.1 or Transformers>=5.1)")

        try:
            sim_simcse = simcse_mod.embed_and_score(text_a, text_b, device=device, task=args.task)
        except Exception:
            LOGGER.exception("[v1] unsup-simcse-ja-large failed")

        try:
            sim_simcse_bert = simcse_bert_mod.embed_and_score(text_a, text_b, device=device, task=args.task)
        except Exception:
            LOGGER.exception("[v1] simcse-ja-bert-base-clcmlp failed")

        try:
            sim_sbert = sbert_v2_mod.embed_and_score(text_a, text_b, device=device, task=args.task)
        except Exception:
            LOGGER.exception("[v1] sentence-bert-base-ja-mean-tokens-v2 failed")

        try:
            sim_sbert_base = sbert_base_mod.embed_and_score(text_a, text_b, device=device, task=args.task)
        except Exception:
            LOGGER.exception("[v1] sbert-base-ja failed")

        try:
            sim_sbert_luke = sbert_luke_lite_mod.embed_and_score(text_a, text_b, device=device, task=args.task)
        except Exception:
            LOGGER.exception("[v1] sbert-jsnli-luke-japanese-base-lite failed")

        try:
            sim_bleu = bleu_mod.embed_and_score(text_a, text_b, device=device, task=args.task, truncate_dim=args.truncate_dim)
        except Exception:
            LOGGER.exception("[v1] BLEU failed")
        try:
            sim_nist = nist_mod.embed_and_score(text_a, text_b, device=device, task=args.task, truncate_dim=args.truncate_dim)
        except Exception:
            LOGGER.exception("[v1] NIST failed")
        try:
            sim_rouge = rouge1_mod.embed_and_score(text_a, text_b, device=device, task=args.task, truncate_dim=args.truncate_dim)
        except Exception:
            LOGGER.exception("[v1] ROUGE-1 F1 failed")

        # 人手評価（ある日付のみ）
        if human_ratio is not None:
            if sim_sar is not None: compare_data["Sarashina-Embedding-v2-1B"].append((human_ratio, sim_sar))
            if sim_sar_v1 is not None: compare_data["Sarashina-Embedding-v1-1B"].append((human_ratio, sim_sar_v1))
            if sim_sta is not None: compare_data["static-embedding-japanese"].append((human_ratio, sim_sta))
            if sim_ruri is not None: compare_data["ruri-base"].append((human_ratio, sim_ruri))
            if sim_ruri_v3 is not None: compare_data["ruri-v3-310m"].append((human_ratio, sim_ruri_v3))
            if sim_plamo is not None: compare_data["PLaMo-Embedding-1B"].append((human_ratio, sim_plamo))
            if sim_e5 is not None: compare_data["multilingual-e5-base"].append((human_ratio, sim_e5))
            if sim_e5_large is not None: compare_data["multilingual-e5-large"].append((human_ratio, sim_e5_large))
            if sim_glucose is not None: compare_data["GLuCoSE-base-ja-v2"].append((human_ratio, sim_glucose))
            if sim_jina is not None: compare_data["jina-embeddings-v3"].append((human_ratio, sim_jina))
            if sim_bge is not None: compare_data["BGE-M3"].append((human_ratio, sim_bge))
            if sim_jacolbert25 is not None: compare_data["JaColBERTv2.5"].append((human_ratio, sim_jacolbert25))
            if sim_jacolbert2 is not None: compare_data["JaColBERTv2"].append((human_ratio, sim_jacolbert2))
            if sim_gemma is not None: compare_data["embeddinggemma-300m"].append((human_ratio, sim_gemma))
            if sim_openai is not None: compare_data["text-embedding-3-large"].append((human_ratio, sim_openai))
            if sim_simcse is not None: compare_data["unsup-simcse-ja-large"].append((human_ratio, sim_simcse))
            if sim_simcse_bert is not None: compare_data["simcse-ja-bert-base-clcmlp"].append((human_ratio, sim_simcse_bert))
            if sim_sbert is not None: compare_data["sentence-bert-base-ja-mean-tokens-v2"].append((human_ratio, sim_sbert))
            if sim_sbert_base is not None: compare_data["sbert-base-ja"].append((human_ratio, sim_sbert_base))
            if sim_sbert_luke is not None: compare_data["sbert-jsnli-luke-japanese-base-lite"].append((human_ratio, sim_sbert_luke))
            if sim_bleu is not None: compare_data["BLEU"].append((human_ratio, sim_bleu))
            if sim_nist is not None: compare_data["NIST"].append((human_ratio, sim_nist))
            if sim_rouge is not None: compare_data["ROUGE-1 F1"].append((human_ratio, sim_rouge))

        print(f"----- Results: {label} -----")
        result_lines.append(f"----- Results: {label} -----")
        result_lines.append(f"Text A path: {path_a}")
        result_lines.append(f"Text B path: {path_b}")
        result_lines.append("Text A (used):")
        result_lines.append(text_a)
        result_lines.append("Text B (used):")
        result_lines.append(text_b)
        _print("Sarashina-Embedding-v1-1B", sim_sar_v1, human_ratio)
        _print("Sarashina-Embedding-v2-1B", sim_sar, human_ratio)
        _print("static-embedding-japanese", sim_sta, human_ratio)
        _print("ruri-base", sim_ruri, human_ratio)
        _print("ruri-v3-310m", sim_ruri_v3, human_ratio)
        _print("PLaMo-Embedding-1B", sim_plamo, human_ratio)
        _print("multilingual-e5-base", sim_e5, human_ratio)
        _print("multilingual-e5-large", sim_e5_large, human_ratio)
        _print("GLuCoSE-base-ja-v2", sim_glucose, human_ratio)
        _print("jina-embeddings-v3", sim_jina, human_ratio)
        _print("BGE-M3", sim_bge, human_ratio)
        _print("JaColBERTv2.5", sim_jacolbert25, human_ratio)
        _print("JaColBERTv2", sim_jacolbert2, human_ratio)
        _print("embeddinggemma-300m", sim_gemma, human_ratio)
        _print("text-embedding-3-large", sim_openai, human_ratio)
        _print("unsup-simcse-ja-large", sim_simcse, human_ratio)
        _print("simcse-ja-bert-base-clcmlp", sim_simcse_bert, human_ratio)
        _print("sentence-bert-base-ja-mean-tokens-v2", sim_sbert, human_ratio)
        _print("sbert-base-ja", sim_sbert_base, human_ratio)
        _print("sbert-jsnli-luke-japanese-base-lite", sim_sbert_luke, human_ratio)
        _print("BLEU", sim_bleu, human_ratio)
        _print("NIST", sim_nist, human_ratio)
        _print("ROUGE-1 F1", sim_rouge, human_ratio)
        print("")

    # ===== 集約サマリ: human_eval(normal_ratio) と各モデルの相関 =====
    def _pearson(pairs: List[Tuple[float, float]]) -> Optional[float]:
        n = len(pairs)
        if n < 2:
            return None
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
        den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
        if den_x == 0 or den_y == 0:
            return None
        return num / (den_x * den_y)

    corr_results: Dict[str, Optional[float]] = {}
    for model_name, data in compare_data.items():
        corr_results[model_name] = _pearson(data)

    best_model = None
    best_corr = None
    for name, r in corr_results.items():
        if r is None:
            continue
        if (best_corr is None) or (r > best_corr):
            best_model = name
            best_corr = r

    print("===== Summary: human_eval 正解率との相関（Pearson） ランキング順=====")
    result_lines.append("===== Summary: human_eval 正解率との相関（Pearson） ランキング順=====")
    result_lines.append("Pearson correlation formula:")
    result_lines.append("r = sum_i (x_i - mean_x) * (y_i - mean_y) / sqrt( sum_i (x_i - mean_x)^2 * sum_i (y_i - mean_y)^2 )")
    result_lines.append("where x_i = human_eval normal_ratio for date i, y_i = model score for the same date i.")
    valid = [(name, r) for name, r in corr_results.items() if r is not None]
    valid.sort(key=lambda x: x[1], reverse=True)
    for idx, (name, r) in enumerate(valid, start=1):
        line = f"{idx}. {name}: Pearson r = {r:.6f}"
        print(line)
        result_lines.append(line)

    na_models = [name for name, r in corr_results.items() if r is None]
    if na_models:
        line = "- N/A: " + ", ".join(na_models)
        print(line)
        result_lines.append(line)

    if best_model is None:
        print("最も人手評価に近いモデル: 判定不可（有効な比較データが不足）")
        result_lines.append("最も人手評価に近いモデル: 判定不可（有効な比較データが不足）")
    else:
        print(f"最も人手評価と似ている評価を行ったモデル: {best_model} (Pearson r={best_corr:.6f})")
        result_lines.append(f"最も人手評価と似ている評価を行ったモデル: {best_model} (Pearson r={best_corr:.6f})")

    expl = "\n算出方法: 各日付に対して、人手評価CSVの normal_ratio（model='gpt v4 Comment:'）と、各埋め込みモデルおよびテキスト指標（BLEU/NIST/ROUGE-1 F1）のスコアとの間でピアソン相関係数を計算。相関が最も高いものを『最も人手評価と似ている』と定義。スケール差の影響を受けにくく、順位関係の整合性を捉えられるため相関を採用。比較は human_eval が存在する日付のみで行い、SELF_CHECK は除外。"
    print(expl)
    result_lines.append(expl)

    try:
        out_path = os.path.join(_THIS_DIR, "result_v1.log")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(result_lines) + "\n")
        print(f"[results] Written to: {out_path}")
    except Exception as e:
        print(f"[results] Failed to write result_v1.log: {e}")


# ===== LLM 生成 + 評価（v2 追加） =====

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _date_key_from_numeric_filename(fname: str) -> Optional[str]:
    """
    '2022-1-1.txt' のようなファイル名から '2022_01_01' に変換。
    """
    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})\.txt", fname)
    if not m:
        return None
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return f"{y:04d}_{mo:02d}_{d:02d}"

def _scan_comment_maps() -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    return (gen_map, org_map)
    - gen_map: YYYY_MM_DD -> ./data/generate_comment/{date}_gpt_generate_v4.txt
    - org_map: YYYY_MM_DD -> ./data/original_comment/{date}_original.txt
    """
    gen_dir = os.path.join(_THIS_DIR, "data", "generate_comment")
    org_dir = os.path.join(_THIS_DIR, "data", "original_comment")
    gen_pat = re.compile(r"^(\d{4}_\d{2}_\d{2})_gpt_generate_v4\.txt$")
    org_pat = re.compile(r"^(\d{4}_\d{2}_\d{2})_original\.txt$")

    gen_map: Dict[str, str] = {}
    org_map: Dict[str, str] = {}

    if os.path.isdir(gen_dir):
        for fname in os.listdir(gen_dir):
            m = gen_pat.match(fname)
            if m:
                date = m.group(1)
                gen_map[date] = os.path.join(gen_dir, fname)
    if os.path.isdir(org_dir):
        for fname in os.listdir(org_dir):
            m = org_pat.match(fname)
            if m:
                date = m.group(1)
                org_map[date] = os.path.join(org_dir, fname)
    return gen_map, org_map


def _score_and_log_all_models(
    text_a: str,
    text_b: str,
    args: argparse.Namespace,
    device: str,
    can_sarashina: bool,
    can_static: bool,
    can_jina: bool,
    can_gemma: bool,
    result_lines: List[str],
    human_ratio: Optional[float] = None,
) -> None:
    """
    main_v1 と同等の全モデルスコア算出とログ追加。
    """
    def _print(name: str, val: Optional[float], human_ratio: Optional[float]) -> None:
        if val is None:
            msg = f"[{name}] FAILED"
        else:
            msg = f"[{name}] Score: {val:.6f}"
        if human_ratio is None:
            msg += " | human_eval: N/A"
        else:
            msg += f" | human_eval normal_ratio: {human_ratio:.6f}"
        print(msg)
        result_lines.append(msg)

    sim_sar_v1 = sim_sar = sim_sta = sim_ruri = sim_ruri_v3 = sim_plamo = sim_e5 = sim_e5_large = sim_jina = sim_bge = sim_gemma = sim_glucose = sim_openai = sim_simcse = sim_simcse_bert = None
    sim_sbert = sim_sbert_base = None
    sim_sbert_luke = None
    sim_bleu = sim_nist = sim_rouge = None
    sim_jacolbert25 = sim_jacolbert2 = None

    if can_sarashina:
        try:
            sim_sar = sarashina.embed_and_score(text_a, text_b, device=device, task=args.task)
        except Exception:
            LOGGER.exception("Sarashina-Embedding-v2-1B failed")
        try:
            sim_sar_v1 = sarashina_v1_mod.embed_and_score(text_a, text_b, device=device, task=args.task)
        except Exception:
            LOGGER.exception("Sarashina-Embedding-v1-1B failed")
    else:
        LOGGER.warning("Sarashina-Embedding skipped")
        print("[Sarashina-Embedding-v2-1B] SKIPPED")
        print("[Sarashina-Embedding-v1-1B] SKIPPED")

    if can_static:
        try:
            sim_sta = static_mod.embed_and_score(text_a, text_b, device=device, truncate_dim=args.truncate_dim)
        except Exception:
            LOGGER.exception("static-embedding-japanese failed")
    else:
        LOGGER.warning("static-embedding-japanese skipped: sentence-transformers>=3.3.1 が必要です。")
        print("[static-embedding-japanese] SKIPPED (requires sentence-transformers>=3.3.1)")

    try:
        sim_ruri = ruri_mod.embed_and_score(text_a, text_b, device=device, task=args.task)
    except Exception:
        LOGGER.exception("ruri-base failed")

    try:
        sim_ruri_v3 = ruri3_mod.embed_and_score(text_a, text_b, device=device, task=args.task)
    except Exception:
        LOGGER.exception("ruri-v3-310m failed")

    try:
        sim_plamo = plamo_mod.embed_and_score(text_a, text_b, device=device, task=args.task)
    except Exception:
        LOGGER.exception("PLaMo-Embedding-1B failed")

    try:
        sim_e5 = e5_mod.embed_and_score(text_a, text_b, device=device, task=args.task)
    except Exception:
        LOGGER.exception("multilingual-e5-base failed")
    try:
        sim_e5_large = e5_large_mod.embed_and_score(text_a, text_b, device=device, task=args.task)
    except Exception:
        LOGGER.exception("multilingual-e5-large failed")

    try:
        sim_glucose = glucose_mod.embed_and_score(text_a, text_b, device=device, task=args.task)
    except Exception:
        LOGGER.exception("GLuCoSE-base-ja-v2 failed")

    try:
        sim_openai = oai_emb.embed_and_score(text_a, text_b, dimensions=args.truncate_dim, device=device)
    except Exception:
        LOGGER.exception("text-embedding-3-large failed")

    if can_jina:
        try:
            sim_jina = jina_mod.embed_and_score(text_a, text_b, device=device, task=args.task, truncate_dim=args.truncate_dim)
        except Exception:
            LOGGER.exception("jina-embeddings-v3 failed")
    else:
        LOGGER.warning("jina-embeddings-v3 skipped: sentence-transformers>=3.0 が必要です。")
        print("[jina-embeddings-v3] SKIPPED (requires sentence-transformers>=3.0)")

    try:
        sim_bge = bge_m3_mod.embed_and_score(text_a, text_b, device=device, task=args.task)
    except Exception:
        LOGGER.exception("BGE-M3 failed")

    try:
        sim_jacolbert25 = jacolbert25_mod.embed_and_score(text_a, text_b, device=device, task=args.task)
    except Exception as e:
        LOGGER.warning(f"JaColBERTv2.5 skipped: {e}")
    try:
        sim_jacolbert2 = jacolbert2_mod.embed_and_score(text_a, text_b, device=device, task=args.task)
    except Exception as e:
        LOGGER.warning(f"JaColBERTv2 skipped: {e}")

    if can_gemma:
        try:
            sim_gemma = gemma_mod.embed_and_score(
                text_a, text_b, device=device, task=args.task, truncate_dim=args.truncate_dim
            )
        except Exception:
            LOGGER.exception("embeddinggemma-300m failed")
    else:
        LOGGER.warning("embeddinggemma-300m skipped: ST>=3.3.1 または TF>=5.1 が必要です。")
        print("[embeddinggemma-300m] SKIPPED (requires Sentence-Transformers>=3.3.1 or Transformers>=5.1)")

    try:
        sim_simcse = simcse_mod.embed_and_score(text_a, text_b, device=device, task=args.task)
    except Exception:
        LOGGER.exception("unsup-simcse-ja-large failed")

    try:
        sim_simcse_bert = simcse_bert_mod.embed_and_score(text_a, text_b, device=device, task=args.task)
    except Exception:
        LOGGER.exception("simcse-ja-bert-base-clcmlp failed")

    try:
        sim_sbert = sbert_v2_mod.embed_and_score(text_a, text_b, device=device, task=args.task)
    except Exception:
        LOGGER.exception("sentence-bert-base-ja-mean-tokens-v2 failed")

    try:
        sim_sbert_base = sbert_base_mod.embed_and_score(text_a, text_b, device=device, task=args.task)
    except Exception:
        LOGGER.exception("sbert-base-ja failed")

    try:
        sim_sbert_luke = sbert_luke_lite_mod.embed_and_score(text_a, text_b, device=device, task=args.task)
    except Exception:
        LOGGER.exception("sbert-jsnli-luke-japanese-base-lite failed")

    try:
        sim_bleu = bleu_mod.embed_and_score(text_a, text_b, device=device, task=args.task, truncate_dim=args.truncate_dim)
    except Exception:
        LOGGER.exception("BLEU failed")
    try:
        sim_nist = nist_mod.embed_and_score(text_a, text_b, device=device, task=args.task, truncate_dim=args.truncate_dim)
    except Exception:
        LOGGER.exception("NIST failed")
    try:
        sim_rouge = rouge1_mod.embed_and_score(text_a, text_b, device=device, task=args.task, truncate_dim=args.truncate_dim)
    except Exception:
        LOGGER.exception("ROUGE-1 F1 failed")

    _print("Sarashina-Embedding-v1-1B", sim_sar_v1, human_ratio)
    _print("Sarashina-Embedding-v2-1B", sim_sar, human_ratio)
    _print("static-embedding-japanese", sim_sta, human_ratio)
    _print("ruri-base", sim_ruri, human_ratio)
    _print("ruri-v3-310m", sim_ruri_v3, human_ratio)
    _print("PLaMo-Embedding-1B", sim_plamo, human_ratio)
    _print("multilingual-e5-base", sim_e5, human_ratio)
    _print("multilingual-e5-large", sim_e5_large, human_ratio)
    _print("GLuCoSE-base-ja-v2", sim_glucose, human_ratio)
    _print("jina-embeddings-v3", sim_jina, human_ratio)
    _print("BGE-M3", sim_bge, human_ratio)
    _print("JaColBERTv2.5", sim_jacolbert25, human_ratio)
    _print("JaColBERTv2", sim_jacolbert2, human_ratio)
    _print("embeddinggemma-300m", sim_gemma, human_ratio)
    _print("text-embedding-3-large", sim_openai, human_ratio)
    _print("unsup-simcse-ja-large", sim_simcse, human_ratio)
    _print("simcse-ja-bert-base-clcmlp", sim_simcse_bert, human_ratio)
    _print("sentence-bert-base-ja-mean-tokens-v2", sim_sbert, human_ratio)
    _print("sbert-base-ja", sim_sbert_base, human_ratio)
    _print("sbert-jsnli-luke-japanese-base-lite", sim_sbert_luke, human_ratio)
    _print("BLEU", sim_bleu, human_ratio)
    _print("NIST", sim_nist, human_ratio)
    _print("ROUGE-1 F1", sim_rouge, human_ratio)


def run_llm_generation_and_evaluation(args: argparse.Namespace) -> None:
    """
    - data/Numerical_weather_data の各 .txt と data/prompt_gpt/v4_instruction.txt を連結し LLM 実行
    - 出力を保存
    - original_comment / generate_comment との類似度を main_v1 と同じ評価系で算出
    """
    # デバイスと能力検出は v1 に合わせる
    st_ver, tf_ver, has_flash_attn = _detect_caps()
    can_static = st_ver >= (3, 3, 1)
    can_jina = st_ver >= (3, 0, 0)
    can_gemma = (st_ver >= (3, 3, 1)) or (tf_ver >= (5, 1, 0))
    can_sarashina = (tf_ver >= (4, 45, 0)) and (not has_flash_attn)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOGGER.info(f"[v2] Selected device: {device}")

    numeric_dir = os.path.join(_THIS_DIR, "data", "Numerical_weather_data")
    instruction_path = os.path.join(_THIS_DIR, "data", "prompt_gpt", "v4_instruction.txt")
    out_dir = os.path.join(_THIS_DIR, "data", "llm_outputs", "weblab10b")
    _ensure_dir(out_dir)

    # コメントの参照パスをマップ化
    gen_map, org_map = _scan_comment_maps()

    # LLM 生成
    produced_map: Dict[str, str] = {}  # date_key -> out_path
    if not os.path.isdir(numeric_dir):
        raise RuntimeError(f"Numerical_weather_data ディレクトリが見つかりません: {numeric_dir}")

    for fname in sorted(os.listdir(numeric_dir)):
        date_key = _date_key_from_numeric_filename(fname)
        if not date_key:
            # 想定外のファイル名はスキップ
            continue
        numeric_path = os.path.join(numeric_dir, fname)
        LOGGER.info(f"[v2] Generating for {date_key} from: {numeric_path}")
        try:
            text_out = llmjp.generate_from_files(
                numeric_data_path=numeric_path,
                instruction_path=instruction_path,
                max_new_tokens=args.llm_max_new_tokens,
                temperature=args.llm_temperature,
            )
        except Exception:
            LOGGER.exception(f"[v2] LLM generation failed: {date_key}")
            continue
        out_path = os.path.join(out_dir, f"{date_key}_weblab10b.txt")
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text_out.strip() + "\n")
            produced_map[date_key] = out_path
            LOGGER.info(f"[v2] Written: {out_path}")
        except Exception:
            LOGGER.exception(f"[v2] Failed to write output: {out_path}")

    if not produced_map:
        print("[v2] 生成結果がありません。ファイル名形式やモデルのロードに失敗していないか確認してください。")
        return

    # 評価
    result_lines: List[str] = []
    human_csv_path = os.path.join(_THIS_DIR, "data", "human_eval.csv")
    human_map = load_human_eval(human_csv_path)

    for date_key, llm_path in sorted(produced_map.items()):
        try:
            llm_text = _read_text(llm_path, use_first_line=False)
        except Exception:
            LOGGER.exception(f"[v2] 読み込み失敗: {llm_path}")
            continue

        # 1) LLM vs ORIGINAL
        org_path = org_map.get(date_key)
        if org_path and os.path.isfile(org_path):
            try:
                org_text = _read_text(org_path, use_first_line=False)
            except Exception:
                LOGGER.exception(f"[v2] original 読み込み失敗: {org_path}")
                org_text = ""
            if org_text:
                human_ratio = human_map.get((date_key, "gpt v4 Comment:"))  # human_eval は gpt v4 参照のみ
                print(f"----- LLM vs ORIGINAL Results: {date_key} -----")
                result_lines.append(f"----- LLM vs ORIGINAL Results: {date_key} -----")
                result_lines.append(f"LLM out path: {llm_path}")
                result_lines.append(f"Original path: {org_path}")
                result_lines.append("LLM text (used):")
                result_lines.append(llm_text)
                result_lines.append("Original text (used):")
                result_lines.append(org_text)
                _score_and_log_all_models(
                    text_a=llm_text,
                    text_b=org_text,
                    args=args,
                    device=device,
                    can_sarashina=can_sarashina,
                    can_static=can_static,
                    can_jina=can_jina,
                    can_gemma=can_gemma,
                    result_lines=result_lines,
                    human_ratio=human_ratio,
                )
                print("")

        # 2) LLM vs GPTv4 generate_comment
        gen_path = gen_map.get(date_key)
        if gen_path and os.path.isfile(gen_path):
            try:
                gen_text = _read_text(gen_path, use_first_line=False)
            except Exception:
                LOGGER.exception(f"[v2] generate_comment 読み込み失敗: {gen_path}")
                gen_text = ""
            if gen_text:
                human_ratio = human_map.get((date_key, "gpt v4 Comment:"))
                print(f"----- LLM vs GPTv4 Results: {date_key} -----")
                result_lines.append(f"----- LLM vs GPTv4 Results: {date_key} -----")
                result_lines.append(f"LLM out path: {llm_path}")
                result_lines.append(f"GPTv4 path: {gen_path}")
                result_lines.append("LLM text (used):")
                result_lines.append(llm_text)
                result_lines.append("GPTv4 text (used):")
                result_lines.append(gen_text)
                _score_and_log_all_models(
                    text_a=llm_text,
                    text_b=gen_text,
                    args=args,
                    device=device,
                    can_sarashina=can_sarashina,
                    can_static=can_static,
                    can_jina=can_jina,
                    can_gemma=can_gemma,
                    result_lines=result_lines,
                    human_ratio=human_ratio,
                )
                print("")

    # LLM 評価結果の保存
    try:
        out_path = os.path.join(_THIS_DIR, "result_v2.log")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(result_lines) + "\n")
        print(f"[v2 results] Written to: {out_path}")
    except Exception as e:
        print(f"[v2 results] Failed to write result_v2.log: {e}")


def main():
    args = parse_args()

    # LLM 実行（指定時）
    if args.run_llm:
        run_llm_generation_and_evaluation(args)

    # 従来の v1 評価（スキップ指定が無ければ実行）
    if not args.skip_v1:
        run_v1_evaluation(args)


if __name__ == "__main__":
    main()
