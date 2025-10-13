# -*- coding: utf-8 -*-
"""
Sarashina-Embedding-v2-1B を使って2つの文章の類似度（コサイン類似度）を算出するスクリプト。

要件:
- ラッパーモジュール: model/Sarashina_Embedding_v2_1B.py を用いる
  （通常の import を使用）
- 比較対象の2文章は以下のファイルから読み込む（本スクリプトファイル基準の相対パス）
  ./data/generate_comment/2022_01_01_gpt_generate_v4.txt
  ./data/original_comment/2022_01_01_original.txt
- 出力: 類似度スコア（-1..1, コサイン類似度）を標準出力へ

実行例:
nohup python main_v1.py > main_v1.log 2>&1 &

"""

import os
import sys
import argparse
import importlib.util
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

# モデルラッパを通常 import
from model import Sarashina_Embedding_v2_1B as sarashina
from model import static_embedding_japanese as static_mod
from model import ruri_base as ruri_mod
from model import PLaMo_Embedding_1B as plamo_mod
from model import Multilingual_E5_base as e5_mod
from model import jina_embeddings_v3 as jina_mod
from model import BGE_M3 as bge_m3_mod
from model import embeddinggemma_300m as gemma_mod
from model import GLuCoSE_base_ja_v2 as glucose_mod
from model import text_embedding_3_large as oai_emb
from model import bleu as bleu_mod
from model import nist as nist_mod
from model import rouge1_f1 as rouge1_mod

LOGGER = setup_logger("embedding_main")

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
    parser = argparse.ArgumentParser(description="Compare two Japanese texts using Sarashina-Embedding-v2-1B.")
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


def main():
    args = parse_args()

    # デバイス自動選択（GPUがあればCUDA、なければCPU）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOGGER.info(f"Selected device: {device}")

    # ライブラリ互換性に基づくモデル有効化フラグ（全ペア共通で一度だけ判定）
    st_ver, tf_ver, has_flash_attn = _detect_caps()
    can_static = st_ver >= (3, 3, 1)          # StaticEmbedding には ST>=3.3.1 が必要
    can_jina = st_ver >= (3, 0, 0)            # Jina v3 は ST>=3 系を前提
    # Embedding Gemma は ST 形式のため、ST>=3.3.1 で trust_remote_code による読込が可能。
    # それ未満の場合のみ Transformers>=5.1 を要求（gemma3_text ネイティブ対応）。
    can_gemma = (st_ver >= (3, 3, 1)) or (tf_ver >= (5, 1, 0))
    # Sarashina(LLaMA系) は TF 4.43 系では flash-attn の import 経路で落ちることがあるため、
    # Transformers>=4.45.0 かつ flash-attn 未検出時のみ許可（保守的）
    can_sarashina = (tf_ver >= (4, 45, 0)) and (not has_flash_attn)

    # 結果のみを収集してファイル出力するためのバッファ
    result_lines: List[str] = []

    # 類似度を各モデルで算出（各モデルの失敗は他へ波及させない）
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
        "static-embedding-japanese": [],
        "ruri-base": [],
        "PLaMo-Embedding-1B": [],
        "multilingual-e5-base": [],
        "GLuCoSE-base-ja-v2": [],
        "jina-embeddings-v3": [],
        "BGE-M3": [],
        "embeddinggemma-300m": [],
        "text-embedding-3-large": [],
        "BLEU": [],
        "NIST": [],
        "ROUGE-1 F1": [],
    }

    pairs = resolve_pair_paths()
    if not pairs:
        raise RuntimeError("比較対象のペアが見つかりません。ディレクトリ構成とファイル名を確認してください。")

    for label, path_a, path_b in pairs:
        LOGGER.info("==================================================")
        LOGGER.info(f"Pair: {label}")
        LOGGER.info(f"Text A: {path_a}")
        LOGGER.info(f"Text B: {path_b}")

        # human_eval 正解率（normal_ratio）取得（自己チェックは N/A）
        m_date = re.fullmatch(r"\d{4}_\d{2}_\d{2}", label)
        date_key = m_date.group(0) if m_date else None
        human_ratio = human_map.get((date_key, CSV_MODEL_KEY)) if date_key else None

        try:
            text_a = _read_text(path_a, use_first_line=args.use_first_line)
            text_b = _read_text(path_b, use_first_line=args.use_first_line)
        except Exception:
            LOGGER.exception("テキスト読み込みに失敗しました")
            print(f"[{label}] FAILED to read texts")
            continue

        if not text_a or not text_b:
            print(f"[{label}] Skipped: 入力文章が空です。--use-first-line を外す/付けるなど調整してください。")
            continue

        sim_sar = sim_sta = sim_ruri = sim_plamo = sim_e5 = sim_jina = sim_bge = sim_gemma = sim_glucose = sim_openai = None
        sim_bleu = sim_nist = sim_rouge = None

        if can_sarashina:
            try:
                sim_sar = sarashina.embed_and_score(text_a, text_b, device=device, task=args.task)
            except Exception:
                LOGGER.exception("Sarashina-Embedding-v2-1B failed")
        else:
            LOGGER.warning("Sarashina-Embedding-v2-1B skipped: flash-attn が現在の PyTorch/CUDA と非互換の可能性が高いためスキップします。")
            print("[Sarashina-Embedding-v2-1B] SKIPPED")

        if can_static:
            try:
                sim_sta = static_mod.embed_and_score(text_a, text_b, device=device, truncate_dim=args.truncate_dim)
            except Exception:
                LOGGER.exception("static-embedding-japanese failed")
        else:
            LOGGER.warning("static-embedding-japanese skipped: sentence-transformers>=3.3.1 が必要です（現行環境では非対応）。")
            print("[static-embedding-japanese] SKIPPED (requires sentence-transformers>=3.3.1)")

        try:
            sim_ruri = ruri_mod.embed_and_score(text_a, text_b, device=device, task=args.task)
        except Exception:
            LOGGER.exception("ruri-base failed")

        try:
            sim_plamo = plamo_mod.embed_and_score(text_a, text_b, device=device, task=args.task)
        except Exception:
            LOGGER.exception("PLaMo-Embedding-1B failed")

        try:
            sim_e5 = e5_mod.embed_and_score(text_a, text_b, device=device, task=args.task)
        except Exception:
            LOGGER.exception("multilingual-e5-base failed")

        # GLuCoSE-base-ja-v2
        try:
            sim_glucose = glucose_mod.embed_and_score(text_a, text_b, device=device, task=args.task)
        except Exception:
            LOGGER.exception("GLuCoCE-base-ja-v2 failed")

        # OpenAI text-embedding-3-large (API)
        try:
            # dimensions には --truncate-dim の値をそのまま利用（未指定ならデフォルト次元=3072）
            sim_openai = oai_emb.embed_and_score(text_a, text_b, dimensions=args.truncate_dim, device=device)
        except Exception:
            LOGGER.exception("text-embedding-3-large failed")

        if can_jina:
            try:
                sim_jina = jina_mod.embed_and_score(text_a, text_b, device=device, task=args.task, truncate_dim=args.truncate_dim)
            except Exception:
                LOGGER.exception("jina-embeddings-v3 failed")
        else:
            LOGGER.warning("jina-embeddings-v3 skipped: sentence-transformers>=3.0 が必要です（現行環境では非対応）。")
            print("[jina-embeddings-v3] SKIPPED (requires sentence-transformers>=3.0)")

        try:
            sim_bge = bge_m3_mod.embed_and_score(text_a, text_b, device=device, task=args.task)
        except Exception:
            LOGGER.exception("BGE-M3 failed")

        if can_gemma:
            try:
                sim_gemma = gemma_mod.embed_and_score(
                    text_a, text_b, device=device, task=args.task, truncate_dim=args.truncate_dim
                )
            except Exception:
                LOGGER.exception("embeddinggemma-300m failed")
        else:
            LOGGER.warning("embeddinggemma-300m skipped: Sentence-Transformers>=3.3.1 または Transformers>=5.1 が必要です。")
            print("[embeddinggemma-300m] SKIPPED (requires Sentence-Transformers>=3.3.1 or Transformers>=5.1)")

        # Text metrics (BLEU/NIST/ROUGE-1 F1)
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

        # 集計（人手評価があり、スコアが算出できたもののみ）
        if human_ratio is not None:
            if sim_sar is not None: compare_data["Sarashina-Embedding-v2-1B"].append((human_ratio, sim_sar))
            if sim_sta is not None: compare_data["static-embedding-japanese"].append((human_ratio, sim_sta))
            if sim_ruri is not None: compare_data["ruri-base"].append((human_ratio, sim_ruri))
            if sim_plamo is not None: compare_data["PLaMo-Embedding-1B"].append((human_ratio, sim_plamo))
            if sim_e5 is not None: compare_data["multilingual-e5-base"].append((human_ratio, sim_e5))
            if sim_glucose is not None: compare_data["GLuCoSE-base-ja-v2"].append((human_ratio, sim_glucose))
            if sim_jina is not None: compare_data["jina-embeddings-v3"].append((human_ratio, sim_jina))
            if sim_bge is not None: compare_data["BGE-M3"].append((human_ratio, sim_bge))
            if sim_gemma is not None: compare_data["embeddinggemma-300m"].append((human_ratio, sim_gemma))
            if sim_openai is not None: compare_data["text-embedding-3-large"].append((human_ratio, sim_openai))
            if sim_bleu is not None: compare_data["BLEU"].append((human_ratio, sim_bleu))
            if sim_nist is not None: compare_data["NIST"].append((human_ratio, sim_nist))
            if sim_rouge is not None: compare_data["ROUGE-1 F1"].append((human_ratio, sim_rouge))

        print(f"----- Results: {label} -----")
        result_lines.append(f"----- Results: {label} -----")
        # Compared texts (the exact strings used for scoring)
        result_lines.append(f"Text A path: {path_a}")
        result_lines.append(f"Text B path: {path_b}")
        result_lines.append("Text A (used):")
        result_lines.append(text_a)
        result_lines.append("Text B (used):")
        result_lines.append(text_b)
        _print("Sarashina-Embedding-v2-1B", sim_sar, human_ratio)
        _print("static-embedding-japanese", sim_sta, human_ratio)
        _print("ruri-base", sim_ruri, human_ratio)
        _print("PLaMo-Embedding-1B", sim_plamo, human_ratio)
        _print("multilingual-e5-base", sim_e5, human_ratio)
        _print("GLuCoSE-base-ja-v2", sim_glucose, human_ratio)
        _print("jina-embeddings-v3", sim_jina, human_ratio)
        _print("BGE-M3", sim_bge, human_ratio)
        _print("embeddinggemma-300m", sim_gemma, human_ratio)
        _print("text-embedding-3-large", sim_openai, human_ratio)
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

    # ベストモデル選定（最も相関が高いモデル）
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
    # Write the exact Pearson correlation formula used
    result_lines.append("Pearson correlation formula:")
    result_lines.append("r = sum_i (x_i - mean_x) * (y_i - mean_y) / sqrt( sum_i (x_i - mean_x)^2 * sum_i (y_i - mean_y)^2 )")
    result_lines.append("where x_i = human_eval normal_ratio for date i, y_i = model score for the same date i.")
    # ランキング（相関が計算できたモデルのみ、降順）
    valid = [(name, r) for name, r in corr_results.items() if r is not None]
    valid.sort(key=lambda x: x[1], reverse=True)
    for idx, (name, r) in enumerate(valid, start=1):
        line = f"{idx}. {name}: Pearson r = {r:.6f}"
        print(line)
        result_lines.append(line)

    # 相関が計算できなかったモデル群（N/A）も参考として列挙
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

    # 結果のみのログを書き出し
    try:
        out_path = os.path.join(_THIS_DIR, "result_v1.log")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(result_lines) + "\n")
        print(f"[results] Written to: {out_path}")
    except Exception as e:
        print(f"[results] Failed to write result_v1.log: {e}")


if __name__ == "__main__":
    main()
