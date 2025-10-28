#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ERA5 種データ取得スクリプト（リクエスト分割版）
- 変数:
  - 2m_temperature (K)              -> 気温関連
  - total_precipitation (m/h)       -> 降水関連（1時間積算）
  - snowfall (m SWE/h)              -> 降雪関連（1時間積算, 水当量）
  - 10m_u_component_of_wind (m/s)   -> 風速 U成分
  - 10m_v_component_of_wind (m/s)   -> 風速 V成分
  - land_sea_mask (0/1, static)     -> 陸海マスク
- 解像度: 0.25° (grid: [0.25, 0.25])
- 範囲: area=[55, 115, 15, 155]  (North, West, South, East; 日本域広め)
- 時間間隔: 1時間（24本/日）、期間内の全日
- 出力:
  - 月別の分割 NetCDF:   ./era5_seed/raw/monthly/era5_seed_{yyyymm}.nc
  -（フォールバック時）月別・変数別 NetCDF: ./era5_seed/raw/monthly_vars/era5_seed_{yyyymm}_{var}.nc
  - 陸海マスク NetCDF:   ./era5_seed/raw/lsm_025_japan.nc
  - 最終マージ NetCDF:   ./seed_era5_data.nc  （全変数・全時間を1ファイルに結合）

実装ポイント:
- 1年一括リクエストでの「cost limits exceeded」(403/413) を回避するため、
  年→月単位に分割。
- さらに月一括（5変数）でも大き過ぎる場合は、月・変数別に分割して取得し、
  月内でマージして1ファイルに統合。
- 取得済みファイルはスキップ。リトライは指数バックオフ。
"""
import os
import sys
import glob
import time
import traceback
import calendar
import re
from typing import Dict, List, Tuple

import cdsapi
import xarray as xr
import numpy as np
from tqdm import tqdm


# ==============================================================================
# 設定
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(SCRIPT_DIR, "era5_seed", "raw")
MONTHLY_DIR = os.path.join(RAW_DIR, "monthly")
MONTHLY_VARS_DIR = os.path.join(RAW_DIR, "monthly_vars")
FINAL_PATH = os.path.join(SCRIPT_DIR, "seed_era5_data.nc")

# デフォルト年範囲（環境変数で上書き可能）
DEFAULT_START_YEAR = int(os.environ.get("START_YEAR", "1950"))
DEFAULT_END_YEAR = int(os.environ.get("END_YEAR", "2024"))

# 対象変数
SINGLE_LEVEL_VARIABLES = [
    "2m_temperature",
    "total_precipitation",
    "snowfall",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]
LSM_VARIABLE = "land_sea_mask"

# 期間/領域/フォーマット
TIMES = [f"{h:02d}:00" for h in range(24)]
AREA = [55, 115, 15, 155]     # [North, West, South, East]
GRID = [0.25, 0.25]
FORMAT = "netcdf"

# リトライ設定（指数バックオフ）
MAX_RETRIES = 3
BACKOFF_BASE_SEC = 10
BACKOFF_CAP_SEC = 120


# ==============================================================================
# ユーティリティ
# ==============================================================================
def ensure_dirs() -> None:
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(MONTHLY_DIR, exist_ok=True)
    os.makedirs(MONTHLY_VARS_DIR, exist_ok=True)


def year_list(start: int, end: int) -> List[int]:
    if end < start:
        raise ValueError(f"END_YEAR({end}) は START_YEAR({start}) 以上である必要があります。")
    return list(range(start, end + 1))


def iter_year_months(start_year: int, end_year: int) -> List[Tuple[int, int]]:
    yms: List[Tuple[int, int]] = []
    for y in year_list(start_year, end_year):
        for m in range(1, 13):
            yms.append((y, m))
    return yms


def yyyymm_str(year: int, month: int) -> str:
    return f"{year:04d}{month:02d}"


def days_in_month(year: int, month: int) -> List[str]:
    _, last_day = calendar.monthrange(year, month)
    return [f"{d:02d}" for d in range(1, last_day + 1)]


def build_common_request_for_month(year: int, month: int) -> Dict:
    return {
        "year": str(year),
        "month": [f"{month:02d}"],
        "day": days_in_month(year, month),
        "time": TIMES,
        "area": AREA,          # N, W, S, E
        "grid": GRID,
        "format": FORMAT,
        "product_type": "reanalysis",
    }


def monthly_combined_path(year: int, month: int) -> str:
    return os.path.join(MONTHLY_DIR, f"era5_seed_{yyyymm_str(year, month)}.nc")


def monthly_var_path(year: int, month: int, var: str) -> str:
    safe_var = var  # 変数名はそのまま使用（cds変数名にASCII以外は含まれない）
    return os.path.join(MONTHLY_VARS_DIR, f"era5_seed_{yyyymm_str(year, month)}_{safe_var}.nc")


def is_cost_or_size_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    patterns = [
        "cost limits exceeded",
        "your request is too large",
        "413",                 # Payload Too Large
        "request too large",
        "payload too large",
        "403",                 # Forbidden (CDSでサイズ超過を403で返す場合あり)
        "forbidden for url",
    ]
    return any(p in msg for p in patterns)


def is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    patterns = [
        "too many requests",
        "429",
        "rate limit",
        "temporarily unavailable",
        "service unavailable",
        "503",
    ]
    return any(p in msg for p in patterns)


def retrieve_with_retry(client: cdsapi.Client, dataset: str, request: dict, target_path: str, abort_on_cost_error: bool = False) -> None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            client.retrieve(dataset, request, target_path)
            return
        except Exception as e:
            if abort_on_cost_error and is_cost_or_size_error(e):
                # サイズ超過は即フォールバック判断
                raise
            wait = min(BACKOFF_BASE_SEC * (2 ** (attempt - 1)), BACKOFF_CAP_SEC)
            kind = "レート制限" if is_rate_limit_error(e) else "エラー"
            print(f"⚠ {kind} {attempt}/{MAX_RETRIES}: {e}")
            if attempt == MAX_RETRIES:
                raise
            time.sleep(wait)


def save_dataset_atomic(ds: xr.Dataset, path: str, encoding: Dict = None) -> None:
    tmp = path + ".part"
    if os.path.exists(tmp):
        try:
            os.remove(tmp)
        except Exception:
            pass
    if encoding is not None:
        ds.to_netcdf(tmp, encoding=encoding)
    else:
        ds.to_netcdf(tmp)
    if not os.path.exists(tmp) or os.path.getsize(tmp) == 0:
        raise RuntimeError(f"一時ファイルが生成できませんでした: {tmp}")
    os.replace(tmp, path)


# ==============================================================================
# 取得処理（月単位 + フォールバック: 月×変数）
# ==============================================================================
def download_month_all_vars(client: cdsapi.Client, year: int, month: int) -> str:
    """
    指定年月の単一レベル（5変数）を1ファイルにまとめて取得（まずはこれを試す）。
    大き過ぎる場合は例外を上位で捕捉してフォールバック。
    """
    out_path = monthly_combined_path(year, month)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"✅ スキップ: {os.path.basename(out_path)} は既に存在します。")
        return out_path

    req = build_common_request_for_month(year, month)
    req.update({"variable": SINGLE_LEVEL_VARIABLES})

    print(f"🚀 月一括ダウンロード開始: {year}-{month:02d} ({', '.join(SINGLE_LEVEL_VARIABLES)})")
    retrieve_with_retry(
        client,
        "reanalysis-era5-single-levels",
        req,
        out_path,
        abort_on_cost_error=True,
    )
    print(f"🎉 成功: {os.path.basename(out_path)}")
    return out_path


def download_month_by_variable(client: cdsapi.Client, year: int, month: int) -> str:
    """
    指定年月を変数別（5ファイル）に分割して取得し、月内でマージして1ファイルに統合。
    """
    combined_path = monthly_combined_path(year, month)
    if os.path.exists(combined_path) and os.path.getsize(combined_path) > 0:
        print(f"✅ スキップ: {os.path.basename(combined_path)} は既に存在します。")
        return combined_path

    req_common = build_common_request_for_month(year, month)

    # 変数別取得
    ds_list: List[xr.Dataset] = []
    for var in SINGLE_LEVEL_VARIABLES:
        var_path = monthly_var_path(year, month, var)
        if os.path.exists(var_path) and os.path.getsize(var_path) > 0:
            print(f"↪️  既存利用: {os.path.basename(var_path)}")
        else:
            print(f"🚀 月×変数ダウンロード: {year}-{month:02d} [{var}]")
            req = {**req_common, "variable": [var]}
            retrieve_with_retry(
                client,
                "reanalysis-era5-single-levels",
                req,
                var_path,
                abort_on_cost_error=False,  # 変数単位であれば基本通る前提
            )
            print(f"   🎉 成功: {os.path.basename(var_path)}")

        # すぐには読み込まず最後にまとめて開くとメモリ効率が良いが、
        # ここでは確実性のため順次開いてmerge（xarrayは遅延読み込み）
        ds = xr.open_dataset(var_path)
        ds_list.append(ds)

    # 月内マージ
    print(f"🧩 月内マージ: {year}-{month:02d} の {len(ds_list)} 変数を統合")
    ds_month = xr.merge(ds_list, compat="no_conflicts", join="inner")

    # 圧縮設定（任意）
    encoding = {v: {"zlib": True, "complevel": 4} for v in ds_month.data_vars}
    save_dataset_atomic(ds_month, combined_path, encoding=encoding)
    print(f"✅ 月ファイル生成: {combined_path}")

    # 後片付け（開いているdsを閉じる）
    for ds in ds_list:
        ds.close()

    return combined_path


def download_month(client: cdsapi.Client, year: int, month: int) -> str:
    """
    月一括 → 失敗なら 月×変数 にフォールバック
    """
    try:
        return download_month_all_vars(client, year, month)
    except Exception as e:
        if is_cost_or_size_error(e):
            print(f"⚠ 大き過ぎるためフォールバックします（{year}-{month:02d} 月×変数）: {e}")
            return download_month_by_variable(client, year, month)
        # その他の例外はそのまま上げる
        raise


def download_lsm_once(client: cdsapi.Client) -> str:
    """
    陸海マスクは静的データのため代表時刻のみで取得
    """
    out_path = os.path.join(RAW_DIR, "lsm_025_japan.nc")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"✅ スキップ: {os.path.basename(out_path)} は既に存在します。")
        return out_path

    req = {
        **build_common_request_for_month(2000, 1),
        # 静的だが、CDS APIの仕様上、年/月/日/時刻を与える必要があるため代表値のみ指定
        "month": ["01"],
        "day": ["01"],
        "time": ["00:00"],
        "variable": [LSM_VARIABLE],
    }
    print(f"🚀 ダウンロード開始: {LSM_VARIABLE}（代表時刻で取得）")
    retrieve_with_retry(
        client,
        "reanalysis-era5-single-levels",
        req,
        out_path,
        abort_on_cost_error=False,
    )
    print(f"🎉 成功: {os.path.basename(out_path)}")
    return out_path


# ==============================================================================
# マージ処理（全月 → 最終）
# ==============================================================================
def open_monthly_files() -> xr.Dataset:
    files = sorted(glob.glob(os.path.join(MONTHLY_DIR, "era5_seed_*.nc")))
    if not files:
        raise FileNotFoundError(f"月別NetCDFが見つかりません: {MONTHLY_DIR}/era5_seed_*.nc")

    print(f"📦 月別ファイルを結合中... ({len(files)} 個)")
    ds = xr.open_mfdataset(
        files,
        combine="by_coords",
        parallel=True,
        join="outer",
        compat="no_conflicts",
        data_vars="minimal",
        coords="minimal",
        combine_attrs="drop_conflicts",
    )
    # ERA5 NetCDFは time / latitude / longitude が基本
    return ds


def open_lsm_file(lsm_path: str) -> xr.Dataset:
    ds = xr.open_dataset(lsm_path)
    # 変数名の正規化: lsm -> land_sea_mask
    if "lsm" in ds.variables and "land_sea_mask" not in ds.variables:
        ds = ds.rename({"lsm": "land_sea_mask"})
    # 代表時刻で取得しているため、time 次元があれば squeeze
    if "time" in ds.dims and ds.dims.get("time", 1) == 1:
        ds = ds.squeeze("time", drop=True)
    return ds[["land_sea_mask"]]


def save_merged(ds: xr.Dataset, path: str) -> None:
    # NetCDF4 圧縮設定（zlib）
    encoding = {}
    for var in ds.data_vars:
        encoding[var] = {"zlib": True, "complevel": 4}
    for coord in ds.coords:
        if coord in ("time", "latitude", "longitude"):
            continue
        encoding.setdefault(coord, {"zlib": True, "complevel": 4})

    save_dataset_atomic(ds, path, encoding=encoding)
    print(f"✅ 保存完了: {path}")


def merge_all(lsm_path: str) -> None:
    print("🔗 マージ処理を開始します...")
    ds_main = open_monthly_files()
    ds_lsm = open_lsm_file(lsm_path)

    # 空間座標の整合（latitude/longitude を ds_main に合わせる）
    for coord in ["latitude", "longitude"]:
        if coord in ds_lsm.coords:
            if not np.array_equal(ds_main[coord].values, ds_lsm[coord].values):
                ds_lsm = ds_lsm.reindex({coord: ds_main[coord].values}, method=None)

    # マージ（陸海は静的: lat/lon のみの2次元、他は time,lat,lon）
    ds_out = xr.merge([ds_main, ds_lsm], compat="no_conflicts", join="inner")

    # 最終保存
    save_merged(ds_out, FINAL_PATH)

    # 簡単な検証ログ
    with xr.open_dataset(FINAL_PATH) as chk:
        expected_vars = {
            "t2m": "2m_temperature",
            "tp": "total_precipitation",
            "sf": "snowfall",
            "u10": "10m_u_component_of_wind",
            "v10": "10m_v_component_of_wind",
            "land_sea_mask": "land_sea_mask",
        }
        present = list(chk.data_vars)
        print(f"📑 出力に含まれる変数: {present}")
        missing_logical = []
        for short, longname in expected_vars.items():
            if short == "land_sea_mask":
                if "land_sea_mask" not in present:
                    missing_logical.append("land_sea_mask")
            else:
                if short not in present and longname not in present:
                    missing_logical.append(f"{short} (or {longname})")
        if missing_logical:
            print(f"⚠ 注意: 期待する変数が見つからない可能性: {missing_logical}")
        print("🔍 time 期間:", str(chk["time"].values[0]), "～", str(chk["time"].values[-1]))
        print("🗺️  格子:", chk.sizes.get("latitude"), "x", chk.sizes.get("longitude"))


# ==============================================================================
# メイン
# ==============================================================================
def main():
    print("=" * 80)
    print("ERA5 種データダウンロード & マージ（分割ダウンロード対応）")
    print("- 変数:", ", ".join(SINGLE_LEVEL_VARIABLES), "+ land_sea_mask")
    print(f"- 解像度: {GRID[0]}°")
    print(f"- 範囲 (N,W,S,E): {AREA}")
    print(f"- 出力: {FINAL_PATH}")
    print("=" * 80)

    start_year = DEFAULT_START_YEAR
    end_year = DEFAULT_END_YEAR
    print(f"対象年: {start_year} ～ {end_year}（{end_year - start_year + 1} 年分）")

    ensure_dirs()

    try:
        client = cdsapi.Client()
    except Exception as e:
        print("❌ cdsapi.Client の初期化に失敗しました。~/.cdsapirc を確認してください。")
        print(e)
        sys.exit(1)

    # 月別の本体データ（先に月一括→駄目なら変数別）
    yms = iter_year_months(start_year, end_year)
    for (y, m) in tqdm(yms, desc="月別ダウンロード"):
        try:
            download_month(client, y, m)
        except Exception:
            print(f"❌ {y}-{m:02d} のダウンロードに失敗しました。")
            traceback.print_exc()
            sys.exit(2)

    # 陸海マスク（代表時刻）
    try:
        lsm_path = download_lsm_once(client)
    except Exception:
        print("❌ 陸海マスクのダウンロードに失敗しました。")
        traceback.print_exc()
        sys.exit(3)

    # マージ & 出力
    try:
        merge_all(lsm_path)
    except Exception:
        print("❌ マージ/出力に失敗しました。")
        traceback.print_exc()
        sys.exit(4)

    print("\nすべての処理が完了しました。")


if __name__ == "__main__":
    main()
