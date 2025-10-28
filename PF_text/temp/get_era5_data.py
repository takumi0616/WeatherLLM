#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ERA5 種データ取得スクリプト
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
  - 年別の分割 NetCDF:   ./era5_seed/raw/era5_seed_{year}.nc
  - 陸海マスク NetCDF:   ./era5_seed/raw/lsm_025_japan.nc
  - 最終マージ NetCDF:   ./seed_era5_data.nc  （全変数・全時間を1ファイルに結合）

参考: src/PressurePattern/Classification/data/download_era5_large.py の設計パターンに準拠
"""
import os
import sys
import glob
import time
import traceback
from typing import List

import cdsapi
import xarray as xr
import numpy as np
from tqdm import tqdm


# ==============================================================================
# 設定
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(SCRIPT_DIR, "era5_seed", "raw")
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
MONTHS = [f"{m:02d}" for m in range(1, 13)]
DAYS = [f"{d:02d}" for d in range(1, 32)]
TIMES = [f"{h:02d}:00" for h in range(24)]
AREA = [55, 115, 15, 155]     # [North, West, South, East]
GRID = [0.25, 0.25]
FORMAT = "netcdf"

# リトライ設定
MAX_RETRIES = 3
RETRY_WAIT_SEC = 15


# ==============================================================================
# ユーティリティ
# ==============================================================================
def ensure_dirs() -> None:
    os.makedirs(RAW_DIR, exist_ok=True)


def year_list(start: int, end: int) -> List[str]:
    if end < start:
        raise ValueError(f"END_YEAR({end}) は START_YEAR({start}) 以上である必要があります。")
    return [str(y) for y in range(start, end + 1)]


def build_common_request():
    return {
        "month": MONTHS,
        "day": DAYS,
        "time": TIMES,
        "area": AREA,          # N, W, S, E
        "grid": GRID,
        "format": FORMAT,
        "product_type": "reanalysis",
    }


def retrieve_with_retry(client: cdsapi.Client, dataset: str, request: dict, target_path: str) -> None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            client.retrieve(dataset, request, target_path)
            return
        except Exception as e:
            print(f"⚠ 失敗 {attempt}/{MAX_RETRIES}: {e}")
            if attempt == MAX_RETRIES:
                raise
            time.sleep(RETRY_WAIT_SEC)


# ==============================================================================
# 取得処理
# ==============================================================================
def download_year_file(client: cdsapi.Client, year: str) -> str:
    """
    指定年の単一レベル（5変数）を1ファイルにまとめて取得
    """
    out_path = os.path.join(RAW_DIR, f"era5_seed_{year}.nc")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"✅ スキップ: {os.path.basename(out_path)} は既に存在します。")
        return out_path

    req = build_common_request()
    req.update({
        "variable": SINGLE_LEVEL_VARIABLES,
        "year": year,
    })
    print(f"🚀 ダウンロード開始: {year}年 ({', '.join(SINGLE_LEVEL_VARIABLES)})")
    retrieve_with_retry(
        client,
        "reanalysis-era5-single-levels",
        req,
        out_path
    )
    print(f"🎉 成功: {os.path.basename(out_path)}")
    return out_path


def download_lsm_once(client: cdsapi.Client) -> str:
    """
    陸海マスクは静的データのため代表時刻のみで取得
    """
    out_path = os.path.join(RAW_DIR, "lsm_025_japan.nc")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"✅ スキップ: {os.path.basename(out_path)} は既に存在します。")
        return out_path

    req = {
        **build_common_request(),
        # 静的だが、CDS APIの仕様上、年/月/日/時刻を与える必要があるため代表値のみ指定
        "year": "2000",
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
        out_path
    )
    print(f"🎉 成功: {os.path.basename(out_path)}")
    return out_path


# ==============================================================================
# マージ処理
# ==============================================================================
def open_yearly_files() -> xr.Dataset:
    files = sorted(glob.glob(os.path.join(RAW_DIR, "era5_seed_*.nc")))
    if not files:
        raise FileNotFoundError(f"年別NetCDFが見つかりません: {RAW_DIR}/era5_seed_*.nc")

    print(f"📦 年別ファイルを結合中... ({len(files)} 個)")
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
    # 念のため座標名はそのまま利用（変換不要の想定）
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
        # 可変長文字列などが無い前提
        encoding[var] = {"zlib": True, "complevel": 4}
    # 座標にも圧縮設定（不要ならスキップ可）
    for coord in ds.coords:
        if coord in ("time", "latitude", "longitude"):
            continue
        # その他の座標がある場合のみ圧縮を付与
        encoding.setdefault(coord, {"zlib": True, "complevel": 4})

    tmp = path + ".part"
    if os.path.exists(tmp):
        try:
            os.remove(tmp)
        except Exception:
            pass

    ds.to_netcdf(tmp)
    if not os.path.exists(tmp) or os.path.getsize(tmp) == 0:
        raise RuntimeError(f"一時ファイルが生成できませんでした: {tmp}")
    os.replace(tmp, path)
    print(f"✅ 保存完了: {path}")


def merge_all(lsm_path: str) -> None:
    print("🔗 マージ処理を開始します...")
    ds_main = open_yearly_files()
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
        # ERA5 NetCDF の変数短名は cds から返る短名（例: t2m, tp, sf, u10, v10, lsm）であることが多い
        # land_sea_mask は本スクリプトで 'land_sea_mask' に統一済み
        missing_logical = []
        for short, longname in expected_vars.items():
            if short == "land_sea_mask":
                if "land_sea_mask" not in present:
                    missing_logical.append("land_sea_mask")
            else:
                if short not in present:
                    # 場合によっては long name のまま（稀）だが、ほぼ short name で返る想定
                    if longname not in present:
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
    print("ERA5 種データダウンロード & マージ")
    print("- 変数:", ", ".join(SINGLE_LEVEL_VARIABLES), "+ land_sea_mask")
    print(f"- 解像度: {GRID[0]}°")
    print(f"- 範囲 (N,W,S,E): {AREA}")
    print(f"- 出力: {FINAL_PATH}")
    print("=" * 80)

    start_year = DEFAULT_START_YEAR
    end_year = DEFAULT_END_YEAR
    yrs = year_list(start_year, end_year)
    print(f"対象年: {yrs[0]} ～ {yrs[-1]}（{len(yrs)} 年分）")

    ensure_dirs()

    try:
        client = cdsapi.Client()
    except Exception as e:
        print("❌ cdsapi.Client の初期化に失敗しました。~/.cdsapirc を確認してください。")
        print(e)
        sys.exit(1)

    # 年別の本体データ
    for y in tqdm(yrs, desc="年別ダウンロード"):
        try:
            download_year_file(client, y)
        except Exception:
            print(f"❌ 年 {y} のダウンロードに失敗しました。")
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
