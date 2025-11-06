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

try:
    import cdsapi  # noqa: F401
except Exception:
    cdsapi = None  # type: ignore
import xarray as xr
import numpy as np
import zipfile
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
# Variables grouped by stepType to avoid cfgrib 'stepType' filtering dropping accum variables
STEP_GROUPS = {
    "instant": ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind"],
    "accum": ["total_precipitation", "snowfall"],
}
# Short names expected in NetCDF after CDS conversion
REQUIRED_SHORTNAMES = {
    "2m_temperature": "t2m",
    "total_precipitation": "tp",
    "snowfall": "sf",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
}

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
    """
    NetCDF を一時ファイルに書き出してから原子的に置換。
    書き出しエンジンは netcdf4 -> h5netcdf -> scipy の順でフォールバック。
    環境変数 ERA5_OUTPUT_ENGINE を指定すると優先される。
    """
    # 出力ディレクトリの存在を保証（念のため）
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    tmp = path + ".part"
    if os.path.exists(tmp):
        try:
            os.remove(tmp)
        except Exception:
            pass

    # エンジン候補
    prefer = []
    env_engine = os.environ.get("ERA5_OUTPUT_ENGINE", "").strip()
    if env_engine:
        prefer.append(env_engine)
    engines = prefer + [e for e in ["netcdf4", "h5netcdf", "scipy"] if e not in prefer]

    last_err: Exception | None = None
    for eng in engines:
        try:
            if eng == "scipy":
                # scipy エンジンは圧縮エンコーディング未対応のため encoding を外す
                ds.to_netcdf(tmp, engine=eng)
            else:
                if encoding is not None:
                    ds.to_netcdf(tmp, engine=eng, encoding=encoding)
                else:
                    ds.to_netcdf(tmp, engine=eng)

            # 作成確認
            if not os.path.exists(tmp) or os.path.getsize(tmp) == 0:
                raise RuntimeError(f"一時ファイルが生成できませんでした: {tmp}")

            os.replace(tmp, path)
            print(f"💾 保存エンジン: {eng} -> {os.path.basename(path)}")
            return
        except Exception as e:
            last_err = e
            print(f"⚠ to_netcdf 失敗 (engine={eng}): {e}")
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass
            # 次のエンジンで再試行
            continue

    # すべて失敗
    raise last_err if last_err else RuntimeError("to_netcdf すべてのエンジンで失敗しました")


# ==============================================================================
# ZIP偽装検出・解凍 / エンジンフォールバック / 既存データからの再構築ユーティリティ
# ==============================================================================
def is_merge_only_mode() -> bool:
    return os.environ.get("ERA5_MERGE_ONLY", "0").lower() in ("1", "true", "yes", "on")


def maybe_extract_nc_from_zip(path: str) -> None:
    """
    .nc 拡張子でも実体が ZIP になっている場合があるため検出して解凍する。
    ZIP 内の最初の .nc を取り出して同じパスに上書きする（原子置換）。
    """
    try:
        if not os.path.isfile(path):
            return
        if not zipfile.is_zipfile(path):
            return
        with zipfile.ZipFile(path, "r") as zf:
            members = [m for m in zf.namelist() if m.lower().endswith(".nc")]
            if not members:
                # .nc が含まれない ZIP はそのままにする（後段の読み込みで失敗させる）
                print(f"⚠ ZIPですが .nc が含まれていません: {os.path.basename(path)}")
                return
            member = members[0]
            tmp_path = path + ".unzip.part"
            with zf.open(member) as src, open(tmp_path, "wb") as dst:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
            if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
                raise RuntimeError(f"Zip解凍に失敗しました: {path}")
            os.replace(tmp_path, path)
            print(f"🧰 解凍: ZIPだったためNetCDFに展開しました -> {os.path.basename(path)}")
    except Exception as e:
        print(f"⚠ ZIP判定/解凍で問題が発生しました: {path}: {e}")


def fix_zip_in_directory(dir_path: str, pattern: str = "era5_seed_*.nc") -> None:
    files = sorted(glob.glob(os.path.join(dir_path, pattern)))
    for f in files:
        maybe_extract_nc_from_zip(f)


def open_dataset_with_fallback(path: str) -> xr.Dataset:
    """
    xarray の IO バックエンドを netcdf4 -> h5netcdf -> scipy の順で試す。
    その前に ZIP 偽装を自動解凍する。
    """
    maybe_extract_nc_from_zip(path)
    engines = ["netcdf4", "h5netcdf", "scipy"]
    last_err = None
    for eng in engines:
        try:
            return xr.open_dataset(path, engine=eng)
        except Exception as e:
            last_err = e
    # 最後に自動判別も試す
    try:
        return xr.open_dataset(path)
    except Exception:
        pass
    raise last_err if last_err else ValueError(f"Cannot open dataset: {path}")


def ensure_combined_from_vars_if_missing() -> None:
    """
    monthly_vars に変数別しか存在しない月があれば、ダウンロードせずに月内マージして
    monthly/era5_seed_YYYYMM.nc を生成する。
    """
    # var 側の ZIP 偽装を正す
    fix_zip_in_directory(MONTHLY_VARS_DIR, "era5_seed_*.nc")
    var_files = sorted(glob.glob(os.path.join(MONTHLY_VARS_DIR, "era5_seed_????????_*.nc")))
    if not var_files:
        return

    groups: Dict[str, List[str]] = {}
    for fp in var_files:
        base = os.path.basename(fp)
        m = re.match(r"era5_seed_(\d{6})_", base)
        if not m:
            continue
        ym = m.group(1)
        groups.setdefault(ym, []).append(fp)

    for ym, fps in sorted(groups.items()):
        year = int(ym[:4]); month = int(ym[4:6])
        combined_path = monthly_combined_path(year, month)
        if os.path.exists(combined_path) and os.path.getsize(combined_path) > 0:
            continue

        ds_list: List[xr.Dataset] = []
        for fp in sorted(fps):
            try:
                ds = open_dataset_with_fallback(fp)
                ds_list.append(ds)
            except Exception as e:
                print(f"⚠ 変数別ファイルを開けませんでした: {fp}: {e}")

        if not ds_list:
            continue

        print(f"🧩 月内マージ（再構築）: {year}-{month:02d} の {len(ds_list)} 変数を統合")
        ds_month = xr.merge(ds_list, compat="no_conflicts", join="inner")
        encoding = {v: {"zlib": True, "complevel": 4} for v in ds_month.data_vars}
        save_dataset_atomic(ds_month, combined_path, encoding=encoding)
        print(f"✅ 月ファイル生成: {combined_path}")

        for ds in ds_list:
            try:
                ds.close()
            except Exception:
                pass


def normalize_coords(ds: xr.Dataset) -> xr.Dataset:
    """
    各ファイルの座標や余計な次元を正規化する:
    - valid_time -> time にリネーム（time が無い場合）
    - number, expver がサイズ1なら squeeze（expver が複数なら最後を採用）
    """
    # 時間座標の正規化
    if ("time" not in ds.dims and "time" not in ds.coords) and ("valid_time" in ds.dims or "valid_time" in ds.coords):
        ds = ds.rename({"valid_time": "time"})

    # expver が複数（例: 2）ある ERA5 特有のケースに対応（最後=最新を採用）
    if "expver" in ds.dims and ds.dims.get("expver", 1) > 1:
        try:
            ds = ds.isel(expver=-1)
        except Exception:
            pass

    # 次元の整理: number/expver が1なら落とす
    for dim in ("number", "expver"):
        if dim in ds.dims:
            if ds.dims[dim] == 1:
                try:
                    ds = ds.squeeze(dim, drop=True)
                except Exception:
                    pass
    return ds


def dataset_has_all_required_vars(ds: xr.Dataset) -> Tuple[bool, List[str]]:
    """
    月ファイルが必要変数（t2m, u10, v10, tp, sf）を全て含むかを判定。
    land_sea_mask は別途マージするため対象外。
    戻り値: (すべて揃っているか, 欠損している CDS 変数名リスト)
    """
    missing: List[str] = []
    for cds_name, short in REQUIRED_SHORTNAMES.items():
        # lsm はここでは扱わない
        if cds_name == LSM_VARIABLE:
            continue
        if (short not in ds.data_vars) and (cds_name not in ds.data_vars):
            missing.append(cds_name)
    return (len(missing) == 0, missing)


def file_missing_variables(path: str) -> List[str]:
    """NetCDF ファイルを開いて欠損している CDS 変数名（REQUIRED_SHORTNAMES のキー）を返す"""
    ds = open_dataset_with_fallback(path)
    try:
        ds = normalize_coords(ds)
        ok, missing = dataset_has_all_required_vars(ds)
        return [] if ok else missing
    finally:
        try:
            ds.close()
        except Exception:
            pass


def download_vars_for_month(client: cdsapi.Client, year: int, month: int, vars_to_get: List[str]) -> List[str]:
    """指定月の欠損変数（CDS 変数名）を monthly_vars に個別取得し、保存パス一覧を返す"""
    paths: List[str] = []
    if not vars_to_get:
        return paths
    req_common = build_common_request_for_month(year, month)
    for var in vars_to_get:
        var_path = monthly_var_path(year, month, var)
        if os.path.exists(var_path) and os.path.getsize(var_path) > 0:
            print(f"↪️  欠損補完: 既存ファイルを使用します: {os.path.basename(var_path)}")
            paths.append(var_path)
            continue
        print(f"🚀 欠損補完ダウンロード: {year}-{month:02d} [{var}]")
        req = {**req_common, "variable": [var]}
        retrieve_with_retry(
            client,
            "reanalysis-era5-single-levels",
            req,
            var_path,
            abort_on_cost_error=False,
        )
        print(f"   🎉 成功: {os.path.basename(var_path)}")
        paths.append(var_path)
    return paths


def merge_extra_vars_into_month(base_month_path: str, extra_var_files: List[str]) -> None:
    """
    既存の月ファイルに、個別取得した変数ファイル群をマージして上書き保存する。
    """
    if not extra_var_files:
        return
    ds_list: List[xr.Dataset] = []
    try:
        base_ds = open_dataset_with_fallback(base_month_path)
        ds_list.append(base_ds)
        for fp in extra_var_files:
            ds_list.append(open_dataset_with_fallback(fp))
        ds_merged = xr.merge(ds_list, compat="no_conflicts", join="outer")
        encoding = {v: {"zlib": True, "complevel": 4} for v in ds_merged.data_vars}
        save_dataset_atomic(ds_merged, base_month_path, encoding=encoding)
    finally:
        for ds in ds_list:
            try:
                ds.close()
            except Exception:
                pass


def check_and_fix_month_file(client: cdsapi.Client, year: int, month: int, combined_path: str) -> None:
    """
    月一括で取得したファイルが 'instant' のみになり、tp/sf が落ちるケースに対応。
    欠損があればその変数だけ個別取得して月ファイルへマージする。
    それでも欠損が解消しなければ例外を投げる。
    """
    missing = file_missing_variables(combined_path)
    if not missing:
        return
    print(f"⚠ 必要変数が不足しています（{year}-{month:02d}）: {missing} -> 個別取得で補完します")
    extra_files = download_vars_for_month(client, year, month, missing)
    merge_extra_vars_into_month(combined_path, extra_files)
    # 再検証
    missing_after = file_missing_variables(combined_path)
    if missing_after:
        raise RuntimeError(f"補完後も変数が不足しています: {missing_after}")


def augment_from_monthly_vars_if_missing(ds_main: xr.Dataset) -> xr.Dataset:
    """
    ds_main に欠けている変数（tp/sf）を monthly_vars から補完する。
    """
    missing_targets = []
    # total_precipitation
    if ("tp" not in ds_main.data_vars) and ("total_precipitation" not in ds_main.data_vars):
        missing_targets.append(("total_precipitation", "tp"))
    # snowfall
    if ("sf" not in ds_main.data_vars) and ("snowfall" not in ds_main.data_vars):
        missing_targets.append(("snowfall", "sf"))

    if not missing_targets:
        return ds_main

    for long_name, short_name in missing_targets:
        pattern = os.path.join(MONTHLY_VARS_DIR, f"era5_seed_????????_{long_name}.nc")
        files = sorted(glob.glob(pattern))
        if not files:
            alt_pattern = os.path.join(MONTHLY_VARS_DIR, f"era5_seed_????????_{short_name}.nc")
            alt_files = sorted(glob.glob(alt_pattern))
            if alt_files:
                files = alt_files
                pattern = alt_pattern  # ログ用
        if not files:
            print(f"⚠ 補完対象の変数ファイルが見つかりませんでした: {long_name}/{short_name} -> {pattern}")
            continue
        print(f"➕ 欠損変数を補完します: {long_name}（{len(files)} 個）")
        try:
            ds_more = xr.open_mfdataset(
                files,
                combine="by_coords",
                parallel=True,
                join="outer",
                compat="no_conflicts",
                data_vars="all",
                coords="minimal",
                combine_attrs="drop_conflicts",
                preprocess=normalize_coords,
            )
            # 変数名が long/short いずれにせよそのまま merge（座標はユニオン）
            ds_main = xr.merge([ds_main, ds_more], compat="no_conflicts", join="outer")
            try:
                ds_more.close()
            except Exception:
                pass
        except Exception as e:
            print(f"⚠ {long_name} の補完に失敗しました: {e}")
    return ds_main

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
    # 単発NetCDFでは cfgrib の stepType=instant のみが残り tp/sf が欠落することがあるため検証・補完
    check_and_fix_month_file(client, year, month, out_path)
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
        ds = open_dataset_with_fallback(var_path)
        ds_list.append(ds)

    # 月内マージ
    print(f"🧩 月内マージ: {year}-{month:02d} の {len(ds_list)} 変数を統合")
    ds_month = xr.merge(ds_list, compat="no_conflicts", join="inner")

    # 圧縮設定（任意）
    encoding = {v: {"zlib": True, "complevel": 4} for v in ds_month.data_vars}
    save_dataset_atomic(ds_month, combined_path, encoding=encoding)
    print(f"✅ 月ファイル生成: {combined_path}")

    # 念のため、必要変数が全て揃っているか検証・補完
    check_and_fix_month_file(client, year, month, combined_path)

    # 後片付け（開いているdsを閉じる）
    for ds in ds_list:
        ds.close()

    return combined_path


def download_month(client: cdsapi.Client, year: int, month: int) -> str:
    """
    月一括 → 失敗なら 月×変数 にフォールバック。
    既存の月ファイルがある場合も中身を検証し、欠損変数があれば個別取得で補完。
    """
    combined_path = monthly_combined_path(year, month)
    if os.path.exists(combined_path):
        size = os.path.getsize(combined_path)
        if size > 0:
            print(f"✅ スキップ: {os.path.basename(combined_path)} は既に存在します（{size} bytes）。検証・補完を実行します。")
            check_and_fix_month_file(client, year, month, combined_path)
            return combined_path

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
    # 既存データの健全化（ZIP偽装の解凍、var→月の再構築）
    ensure_dirs()
    ensure_combined_from_vars_if_missing()
    fix_zip_in_directory(MONTHLY_DIR, "era5_seed_*.nc")

    files = sorted(glob.glob(os.path.join(MONTHLY_DIR, "era5_seed_*.nc")))
    if not files:
        raise FileNotFoundError(f"月別NetCDFが見つかりません: {MONTHLY_DIR}/era5_seed_*.nc")

    print(f"📦 月別ファイルを結合中... ({len(files)} 個)")
    # 各ファイルの valid_time -> time 変換や不要次元除去を preprocess で正規化
    ds = xr.open_mfdataset(
        files,
        combine="by_coords",
        parallel=True,
        join="outer",
        compat="no_conflicts",
        data_vars="all",
        coords="minimal",
        combine_attrs="drop_conflicts",
        preprocess=normalize_coords,
    )
    # 欠損変数（tp/sf）があれば monthly_vars から補完
    ds = augment_from_monthly_vars_if_missing(ds)
    return ds


def open_lsm_file(lsm_path: str) -> xr.Dataset:
    maybe_extract_nc_from_zip(lsm_path)
    ds = open_dataset_with_fallback(lsm_path)
    # 変数名の正規化: lsm -> land_sea_mask
    if "lsm" in ds.variables and "land_sea_mask" not in ds.variables:
        ds = ds.rename({"lsm": "land_sea_mask"})
    # 代表時刻で取得しているため、time/valid_time 次元があれば squeeze（静的化）
    if "time" in ds.dims and ds.dims.get("time", 1) == 1:
        ds = ds.squeeze("time", drop=True)
    if "valid_time" in ds.dims and ds.dims.get("valid_time", 1) == 1:
        ds = ds.squeeze("valid_time", drop=True)
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


# ==============================================================================
# サニティチェック（不自然値の検出）
# ==============================================================================

def _detect_time_dim(da: xr.DataArray) -> str | None:
    for k in ("time", "valid_time"):
        if k in da.dims:
            return k
    return None


def _safe_min_max_nan(da: xr.DataArray, chunk: int = 744) -> Tuple[float, float, float]:
    """
    大容量データでも扱えるように time 次元をチャンク分割して min/max/NaN率を算出。
    time 次元が無い場合は一括で計算。
    戻り値: (min, max, nan_ratio)
    """
    time_dim = _detect_time_dim(da)
    if time_dim is None:
        arr = da.values
        if arr.size == 0:
            return float("nan"), float("nan"), float("nan")
        if np.isnan(arr).all():
            return float("inf"), float("-inf"), 1.0
        return float(np.nanmin(arr)), float(np.nanmax(arr)), float(np.isnan(arr).sum() / arr.size)

    ntime = da.sizes[time_dim]
    cur_min = float("inf")
    cur_max = float("-inf")
    nan_count = 0
    total_count = 0
    step = max(1, int(os.environ.get("CHECK_CHUNK_HOURS", str(chunk))))
    for i in range(0, ntime, step):
        sl = slice(i, min(i + step, ntime))
        sub = da.isel({time_dim: sl}).values
        if sub.size == 0:
            continue
        nan_mask = np.isnan(sub)
        nan_count += int(nan_mask.sum())
        total_count += sub.size
        if (~nan_mask).any():
            cur_min = min(cur_min, float(np.nanmin(sub)))
            cur_max = max(cur_max, float(np.nanmax(sub)))
    if total_count == 0:
        return float("nan"), float("nan"), float("nan")
    return cur_min, cur_max, nan_count / total_count


def _count_out_of_range_chunked(da: xr.DataArray, vmin: float, vmax: float, tol: float = 1e-12, chunk: int = 744) -> Tuple[int, int, int]:
    """範囲外データ数（下回る/上回る/総数）を time チャンクで集計"""
    time_dim = _detect_time_dim(da)
    if time_dim is None:
        arr = da.values
        mask = ~np.isnan(arr)
        total = int(mask.sum())
        if total == 0:
            return 0, 0, 0
        low = int(((arr < vmin - tol) & mask).sum())
        high = int(((arr > vmax + tol) & mask).sum())
        return low, high, total

    ntime = da.sizes[time_dim]
    step = max(1, int(os.environ.get("CHECK_CHUNK_HOURS", str(chunk))))
    low = high = total = 0
    for i in range(0, ntime, step):
        sl = slice(i, min(i + step, ntime))
        sub = da.isel({time_dim: sl}).values
        if sub.size == 0:
            continue
        mask = ~np.isnan(sub)
        total += int(mask.sum())
        if total == 0:
            continue
        low += int(((sub < vmin - tol) & mask).sum())
        high += int(((sub > vmax + tol) & mask).sum())
    return low, high, total


def run_sanity_checks(nc_path: str) -> None:
    """
    最終出力 seed_era5_data.nc に対して、不自然値がないかを全変数・全時刻・全格子で検査する。
    - 閾値（安全側）:
      t2m[K]: 180～330, u10/v10[m/s]: -80～80, tp/sf[m]: 0～0.5, land_sea_mask: 0～1
    - NaN 率、min/max、範囲外件数を集計して出力。
    併せて <同ディレクトリ>/seed_era5_data_sanity.txt にレポートを書き出します。
    """
    try:
        ds = open_dataset_with_fallback(nc_path)
    except Exception as e:
        print(f"❌ サニティチェック: データセットを開けませんでした: {e}")
        return
    report_lines: List[str] = []
    report_lines.append("ERA5 seed データ サニティチェック")
    report_lines.append(f"対象: {nc_path}")

    # 閾値定義（CDS短名で見る。長名が存在する場合は短名優先で拾う）
    thresholds = {
        "t2m": (180.0, 330.0),
        "u10": (-80.0, 80.0),
        "v10": (-80.0, 80.0),
        "tp": (0.0, 0.5),
        "sf": (0.0, 0.5),
        "land_sea_mask": (0.0, 1.0),
    }
    # 可能なら短名→長名の対応もチェック
    alternates = {
        "t2m": ["2m_temperature"],
        "u10": ["10m_u_component_of_wind"],
        "v10": ["10m_v_component_of_wind"],
        "tp": ["total_precipitation"],
        "sf": ["snowfall"],
        "land_sea_mask": ["lsm"],
    }

    all_ok = True
    for key, (vmin, vmax) in thresholds.items():
        var_name = None
        if key in ds.data_vars:
            var_name = key
        else:
            for alt in alternates.get(key, []):
                if alt in ds.data_vars:
                    var_name = alt
                    break
        if var_name is None:
            print(f"⚠ サニティチェック: 変数が見つかりません: {key}")
            report_lines.append(f"[WARN] {key}: 変数が存在しません")
            all_ok = False
            continue
        da = ds[var_name]
        mn, mx, nan_ratio = _safe_min_max_nan(da)
        low, high, total = _count_out_of_range_chunked(da, vmin, vmax)
        msg = (
            f"{var_name}: min={mn:.3f}, max={mx:.3f}, NaN率={nan_ratio*100:.3f}%"
            f", 範囲外(下)={low}, 範囲外(上)={high}, 総数={total}"
        )
        print("🔎 ", msg)
        report_lines.append(msg)
        if low > 0 or high > 0:
            all_ok = False

    # レポート保存
    try:
        report_path = nc_path.replace(".nc", "_sanity.txt")
        with open(report_path, "w", encoding="utf-8") as fw:
            fw.write("\n".join(report_lines) + "\n")
        print(f"📝 サニティチェックレポートを保存しました: {report_path}")
    except Exception as e:
        print(f"⚠ サニティチェックレポートの保存に失敗しました: {e}")

    if all_ok:
        print("✅ サニティチェック: 不自然値は検出されませんでした")
    else:
        print("⚠ サニティチェック: 一部に閾値外の値が見つかりました。レポートをご確認ください。")


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
        time_key = "time" if ("time" in chk.coords or "time" in chk.dims) else ("valid_time" if ("valid_time" in chk.coords or "valid_time" in chk.dims) else None)
        if time_key is not None and chk.sizes.get(time_key, 0) > 0:
            print("🔍 time 期間:", str(chk[time_key].values[0]), "～", str(chk[time_key].values[-1]))
        else:
            print("🔍 時間座標なし: time/valid_time が存在しません")
        print("🗺️  格子:", chk.sizes.get("latitude"), "x", chk.sizes.get("longitude"))
    # 追加: サニティチェックを実行
    print("🧪 サニティチェックを実行します...")
    run_sanity_checks(FINAL_PATH)


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

    # 既存データのみでマージするモード（途中再開・再実行向け）
    merge_only = is_merge_only_mode()
    if merge_only:
        print("🔁 マージ専用モード: 既存ファイルのみを用いてマージを実行します（ダウンロード無し）。")
        lsm_path = os.path.join(RAW_DIR, "lsm_025_japan.nc")
        if not (os.path.exists(lsm_path) and os.path.getsize(lsm_path) > 0):
            print("❌ ERA5_MERGE_ONLY=1 ですが、陸海マスクが見つかりませんでした。期待パス:", lsm_path)
            sys.exit(3)
        try:
            merge_all(lsm_path)
        except Exception:
            print("❌ マージ/出力に失敗しました。")
            traceback.print_exc()
            sys.exit(4)
        print("\nすべての処理が完了しました。")
        return

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
