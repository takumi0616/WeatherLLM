#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ERA5 データ変換スクリプト
対象: ./seed_era5_data.nc
処理内容:
  1. 単位変換
     - t2m (K) -> (℃) : -273.15
     - tp (m)  -> (mm): * 1000
     - sf (m)  -> (cm): * 100  (水換算)
  2. 時間変換
     - UTC -> JST (UTC+9)
出力:
  - 1950_2024_era5_converted.nc
"""
import os
import sys
import xarray as xr
import numpy as np
from datetime import timedelta
from dask.diagnostics import ProgressBar

# ==============================================================================
# 設定
# ==============================================================================
INPUT_FILE = "./seed_era5_data.nc"
OUTPUT_FILE = "1950_2024_era5_converted.nc"

# メモリ不足を防ぐためのチャンクサイズ（時間の分割単位）
# 24時間 * 30日 = 720時間（1ヶ月分ずつ処理するイメージ）
CHUNK_SIZE = {"time": 720}

def main():
    print("=" * 60)
    print("ERA5 データ変換処理 (単位変換 & JST化)")
    print(f"入力ファイル: {INPUT_FILE}")
    print(f"出力ファイル: {OUTPUT_FILE}")
    print("=" * 60)

    if not os.path.exists(INPUT_FILE):
        print(f"エラー: 入力ファイルが見つかりません: {INPUT_FILE}")
        sys.exit(1)

    # 1. データ読み込み（chunksを指定して遅延読み込みにする）
    try:
        ds = xr.open_dataset(INPUT_FILE, chunks=CHUNK_SIZE)
    except Exception as e:
        print(f"ファイルを開けませんでした: {e}")
        sys.exit(1)

    print("🔄 変換処理を定義中...")

    # 2. 時間変換: UTC -> JST (+9時間)
    # time座標そのものに9時間を加算します
    print("   - 時刻を JST (UTC+9) に変換")
    ds["time"] = ds["time"] + np.timedelta64(9, "h")
    ds["time"].attrs["timezone"] = "JST"

    # 3. 単位変換
    # 計算式を適用し、属性(units)も書き換えます

    # --- t2m: K -> ℃ ---
    if "t2m" in ds:
        print("   - t2m: Kelvin -> Celsius (-273.15)")
        ds["t2m"] = ds["t2m"] - 273.15
        ds["t2m"].attrs["units"] = "degC"
        ds["t2m"].attrs["long_name"] = "2 metre temperature (Celsius)"

    # --- tp: m -> mm ---
    if "tp" in ds:
        print("   - tp: m -> mm (* 1000)")
        ds["tp"] = ds["tp"] * 1000.0
        ds["tp"].attrs["units"] = "mm"
        ds["tp"].attrs["long_name"] = "Total precipitation (mm)"

    # --- sf: m -> cm ---
    if "sf" in ds:
        print("   - sf: m -> cm (* 100)")
        ds["sf"] = ds["sf"] * 100.0
        ds["sf"].attrs["units"] = "cm (water equivalent)"
        ds["sf"].attrs["long_name"] = "Snowfall (cm water equivalent)"
    
    # --- 他の変数はそのまま ---
    # u10, v10, land_sea_mask は変更なしで保持されます

    # 4. 保存処理
    print(f"💾 保存を開始します: {OUTPUT_FILE}")
    print("   ※データ量が大きいため時間がかかります。進捗バーが表示されます。")

    # 圧縮設定（ファイルサイズ削減のためzlib圧縮を有効化）
    encoding = {}
    for var in ds.data_vars:
        encoding[var] = {"zlib": True, "complevel": 4}
    
    # 座標変数のエンコーディング指定（エラー回避のため time の units を明示する場合があるが、xarrayに任せる）
    
    try:
        # ProgressBarを使って進捗を表示しながら保存
        with ProgressBar():
            ds.to_netcdf(OUTPUT_FILE, encoding=encoding)
        print("\n✅ 変換完了しました！")
        
    except KeyboardInterrupt:
        print("\n⚠️ ユーザーによって中断されました。")
        # 中途半端なファイルを削除
        if os.path.exists(OUTPUT_FILE):
            os.remove(OUTPUT_FILE)
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 保存中にエラーが発生しました: {e}")
        sys.exit(1)

    # 5. 簡易確認
    print("-" * 60)
    print("🔍 出力ファイルの確認 (最初の1時刻)")
    try:
        with xr.open_dataset(OUTPUT_FILE) as check_ds:
            print(check_ds)
            print(f"Time range: {check_ds.time.values[0]} ~ {check_ds.time.values[-1]}")
    except Exception:
        pass

if __name__ == "__main__":
    main()