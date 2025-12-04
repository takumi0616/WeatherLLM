#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ERA5 変換済みデータの年別分割スクリプト

入力: 1950_2024_era5_converted.nc (125GB超の大容量ファイル)
処理: 年ごとにデータを分割して個別のNetCDFファイルに保存
出力: era5_converted_YYYY.nc (各年ごとのファイル)

【元データについて】
get_era5_data.py:
  - CDS APIからERA5気象再解析データを取得
  - 対象変数: 2m気温、全降水量、降雪、10m風速(U/V成分)、陸海マスク
  - 空間解像度: 0.25度グリッド (161×161グリッドポイント)
  - 空間範囲: 日本域 (北緯15-55度、東経115-155度)
  - 時間範囲: 1950-2024年、1時間間隔
  - 月別にダウンロードし、最終的にseed_era5_data.ncに統合

era5_data_conversion.py:
  - seed_era5_data.ncを読み込んで単位変換とタイムゾーン変換
  - 単位変換:
    * t2m: K → ℃ (ケルビンから摂氏へ -273.15)
    * tp: m → mm (メートルからミリメートルへ ×1000)
    * sf: m → cm (メートルからセンチメートルへ ×100)
  - 時刻変換: UTC → JST (協定世界時から日本標準時へ +9時間)
  - 出力: 1950_2024_era5_converted.nc (125.81GB)

【本スクリプトの処理アルゴリズム】
1. 大容量ファイルをchunks指定で遅延読み込み（メモリ効率化）
2. 時間座標から全ての年リストを抽出
3. 各年ごとにループ:
   a. その年のデータのみをselect（時刻でフィルタ）
   b. land_sea_maskは時間次元を持たないため特別処理
   c. 圧縮設定を適用してNetCDFに保存
   d. 進捗表示
4. 分割完了後、各ファイルのサイズと期間を表示


notify-run yt03 -- nohup python separate_data.py  > separate_data.out 2>&1 &
"""

import os
import sys
import xarray as xr
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ==============================================================================
# 設定
# ==============================================================================
# 作業ディレクトリを取得（このスクリプトと同じディレクトリ）
SCRIPT_DIR = Path(__file__).parent.resolve()

# 入力ファイル（統合された変換済みERA5データ）
INPUT_FILE = SCRIPT_DIR / "1950_2024_era5_converted.nc"

# 出力ディレクトリ（年別ファイルを格納）
OUTPUT_DIR = SCRIPT_DIR / "yearly_data"

# メモリ効率のためのチャンクサイズ
# 720時間 = 約1ヶ月分のデータを一度にメモリに読み込む
CHUNK_SIZE = {"time": 720, "latitude": 161, "longitude": 161}


def main():
    print("=" * 80)
    print("ERA5 変換済みデータの年別分割処理")
    print(f"入力ファイル: {INPUT_FILE}")
    print(f"出力ディレクトリ: {OUTPUT_DIR}")
    print("=" * 80)

    # 1. 入力ファイルの存在確認
    if not INPUT_FILE.exists():
        print(f"❌ エラー: 入力ファイルが見つかりません: {INPUT_FILE}")
        sys.exit(1)

    # ファイルサイズを表示
    file_size_gb = INPUT_FILE.stat().st_size / (1024**3)
    print(f"📊 入力ファイルサイズ: {file_size_gb:.2f} GB")

    # 2. 出力ディレクトリの作成
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📁 出力ディレクトリを作成: {OUTPUT_DIR}")

    # 3. データセットの読み込み（遅延読み込み）
    print("\n🔄 データセットを読み込み中...")
    print("   ※chunksを指定して遅延読み込みするため、メモリを節約できます")
    
    try:
        ds = xr.open_dataset(INPUT_FILE, chunks=CHUNK_SIZE)
    except Exception as e:
        print(f"❌ ファイルを開けませんでした: {e}")
        sys.exit(1)

    print(f"✅ データセット読み込み完了")
    print(f"   時間範囲: {ds.time.values[0]} ～ {ds.time.values[-1]}")
    print(f"   時間ステップ数: {len(ds.time)}")
    print(f"   格子サイズ: {len(ds.latitude)} × {len(ds.longitude)}")
    print(f"   変数: {', '.join(ds.data_vars)}")

    # 4. 年リストの抽出
    # 時間座標からユニークな年のリストを取得
    time_values = ds.time.values
    # numpy.datetime64から年を抽出
    years = np.unique(np.array([np.datetime64(t, 'Y').astype(int) + 1970 
                                 for t in time_values]))
    
    print(f"\n📅 対象年: {years[0]} ～ {years[-1]} ({len(years)} 年分)")

    # 5. 年ごとに分割して保存
    print("\n" + "=" * 80)
    print("年別ファイルの作成を開始します...")
    print("=" * 80)

    # land_sea_maskを事前に取り出す（時間次元を持たない静的データ）
    if "land_sea_mask" in ds.data_vars:
        land_sea_mask = ds["land_sea_mask"]
        print("🗺️  land_sea_mask（陸海マスク）は静的データのため全年で共通です")
    else:
        land_sea_mask = None
        print("⚠️  land_sea_mask が見つかりませんでした")

    # 各年ごとに処理
    for year in tqdm(years, desc="年別分割進行中"):
        output_file = OUTPUT_DIR / f"era5_converted_{year}.nc"
        
        # 既に存在する場合はスキップ
        if output_file.exists():
            existing_size_mb = output_file.stat().st_size / (1024**2)
            print(f"  ⏭️  {year}: スキップ（既存ファイル {existing_size_mb:.1f} MB）")
            continue

        try:
            # その年のデータを抽出
            # numpy.datetime64で年の範囲を指定
            year_start = np.datetime64(f'{year}-01-01')
            year_end = np.datetime64(f'{year+1}-01-01')
            
            # 時間次元でフィルタリング
            ds_year = ds.sel(time=slice(year_start, year_end))
            
            # 実際にデータが含まれているか確認
            if len(ds_year.time) == 0:
                print(f"  ⚠️  {year}: データが見つかりません（スキップ）")
                continue

            # land_sea_maskを追加（静的データなので年ごとに同じものを付与）
            if land_sea_mask is not None:
                ds_year = ds_year.assign(land_sea_mask=land_sea_mask)

            # 圧縮設定（ファイルサイズ削減のため）
            encoding = {}
            for var in ds_year.data_vars:
                encoding[var] = {
                    "zlib": True,        # 圧縮を有効化
                    "complevel": 4,      # 圧縮レベル（1-9、4は標準的）
                    "shuffle": True,     # シャッフルフィルタ（圧縮率向上）
                }

            # NetCDFファイルに保存
            # compute=Trueで実際の計算を実行（遅延評価を解消）
            ds_year.to_netcdf(
                output_file,
                encoding=encoding,
                engine='netcdf4'
            )

            # ファイルサイズを取得して表示
            file_size_mb = output_file.stat().st_size / (1024**2)
            time_count = len(ds_year.time)
            
            print(f"  ✅ {year}: {time_count} 時間ステップ → {output_file.name} ({file_size_mb:.1f} MB)")

        except KeyboardInterrupt:
            print("\n⚠️  ユーザーによって中断されました")
            # 不完全なファイルを削除
            if output_file.exists():
                output_file.unlink()
            sys.exit(1)
            
        except Exception as e:
            print(f"  ❌ {year}: エラーが発生しました: {e}")
            # エラーが発生した場合も不完全なファイルを削除
            if output_file.exists():
                output_file.unlink()
            continue

    # 6. 完了メッセージと統計情報
    print("\n" + "=" * 80)
    print("✅ 年別分割処理が完了しました！")
    print("=" * 80)

    # 生成されたファイルの一覧と合計サイズ
    output_files = sorted(OUTPUT_DIR.glob("era5_converted_*.nc"))
    
    if output_files:
        print(f"\n📦 生成されたファイル: {len(output_files)} 個")
        total_size_gb = 0
        
        print("\n【ファイル一覧】")
        for f in output_files:
            size_mb = f.stat().st_size / (1024**2)
            total_size_gb += size_mb / 1024
            
            # 各ファイルの時間範囲を確認（最初と最後の3ファイルのみ詳細表示）
            if output_files.index(f) < 3 or output_files.index(f) >= len(output_files) - 3:
                try:
                    with xr.open_dataset(f) as ds_check:
                        time_range = f"{ds_check.time.values[0]} ～ {ds_check.time.values[-1]}"
                        print(f"  {f.name}: {size_mb:.1f} MB ({len(ds_check.time)} 時間ステップ)")
                        print(f"    時間範囲: {time_range}")
                except Exception:
                    print(f"  {f.name}: {size_mb:.1f} MB")
            elif output_files.index(f) == 3:
                print(f"  ... ({len(output_files) - 6} ファイル省略) ...")
        
        print(f"\n📊 合計サイズ: {total_size_gb:.2f} GB")
        print(f"   元のファイル: {file_size_gb:.2f} GB")
        print(f"   圧縮による差分: {file_size_gb - total_size_gb:.2f} GB")
        
        print(f"\n💡 使い方:")
        print(f"   各年のデータは {OUTPUT_DIR} に保存されています")
        print(f"   例: {output_files[0].name}")
    else:
        print("\n⚠️  生成されたファイルが見つかりませんでした")

    print("\n処理完了！")


if __name__ == "__main__":
    main()
