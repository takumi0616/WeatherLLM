#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
ERA5 seed データ（UTC）の年別分割スクリプト

入力: seed_era5_data.nc（get_era5_data.py が生成する統合ファイル / 超大容量）
処理: 年ごとにデータを分割して個別の NetCDF ファイルに保存
出力: era5_seed_YYYY.nc（各年ごとのファイル）

【元データについて】
get_era5_data.py:
  - CDS APIからERA5気象再解析データを取得
  - 対象変数: 2m気温、全降水量、降雪、10m風速(U/V成分)、陸海マスク
  - 空間解像度: 0.25度グリッド (161×161グリッドポイント)
  - 空間範囲: 日本域 (北緯15-55度、東経115-155度)
  - 時間範囲: 1950-2024年、1時間間隔
  - 月別にダウンロードし、最終的にseed_era5_data.ncに統合

※本スクリプトは "converted（JST化や単位変換後）" ではなく、
  UTC のままの seed_era5_data.nc を年別に切り分けます。

【本スクリプトの処理アルゴリズム】
1. 大容量ファイルを chunks 指定で遅延読み込み（メモリ効率化）
2. time 座標から年リストを抽出
3. 各年ごとにループ:
   a. その年のデータのみを抽出（右端排他: [year_start, next_year_start)）
   b. land_sea_mask は時間次元を持たないため特別処理（各年ファイルに付与）
   c. .part に保存してから os.replace で原子的にリネーム（途中失敗で壊れたファイルを残さない）
   d. 進捗表示
4. 分割完了後、各ファイルのサイズと期間を表示


notify-run via-tml2 -- nohup python separate_data.py  > separate_data.out 2>&1 &
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

# 入力ファイル（統合された seed ERA5 データ）
INPUT_FILE = SCRIPT_DIR / "seed_era5_data.nc"

# 出力ディレクトリ（年別ファイルを格納）
OUTPUT_DIR = SCRIPT_DIR / "yearly_seed_data"

# メモリ効率のためのチャンクサイズ
# 720時間 = 約1ヶ月分のデータを一度にメモリに読み込む
CHUNK_SIZE = {"time": 720, "latitude": 161, "longitude": 161}


def _atomic_to_netcdf(ds: xr.Dataset, final_path: Path, *, encoding: dict, engine: str | None = None) -> None:
    """一時ファイルに書いてから原子的に置換する（途中失敗で壊れた成果物を残さない）"""
    final_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = final_path.with_suffix(final_path.suffix + ".part")

    # 以前のゴミがあれば消す
    try:
        if part_path.exists():
            part_path.unlink()
    except Exception:
        pass

    try:
        kwargs = {}
        if engine is not None:
            kwargs["engine"] = engine
        ds.to_netcdf(part_path, encoding=encoding, **kwargs)

        # 簡易チェック: 0バイトを防ぐ
        if (not part_path.exists()) or part_path.stat().st_size == 0:
            raise RuntimeError(f"write failed: part file not created or empty: {part_path}")

        os.replace(part_path, final_path)
    except Exception:
        # 失敗時は part を消す（残骸を残さない）
        try:
            if part_path.exists():
                part_path.unlink()
        except Exception:
            pass
        raise


def _is_healthy_year_file(path: Path, year: int) -> bool:
    """既存ファイルをスキップしてよいかの最低限チェック（壊れたファイルのスキップ防止）"""
    try:
        if not path.exists() or path.stat().st_size == 0:
            return False
        # netcdf4 が無い環境もあるので、エンジン自動判別で開いてみる
        ds = xr.open_dataset(path)
        try:
            if "time" not in ds.coords and "time" not in ds.dims:
                return False
            t0 = np.datetime64(f"{year}-01-01T00:00")
            t1 = np.datetime64(f"{year+1}-01-01T00:00")
            # 年ファイルがその範囲内に収まっていること（右端排他）
            if ds.time.values.size == 0:
                return False
            if not (ds.time.values[0] >= t0 and ds.time.values[-1] < t1):
                return False
            return True
        finally:
            ds.close()
    except Exception:
        return False


def main():
    print("=" * 80)
    print("ERA5 seed データ（UTC）の年別分割処理")
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

    # 主要座標名の存在確認（seed_era5_data.nc では time/latitude/longitude が想定）
    if "time" not in ds.coords and "time" not in ds.dims:
        print("❌ エラー: time 座標が見つかりません。入力ファイルが想定と違う可能性があります")
        sys.exit(1)
    if "latitude" not in ds.coords or "longitude" not in ds.coords:
        print("❌ エラー: latitude/longitude 座標が見つかりません。入力ファイルが想定と違う可能性があります")
        sys.exit(1)

    print(f"✅ データセット読み込み完了")
    print(f"   時間範囲(UTC): {ds.time.values[0]} ～ {ds.time.values[-1]}")
    print(f"   時間ステップ数: {len(ds.time)}")
    print(f"   格子サイズ: {len(ds.latitude)} × {len(ds.longitude)}")
    print(f"   変数: {', '.join(ds.data_vars)}")

    # 4. 年リストの抽出（UTCの年）
    # 大量の時刻配列を Python ループで回すのは避け、numpy の年月単位に丸めて抽出
    time_values = ds.time.values
    years = np.unique(time_values.astype("datetime64[Y]").astype(int) + 1970)
    
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
        output_file = OUTPUT_DIR / f"era5_seed_{year}.nc"
        
        # 既に存在する場合は健全性チェックしてからスキップ
        if _is_healthy_year_file(output_file, int(year)):
            existing_size_mb = output_file.stat().st_size / (1024**2)
            print(f"  ⏭️  {year}: スキップ（既存の健全なファイル {existing_size_mb:.1f} MB）")
            continue

        try:
            # その年のデータを抽出
            # numpy.datetime64で年の範囲を指定
            year_start = np.datetime64(f"{year}-01-01T00:00")
            next_year_start = np.datetime64(f"{year+1}-01-01T00:00")

            # 時間次元でフィルタリング（右端排他）
            # slice の両端inclusive挙動による重複を避けるため、boolean mask を使って確実に [start, next) にする
            t = ds["time"]
            ds_year = ds.where((t >= year_start) & (t < next_year_start), drop=True)
            
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

            # NetCDFファイルに保存（原子的に置換）
            # netcdf4 エンジンが無い環境もあるため engine は固定しない（自動判別）
            _atomic_to_netcdf(ds_year, output_file, encoding=encoding, engine=None)

            # ファイルサイズを取得して表示
            file_size_mb = output_file.stat().st_size / (1024**2)
            time_count = len(ds_year.time)
            
            # 右端排他になっているか最終確認（軽いチェック）
            if not (ds_year.time.values[-1] < next_year_start):
                print(f"  ⚠️  {year}: 年境界チェックに失敗しました（右端排他になっていない可能性）")

            print(f"  ✅ {year}: {time_count} 時間ステップ → {output_file.name} ({file_size_mb:.1f} MB)")

        except KeyboardInterrupt:
            print("\n⚠️  ユーザーによって中断されました")
            # 不完全なファイルを削除
            if output_file.exists():
                output_file.unlink()
            sys.exit(1)
            
        except Exception as e:
            print(f"  ❌ {year}: エラーが発生しました: {e}")
            # エラーが発生した場合も不完全なファイルを削除（念のため）
            try:
                if output_file.exists():
                    output_file.unlink()
            except Exception:
                pass
            continue

    # 6. 完了メッセージと統計情報
    print("\n" + "=" * 80)
    print("✅ 年別分割処理が完了しました！")
    print("=" * 80)

    # 生成されたファイルの一覧と合計サイズ
    output_files = sorted(OUTPUT_DIR.glob("era5_seed_*.nc"))
    
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
