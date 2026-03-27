# PF_text 先行研究プログラム 再現実装計画

> 作成日: 2026-03-04  
> 対象: `document/program/src/` 以下の先行研究コード一式の再現

---

## 1. 全体概要・目的

先行研究 (`document/program/src/`) では、以下のパイプラインで「きのう・きょう・あした」の天気コメントをJSON生成する。

```
PF予報データ (Tokyo Cabinet)
    ↓
AMeDAS観測データ (PostgreSQL: amedas_tsdb)
    + MSM NetCDF (降雪補完)
    ↓
平年値・分位情報 (PostgreSQL: m_nml_*テーブル)
    ↓
重要度計算 (get.py内アルゴリズム)
    ↓
moji天気API → 文字天気・天気コード
    + PR1H3H CSV → 代表天気コード補正
    ↓
SOM気圧配置分類 (batchSOM.py: GSM GPV GRIB2使用)
    + 冬型整合チェック (check_winter_pattern.py)
    ↓
キーワード生成 → LLM用テキスト生成
    ↓
OpenAI API (chatgpt.py) → 天気コメント
    ↓
JSON / JSON.GZ 出力 (main.py)
```

### 最大の課題: 外部依存性

先行研究は**社内専用インフラ**に強く依存しており、そのまま実行不可能。

| 依存コンポーネント | 用途 | アクセス可否 |
|---|---|---|
| PostgreSQL `amedas_tsdb` (192.168.110.74) | 観測データ + 平年値 | ❌ 不可 |
| Tokyo Cabinet/Tyrant (192.168.110.34) | PF予報データ | ❌ 不可 |
| GSM GPV GRIB2 (/mnt/nfs41, /work_ytakano) | SOM気圧分類 | ❌ 不可 |
| MSM NetCDF (/mnt/calc43) | 降雪補完 | ❌ 不可 |
| PR1H3H CSV (/mnt/isilon) | 天気コード補正 | ❌ 不可 |
| moji天気API (hima.weathermap.co.jp) | ピンポイント天気 | △ 社内API |
| OpenAI API | LLM天気コメント | ✅ 可(APIキー要) |

---

## 2. 現在の手元資産の確認

### 2.1 既存ファイル

```
src/WeatherLLM/PF_text/
├── document/program/src/          ← 先行研究コード（全17ファイル、完全閲覧済み）
│   ├── get.py                     ← メインオーケストレーション
│   ├── main.py                    ← バッチ処理
│   ├── front_line.py              ← 前線推定（未連携）
│   ├── get_latest.py              ← 最新時刻CGI
│   └── app/
│       ├── amedas_tsdb.py         ← PostgreSQL接続
│       ├── batchSOM.py            ← SOM気圧配置分類
│       ├── calculation.py         ← 当日補完処理
│       ├── chatgpt.py             ← OpenAI API呼び出し
│       ├── check_winter_pattern.py← 冬型整合チェック
│       ├── create_keywords.py     ← キーワード抽出
│       ├── create_weather_text.py ← LLM入力テキスト生成
│       ├── forecast.py            ← PF予報読み込み
│       ├── moji_weather.py        ← ピンポイント天気API
│       ├── normal_day.py          ← 日別平年値
│       ├── normal_month.py        ← 月別平年値
│       ├── normal_season.py       ← 季節別平年値
│       ├── normal_year.py         ← 年別標準偏差
│       ├── observation.py         ← 観測データ取得
│       ├── tokyocabinet.py        ← TCH/TT ctypesラッパ
│       └── weather_code.py        ← 天気コード補正
├── document/program/json/
│   └── latest_data.json           ← ✅ PF予報JSONサンプル（東京: A44132）
├── temp/
│   ├── README.md                  ← ERA5データ取得・平年値計算計画書
│   └── yearly_seed_data/
│       └── era5_seed_1950.nc      ← ✅ ERA5データ（1950年分）がダウンロード済み
├── coordinate/
│   └── kanku_chihou_56.json       ← 地点座標情報
└── data/
    └── README.md                  ← JMAオープンデータ形式の説明
```

### 2.2 `latest_data.json` の内容（重要）

東京（A44132）の2024-10-11_17 発表データが格納されている。このファイルは：
- `info.time_def.timeA/B/C/D/E`：時刻定義（A=時別風、E=時別降水）
- `A44132.F_daily_2`：D-1〜D16の日別予報（T_min/T_max/R_sum/W_spd/W_dir）
- `A44132.F_hourly.wind/rain1h/temp`：時別予報
- `A44132.location.lon/lat`：地点座標

→ **forecast.py の `read_PF()` が返す構造と完全一致** → Tokyo Cabinet不要でテスト可能

### 2.3 ERA5データの現状

`temp/yearly_seed_data/era5_seed_1950.nc` が存在。  
`temp/README.md` に詳細なダウンロード計画（1940〜2024年, 85年分）が記載済み。  
変数: `2m_temperature`, `total_precipitation`, `snowfall`, `u10`, `v10`, `land_sea_mask`

---

## 3. 再現戦略：依存性を段階的に置換

### 3.1 依存性置換マップ

| 元の依存 | 再現時の代替 | 優先度 |
|---|---|---|
| PostgreSQL → `m_nml_*` 平年値テーブル | **ERA5から算出したCSV/NetCDF** | ⭐⭐⭐ 最優先 |
| Tokyo Cabinet → PF予報データ | **`latest_data.json` 形式のJSONファイル** | ⭐⭐⭐ 最優先 |
| AMeDAS観測DB | **気象庁オープンデータCSV** または **ERA5再解析値** | ⭐⭐ 重要 |
| MSM NetCDF (降雪) | **ERA5 snowfall変数** | ⭐⭐ 重要 |
| GSM GPV GRIB2 (SOM) | **ERA5 mean_sea_level_pressure** | ⭐⭐ 重要 |
| moji天気API | **気象庁天気API** または **ERA5天気カテゴリ** | ⭐ 後回し可 |
| PR1H3H CSV | **天気コード固定値 or ERA5推定** | ⭐ 後回し可 |
| OpenAI API | **そのまま使用（APIキー設定）** | ✅ 対応済み |

---

## 4. 具体的な実装計画

### Phase 1: データ基盤の構築（最重要）

#### 1-A. ERA5から平年値テーブルを生成する

**`temp/README.md` の計画を実装する。**

必要な出力テーブル（DBの代替として NetCDF or CSV で作成）:

```
m_nml_temp_day:
  - 地点/グリッド, record_month, record_day
  - mintemp_nml, maxtemp_nml
  - mintemp_q000,q010,q033,q067,q090,q100
  - maxtemp_q000,q010,q033,q067,q090,q100
  - stat_years

m_nml_temp_month:
  - 地点/グリッド, record_month
  - maxtemp_lt0 (真冬日日数), maxtemp_ge25 (夏日), maxtemp_ge30, maxtemp_ge35
  - mintemp_lt0 (冬日), mintemp_ge25 (熱帯夜)
  - stat_years

m_nml_prcp_day_pp:
  - 地点/グリッド, record_month, record_day
  - prcp_q000,q010,q033,q067,q090,q100
  - maxprcp_1h_q000,q010,q033,q067,q090,q100
  - stat_days

m_nml_snow_day_pp:
  - 地点/グリッド, record_month, record_day
  - snow_q000,q010,q033,q067,q090,q100
  - stat_days

m_nml_wind_mb10d:
  - 地点/グリッド, record_month, record_day (1/11/21)
  - meanwind_q000,q010,q033,q067,q090,q100
  - stat_years

m_nml_temp_year_pp:
  - 地点/グリッド
  - mintemp_0009_diff_sd (最低気温前日差SD)
  - maxtemp_0918_diff_sd (最高気温前日差SD)
```

**実装スクリプト計画: `src/WeatherLLM/PF_text/temp/compute_era5_normals.py`**

```python
# 処理フロー:
# 1. era5_seed_YYYY.nc を年順に読み込む（1940〜2024年）
# 2. UTC→JST変換（+9時間シフト）
# 3. 日最高・日最低気温を日毎に計算
# 4. 降水合計・最大1時間降水を日毎に計算
# 5. 降雪合計を日毎に計算（snowfall×100でcm換算）
# 6. 風速(sqrt(u10^2+v10^2))の日最大を計算
# 7. 85年分を DOY別・月別に集約して分位点・標準偏差を算出
# 8. 25地点の最近傍グリッドを coordinate/ から特定して抽出
```

**入力**: `temp/yearly_seed_data/era5_seed_YYYY.nc` (1940〜2024)  
**出力**: `temp/era5_normals/` 配下にNetCDF or CSV群

#### 1-B. PF予報データのJSONアダプター作成

先行研究の `forecast.py:read_PF()` はTokyo Cabinetを叩くが、  
`latest_data.json` と同じ構造のJSONファイルを読むアダプターを作成する。

**`src/WeatherLLM/PF_text/app/forecast_json.py`**（新規作成）:

```python
def read_PF_from_json(json_path: str, points: list[str]) -> dict:
    """
    latest_data.jsonと同じ構造のJSONファイルを読み込む
    → forecast.py の read_PF() と同じ戻り値を返す
    """
    with open(json_path) as f:
        return json.load(f)
```

→ `forecast.py` の `read_PF()` 内部で `ann == "json:パス"` のようなプレフィックスで分岐させる

#### 1-C. 観測データの代替

**選択肢A: JMAオープンデータ（推奨）**
- `https://www.jma.go.jp/bosai/amedas/data/latest_time.json` 等
- 気象庁の各種CSV（`data/README.md` に記載のtmax/tmin/pre1h等の形式）
- `coordinate/smaster.index` の地点情報と組み合わせ

**選択肢B: ERA5再解析値で近似**
- ERA5は2m気温・降水量・風速を持つ → 観測の代替として使用可能
- 地点ごとに最近傍格子の値を抽出

---

### Phase 2: コアモジュールの再実装（モジュール別）

#### 2-A. `app/amedas_tsdb.py` の置換

```
旧: connect_amedas_tsdb() → psycopg2でPostgreSQL接続

新: connect_era5_normals() → ERA5由来のNetCDF/CSVを返すインターフェース
    + connect_jma_obs() → JMAオープンデータ or ERA5観測値
```

**アダプターパターン**: 既存の `normal_day.py`, `normal_month.py`, `normal_season.py`, `normal_year.py`, `observation.py` が受け取る `cur` 引数を、  
ERA5/JMAデータを読むカーソル相当オブジェクトに置き換える。

具体的には、`cur.execute(sql, params)` / `cur.fetchone()` を模倣する  
**ダックタイピングのカーソルクラス** を作成する:

```python
class ERA5Cursor:
    """psycopg2 cursorと同じインターフェースでERA5データを返す"""
    def __init__(self, normals_path):
        self.normals = load_era5_normals(normals_path)
        self._result = None
    
    def execute(self, sql, params):
        # SQLのテーブル名とパラメータから対応データを返す
        table = extract_table_from_sql(sql)
        self._result = self.normals.query(table, params)
    
    def fetchone(self):
        return self._result
```

#### 2-B. `app/batchSOM.py` の置換

SOM気圧配置分類で必要なデータ:

```
学習用: PRMSL NetCDF (2016-03-01〜2024-10-31)
推論用: 各日の PRMSL データ
```

**ERA5による代替**:
- `reanalysis-era5-single-levels` の `mean_sea_level_pressure` 変数を使用
- 領域: 15-55N, 115-155E (先行研究と同一)
- 解像度: 0.25°（ERA5デフォルト）→ 0.5°に再補間（先行研究と合わせる）

`train_som_classifier()` および `judgePressurePattern()` は元コードのロジックをそのまま使用可能、  
**変わるのはGRIB2ファイル→NetCDFへの変換部分のみ**（`wgrib2`コマンド不要になる）。

#### 2-C. `app/calculation.py` の降雪補完置換

```
旧: MSM NetCDF (/mnt/calc43/archives/GUID_grid_nc/MSM/...)
新: ERA5 snowfall NetCDF
    → get_snow_forecast_from_netcdf() の file_path 生成部分のみ修正
```

#### 2-D. `app/moji_weather.py` の置換

**選択肢A**: 気象庁の天気予報API
```
https://www.jma.go.jp/bosai/forecast/data/forecast/{area_code}.json
```
→ 地点コードから都道府県コードへのマッピングが必要

**選択肢B**: ERA5の天気カテゴリ（precipitation, cloud cover等）から自動生成  
- Wd/Ws の変換テーブルは `moji_weather.py` にそのまま定義済み
- 入力値をERA5から生成するロジックを作成

**選択肢C（最小工数）**: 固定の moji_data サンプル（`document/program/json/moji.json` など）を使用してテスト

#### 2-E. `app/weather_code.py` の置換

PR1H3H CSVは社内データ→アクセス不可  
→ moji_weather.pyのWd値をそのまま代表天気コードとして使用する簡略版を作成  
→ `weather_code_fix()` ロジックはそのまま活用

---

### Phase 3: 統合テスト用設定

#### 3-A. テスト実行環境の構成

```
src/WeatherLLM/PF_text/
├── src_new/                   ← 新規再現コード（先行研究と平行開発）
│   ├── app/
│   │   ├── era5_cursor.py     ← ERA5カーソルアダプター（DB置換）
│   │   ├── forecast_json.py   ← JSON版PF読み込み
│   │   ├── jma_obs.py         ← JMA観測データ取得
│   │   ├── batchSOM_era5.py   ← ERA5版SOM
│   │   └── ... (その他は先行研究コードをそのままコピー)
│   ├── get.py                 ← 先行研究get.pyを依存先変更のみ修正
│   └── main.py
└── temp/
    ├── compute_era5_normals.py ← 平年値計算スクリプト
    └── era5_normals/          ← 計算済み平年値データ（出力先）
```

#### 3-B. 設定ファイルの作成

```yaml
# config.yaml
data_sources:
  pf_json_path: "document/program/json/latest_data.json"
  era5_normals_dir: "temp/era5_normals/"
  era5_prmsl_dir: "temp/era5_prmsl/"
  
openai:
  api_key_env: "OPENAI_KEY_TOKEN"
  model: "gpt-4o-mini"  # コスト削減: o1-previewの代替

points:
  - ["A44132", "東京"]    # まずは1地点でテスト
  - ["A14163", "札幌"]
  # ... 25地点
```

---

### Phase 4: LLM天気コメント生成

`chatgpt.py` は構造を変えずそのまま使用可能。  
変更点:
- `model: "o1-preview"` → コスト削減のため `"gpt-4o-mini"` に変更検討
- APIキーは `.env` ファイルまたは環境変数 `OPENAI_KEY_TOKEN` で設定

---

## 5. 実装優先度と工数見積もり

### 優先度 ★★★（システムが動く最小構成）

| タスク | 実装内容 | 推定工数 |
|---|---|---|
| **ERA5平年値計算スクリプト** | `compute_era5_normals.py` 作成 | 3〜5日 |
| **PF-JSON読み込みアダプター** | `forecast_json.py` + `forecast.py`修正 | 0.5日 |
| **ERA5カーソルアダプター** | `era5_cursor.py` (SQLパーサー代替) | 2〜3日 |
| **main pipelineテスト** | 東京1地点で end-to-end 動作確認 | 1日 |

### 優先度 ★★（精度向上）

| タスク | 実装内容 | 推定工数 |
|---|---|---|
| **SOM学習のERA5対応** | `batchSOM_era5.py` (GRIB2→ERA5置換) | 1〜2日 |
| **JMA観測データ取得** | `jma_obs.py` (オープンデータCSV取得) | 1〜2日 |
| **降雪ERA5対応** | `calculation.py` のNetCDF参照先変更 | 0.5日 |

### 優先度 ★（拡張・改良）

| タスク | 実装内容 | 推定工数 |
|---|---|---|
| **気象庁天気API対応** | `moji_weather.py` 代替 | 1〜2日 |
| **25地点バッチ処理** | `main.py` の全地点対応 | 0.5日 |
| **前線推定の統合** | `front_line.py` を get.py に組み込み | 1〜2日 |

---

## 6. 具体的な実装手順（ステップバイステップ）

### Step 1: ERA5平年値の計算

```bash
# 必要パッケージ確認
conda activate weather_env
pip install cdsapi xarray numpy scipy pandas netcdf4

# ERA5データの追加ダウンロード（1951〜2024年分が未取得の場合）
cd src/WeatherLLM/PF_text/temp
python get_era5_data.py  # 既存スクリプト

# 平年値計算（新規作成スクリプト）
python compute_era5_normals.py
```

**`compute_era5_normals.py` の核心ロジック**:
```python
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path

# 25地点の座標 (coordinate/kanku_chihou_56.json から取得)
STATIONS = {
    "A44132": {"lat": 35.692, "lon": 139.750, "name": "東京"},
    "A14163": {"lat": 43.060, "lon": 141.328, "name": "札幌"},
    # ... 全25地点
}

def compute_normals(years_range=(1940, 2024)):
    all_data = {}
    for year in range(*years_range):
        nc_path = f"yearly_seed_data/era5_seed_{year}.nc"
        if not Path(nc_path).exists():
            continue
        ds = xr.open_dataset(nc_path)
        
        # UTC→JST (+9h)
        ds = ds.assign_coords(time=ds.time + np.timedelta64(9, 'h'))
        
        # 2m気温: K→℃
        t2m = ds['t2m'] - 273.15
        
        # 日次統計（JST日付で集約）
        daily_tmax = t2m.resample(time='1D').max()
        daily_tmin = t2m.resample(time='1D').min()
        daily_prcp = (ds['tp'] * 1000).resample(time='1D').sum()  # m→mm
        daily_maxprcp1h = (ds['tp'] * 1000).resample(time='1D').max()
        daily_snow = (ds['sf'] * 100).resample(time='1D').sum()  # m→cm
        
        u10, v10 = ds['u10'], ds['v10']
        wind_speed = np.sqrt(u10**2 + v10**2)
        daily_wind = wind_speed.resample(time='1D').max()
        
        # 25地点ごとに最近傍グリッドを選択
        for point_id, info in STATIONS.items():
            station_data = {
                'tmax': daily_tmax.sel(lat=info['lat'], lon=info['lon'], method='nearest').values,
                'tmin': daily_tmin.sel(lat=info['lat'], lon=info['lon'], method='nearest').values,
                'prcp': daily_prcp.sel(lat=info['lat'], lon=info['lon'], method='nearest').values,
                'maxprcp1h': daily_maxprcp1h.sel(...).values,
                'snow': daily_snow.sel(...).values,
                'wind': daily_wind.sel(...).values,
                'dates': daily_tmax.time.values,
            }
            # all_data に追加
    
    # DOY別・月別に分位点を計算
    # → m_nml_temp_day, m_nml_temp_month, etc. に対応するCSVを出力
```

### Step 2: PF-JSONアダプター作成

`forecast.py` の `read_PF()` を修正:

```python
def read_PF(ann: str, points: list[str]):
    # 新規: JSONファイルから読み込む
    if ann.startswith("json:"):
        json_path = ann[5:]  # "json:path/to/file.json"
        with open(json_path) as f:
            return json.load(f)
    
    # 既存: latest → Tokyo Tyrant
    if ann == "latest":
        TC = tokyocabinet.TT("192.168.110.34", 17851)
    # 既存: アーカイブ → TCH
    else:
        ...
```

### Step 3: ERA5カーソルアダプター作成

`normal_day.py` 等が使う `cur.execute(sql)` / `cur.fetchone()` を模倣:

```python
class ERA5NormalsCursor:
    """
    ERA5から計算した平年値CSVをSQLクエリ風に問い合わせるカーソル
    """
    def __init__(self, normals_dir: str):
        self.normals_dir = Path(normals_dir)
        self._result = None
    
    def execute(self, sql: str, params: tuple):
        # SQLからテーブル名を抽出
        table = self._extract_table(sql)
        amedas, *rest = params
        
        if table == 'm_nml_temp_day':
            month, day = rest
            df = pd.read_csv(self.normals_dir / 'temp_day.csv')
            row = df[(df.amedas == amedas) & 
                     (df.record_month == month) & 
                     (df.record_day == day)]
            self._result = row.iloc[0].values.tolist() if len(row) > 0 else None
        
        elif table == 'm_nml_temp_month':
            month = rest[0]
            df = pd.read_csv(self.normals_dir / 'temp_month.csv')
            ...
        
        # 他のテーブルも同様
    
    def fetchone(self):
        return self._result
```

### Step 4: end-to-end テスト

```python
# test_pipeline.py
from src_new.app.era5_cursor import ERA5NormalsCursor
from src_new.get import get_dict

# ERA5カーソルを使ってテスト
cur = ERA5NormalsCursor("temp/era5_normals/")

# PF JSONファイルを使ってテスト（ann に json: プレフィックス）
result = get_dict("A44132", "json:document/program/json/latest_data.json")
print(result)
```

---

## 7. 注意点・実装時のリスク

### 7.1 ERA5データ量の問題

`temp/README.md` 記載の通り、85年分の毎時データは**数TB規模**。

**対応策**:
- 年ごとに分割処理（yearly_seed_data/ ディレクトリの既存構造）
- まず1940〜1960年の20年分で平年値をテスト計算
- 最終的には1991-2020年の30年平均（気象庁標準の平年値期間）で計算

### 7.2 地点とグリッドの対応

ERA5の0.25°グリッドとAMeDAS地点は正確に一致しない。  
→ `method='nearest'` による最近傍選択で対応（誤差は±14km程度）  
→ `coordinate/kanku_chihou_56.json` に25地点の座標が記載済み

### 7.3 SOM再学習の問題

`batchSOM.py` の SOM モデルは2016-2024年の GSM GPV データで学習済み（先行研究のピクル）。  
→ ERA5対応版で再学習が必要（ERA5 PRMSL 2016-2024年分が必要）  
→ ERA5ダウンロード済みデータに PRMSL が含まれていれば流用可能

### 7.4 天気コード（PR1H3H CSV）問題

PR1H3H CSVは気象庁の内部データ→再現不可。  
→ `weather_code.py` の `get_weather_code()` をスキップし、  
　`moji_weather.py` の `Wd` 値のみから `weather_code_fix()` を実行する簡略版で対応。  
→ `FN=1` のケース（天気コード1コマのみ）として処理

### 7.5 `check_winter_pattern.py` の NetCDF

```
/home/devel/contents_C/PF_text/wm_nc/winter_weather_pattern.nc
/home/devel/contents_C/PF_text/wm_nc/all_nodes_weather_patterns.nc
```
→ 社内NetCDFファイルにアクセス不可  
→ 冬型整合チェックを `is_*_winter_pressuer_pattern = 0` (固定値) で代替するか、  
　ERA5と統計的手法で類似マップを作成する（発展的課題）

---

## 8. 推奨実装順序まとめ

```
Week 1:
  [x] ERA5データの構造確認 (era5_seed_1950.nc)
  [ ] compute_era5_normals.py の実装・テスト (1年分で動作確認)
  [ ] 25地点の座標マッピング確認 (coordinate/ から抽出)

Week 2:
  [ ] ERA5平年値の全期間計算 (メモリ・時間次第で並列化)
  [ ] ERA5カーソルアダプター (era5_cursor.py) の実装
  [ ] forecast_json.py の実装
  [ ] get.py の依存先切り替え (PostgreSQL→ERA5カーソル)

Week 3:
  [ ] JMA観測データ取得 (オープンデータAPI)
  [ ] observation.py のアダプター実装
  [ ] end-to-end テスト (東京 A44132 で動作確認)

Week 4:
  [ ] batchSOM_era5.py (ERA5版気圧配置分類)
  [ ] 25地点バッチ処理テスト
  [ ] LLMコメント生成の最終確認
```

---

## 9. コード再利用性の評価

先行研究コードのうち、**変更なしで再利用可能**なもの:

- `app/chatgpt.py` ✅ そのまま使用可
- `app/create_keywords.py` ✅ そのまま使用可
- `app/create_weather_text.py` ✅ そのまま使用可
- `app/moji_weather.py` ✅ 変換テーブルはそのまま（API呼び出し部分のみ修正）
- `app/weather_code.py` ✅ `weather_code_fix()` ロジックはそのまま使用可
- `front_line.py` ✅ そのまま使用可（未連携のため優先度低）
- `get.py` の重要度計算アルゴリズム ✅ 変更不要
- `main.py` の `adjust_time()`, `replace_nan()` ✅ 変更不要

**修正が必要**なもの（インターフェース置換のみ）:

- `app/amedas_tsdb.py` → ERA5カーソルアダプターに差し替え
- `app/forecast.py` → JSON読み込み対応追加
- `app/observation.py` → JMA/ERA5観測データ対応
- `app/batchSOM.py` → ERA5 PRMSL対応（wgrib2部分のみ）
- `app/calculation.py` → MSM→ERA5 snowfall参照先変更

---

## 10. 参考: 先行研究のデータフロー図（再確認）

```
get.py (get_dict)
│
├─ forecast.py:get_forecast(ann, point)
│   └─ tokyocabinet → PF JSON (latest_data.json で代替可)
│
├─ observation.py:get_observation(cur, amedas, date_list, ...)
│   ├─ amedas_tsdb.py → PostgreSQL (ERA5カーソルで代替)
│   └─ calculation.py
│       ├─ get_today_prcp()
│       ├─ get_today_wind()
│       ├─ get_today_snow() → MSM NetCDF (ERA5 snowfallで代替)
│       └─ get_today_maxprcp_1h()
│
├─ normal_day/month/season/year.py (ERA5カーソルで代替)
│
├─ moji_weather.py:get_moji_data() → 外部API (気象庁APIで代替)
├─ weather_code.py:get_weather_code() → PR1H3H CSV (簡略版で代替)
│
├─ batchSOM.py:train_som_classifier() → GSM GPV (ERA5 PRMSLで代替)
├─ batchSOM.py:judgePressurePattern() → GSM GPV (ERA5で代替)
├─ check_winter_pattern.py → 社内NetCDF (暫定: 固定値0で代替)
│
├─ create_keywords.py (変更不要)
├─ create_weather_text.py (変更不要)
└─ chatgpt.py:generate_weather_comment() (変更不要)
```

---

*以上が先行研究プログラムの再現実装計画です。*  
*最初のマイルストーンは「ERA5平年値計算スクリプトの完成」と「PF-JSONアダプターを使ったend-to-endテスト」です。*
