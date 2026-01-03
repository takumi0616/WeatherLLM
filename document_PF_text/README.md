# PF_text プログラム詳細ドキュメント

## 概要

PF_text は、気象データを収集・分析し、OpenAI GPT API を使用して自然言語による天気コメントを自動生成するシステムです。「きのう」「きょう」「あした」の 3 日間の天気情報を処理し、各地点ごとのパーソナライズされた天気解説文を作成します。

---

## システム構成図

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              main.py                                        │
│                    (バッチ実行・複数地点処理)                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                               get.py                                        │
│                    (メインデータ処理・統合処理)                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
          ┌─────────────┬───────────┼───────────┬─────────────┐
          ▼             ▼           ▼           ▼             ▼
    ┌──────────┐  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
    │ forecast │  │observation│ │ normal_* │ │ moji_    │ │ batchSOM │
    │   .py    │  │   .py     │ │   .py    │ │weather.py│ │   .py    │
    └──────────┘  └──────────┘ └──────────┘ └──────────┘ └──────────┘
          │             │           │           │             │
          ▼             ▼           ▼           ▼             ▼
    ┌──────────────────────────────────────────────────────────────┐
    │                      データソース                             │
    │  ・TokyoCabinet (PF予報)  ・PostgreSQL (観測・平年値)         │
    │  ・Moji API (天気予報)    ・NetCDF (気圧・降雪予報)          │
    └──────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌──────────────────────────────────────────────────────────────┐
    │                  天気コメント生成パイプライン                   │
    │  create_keywords → create_weather_text → chatgpt.py         │
    └──────────────────────────────────────────────────────────────┘
```

---

## ファイル構成と役割

### エントリーポイント

| ファイル        | 役割                                                           |
| --------------- | -------------------------------------------------------------- |
| `main.py`       | バッチ実行スクリプト。25 地点を順次処理し、JSON ファイルを出力 |
| `get.py`        | メインのデータ処理スクリプト。全モジュールを統合               |
| `get_latest.py` | 最新の初期時間情報を JSON 形式で取得する CGI スクリプト        |

### app ディレクトリ（モジュール群）

| ファイル                  | 役割                                        |
| ------------------------- | ------------------------------------------- |
| `__init__.py`             | パッケージ初期化（空ファイル）              |
| `amedas_tsdb.py`          | PostgreSQL データベース接続                 |
| `batchSOM.py`             | 自己組織化マップ(SOM)による気圧パターン分類 |
| `calculation.py`          | 降水量・風速・積雪量の計算処理              |
| `chatgpt.py`              | OpenAI GPT API による天気コメント生成       |
| `check_winter_pattern.py` | 冬型気圧配置パターンの判定                  |
| `create_keywords.py`      | 気象データからキーワード生成                |
| `create_weather_text.py`  | 気象データをテキスト形式に変換              |
| `forecast.py`             | PF（ポイント予報）データの読み込み          |
| `moji_weather.py`         | 天気コード変換・Moji データ取得             |
| `normal_day.py`           | 日別平年値取得                              |
| `normal_month.py`         | 月別平年値取得                              |
| `normal_season.py`        | 季節別平年値取得                            |
| `normal_year.py`          | 年別平年値取得                              |
| `observation.py`          | 観測データ取得                              |
| `tokyocabinet.py`         | TokyoCabinet/TokyoTyrant ラッパークラス     |
| `weather_code.py`         | 天気コード変換アルゴリズム                  |

### その他

| ファイル        | 役割                       |
| --------------- | -------------------------- |
| `front_line.py` | 前線位置解析（現在未使用） |

---

## 主要な変数とデータ構造

### 入力変数

| 変数名   | 型    | 説明                                          | 取得元           |
| -------- | ----- | --------------------------------------------- | ---------------- |
| `point`  | `str` | 地点コード（例: "A44132"）                    | 引数またはクエリ |
| `ann`    | `str` | 発表時刻（例: "2023-12-17_08"）または"latest" | 引数またはクエリ |
| `amedas` | `int` | AMeDAS 地点番号（point の数字部分）           | point から抽出   |

### 中間データ構造

```python
# 予報データ
class Forecast(TypedDict):
    date: str              # 日付（YYYY-MM-DD）
    mintemp: Decimal       # 最低気温
    maxtemp: Decimal       # 最高気温
    prcp: Decimal          # 降水量
    wind: Decimal          # 風速
    wind_dir: str          # 風向
    snow: Decimal          # 積雪量
    maxprcp_1h: Decimal    # 最大1時間降水量
    time_maxprcp_1h: str   # 最大1時間降水量の時刻

# 観測データ
class Observation(TypedDict):
    mintemp: Decimal       # 最低気温（観測値）
    maxtemp: Decimal       # 最高気温（観測値）
    prcp: Decimal          # 降水量
    wind: Decimal          # 風速
    wind_dir: str          # 風向
    snow: Decimal          # 積雪量
    maxprcp_1h: Decimal    # 最大1時間降水量
    time_maxprcp_1h: str   # 発生時刻
    maxprcp_1h_kind: str   # データ種別（observation/forecast）

# 日別平年値
class NormalDay(TypedDict):
    nml: Decimal           # 平年値
    quantile: tuple        # 分位数（6個）
    stat_years: int        # 統計年数

# 月別平年値
class NormalMonth(TypedDict):
    maxtemp_lt0_frequency: float    # 真冬日頻度
    maxtemp_ge25_frequency: float   # 夏日頻度
    maxtemp_ge30_frequency: float   # 真夏日頻度
    maxtemp_ge35_frequency: float   # 猛暑日頻度
    mintemp_lt0_frequency: float    # 冬日頻度
    mintemp_ge25_frequency: float   # 熱帯夜頻度
    stat_years: int                 # 統計年数

# 季節別平年値
class NormalSeason(TypedDict):
    quantile: tuple        # 分位数（6個）
    stat_days: int         # 統計日数
    stat_years: int        # 統計年数（風速用）
```

### 出力データ構造

```python
# 最終出力（JSON）
{
    "info": {
        "point": str,          # 地点コード
        "name": str,           # 地点名
        "date": list[str],     # 日付リスト（3日分）
        "ann_time": str,       # 発表時刻
        "ini_time": str,       # 初期時刻
        "amedas_time": str     # AMeDAS最新時刻
    },
    "weather_comment": {
        "yesterday": str,      # きのうのコメント
        "today": str,          # きょうのコメント
        "tomorrow": str        # あしたのコメント
    },
    "yesterday_Pressure_Pattern": str,  # きのうの気圧配置
    "today_Pressure_Pattern": str,      # きょうの気圧配置
    "tomorrow_Pressure_Pattern": str,   # あしたの気圧配置
    # ... その他の気象データ
}
```

---

## 処理フロー詳細

### 1. 初期化フェーズ（main.py）

```
1. コマンドライン引数から ann_time を取得
   - 例: "2024-10-11_15" または "latest"

2. 25地点のリストをループ処理
   - 各地点: ["A11016", "稚内"], ["A14163", "札幌"], ...

3. 時刻調整処理（adjust_time関数）
   - 有効時刻: 2, 8, 14, 20時に調整
   - 例: 15時 → 14時に調整

4. 出力ファイルパス構築
   - JSON: /home/devel/mnt/optn/archives/PF_text/v3/json/YYYY-MM/DD_HH/
   - GZIP: /home/devel/mnt/optn/archives/PF_text/v3/YYYY-MM/DD_HH/
```

### 2. データ取得フェーズ（get.py → get_dict 関数）

```
1. AMeDASデータベース接続
   └─ amedas_tsdb.py: connect_amedas_tsdb()
      - ホスト: 192.168.110.74:5432
      - データベース: amedas_tsdb

2. 予報データ取得
   └─ forecast.py: get_forecast()
      - TokyoCabinet/TokyoTyrantから読み込み
      - ファイル: /mnt/isilon/archives/PF3.2/YYYY-MM/PF_YYYY-MM-DD_HH.tch
      - 最新: 192.168.110.34:17851

3. 観測データ取得
   └─ observation.py: get_observation()
      - PostgreSQLから読み込み
      - テーブル: t_obs_temp_10min_pp, v_obs_prcp_10min, v_obs_wind_10min, v_obs_snow_hour

4. 平年値データ取得
   ├─ normal_day.py: get_normal_day()      → m_nml_temp_day
   ├─ normal_month.py: get_normal_month()  → m_nml_temp_month
   ├─ normal_season.py: get_normal_season_*() → m_nml_prcp_day_pp, m_nml_wind_mb10d, m_nml_snow_day_pp
   └─ normal_year.py: get_normal_year()    → m_nml_temp_year_pp

5. 天気予報データ取得
   └─ moji_weather.py: get_moji_data()
      - API: https://hima.weathermap.co.jp/work/moji/get.php
```

### 3. 気圧パターン分類フェーズ（batchSOM.py）

```
1. SOMモデルの初期化/読み込み
   └─ train_som_classifier()
      - モデルファイル: app/som.pkl, scaler.pkl, pca.pkl, som_params.pkl
      - 存在しない場合は学習を実行

2. 気圧データ取得
   - GSM GPVファイルから海面気圧データを取得
   - パス: /home/devel/mnt/nfs41/center/GSM_gl/ または /home/devel/work_C/work_ytakano/GSMGPV/GSM_gl/

3. 前処理
   - GRIB2 → NetCDF変換（wgrib2コマンド）
   - 緯度経度範囲: 15-55°N, 115-155°E
   - 領域平均を引いて偏差を計算

4. SOM分類
   - 10×10のSOMマップにマッピング
   - ノード位置から気圧配置タイプを判定:
     - 1: 冬型の気圧配置
     - 2: 夏型の気圧配置
     - 3: 梅雨型の気圧配置
     - 4: 台風
     - 0: 該当なし
```

### 4. 天気コード処理フェーズ（weather_code.py）

```
1. CSVから天気コード取得
   └─ get_weather_code()
      - ファイル: /home/devel/mnt/isilon/archives/PR1H3H/YYYY-MM/PR-city_YYYY-MM-DD.csv
      - 地点コード変換: A44132 → C13221

2. 天気コード補正
   └─ weather_code_fix()
      - 8コマ分の天気マーク（3時間ごと）を1つのテロップに統合
      - 晴れ・曇り・雨・雪・雷・暴風雨などの複合判定
```

### 5. 重要度計算フェーズ（get.py 内の各関数）

```
1. 気温の重要度計算
   ├─ get_temp_anom_normal(): 平年差の重要度
   │   - 分位数から確率を算出
   │   - importance = -log10(p) (情報量)
   └─ get_temp_anom_prev(): 前日差の重要度
       - 正規分布でz値を計算

2. 降水量の重要度計算
   └─ get_prcp_day(), get_maxprcp_1h_day()
       - 対数変換した分位数補間で確率算出

3. 風速・積雪の重要度計算
   └─ get_wind_day(), get_snow_day()
       - 同様に分位数補間で確率算出
```

### 6. キーワード生成フェーズ（create_keywords.py）

```
入力: weather_data（日ごとの気象データ）

処理:
1. 気温偏差からキーワード生成
   - -5℃未満: "平年よりも大幅に低い"
   - -5～-1.5℃: "平年よりも低い"
   - ±1.5℃以内: "平年並み"
   - 1.5～5℃: "平年よりも高い"
   - 5℃以上: "平年よりも大幅に高い"

2. 降水量からキーワード生成
   - 30mm以上: "大雨"
   - 1mm以下: "小雨"

3. 風速からキーワード生成
   - 5-10m/s: "穏やかな風"
   - 10-15m/s: "やや強い風"
   - 15-20m/s: "強い風"
   - 20-30m/s: "非常に強い風"
   - 30m/s以上: "猛烈な風"

4. 気圧配置からキーワード生成
   - 1: "冬型の気圧配置"
   - 2: "夏型の気圧配置"
   - 3: "梅雨型の気圧配置"
   - 4: "台風"

出力: キーワード辞書
```

### 7. テキスト生成フェーズ（create_weather_text.py）

```
入力: weather_data + keywords

処理:
1. 天気予報情報をテキスト化
2. 最低気温・最高気温の情報を整形
3. 降水量・最大1時間降水量の情報を整形
4. 風速・積雪量の情報を整形
5. 重要度を付記

出力: プロンプト用テキスト
```

### 8. コメント生成フェーズ（chatgpt.py）

```
1. プロンプト構築
   - システムプロンプト: 気象予報士として振る舞う指示
   - ユーザープロンプト: 気象データテキスト

2. API呼び出し
   - モデル: o1-preview
   - APIキー: 環境変数 OpenAI_KEY_TOKEN または .envファイル

3. 後処理
   - マークダウン記号の除去
   - 冒頭文字のチェック（"きのうは"/"きょうは"/"あしたは"で始まるか）
   - 不適切な場合は再生成（最大10000回）
```

---

## ファイル間の依存関係

```
main.py
  └── get.py
        ├── app/amedas_tsdb.py ─────────────────┐
        ├── app/forecast.py                      │
        │     └── app/tokyocabinet.py ──────────┤
        ├── app/observation.py                   │
        │     └── app/calculation.py             ├── PostgreSQL (amedas_tsdb)
        │           └── app/forecast.py          │
        ├── app/normal_day.py ──────────────────┤
        ├── app/normal_month.py ────────────────┤
        ├── app/normal_season.py ───────────────┤
        ├── app/normal_year.py ─────────────────┘
        ├── app/moji_weather.py ──────────────────── Moji API
        ├── app/weather_code.py ──────────────────── PR1H3H CSV
        ├── app/batchSOM.py ──────────────────────── GSM GPV (GRIB2/NetCDF)
        │     └── app/check_winter_pattern.py ────── winter_weather_pattern.nc
        ├── app/create_keywords.py
        ├── app/create_weather_text.py
        └── app/chatgpt.py ───────────────────────── OpenAI API

get_latest.py
  └── app/tokyocabinet.py ────────────────────────── TokyoTyrant

front_line.py（未使用）
  └── NetCDFファイル
```

---

## データソース詳細

### 1. PostgreSQL データベース（amedas_tsdb）

| テーブル/ビュー       | 用途                                    |
| --------------------- | --------------------------------------- |
| `t_obs_temp_10min_pp` | 10 分毎気温観測（日最低・最高気温含む） |
| `v_obs_prcp_10min`    | 10 分毎降水量観測                       |
| `v_obs_wind_10min`    | 10 分毎風速観測                         |
| `v_obs_snow_hour`     | 1 時間毎積雪観測                        |
| `t_sta_temp_day_pp`   | 日統計気温                              |
| `v_sta_prcp_day`      | 日統計降水量                            |
| `v_sta_wind_day`      | 日統計風速                              |
| `v_sta_snow_day`      | 日統計積雪                              |
| `t_sta_prcp_10min`    | 10 分毎降水量統計（最大 1 時間降水量）  |
| `m_nml_temp_day`      | 日別気温平年値                          |
| `m_nml_temp_month`    | 月別気温平年値                          |
| `m_nml_prcp_day_pp`   | 日別降水量平年値                        |
| `m_nml_wind_mb10d`    | 旬別風速平年値                          |
| `m_nml_snow_day_pp`   | 日別積雪平年値                          |
| `m_nml_temp_year_pp`  | 年別気温統計                            |
| `v_obs_latest`        | 最新観測時刻                            |

### 2. TokyoCabinet/TokyoTyrant（PF 予報）

| 接続先                                                    | 用途                        |
| --------------------------------------------------------- | --------------------------- |
| `192.168.110.34:17851`                                    | 最新 PF 予報（TokyoTyrant） |
| `/mnt/isilon/archives/PF3.2/YYYY-MM/PF_YYYY-MM-DD_HH.tch` | 過去 PF 予報ファイル        |
| `/mnt/isilon/archives/PF3.1/YYYY-MM/PF_YYYY-MM-DD_HH.tch` | 旧バージョン予報ファイル    |

### 3. GSM GPV データ

| パス                                                    | 用途                    |
| ------------------------------------------------------- | ----------------------- |
| `/home/devel/mnt/nfs41/center/GSM_gl/`                  | 最近 5 日以内の GSM GPV |
| `/home/devel/work_C/work_ytakano/GSMGPV/GSM_gl/YYYYMM/` | 過去の GSM GPV          |

### 4. その他ファイル

| パス                                                         | 用途                          |
| ------------------------------------------------------------ | ----------------------------- |
| `/mnt/isilon/archives/PR1H3H/YYYY-MM/PR-city_YYYY-MM-DD.csv` | 3 時間毎天気予報              |
| `/mnt/calc43/archives/GUID_grid_nc/MSM/`                     | MSM 降雪予報（NetCDF）        |
| `/home/devel/contents_C/PF_text/wm_nc/`                      | 気圧配置パターン判定用 NetCDF |

---

## 対象地点一覧

```python
points = [
    ["A11016", "稚内"],    ["A14163", "札幌"],    ["A12442", "旭川"],
    ["A19432", "釧路"],    ["A23232", "函館"],    ["A31312", "青森"],
    ["A33431", "盛岡"],    ["A34392", "仙台"],    ["A43056", "熊谷"],
    ["A44132", "東京"],    ["A45212", "千葉"],    ["A46106", "横浜"],
    ["A48156", "長野"],    ["A54232", "新潟"],    ["A56227", "金沢"],
    ["A51106", "名古屋"],  ["A62078", "大阪"],    ["A67437", "広島"],
    ["A68132", "松江"],    ["A74182", "高知"],    ["A82182", "福岡"],
    ["A88317", "鹿児島"],  ["A88836", "名瀬"],    ["A91197", "那覇"],
    ["A94081", "石垣島"]
]
```

---

## 重要度（importance）計算アルゴリズム

### 情報量ベースの重要度

```
importance = information + impact

information = -log10(p)
  - p: 確率（稀少さの指標）
  - 確率が低いほど情報量が高い

impact = 0（現在は未使用）
```

### 分位数からの確率算出

```python
# 気温の場合
if value <= q000:
    p = 1/stat_years  # 最小値以下
elif value <= q010:
    # q000-q010間を正規分布で補間
    n = n000 + fac * (n010 - n000)
    p = norm.cdf(n)
# ... 以下同様

# 降水量・積雪・風速の場合
# 対数変換した値で線形補間
lnv = np.log(v + 1)
lnq = np.log(q + 1)
fac = (lnv - lnq_lower) / (lnq_upper - lnq_lower)
p = p_lower + fac * (p_upper - p_lower)
```

---

## エラーハンドリング

### main.py

- 各地点の処理で Exception をキャッチし、エラーメッセージを出力して次の地点へ継続

### get.py

- トップレベルで Exception をキャッチし、トレースバックを JSON 形式で返却
- データ不足時は None 値で代替

### chatgpt.py

- API エラー時はエラーメッセージを返却
- 不適切な応答は最大 10000 回再生成を試行

---

## 実行方法

```bash
# 最新データで実行
python main.py latest

# 特定日時で実行
python main.py 2024-10-11_15

# 単一地点のテスト（get.py直接実行）
python get.py
# ※ get.py内のget_query()でデフォルト値を設定
```

---

## 出力ファイル

```
/home/devel/mnt/optn/archives/PF_text/v3/
├── json/
│   └── YYYY-MM/
│       └── DD_HH/
│           └── PF-text_YYYY-MM-DD_HH_Axxxxx.json
└── YYYY-MM/
    └── DD_HH/
        └── PF-text_YYYY-MM-DD_HH_Axxxxx.json.gz
```

---

## 注意事項・改善点

1. **chatgpt.py の再試行回数**: 最大 10000 回の再生成は過剰。適切な上限（10-20 回程度）に変更推奨

2. **front_line.py**: 現在 get.py から呼び出されていないが、前線解析機能として実装済み。必要に応じて統合可能

3. **エラーログ**: main.py のエラー出力はコメントアウトされている部分あり。本番環境では適切なログ出力を推奨

4. **API キー管理**: .env ファイルまたは環境変数での管理。セキュリティ上、環境変数を推奨

5. **データベース接続**: 接続情報がハードコードされている。設定ファイルまたは環境変数での管理を推奨

---

## 更新履歴

| 日付       | 内容     |
| ---------- | -------- |
| 2024-12-29 | 初版作成 |
