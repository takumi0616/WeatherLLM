# WeatherLLM PF_text プログラム仕様書

本ドキュメントは、`src/WeatherLLM/document_PF_text/document/program` 以下の一連のプログラム群の処理内容・入出力・変数の流れ・アルゴリズム・ファイル間依存関係を詳細に整理したものです。最終成果物は各地点・各発表時刻に対応する「PF-text*YYYY-MM-DD_HH*<POINT>.json(.gz)」で、地点ごとの「きのう・きょう・あした」の数値情報と重要度評価、さらに LLM で生成した天気コメントを含みます。

対象ファイル一覧（実装の主な役割）

- app/amedas_tsdb.py: AMeDAS DB(PostgreSQL)接続
- app/batchSOM.py: PRMSL（海面更正気圧）SOM 分類・天気図作成・圧配置判定
- app/calculation.py: 当日の降水/風/降雪（欠測補完を含む）集計、MSM NetCDF からの降雪予報取得
- app/chatgpt.py: OpenAI API を用いた天気コメント生成
- app/check_winter_pattern.py: 冬型気圧配置 × ノード × 地点の整合チェック
- app/create_keywords.py: キーワード抽出（閾値ベース）
- app/create_weather_text.py: LLM 入力用「気象テキスト」生成
- app/forecast.py: PF(Tokyo Cabinet)からの予報読み込み・日毎予報作成
- app/moji_weather.py: Web API からピンポイント天気「文字天気」取得・コード → 日本語変換
- app/normal_day.py / normal_month.py / normal_season.py / normal_year.py: 平年値・分位点等の取得
- app/observation.py: 観測値取得と日別集約
- app/tokyocabinet.py: Tokyo Cabinet/Tyrant ctypes ラッパ
- app/weather_code.py: PR1H3H CSV から天気コード取得 + コード補正ロジック
- front_line.py: 前線推定（現時点では main から未連携）
- get.py: API ハンドラ相当（1 地点分の JSON 組み立て）
- main.py: 複数地点一括処理し JSON/JSON.GZ 保存
- get_latest.py: 最新 ini_time 返却 CGI
- pressure_pattern_label.py: SOM 学習用ラベル・日付リスト群

---

## 全体フロー（概要）

1. 入力決定

- 地点 `point`（例 "A44132"）と発表時刻 `ann`（例 "2023-12-17_08" or "latest"）を受け取る（get.py → get_query）。

2. 予報データの読み込み

- PF（Tokyo Cabinet/Tyrant）から「日別」予報を構築（forecast.py）。
  - D-1, D0(きょう), D1(あした) について、T_min/T_max/R_sum/W_spd/W_dir 等を取得。
  - 1 時間最大降水は timeE 軸を用い 1 時間ごとに探索（get_max_hourly_prcp_forecast）。

3. 観測データの取得・当日補完

- AMeDAS DB(PostgreSQL)から日別の観測集計を取得（observation.py）。
  - t*obs_temp_10min_pp, v_obs_prcp_10min, v_obs_wind_10min, v_obs_snow_hour, t_sta*_ / v*sta*_ 系を利用し、フラグ（_\_aqc / _\_flg）で品質の良い値を採用。
- 当日の欠測部分は PF の時系列（timeA/timeE）で補完（calculation.py）。
  - 降雪は MSM NetCDF から地点最近傍の時間帯合計を取得し補完（get_snow_forecast_from_netcdf）。

4. 平年値・分位情報の取得

- 最低/最高気温の平年値・分位点（m_nml_temp_day）
- 月別の頻度（夏日/真夏日/猛暑日/真冬日/熱帯夜）（m_nml_temp_month）
- 降水・最大 1 時間降水・降雪・風速の分位/統計日数（m_nml_prcp_day_pp, m_nml_snow_day_pp, m_nml_wind_mb10d）
- 日次差の標準偏差（m_nml_temp_year_pp）

5. 重要度（information）評価

- 分位に基づく -log10(確率) 等で「情報量」を計算（get.py の各 get\_\*\_day 系）。
- 影響度は現状 0 として合算 → importance として記録。

6. ピンポイント天気・天気コード処理

- 文字天気 API 取得（moji_weather.py）→「Wd/Ws の日本語変換」「時間帯別 Ws_text」「簡易 importance=1」。
- PR1H3H CSV から地点（A→C 変換）で天気コード 8 枠（03,06,...,24 時）を取得（weather_code.py）。
- moji Wd と組合せ、PHP ロジック移植の `weather_code_fix` で代表天気コード（telop）を補正。

7. 気圧配置判定（SOM）

- 2016-03-01〜2024-10-31 の PRMSL（海面更正気圧）データから PCA→MiniSom 学習 or 読込（batchSOM.py）。
- 最新 3 日（きのう/きょう/あした）について、GSM GPV_GRIB2→wgrib2 で PRMSL 抽出 →NetCDF→ 共通格子再補間 → 領域平均偏差 →PCA→SOM へ写像。
- 指定ノード集合に基づき「冬(1)/夏(2)/梅雨(3)/台風(4)/その他(0)」判定。勝者ノード(row,col)も返す。
- 同時に SLP 偏差図を保存（/home/devel/contents_C/PF_text/map/）。

8. 冬型整合チェック

- `check_winter_pattern.py` で
  - 雪/雨/晴れの粗いカテゴリ化（Wd 先頭桁 → 1:晴 2:曇 3:雨/雪）
  - 全国冬型マップとノード別マップ（NetCDF）から該地点格子の weather_code を比較
  - 一致すれば 1、違えば 0。

9. LLM 用テキストとキーワード

- `create_keywords.py` の閾値ベースで文言候補を付与。
- `create_weather_text.py` で、日別の値・重要度・キーワード・時間帯天気（Ws_text）などを含む日本語テキストを生成。

10. 天気コメント生成（LLM）

- `chatgpt.py` が OpenAI Chat Completions を呼び、プロンプト条件（最初の文字は「〇〇は、」等）に合わせて きのう/きょう/あした の自然文コメントを生成。
- 「あした」生成時に「きのう」の語が混入したら繰返し生成。

11. JSON 出力

- `get.py` は 1 地点分の JSON を組み立てて返却。
- `main.py` は 25 地点リストを順次処理し、ann*time を 2/8/14/20 時に丸め、`/home/devel/mnt/optn/archives/PF_text/v3/{YYYY-MM}/{DD}*{HH}/PF-text*{YYYY-MM-DD_HH}*{POINT}.json(.gz)` として保存。

---

## 主要出力の構造（get.py の戻り JSON）

info

- point: 入力地点（例 "A44132"）
- name: 地点名（PF から）
- date: ["YYYY-MM-DD", ...] 3 要素（D-1, D0, D1）
- ann_time: 発表時刻（"YYYY-MM-DD_HH"、"時発表"除去済）
- ini_time: モデル初期時刻（PF 内 info）
- amedas_time: AMeDAS 最新 record_time

weather_comment

- yesterday/today/tomorrow: LLM 生成の日本語文（プロンプト制約済）

yesterday_Pressure_Pattern / today_Pressure_Pattern / tomorrow_Pressure_Pattern

- "冬型の気圧配置" / "夏型の気圧配置" / "梅雨型の気圧配置" / "台風" / "該当なし"

is\_\*\_winter_pressuer_pattern

- きのう/きょう/あした それぞれで `check_winter_pattern` 一致判定（0/1）

mintemp / maxtemp

- 各日の {value: Decimal, kind: "observation"/"forecast"}

mintemp_day / maxtemp_day

- {"value": "冬日/真冬日/夏日/真夏日/猛暑日/熱帯夜 または 空文字", "importance": float}

mintemp_anom_normal / maxtemp_anom_normal

- {"value": 観測(予報) − 平年値, "importance": float}

mintemp_anom_prev / maxtemp_anom_prev

- 前日差 {"value": today - yesterday, "importance": float}（標準偏差で正規化し N(0,1)前提の片側確率を情報量化）

prcp_day

- {"value": 日降水量 mm, "kind": "observation"/"forecast", "importance": float}

maxprcp_1h

- {"value": 最大 1h 降水 mm, "time": "YYYY-MM-DD HH:MM:SS"/"none", "kind": "observation"/"forecast", "importance": float}

wind

- {"value": 平均風速 m/s, "dir": 16 方位の英略（N, NE...）/ "none", "kind": "...", "importance": float}

snow

- {"value": 日降雪量 cm, "kind": "...", "importance": float}

weather

- 文字天気（moji）由来の辞書に Ws_text 等を追加し、importance=1 を付与済（moji_weather.py）

---

## 重要な変数と由来・加工・出力先

- point（文字列）: 入力地点。`get_query()` で環境変数 QUERY_STRING またはデフォルト。
- ann（文字列）: 発表時刻。例 "2023-12-17_08" / "latest"。
- forecast_list（list[Forecast]）
  - forecast.py / PF（Tokyo Cabinet/Tyrant）由来。各日 {"date","mintemp","maxtemp","prcp","wind","wind_dir","snow","maxprcp_1h","time_maxprcp_1h"}。
  - D0/D1 の snow は MSM NetCDF 合計（calculation.get_snow_forecast_from_netcdf）。
  - D-1 は数値なし（None）。
- observation_list（list[Observation]）
  - observation.py / AMeDAS DB 由来。日別に観測か統計（t_sta/v_sta）を採用。
  - 当日（current_date）は calculation.py の補完ロジックで PF から穴埋め。
- normal\_\* 系（NormalDay/Month/Season/Year）
  - AMeDAS 正規化用の平年値・分位点、日次差の SD を DB から取得。
- date_list（list[str]）
  - forecast_list から抽出。D-1, D0, D1。
- moji_data_list（list[dict|None]）
  - 文字天気 API（moji_weather.get_moji_data）由来 + get_weather_day で日本語化・整形。
- weather_codes（list[str]）, wd_values（list[int|None]）
  - PR1H3H CSV（/home/devel/mnt/isilon/archives/PR1H3H/）から A→C コード変換で抽出 → moji Wd と組合せ weather_code_fix で補正 telop。
- pressuer_pattern（list[int]）
  - タイポに注意（pressuer）。きのう/きょう/あしたごとに 0/1/2/3/4（その他/冬/夏/梅雨/台風）。
  - batchSOM.judgePressurePattern の出力に基づく。
- node*row/col*\*（int|None）
  - SOM 勝者ノード座標。`check_winter_pattern` に渡される。
- weather_comment（dict）
  - chatgpt.generate_weather_comment の最終生成文を格納。

---

## 主要モジュール詳細

### 1) app/amedas_tsdb.py

- `connect_amedas_tsdb() -> (conn, cur)`
  - DSN: postgresql://read_only:read_only@192.168.110.74:5432/amedas_tsdb
  - 返却: psycopg2 接続・カーソル

### 2) app/forecast.py

- `read_PF(ann, points)`
  - ann="latest" → Tokyo Tyrant 192.168.110.34:17851 から key="info" と各 point 取得。
  - ann 固定 → PF3.2/3.1 アーカイブ `.tch` を `TCH` で読む。
- `get_forecast(ann, point)`
  - PF から D-1, D0, D1 を作り、D0/D1 の温湿・降水・風等を `format_PF_temp()` で Decimal 化。
  - D0/D1 の最大 1h 降水は `get_max_hourly_prcp_forecast()` が timeE キー（"prev_current"）対応で全時間検索して最大値と時刻を返す。
  - D1 の snow は calculation.get_snow_forecast_from_netcdf を使用。
  - 返却: (list[Forecast], 地点名, ann_time(整形), ini_time, 経度, 緯度)
- `format_PF_temp(s: str) -> Decimal|None` 空文字は None、それ以外 Decimal

### 3) app/observation.py

DB から日別観測/統計を取得し、品質フラグで採用/棄却。

- 気温:
  - 最低: t_obs_temp_10min_pp.mintemp_0009（07-09 時のレコード範囲で最新 1 件）
  - 最高: t_obs_temp_10min_pp.maxtemp_0918（16-18 時で最新 1 件）
- 降水:
  - 日降水: v_obs_prcp_10min.prcp_24h（翌日 00:00:00）
  - 最大 1h: t_sta_prcp_10min の範囲検索で最大値（時間も）
- 風:
  - v_obs_wind_10min を当日 00-24h で抽出し、品質<=1 の平均と最頻風向（dir16）を算出（四捨五入 0.1）。
- 雪:
  - v_obs_snow_hour.snow_24h（翌日 00:00:00）
- 当日（current*date）のみ calculation.py の `get_today*\*` に委譲し、観測欠損は PF で補完（降雪は MSM NetCDF）。
- `get_amedas_record_time` で v_obs_latest 最新時刻を取得（get.py でも同関数名あり）。

戻り(Observation):

- {"mintemp","maxtemp","prcp","wind","wind_dir","snow","maxprcp_1h","time_maxprcp_1h","maxprcp_1h_kind"}

注意:

- 型注釈の戻りタプル記述が古いが、実処理は `obs_list` のみ返却。

### 4) app/calculation.py

当日（きょう）の「観測＋ PF 補完」処理。

- `get_today_prcp`:
  - 1-24 時の各スロットが無ければ PF timeE から "prev_current" キーで rain1h を補完し合計。
- `get_today_wind`:
  - 観測の品質<=1 のみ平均風速・最頻風向。欠損は PF timeA の wind/wind_dir で補完。
- `get_today_snow`:
  - 観測 snow_1h の欠測スロットは MSM NetCDF（/mnt/calc43/archives/GUID_grid_nc/MSM/... snow 変数, 最近傍格子, 時間範囲）で補完。合計を cm 表現（値 ×100）で Decimal 化。
- `get_snow_forecast_from_netcdf`:
  - start_time を 3 時間刻みに下げ、JST→UTC→UNIX 秒で time 選択し合計。
- `get_today_maxprcp_1h`:
  - 当日日中までの観測最大 1h 降水と、最新観測時刻以降の PF 最大 1h 予報を比較し大きい方を採用。kind="observation"/"forecast"/"none"。

### 5) app/normal_day.py / normal_month.py / normal_season.py / normal_year.py

- normal_day: m_nml_temp_day から最低/最高気温の平年値と 6 分位点、統計年数。
- normal_month: m_nml_temp_month から月内頻度（日数/当月日数）へ換算。
- normal_season:
  - prcp / maxprcp_1h: m_nml_prcp_day_pp の 6 分位 + 統計日数
  - wind: m_nml_wind_mb10d（上中下旬単位）6 分位 + 統計年数
  - snow: m_nml_snow_day_pp 6 分位 + 統計日数
- normal_year: m_nml_temp_year_pp から日次差の標準偏差を取得（最低 0-9 時, 最高 9-18 時）。

### 6) app/moji_weather.py

- `get_moji_data(point, ann)`:
  - https://hima.weathermap.co.jp/work/moji/get.php?P=<POINT>&ann=<ANN> に GET（JSON）。
- `get_weather_day(moji_data_list)`:
  - Wd/Ws を日本語に変換（大きな変換テーブル内蔵）。
  - 8 つの時間帯（未明〜夜遅く）に対し Ws_text を作成。
  - importance=1 を付与（情報量は簡易的）。
- `get_moji_list(moji)`:
  - API が dict/配列どちらでも配列に正規化。

### 7) app/weather_code.py

- `convert_points`: A→C 地点番号変換表（PR1H3H CSV の地点番号は C 系）。
- `get_weather_code(moji_data_list, first_date, point)`:
  - `/home/devel/mnt/isilon/archives/PR1H3H/{YYYY-MM}/PR-city_{YYYY-MM-DD}.csv` を開き、地点一致行から「天気 03,06,...,24」を抽出。
  - moji `Wd` も横並びに抽出。
- `weather_code_fix(weather_codes, wd_values)`:
  - 8 コマ（FN=8）想定の分布・重み付け・雷/暴風/吹雪のルールベースで telop を導出する大規模ロジック（PHP 相当の移植）。
  - 最終的に `wd_values[0] = telop` を返し、代表コードを 1 件に集約。

### 8) app/batchSOM.py

学習（初回） or 保存済モデル（som/scaler/pca/パラメータ）を読み込み:

- 学習データ
  - `/home/devel/work_C/work_ytakano/GSMGPV/nc/prmsl/` 下の NetCDF (2016-03-01〜2024-10-31)。
  - PRMSL_meansealevel[hPa] を 15-55N, 115-155E で切り出し → 共通格子(0.5°)に補間 → 領域平均を引いた偏差 → フラット化 →StandardScaler→PCA(n=20)。
- SOM
  - MiniSom(10x10, sigma=3, lr=1, batch 100,000 iter, random_seed=0)。
  - 勝者ノードに対し、冬/夏/梅雨/台風の手動ノード集合（winter_nodes 等）を定義。
- ラベル教師情報
  - `pressure_pattern_label.py` の data_label_dict（英字ラベル）および winter/summer/rain/typhoon_dates_list を併用し期間内フィルタ → ノードごとの出現数参考。

関数

- `train_som_classifier() -> (classify_new_data, scaler, pca, common_lat, common_lon, expected_shape, lat_range, lon_range)`
  - 学習済を pickle 読み込み or 新規学習し保存（app/som.pkl, scaler.pkl, pca.pkl, som_params.pkl）。
  - `classify_new_data(new_data_array)` はラベル(0〜4)と勝者ノード(row,col)を返す。
- `judgePressurePattern(date_list, ann_time, ...)`
  - 入力: 3 日の日付・ann_time。各日 "YYYY-MM-DD_HH" を JST→UTC、基準日時（first day 0UTC or first_day 18UTC 等）に合わせ FD コード（予報リード）を算出し、GSM GPV の GRIB2 ファイル名を決定。
  - 直近 5 日以内: `/home/devel/mnt/nfs41/center/GSM_gl/`、それ以前: `/home/devel/work_C/work_ytakano/GSMGPV/GSM_gl/{YYYYMM}` を探索。
  - `wgrib2` (ncpu 1, ":PRMSL mean sea level:" match) で NetCDF 変換（保存先 `/home/devel/contents_C/PF_text/nc`）。
  - NetCDF 読み込み → 座標名（lat/lon → latitude/longitude）正規化 → 日本付近切出し → 共通格子に補間 → 領域平均を引いた偏差を scaler→pca→som へ。
  - 各日の SLP 偏差マップを作図・保存（等値塗り＋等値線、coastline、範囲 115-155E/15-55N）。
  - 出力: 各日の圧配置ラベル(0-4)と SOM ノード(row,col)。

### 9) app/check_winter_pattern.py

- `/home/devel/contents_C/PF_text/wm_nc/winter_weather_pattern.nc`（全国の冬型 weather_code マップ）
- `/home/devel/contents_C/PF_text/wm_nc/all_nodes_weather_patterns.nc`（SOM ノード別マップ）
- 入力: (lon, lat, wd_value, node_row, node_col)
  - `wd_value`は 3 桁コードの先頭桁から 1:晴/2:曇/3:雨/雪 に変換。
  - 該地点の winter/weather_code, ノード subset の weather_code を最近傍で取得。
  - 3 者一致なら 1、不一致なら 0。

### 10) app/create_keywords.py

- 閾値ルールに基づきキーワード配列を作成:
  - 最低/最高気温の平年差・前日差に応じた「平年より高い/低い」「きのうより高い/低い」など
  - 日降水・最大 1 時間降水（強度語彙と時間帯表現）
  - 風速（強度語彙）+ 風向 16 方位を日本語化して附加
  - 降雪（強度語彙）
  - 気圧配置ラベル（1:冬/2:夏/3:梅雨/4:台風）→ 対応語句
- 戻り値を weather_data に `.update()` して LLM 入力文へ反映。

### 11) app/create_weather_text.py

- LLM への「説明用テキスト」を組み立て（Markdown 行）。
- 含むもの:
  - 日付・天気予報見出し
  - 気圧配置キーワード
  - 最低/最高気温（観測/予想、平年差・前日差と重要度）
  - 日降水・最大 1h 降水（重要度・発生時刻）
  - 風速（重要度）
  - 積雪（重要度）
  - 文字天気の時間帯 Ws_text
- print でデバッグ出力（標準出力）。

### 12) app/chatgpt.py

- OpenAI API キーの探索順:
  - 環境変数 OpenAI_KEY_TOKEN / OPENAI_KEY_TOKEN
  - .env 探索（python-dotenv 試行）or 直読み
- `generate_weather_comment(weather_text, date_label, date, point, point_place)`
  - 日本語の厳格なルールをプロンプト化（最初の文頭は「{date_label}は、」など）。
  - model: "o1-preview"（API v1/chat/completions）。
  - 返答が「きのうは/きょうは/あしたは」で始まらない、または「あした」で「きのう」が混じる場合は再生成（最大 10000 試行）。
  - Markdown 記号除去の `clean_comment` を適用。

### 13) app/tokyocabinet.py

- ctypes による TCH (HDB) / TCT (TDB) / TT (Tokyo Tyrant) の薄いラッパ。
- `get/put/open/close` と簡易の errck を実装。

### 14) front_line.py

- `refined_202306010600.nc`（probabilities[time,class,lat,lon]）から前線領域(>0)を抽出し連結成分ごとに:
  - 最近傍距離（km）、領域長（最大点間距離）、PCA で主軸方位 →16 方位に変換。
- 現状 main/get からは未使用。将来的に「前線の有無/距離/方位」特徴量として利用可能。

### 15) get.py（オーケストレーション）

- `get_query()` 環境変数 or デフォルト `point="A31312"`, `ann="2023-12-17_08"`.
- `get_dict(point, ann)` の処理手順（前節「全体フロー」を実装）
  - 途中変数:
    - `forecast_list, name, ann_time, ini_time, longitude, latitude`
    - `date_list`（D-1,D0,D1）
    - `latest_record_time`（観測最新）
    - `observation_list`
    - `normal_season_*_list`, `mintemp_normal_day_list`, `maxtemp_normal_day_list`, `normal_month_list`, `normal_year`
    - 派生の `mintemp_list` 等一式（観測優先/なければ予報、なければ 0）
    - `*_anom_normal_list`（平年差情報量付き）
    - `*_anom_prev_list`（前日差情報量付き）
    - `prcp_day_list`, `maxprcp_1h_day_list`, `wind_day_list`, `snow_day_list`（分位から情報量計算）
    - 文字天気 `moji_data_list`（first_date は ann_first = date_list[0] + "\_02" から moji を取る特別処理）
    - `weather_codes, wd_values`（PR1H3H CSV ＋ moji Wd → fix）
    - SOM 分類 `is_*_pressure_pattern` と `node_row/col_*`
    - 冬型整合 `is_*_winter_pressuer_pattern`
    - LLM 入力文 `weather_text_*` と出力 `comment_*`
  - 返却 JSON を構築。
- `main()`（スクリプト実行時）
  - 複数地点を `ann_arg`（latest も可）で処理。
  - `adjust_time` で ann_time を 2,8,14,20 の直前時刻に揃え、0 時未満は前日 20 時へシフト。
  - JSON と JSON.GZ を `/home/devel/mnt/optn/archives/PF_text/v3/{YYYY-MM}/{DD}_{HH}/` に保存。

### 16) get_latest.py

- Tyrant から `info.ini_time` を JSON 出力する簡易 CGI。

### 17) pressure_pattern_label.py

- 学習期間の「圧配置ラベル」辞書（大量）。
- 冬/夏/梅雨/台風の日付リスト（重複除去・期間内抽出は batchSOM 側）。

---

## 重要なアルゴリズム

- 欠測補完（calculation.py）
  - 降水: 観測の欠けは PF の timeE "prev_current" キーで 1 時間値を補完し日合計。
  - 風: 観測品質<=1 のみ平均。欠けは PF timeA で補完。風向は最頻値。
  - 降雪: MSM NetCDF の `snow` を地点最近傍・時間帯合計（JST→UTC→UNIX 秒で切出）。
- 重要度（情報量）
  - 一般に `information = -log10(1 - 累積確率)`（極端さの指標）。気温の平年差は両側 p を評価し上限 2。
  - 前日差は 正規分布 N(0,sd) を仮定した片側 p。
- SOM 圧配置分類
  - 共通格子(0.5°)の PRMSL 偏差（領域平均差し引き）を PCA(20)で圧縮 → SOM(10x10) でクラスタリング。
  - ノードに対し、冬/夏/梅雨/台風の手動リストでラベル付与。
- 冬型整合チェック
  - 「冬型マップ」×「ノード別マップ」×「地点の代表天気コード（粗カテゴリ）」が一致 →1。
- 代表天気コード補正（weather_code_fix）
  - 8 コマ配列の「はれ率」「降水割合」「雷/暴風/吹雪変換」「昼補正」等の規則で telop（100〜800 等）を最終決定。

---

## 外部依存とパス

- PostgreSQL: 192.168.110.74:5432 / DB: amedas_tsdb / read_only
- Tokyo Tyrant: 192.168.110.34:17851（latest の PF）
- PF アーカイブ: /mnt/isilon/archives/PF3.2 or PF3.1/{YYYY-MM}/PF\_{ann}.tch
- PR1H3H CSV: /home/devel/mnt/isilon/archives/PR1H3H/{YYYY-MM}/PR-city\_{YYYY-MM-DD}.csv
- GSM GPV GRIB2: 直近 5 日 /home/devel/mnt/nfs41/center/GSM_gl、それ以前 /home/devel/work_C/work_ytakano/GSMGPV/GSM_gl/{YYYYMM}
- wgrib2 必須（PRMSL を NetCDF 化）
  - 出力: /home/devel/contents_C/PF_text/nc
- SLP 図保存: /home/devel/contents_C/PF_text/map
- 冬型/ノードマップ NetCDF: /home/devel/contents_C/PF_text/wm_nc/...
- MSM NetCDF（降雪）: /mnt/calc43/archives/GUID*grid_nc/MSM/{YYYYMM}/MSM_GUID_Rjp_txy_snow*{YYYYMMDD}{HHMMSS}.nc
- OpenAI API: `OPENAI_KEY_TOKEN` or `.env`（OpenAI_KEY_TOKEN/他）

---

## 出力ファイル命名と保存先（main.py）

- ann_time の丸め: 2,8,14,20 の直前時刻。日付跨ぎ時は前日 20 時へ調整。
- 保存ディレクトリ:
  - JSON: /home/devel/mnt/optn/archives/PF*text/v3/json/{YYYY-MM}/{DD}*{HH}/PF-text*{YYYY-MM-DD_HH}*{POINT}.json
  - JSON.GZ: /home/devel/mnt/optn/archives/PF*text/v3/{YYYY-MM}/{DD}*{HH}/PF-text*{YYYY-MM-DD_HH}*{POINT}.json.gz

---

## ファイル間の依存関係（概要）

- get.py（中心）

  - DB: app.amedas_tsdb.connect_amedas_tsdb
  - 予報: app.forecast.get_forecast → app.forecast.read_PF → app.tokyocabinet
  - 観測: app.observation.get_observation（内部で app.calculation の当日補完ロジック）
  - 平年値: app.normal\_\* 系
  - 文字天気: app.moji_weather.get_moji_data/get_weather_day
  - 天気コード: app.weather_code.get_weather_code, weather_code_fix
  - SOM: app.batchSOM.train_som_classifier → classify_new_data / judgePressurePattern
  - 冬型整合: app.check_winter_pattern
  - キーワード: app.create_keywords
  - LLM 文面: app.create_weather_text → app.chatgpt.generate_weather_comment

- main.py
  - get.py を地点リストで繰返し、ファイル出力・圧縮保存。

---

## 注意点・実装上の留意事項

- フラグ評価
  - 観測は _\_aqc または _\_flg が 0/1（良）を採用。悪品質は None。
- None/0 の扱い
  - 多くの派生関数は None を 0 として扱うか、情報量=0 として扱う（降水・風・雪）。
- 型
  - Decimal を JSON 化のため float へ置換（main.py の replace_nan）。
- 時間軸
  - PF の timeA（風/風向）と timeE（1 時間降水）はキー形式が異なるので注意（timeE は "prev_current"）。
  - MSM は JST→UTC→UNIX 秒で NetCDF time を切取り。
- chatgpt.py
  - model 名や API 仕様変更に注意。出力冒頭の規約に合うまで最大 10000 試行。
- batchSOM.py
  - wgrib2 が必要。PRMSL の変数名はファイルによって異なるため "PRMSL" を含む変数を探索。
  - lat/lon 昇順・降順に対応して slice を切る。
  - 画像生成は依存ライブラリ（cartopy, matplotlib, seaborn）が必要。
- 命名の表記揺れ
  - `pressuer_pattern`（typo）でキーが作られている。外部連携時はキー名に注意。

---

## 実行例

- 1 地点の JSON を標準出力（CGI 想定）

  - `get.py` を Web 経由で呼出（QUERY_STRING `point=A44132&ann=latest` 等）

- バッチ生成
  - `main.py latest`
  - 25 地点すべてについて ann_time を丸め、所定ディレクトリに JSON / JSON.GZ を保存。

---

## 用語・略記

- AMeDAS: 気象庁 地域気象観測所
- PF: ピンポイント予報データ（Tokyo Cabinet/Tyrant）
- PRMSL: 海面更正気圧
- SOM: Self-Organizing Map（自己組織化マップ）
- MSM: メソ数値予報モデル（NetCDF snow）

---

## 付録：主要 SQL（抜粋・参照）

- v_obs_prcp_10min: (record_time, prcp_1h, prcp_24h)
- v_obs_wind_10min: (wind_speed, wind_speed_aqc, wind_dir16, wind_dir16_aqc)
- v_obs_snow_hour: (snow_1h, snow_24h)
- t_sta_prcp_10min: (maxprcp_1h, maxprcp_1h_time)
- t_obs_temp_10min_pp: (mintemp_0009, mintemp_0009_aqc, maxtemp_0918, maxtemp_0918_aqc)
- v_sta_prcp_day: (prcp, prcp_flg)
- v_sta_wind_day: (meanwind, meanwind_flg, modedir16, modedir16_flg)
- v_sta_snow_day: (snow, snow_flg)
- v_obs_latest: (record_time)
- m_nml_temp_day: （最低/最高気温 平年値と 6 分位, stat_years）
- m_nml_temp_month: （各種日数, stat_years）
- m_nml_prcp_day_pp: （降水/最大 1h 降水 6 分位, stat_days）
- m_nml_wind_mb10d: （風速 6 分位, stat_years）
- m_nml_snow_day_pp: （降雪 6 分位, stat_days）
- m_nml_temp_year_pp: （mintemp_0009_diff_sd, maxtemp_0918_diff_sd）

本仕様書は実装の全体像とデータフロー、外部依存、変数の由来と利用先を俯瞰できるよう構成しています。運用・拡張時は各モジュールの I/O と副作用（ファイル保存や API 呼出、DB 接続）を確認してください。
