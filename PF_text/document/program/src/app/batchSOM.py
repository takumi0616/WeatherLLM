import os
import pickle  # 追加
import random
import subprocess
from collections import defaultdict
from datetime import datetime, timedelta

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def judgePressurePattern(
    date_list,
    ann_time,
    classify_new_data,
    scaler,
    pca,
    common_lat,
    common_lon,
    expected_shape,
    lat_range,
    lon_range,
):
    # 現在の日付を取得
    current_date = datetime.now().date()

    judge_date_list = []
    for idx, date in enumerate(date_list):
        if idx == 1:
            # ann_time から日付部分と時刻部分を抽出
            date_part, time_part = ann_time.split("_")
            hour = int(time_part)
            # 最近接の時刻（3, 9, 15, 21時）に調整
            possible_hours = [3, 9, 15, 21]
            adjusted_hour = min(possible_hours, key=lambda x: abs(x - hour))
            # 時刻をゼロ埋めして再構成
            adjusted_time = f"{adjusted_hour:02d}"
            adjusted_ann_time = f"{date_part}_{adjusted_time}"
            judge_date_list.append(adjusted_ann_time)
        else:
            # '_09' を付加する
            judge_date_list.append(f"{date}_09")

    # ファイルパス名を作成する処理
    file_pass = {}
    for idx, date_time_str in enumerate(judge_date_list):
        # date_time_str は 'YYYY-MM-DD_HH' の形式
        date_part, hour_part = date_time_str.split("_")
        hour_jst = int(hour_part)

        # JSTの日時オブジェクトを作成し、UTCに変換
        date_time_jst = datetime.strptime(f"{date_part}_{hour_jst:02d}", "%Y-%m-%d_%H")
        date_time_utc = date_time_jst - timedelta(hours=9)

        if idx == 0:
            # 最初のエントリの場合、基準時刻は同日の0時UTC
            base_time_utc = datetime.combine(date_time_utc.date(), datetime.min.time())
        else:
            # 他のエントリの場合、最初の日付の18時UTCが基準
            first_date_part, _ = judge_date_list[0].split("_")
            base_date = datetime.strptime(first_date_part, "%Y-%m-%d").date()
            base_time_utc = datetime.combine(
                base_date, datetime.min.time()
            ) + timedelta(hours=18)

        # 予報リードタイムを計算
        lead_time = date_time_utc - base_time_utc
        if lead_time.total_seconds() < 0:
            # リードタイムが負の場合、基準時刻を1日前に調整
            base_time_utc -= timedelta(days=1)
            lead_time = date_time_utc - base_time_utc

        # リードタイムを日と時間に変換
        total_hours = int(lead_time.total_seconds() // 3600)
        days_ahead = total_hours // 24
        hours_ahead = total_hours % 24

        # FDコードを生成
        fd_code = f"{days_ahead:02d}{hours_ahead:02d}"

        # 基準時刻をファイル名形式に変換 YYYYMMDDHH0000
        base_time_str = base_time_utc.strftime("%Y%m%d%H") + "0000"

        # ファイル名を作成
        filename = f"Z__C_RJTD_{base_time_str}_GSM_GPV_Rgl_FD{fd_code}_grib2.bin"

        # 'yesterday', 'today', 'tomorrow' にマッピング
        if idx == 0:
            key = "yesterday"
        elif idx == 1:
            key = "today"
        elif idx == 2:
            key = "tomorrow"
        else:
            key = f"day{idx}"

        # データの日付を取得
        data_date = date_time_jst.date()
        # 現在の日付との差を計算
        delta_days = (current_date - data_date).days

        # データのパスを設定
        if 0 <= delta_days <= 5:
            # 5日前以内の場合
            base_dir = "/home/devel/mnt/nfs41/center/GSM_gl"
            file_path = os.path.join(base_dir, filename)
        else:
            # それより前の場合
            yyyy_mm = data_date.strftime("%Y%m")
            base_dir = f"/home/devel/work_C/work_ytakano/GSMGPV/GSM_gl/{yyyy_mm}"
            file_path = os.path.join(base_dir, filename)

        # ファイルパスを辞書に格納
        file_pass[key] = file_path

    # NetCDFファイルの保存先ディレクトリを設定
    nc_base_dir = "/home/devel/contents_C/PF_text/nc"

    # NetCDFファイルの保存先ディレクトリが存在しない場合は作成
    os.makedirs(nc_base_dir, exist_ok=True)

    # 緯度・経度の範囲を設定（日本付近）
    lat_min, lat_max = 15, 55  # 緯度の範囲
    lon_min, lon_max = 115, 155  # 経度の範囲

    # データを格納するディクショナリを初期化
    data_dict = {}

    for key, file_path in file_pass.items():
        # NetCDFファイルのパスを作成（nc_base_dir に保存）
        filename = os.path.basename(file_path)
        nc_filename = filename.replace(".bin", ".nc")
        nc_file_path = os.path.join(nc_base_dir, nc_filename)

        # ファイルの存在を確認
        if not os.path.exists(file_path):
            print(f"{file_path} が存在しません。スキップします。")
            continue

        # wgrib2コマンドでGRIB2ファイルをNetCDFに変換（CPU使用制限付き）
        if not os.path.exists(nc_file_path):
            wgrib2_command = [
                "wgrib2",
                file_path,
                "-ncpu",
                "1",
                "-match",
                ":(PRMSL|PRMSL mean sea level):",
                "-netcdf",
                nc_file_path,
            ]

            try:
                result = subprocess.run(
                    wgrib2_command,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                # コマンドの出力を表示（デバッグ用）
                print(result.stdout)
            except subprocess.CalledProcessError as e:
                print(f"Error converting {file_path} to NetCDF: {e.stderr}")
                continue
        else:
            print(f"{nc_file_path} は既に存在します。変換をスキップします。")

        # NetCDFファイルを読み込む
        try:
            ds = xr.open_dataset(nc_file_path)
            # 座標名を標準化
            if "lat" in ds.coords and "lon" in ds.coords:
                ds = ds.rename({"lat": "latitude", "lon": "longitude"})
            elif "nav_lat" in ds.coords and "nav_lon" in ds.coords:
                ds = ds.rename({"nav_lat": "latitude", "nav_lon": "longitude"})

            # データセット内の変数名を確認
            print(f"{nc_file_path} の変数一覧: {list(ds.data_vars)}")

            # PRMSLの変数名を取得
            prmsl_var_name = None
            for var in ds.data_vars:
                if "PRMSL" in var:
                    prmsl_var_name = var
                    break
            if prmsl_var_name is None:
                print(f"{nc_file_path} にPRMSLの変数が見つかりませんでした。")
                continue

            # 緯度・経度の範囲でデータを選択
            # 緯度・経度の並び順を確認
            lat_ascending = (ds.latitude[1] - ds.latitude[0]).values > 0
            lon_ascending = (ds.longitude[1] - ds.longitude[0]).values > 0

            if lat_ascending:
                lat_slice = slice(lat_min, lat_max)
            else:
                lat_slice = slice(lat_max, lat_min)

            if lon_ascending:
                lon_slice = slice(lon_min, lon_max)
            else:
                lon_slice = slice(lon_max, lon_min)

            ds_region = ds.sel(latitude=lat_slice, longitude=lon_slice)

            # データの形状を表示
            print(f"{key} のデータ形状: {ds_region[prmsl_var_name].shape}")

            # データを辞書に格納
            data_dict[key] = ds_region[prmsl_var_name]
        except Exception as e:
            print(f"{nc_file_path} の読み込み中にエラーが発生しました: {e}")
            continue

    # 読み込んだデータの形状を表示
    print("\\nData Shapes:")
    for key, data in data_dict.items():
        print(f"{key}: {data.dims}")

    # データの前処理と分類を行う
    new_data_list = []
    date_keys = ["yesterday", "today", "tomorrow"]
    slp_data_list = []
    for idx, key in enumerate(date_keys):
        if key in data_dict:
            ds = data_dict[key]
            # 前処理
            if "lat" in ds.coords and "lon" in ds.coords:
                ds = ds.rename({"lat": "latitude", "lon": "longitude"})

            slp = (ds / 100).isel(time=0).sel(latitude=lat_range, longitude=lon_range)

            if slp.size == 0:
                print(
                    f"警告: slp データが空です。キー {key} のデータをスキップします。"
                )
                new_data_list.append(None)
                slp_data_list.append(None)
                continue

            slp_interp = slp.interp(latitude=common_lat, longitude=common_lon)
            slp_shape = slp_interp.values.shape

            if slp_shape != expected_shape:
                print(
                    f"キー {key}: データ形状が {slp_shape} で、期待される形状 {expected_shape} と一致しません。"
                )
                new_data_list.append(None)
                slp_data_list.append(None)
                continue

            # 領域平均を引く（分類用）
            slp_mean = slp_interp.mean()
            slp_anomaly = slp_interp - slp_mean

            new_data_list.append(slp_anomaly.values.flatten())
            # プロット用にも領域平均を引いたデータを使用
            slp_data_list.append(slp_anomaly)
        else:
            new_data_list.append(None)
            slp_data_list.append(None)

    valid_data = [data for data in new_data_list if data is not None]
    if not valid_data:
        print("分類するデータがありません。")
        result_dict = {"yesterday": 0, "today": 0, "tomorrow": 0}
        node_row_dict = {"yesterday": None, "today": None, "tomorrow": None}
        node_col_dict = {"yesterday": None, "today": None, "tomorrow": None}
    else:
        new_data_array = np.array(valid_data)

        # データを分類
        classified_output = classify_new_data(new_data_array)
        classified_labels = classified_output[0]
        node_row_list = classified_output[1]
        node_col_list = classified_output[2]

        result_dict = {}
        node_row_dict = {}
        node_col_dict = {}
        idx = 0
        for key in date_keys:
            if new_data_list[idx] is not None:
                result_dict[key] = classified_labels[idx]
                node_row_dict[key] = node_row_list[idx]
                node_col_dict[key] = node_col_list[idx]
            else:
                result_dict[key] = 0  # データがない場合は0（当てはまらない）とする
                node_row_dict[key] = None
                node_col_dict[key] = None
            idx += 1

    is_yesterday_pressure_pattern = result_dict.get("yesterday", 0)
    is_today_pressure_pattern = result_dict.get("today", 0)
    is_tomorrow_pressure_pattern = result_dict.get("tomorrow", 0)

    node_row_yesterday = node_row_dict.get("yesterday")
    node_col_yesterday = node_col_dict.get("yesterday")
    node_row_today = node_row_dict.get("today")
    node_col_today = node_col_dict.get("today")
    node_row_tomorrow = node_row_dict.get("tomorrow")
    node_col_tomorrow = node_col_dict.get("tomorrow")

    # プロットの保存先ディレクトリを作成
    map_dir = "/home/devel/contents_C/PF_text/map"
    os.makedirs(map_dir, exist_ok=True)

    # プロットの準備
    cmap = sns.color_palette("RdBu_r", as_cmap=True)

    # カラーバーの範囲を統一
    pressure_vmin = -40
    pressure_vmax = 40
    pressure_levels = np.linspace(
        pressure_vmin, pressure_vmax, 26
    )  # -40 から 40 を 25 分割
    pressure_norm = Normalize(vmin=pressure_vmin, vmax=pressure_vmax)

    # 各日の地上天気図をプロットして保存
    for idx, key in enumerate(date_keys):
        slp_interp = slp_data_list[idx]
        if slp_interp is not None:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
            slp_values = slp_interp.values
            longitude = slp_interp.longitude.values
            latitude = slp_interp.latitude.values

            # 塗りつぶし等高線プロット
            cont = ax.contourf(
                longitude,
                latitude,
                slp_values,
                levels=pressure_levels,
                cmap=cmap,
                extend="both",
                norm=pressure_norm,
                transform=ccrs.PlateCarree(),
            )
            # 等高線（線のみ、黒色）
            ax.contour(
                longitude,
                latitude,
                slp_values,
                levels=pressure_levels,
                colors="k",
                linewidths=0.5,
                transform=ccrs.PlateCarree(),
            )
            # 海岸線をプロット
            ax.add_feature(cfeature.COASTLINE.with_scale("10m"), edgecolor="black")
            # 軸設定
            ax.set_extent([115, 155, 15, 55], crs=ccrs.PlateCarree())
            ax.set_xticks([])
            ax.set_yticks([])

            # カラーバーを追加
            cbar = plt.colorbar(cont, ax=ax, orientation="vertical", pad=0.02)
            cbar.set_label("Sea Level Pressure Anomaly (hPa)")

            # タイトルを追加
            date_str = judge_date_list[idx]
            plt.title(
                f"{key.capitalize()} Sea Level Pressure Anomaly\\n{date_str}",
                fontsize=16,
            )

            # ファイル名に日時を含める
            # date_str は 'YYYY-MM-DD_HH' の形式なので、ファイル名に使用できるように整形
            date_part, hour_part = date_str.split("_")
            date_formatted = date_part.replace("-", "")  # 'YYYYMMDD'
            hour_formatted = hour_part.zfill(2)  # 'HH'

            map_filename = f"SLP_{key}_{date_formatted}_{hour_formatted}.png"
            map_filepath = os.path.join(map_dir, map_filename)

            # ファイルが既に存在する場合は作成をスキップ
            if os.path.exists(map_filepath):
                print(f"{map_filepath} は既に存在するため、作成をスキップします。")
            else:
                # ファイルを保存
                plt.savefig(map_filepath)
                plt.close()
                print(f"{key} の地上天気図を保存しました: {map_filepath}")
        else:
            print(f"{key} のデータがないため、地上天気図を作成できません。")

    return (
        is_yesterday_pressure_pattern,
        is_today_pressure_pattern,
        is_tomorrow_pressure_pattern,
        node_row_yesterday,
        node_col_yesterday,
        node_row_today,
        node_col_today,
        node_row_tomorrow,
        node_col_tomorrow,
    )


def train_som_classifier():
    import os
    import pickle  # 追加
    import random
    from collections import OrderedDict, defaultdict
    from datetime import datetime, timedelta

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import xarray as xr
    from matplotlib.colors import Normalize
    from minisom import MiniSom
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # 乱数シードの固定
    random.seed(0)
    np.random.seed(0)

    # data_label_dict と各種日付リストをインポート
    from WeatherLLM.document_PF_text.pressure_pattern_label import (
        data_label_dict,
        rain_dates_list,
        summer_dates_list,
        typhoon_dates_list,
        winter_dates_list,
    )

    # ファイルパスを定義
    som_model_path = os.path.join("app", "som.pkl")
    scaler_path = os.path.join("app", "scaler.pkl")
    pca_path = os.path.join("app", "pca.pkl")
    params_path = os.path.join("app", "som_params.pkl")

    # モデルが既に保存されている場合は読み込む
    if (
        os.path.exists(som_model_path)
        and os.path.exists(scaler_path)
        and os.path.exists(pca_path)
        and os.path.exists(params_path)
    ):
        print("保存されたモデルを読み込みます。")
        with open(som_model_path, "rb") as som_file:
            som = pickle.load(som_file)
        with open(scaler_path, "rb") as scaler_file:
            scaler = pickle.load(scaler_file)
        with open(pca_path, "rb") as pca_file:
            pca = pickle.load(pca_file)
        with open(params_path, "rb") as params_file:
            params = pickle.load(params_file)
        common_lat = params["common_lat"]
        common_lon = params["common_lon"]
        expected_shape = params["expected_shape"]
        lat_range = params["lat_range"]
        lon_range = params["lon_range"]
        winter_nodes = params["winter_nodes"]
        summer_nodes = params["summer_nodes"]
        rain_nodes = params["rain_nodes"]
        typhoon_nodes = params["typhoon_nodes"]

        # 新しいデータを分類する関数を定義
        def classify_new_data(new_data_array):
            # データの正規化
            new_data_array_norm = scaler.transform(new_data_array)
            # PCAによる次元削減
            new_data_array_pca = pca.transform(new_data_array_norm)
            # SOMによるノードへのマッピング
            new_mapped_nodes = [som.winner(x) for x in new_data_array_pca]
            # マッピングされたノードを表示
            classified_labels = []
            node_row_list = []
            node_col_list = []
            for idx, node in enumerate(new_mapped_nodes):
                if node in winter_nodes:
                    label = 1  # 冬型
                    label_name = "冬型の気圧配置"
                elif node in summer_nodes:
                    label = 2  # 夏型
                    label_name = "夏型の気圧配置"
                elif node in rain_nodes:
                    label = 3  # 梅雨型の気圧配置
                    label_name = "梅雨型の気圧配置"
                elif node in typhoon_nodes:
                    label = 4  # 台風型
                    label_name = "台風"
                else:
                    label = 0  # その他
                    label_name = "該当なし"
                classified_labels.append(label)
                node_row_list.append(node[0])
                node_col_list.append(node[1])
                print(
                    f"データポイント {idx} はノード {node} に分類され、{label_name} と判定されました。"
                )
            return classified_labels, node_row_list, node_col_list

    else:
        # モデルがない場合は学習する
        print("モデルが存在しないため、新たに学習を行います。")
        # データのパス設定
        data_dir = "/home/devel/work_C/work_ytakano/GSMGPV/nc/prmsl/"

        # 解析期間の設定
        start_date = datetime.strptime("2016-03-01", "%Y-%m-%d")
        end_date = datetime.strptime("2024-10-31", "%Y-%m-%d")

        # 使用する緯度・経度の範囲を設定（日本付近）
        lat_range = slice(15, 55)
        lon_range = slice(115, 155)

        # データファイルの一覧を取得（サブディレクトリ内も含める）
        file_paths = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".nc"):
                    file_paths.append(os.path.join(root, file))

        # ファイルパスをソート
        file_paths = sorted(file_paths)

        # ファイル名から日付を取得し、辞書を作成
        file_date_dict = {}
        for file_path in file_paths:
            file_name = os.path.basename(file_path)
            date_str = file_name.replace("prmsl_", "").replace(".nc", "")
            try:
                file_date = datetime.strptime(date_str, "%Y%m%d").date()
                file_date_dict[file_date] = file_path
            except ValueError:
                print(f"ファイル名から日付を抽出できませんでした: {file_name}")

        # 共通の緯度・経度グリッドを定義
        common_lat = np.arange(15, 55.5, 0.5)
        common_lon = np.arange(115, 155.5, 0.5)

        # データを格納するリスト
        data_list = []
        date_list = []

        # 期待されるデータの形状を初期化
        expected_shape = None

        # データの読み込み
        print("データの読み込み")
        current_date = start_date
        while current_date <= end_date:
            current_date_date = current_date.date()
            if current_date_date in file_date_dict:
                nc_file = file_date_dict[current_date_date]
                ds = xr.open_dataset(nc_file)

                # 座標名を標準化
                if "lat" in ds.coords and "lon" in ds.coords:
                    ds = ds.rename({"lat": "latitude", "lon": "longitude"})

                # 海面更正気圧データを取得し、指定の緯度経度範囲でスライス
                slp = (
                    (ds["PRMSL_meansealevel"] / 100)
                    .isel(time=0)
                    .sel(latitude=lat_range, longitude=lon_range)
                )

                # slp データの存在確認
                if slp.size == 0:
                    print(
                        f"警告: slp データが空です。日付 {current_date_date} のデータをスキップします。"
                    )
                    current_date += timedelta(days=1)
                    continue

                # データを共通グリッドに再補間
                slp_interp = slp.interp(latitude=common_lat, longitude=common_lon)
                slp_shape = slp_interp.values.shape

                # 期待されるデータ形状を設定
                if expected_shape is None:
                    expected_shape = slp_shape
                else:
                    if slp_shape != expected_shape:
                        print(
                            f"日付 {current_date_date}: データ形状が {slp_shape} で、期待される形状 {expected_shape} と一致しません。"
                        )
                        current_date += timedelta(days=1)
                        continue

                # 領域平均を引く（再補間後のデータで）
                slp_mean = slp_interp.mean()
                slp_anomaly = slp_interp - slp_mean

                # データを1次元配列にフラット化してリストに追加
                data_list.append(slp_anomaly.values.flatten())
                date_list.append(current_date_date)

            else:
                print(f"データが存在しない日付: {current_date_date}")
            current_date += timedelta(days=1)

        # リストをNumPy配列に変換
        data_array = np.array(data_list)
        print("data_array の形状:", data_array.shape)

        # データの正規化（平均0、標準偏差1）
        scaler = StandardScaler()
        if data_array.size == 0:
            print("エラー: data_array が空です。データの読み込みに問題があります。")
            return None
        else:
            data_array_norm = scaler.fit_transform(data_array)

            # 主成分分析による次元削減
            pca_components = 20
            pca = PCA(n_components=pca_components, svd_solver="full", random_state=0)
            data_array_pca = pca.fit_transform(data_array_norm)
            print(f"データの形状（PCA後）: {data_array_pca.shape}")

        # SOMのサイズを設定
        som_rows = 10
        som_cols = 10

        # 近傍関数の範囲（sigma）と学習率（learning_rate）
        sigma = 3.0
        learning_rate = 1.0

        # イテレーション数
        num_iterations = 100000

        # MiniSomのインスタンスを作成
        som = MiniSom(
            som_rows,
            som_cols,
            data_array_pca.shape[1],
            sigma=sigma,
            learning_rate=learning_rate,
            neighborhood_function="gaussian",
            random_seed=0,
        )

        # 初期化
        som.random_weights_init(data_array_pca)

        # 学習（バッチ学習）
        print("SOMの学習を開始します...")
        som.train_batch(data_array_pca, num_iterations, verbose=True)
        print("SOMの学習が完了しました。")

        # 各データポイントの対応するノードを取得
        mapped_nodes = [som.winner(x) for x in data_array_pca]

        # 開始日と終了日を date 型に変換
        start_date_date = start_date.date()
        end_date_date = end_date.date()

        # 冬型の気圧配置の日付を抽出し、解析期間内にフィルタリング
        dates_with_label_A = [
            date for date, label in data_label_dict.items() if label == "A"
        ]
        dates_with_label_A_str = [
            date.strftime("%Y-%m-%d") for date in dates_with_label_A
        ]
        all_winter_dates_list = list(
            OrderedDict.fromkeys(winter_dates_list + dates_with_label_A_str)
        )
        filtered_winter_dates_list = [
            date_str
            for date_str in all_winter_dates_list
            if start_date_date
            <= datetime.strptime(date_str, "%Y-%m-%d").date()
            <= end_date_date
        ]
        print(f"冬型の気圧配置の日付の総数: {len(filtered_winter_dates_list)} 日")
        winter_dates_set = set(
            datetime.strptime(date_str, "%Y-%m-%d").date()
            for date_str in filtered_winter_dates_list
        )
        is_winter_type = [date in winter_dates_set for date in date_list]

        # 夏型の気圧配置の日付を抽出し、解析期間内にフィルタリング
        dates_with_label_L = [
            date for date, label in data_label_dict.items() if label == "L"
        ]
        dates_with_label_L_str = [
            date.strftime("%Y-%m-%d") for date in dates_with_label_L
        ]
        all_summer_dates_list = list(
            OrderedDict.fromkeys(summer_dates_list + dates_with_label_L_str)
        )
        filtered_summer_dates_list = [
            date_str
            for date_str in all_summer_dates_list
            if start_date_date
            <= datetime.strptime(date_str, "%Y-%m-%d").date()
            <= end_date_date
        ]
        print(f"夏型の気圧配置の日付の総数: {len(filtered_summer_dates_list)} 日")
        summer_dates_set = set(
            datetime.strptime(date_str, "%Y-%m-%d").date()
            for date_str in filtered_summer_dates_list
        )
        is_summer_type = [date in summer_dates_set for date in date_list]

        # 梅雨型の気圧配置の日付を抽出し、解析期間内にフィルタリング
        dates_with_label_JK = [
            date for date, label in data_label_dict.items() if label in ("J", "K")
        ]
        dates_with_label_JK_str = [
            date.strftime("%Y-%m-%d") for date in dates_with_label_JK
        ]
        all_rain_dates_list = list(
            OrderedDict.fromkeys(rain_dates_list + dates_with_label_JK_str)
        )
        filtered_rain_dates_list = [
            date_str
            for date_str in all_rain_dates_list
            if start_date_date
            <= datetime.strptime(date_str, "%Y-%m-%d").date()
            <= end_date_date
        ]
        print(f"梅雨型の気圧配置の日付の総数: {len(filtered_rain_dates_list)} 日")
        rain_dates_set = set(
            datetime.strptime(date_str, "%Y-%m-%d").date()
            for date_str in filtered_rain_dates_list
        )
        is_rain_type = [date in rain_dates_set for date in date_list]

        # 台風型の気圧配置の日付を抽出し、解析期間内にフィルタリング
        dates_with_label_MNO = [
            date for date, label in data_label_dict.items() if label in ("M", "N", "O")
        ]
        dates_with_label_MNO_str = [
            date.strftime("%Y-%m-%d") for date in dates_with_label_MNO
        ]
        all_typhoon_dates_list = list(
            OrderedDict.fromkeys(typhoon_dates_list + dates_with_label_MNO_str)
        )
        filtered_typhoon_dates_list = [
            date_str
            for date_str in all_typhoon_dates_list
            if start_date_date
            <= datetime.strptime(date_str, "%Y-%m-%d").date()
            <= end_date_date
        ]
        print(f"台風型の気圧配置の日付の総数: {len(filtered_typhoon_dates_list)} 日")
        typhoon_dates_set = set(
            datetime.strptime(date_str, "%Y-%m-%d").date()
            for date_str in filtered_typhoon_dates_list
        )
        is_typhoon_type = [date in typhoon_dates_set for date in date_list]

        # ノードごとに各タイプのデータ数をカウントする配列を作成
        node_total_counts = np.zeros((som_rows, som_cols), dtype=int)
        node_winter_counts = np.zeros((som_rows, som_cols), dtype=int)
        node_summer_counts = np.zeros((som_rows, som_cols), dtype=int)
        node_rain_counts = np.zeros((som_rows, som_cols), dtype=int)
        node_typhoon_counts = np.zeros((som_rows, som_cols), dtype=int)

        # ノードごとのデータインデックスのリストを作成
        node_data_indices = defaultdict(list)

        for idx, node in enumerate(mapped_nodes):
            i, j = node
            node_total_counts[i, j] += 1
            node_data_indices[(i, j)].append(idx)
            if is_winter_type[idx]:
                node_winter_counts[i, j] += 1
            if is_summer_type[idx]:
                node_summer_counts[i, j] += 1
            if is_rain_type[idx]:
                node_rain_counts[i, j] += 1
            if is_typhoon_type[idx]:
                node_typhoon_counts[i, j] += 1

        # 各タイプのノードを手動で指定
        # 必要に応じてノードのリストを更新してください
        winter_nodes = [
            (6, 0),
            (7, 0),
            (8, 0),
            (9, 0),
            (5, 1),
            (6, 1),
            (7, 1),
            (8, 1),
            (5, 2),
            (6, 2),
            (7, 2),
            (6, 3),
            (7, 3),
            (6, 4),
            (5, 3),
        ]
        summer_nodes = [
            (8, 6),
            (9, 6),
            (8, 7),
            (9, 7),
            (8, 8),
            (9, 8),
            (9, 9),
            (7, 7),
            (9, 5),
            (9, 4),
            (9, 3),
        ]
        rain_nodes = [
            (6, 7),
            (5, 8),
            (6, 8),
            (7, 8),
            (6, 9),
            (7, 9),
            (8, 9),
            (5, 7),
            (5, 9),
            (4, 7),
            (4, 8),
            (4, 9),
            (3, 7),
            (3, 8),
            (3, 9),
        ]
        typhoon_nodes = [
            (0, 6),
            (0, 7),
            (1, 7),
            (0, 8),
            (1, 8),
            (2, 8),
            (1, 9),
            (2, 9),
            (0, 9),
            (1, 6),
            (0, 4),
            (0, 5),
        ]

        # 指定された各タイプのノードを表示
        print(f"指定された冬型の気圧配置ノード: {winter_nodes}")
        print(f"指定された夏型の気圧配置ノード: {summer_nodes}")
        print(f"指定された梅雨の気圧配置ノード: {rain_nodes}")
        print(f"指定された台風の気圧配置ノード: {typhoon_nodes}")

        # 新しいデータを分類する関数を定義
        def classify_new_data(new_data_array):
            # データの正規化
            new_data_array_norm = scaler.transform(new_data_array)
            # PCAによる次元削減
            new_data_array_pca = pca.transform(new_data_array_norm)
            # SOMによるノードへのマッピング
            new_mapped_nodes = [som.winner(x) for x in new_data_array_pca]
            # マッピングされたノードを表示
            classified_labels = []
            node_row_list = []
            node_col_list = []
            for idx, node in enumerate(new_mapped_nodes):
                if node in winter_nodes:
                    label = 1  # 冬型
                    label_name = "冬型の気圧配置"
                elif node in summer_nodes:
                    label = 2  # 夏型
                    label_name = "夏型の気圧配置"
                elif node in rain_nodes:
                    label = 3  # 梅雨型の気圧配置
                    label_name = "梅雨型の気圧配置"
                elif node in typhoon_nodes:
                    label = 4  # 台風型
                    label_name = "台風"
                else:
                    label = 0  # その他
                    label_name = "該当なし"
                classified_labels.append(label)
                node_row_list.append(node[0])
                node_col_list.append(node[1])
                print(
                    f"データポイント {idx} はノード {node} に分類され、{label_name} と判定されました。"
                )
            return classified_labels, node_row_list, node_col_list

        # 学習済みのモデルやパラメータを保存
        with open(som_model_path, "wb") as som_file:
            pickle.dump(som, som_file)
        with open(scaler_path, "wb") as scaler_file:
            pickle.dump(scaler, scaler_file)
        with open(pca_path, "wb") as pca_file:
            pickle.dump(pca, pca_file)
        params = {
            "common_lat": common_lat,
            "common_lon": common_lon,
            "expected_shape": expected_shape,
            "lat_range": lat_range,
            "lon_range": lon_range,
            "winter_nodes": winter_nodes,
            "summer_nodes": summer_nodes,
            "rain_nodes": rain_nodes,
            "typhoon_nodes": typhoon_nodes,
        }
        with open(params_path, "wb") as params_file:
            pickle.dump(params, params_file)

    return (
        classify_new_data,
        scaler,
        pca,
        common_lat,
        common_lon,
        expected_shape,
        lat_range,
        lon_range,
    )
