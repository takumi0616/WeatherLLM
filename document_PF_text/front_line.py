# front_line.py

import xarray as xr
import numpy as np
from scipy.ndimage import label
from sklearn.decomposition import PCA
import math

def front_line_position(longitude, latitude):
    """
    与えられた緯度経度の地点における前線の位置関係を取得し、
    前線の塊（連続した領域）を分析する関数

    Parameters:
        longitude (float): 経度（例：139.02）
        latitude (float): 緯度（例：37.89）

    Returns:
        list: 各前線の情報（種類、方向、距離、長さ、位置）のリスト
    """
    # ncファイルを読み込む
    ds = xr.open_dataset('./refined_202306010600.nc')

    # 経度と緯度のデータを取得
    lons = ds['lon'].values  # (lon,) の形状
    lats = ds['lat'].values  # (lat,) の形状
    lon_grid, lat_grid = np.meshgrid(lons, lats)  # (lat, lon) の形状

    # 前線のクラス番号と前線名の対応
    class_names = {
        1: '温暖前線',
        2: '寒冷前線',
        3: '停滞前線',
        4: '閉塞前線'
    }

    # 結果を格納するリスト
    fronts_info = []

    # 与えられた地点から各格子点への距離を計算
    distances = haversine_array(longitude, latitude, lon_grid, lat_grid)

    # 各前線クラスについて処理
    for cls_index in range(1, 5):  # クラス1〜4
        # クラスごとの存在確率を取得
        # 'class'はPythonの予約語のため、辞書を使って指定
        prob_data = ds['probabilities'].isel(time=0, **{'class': cls_index}).values  # (lat, lon)

        # 確率が0より大きい箇所を前線ありとする（0/1マスク）
        front_mask = prob_data > 0

        # 連続した領域をラベル付け
        labeled_array, num_features = label(front_mask)

        # 各連結領域（前線の塊）について
        for region_label in range(1, num_features + 1):
            # 領域のマスク
            region_mask = labeled_array == region_label

            # 領域内の格子点の緯度経度を取得
            region_lat = lat_grid[region_mask]
            region_lon = lon_grid[region_mask]

            # 領域の中心を計算
            center_lat = region_lat.mean()
            center_lon = region_lon.mean()

            # 与えられた地点との距離を計算（領域内の最小距離）
            region_distances = distances[region_mask]
            min_distance = region_distances.min()

            # 領域の長さ（最大距離）を計算
            max_distance = max_pairwise_distance(region_lon, region_lat)

            # 領域の主成分分析による方向を計算
            orientation = calculate_orientation(region_lon, region_lat)

            # 方位を16方位に変換
            direction_str = bearing_to_direction(orientation)

            # 結果をリストに追加
            fronts_info.append({
                'front_type': class_names[cls_index],
                'distance_km': min_distance,
                'length_km': max_distance,
                'orientation': direction_str,
                'center_lat': center_lat,
                'center_lon': center_lon
            })

    # 距離でソート
    fronts_info.sort(key=lambda x: x['distance_km'])

    return fronts_info

def haversine(lon1, lat1, lon2, lat2):
    """
    2点間の距離を計算（km）
    """
    R = 6371  # 地球の平均半径（km）
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)
    a = np.sin(delta_lat/2)**2 + np.cos(lat1_rad)*np.cos(lat2_rad)*np.sin(delta_lon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = R * c
    return distance

def haversine_array(lon1, lat1, lon2_array, lat2_array):
    """
    配列との間の距離を計算（km）
    """
    R = 6371  # 地球の平均半径（km）
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2_array)
    delta_lat = np.radians(lat2_array - lat1)
    delta_lon = np.radians(lon2_array - lon1)
    a = np.sin(delta_lat/2)**2 + np.cos(lat1_rad)*np.cos(lat2_rad)*np.sin(delta_lon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = R * c
    return distance

def max_pairwise_distance(lon_array, lat_array):
    """
    領域内の最大2点間距離を計算（km）
    """
    num_points = len(lon_array)
    max_distance = 0
    for i in range(num_points):
        for j in range(i+1, num_points):
            distance = haversine(lon_array[i], lat_array[i], lon_array[j], lat_array[j])
            if distance > max_distance:
                max_distance = distance
    return max_distance

def calculate_orientation(lon_array, lat_array):
    """
    領域の主成分分析による方向（方位角）を計算
    """
    # 緯度経度をxy平面上の座標に変換
    x = lon_array
    y = lat_array
    coords = np.column_stack((x, y))
    # 平均を引く
    coords_centered = coords - coords.mean(axis=0)
    # PCA
    pca = PCA(n_components=2)
    pca.fit(coords_centered)
    # 第一主成分の方向
    angle_rad = np.arctan2(pca.components_[0,1], pca.components_[0,0])
    angle_deg = np.degrees(angle_rad)
    # 方位角に変換（北=0度、時計回り）
    bearing = (90 - angle_deg) % 360
    return bearing

def bearing_to_direction(bearing):
    """
    方位角から16方位の方向を取得
    """
    dirs = ['北', '北北東', '北東', '東北東', '東', '東南東', '南東', '南南東',
            '南', '南南西', '南西', '西南西', '西', '西北西', '北西', '北北西']
    ix = int((bearing + 11.25)/22.5) % 16
    return dirs[ix]

# テスト用コード
if __name__ == "__main__":
    # 一旦以下のように定義
    longitude = 139.02
    latitude = 37.89

    # 関数を実行して結果を取得
    fronts_info = front_line_position(longitude, latitude)

    # 結果を表示
    if fronts_info:
        for info in fronts_info:
            print(f"{info['front_type']}があります。")
            print(f"  距離: {info['distance_km']:.2f} km")
            print(f"  長さ: {info['length_km']:.2f} km")
            print(f"  方向: {info['orientation']}")
            print(f"  中心位置: 緯度 {info['center_lat']:.2f}, 経度 {info['center_lon']:.2f}")
    else:
        print("周囲に前線はありません。")
