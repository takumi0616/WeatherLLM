# check_winter_pattern.py

import xarray as xr
import numpy as np

def check_winter_pattern(
    longitude: float,
    latitude: float,
    wd_value: str,
    node_row: int,
    node_col: int,
) -> int:
    # node_row または node_col が None の場合、データがないため 0 を返す
    if node_row is None or node_col is None:
        return 0  # もしくは適切なデフォルト値

    # winter_weather_pattern.nc を読み込みます
    ds_winter = xr.open_dataset("/home/devel/contents_C/PF_text/wm_nc/winter_weather_pattern.nc")

    # 指定された経度・緯度に最も近い値を取得します
    value_winter = ds_winter['weather_code'].sel(
        longitude=longitude, latitude=latitude, method='nearest').values.item()

    print("value_winter")
    print(value_winter)

    # もし値が0であれば、処理を終了します
    if value_winter == 0:
        return 0

    # all_nodes_weather_patterns.nc を読み込みます
    ds_nodes = xr.open_dataset("/home/devel/contents_C/PF_text/wm_nc/all_nodes_weather_patterns.nc")

    # wd_value を整数に変換します
    wd_value_int = int(wd_value)
    print("wd_value_int")
    print(wd_value_int)

    # wd_value_intの変換処理
    # 3桁の天気コードの1桁目を取得して対応するコードに変換します
    first_digit = int(str(wd_value_int)[0])
    if first_digit == 1:
        wd_value_int_new = 1  # 晴れ
    elif first_digit == 2:
        wd_value_int_new = 2  # くもり
    elif first_digit == 3 or first_digit == 4 or first_digit == 8:
        wd_value_int_new = 3  # 雨/雪
    else:
        wd_value_int_new = 0  # 該当なし

    print("wd_value_int_new ")
    print(wd_value_int_new )

    # 指定された node_row と node_col を使用してデータをサブセットします
    ds_subset = ds_nodes.sel(node_row=node_row, node_col=node_col)

    # 指定された経度・緯度に最も近い値を取得します
    value_node = ds_subset['weather_code'].sel(
        longitude=longitude, latitude=latitude, method='nearest').values.item()

    print("value_node")
    print(value_node)

    # もし value_node が0であれば、0を返します
    if value_node == 0:
        return 0
    else:
        # 3つの値を比較します
        if wd_value_int_new == value_winter == value_node:
            return 1
        else:
            return 0
