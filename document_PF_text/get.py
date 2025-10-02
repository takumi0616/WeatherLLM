#!/home/devel/miniconda3/envs/weather310/bin/python

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "app"))

import math
from decimal import Decimal
from typing import TypedDict
from urllib import parse

import requests
import simplejson as json
from psycopg2.extensions import cursor
from scipy.stats import norm

from app.amedas_tsdb import connect_amedas_tsdb
from app.batchSOM import judgePressurePattern, train_som_classifier
from app.chatgpt import generate_weather_comment
from app.check_winter_pattern import check_winter_pattern
from app.create_keywords import generate_keywords
from app.create_weather_text import create_weather_text
from app.forecast import Forecast, get_forecast
from app.moji_weather import get_moji_data, get_moji_list, get_weather_day
from app.normal_day import NormalDay, get_normal_day
from app.normal_month import NormalMonth, get_normal_month
from app.normal_season import (
    NormalSeason,
    get_normal_season_maxprcp_1h,
    get_normal_season_prcp,
    get_normal_season_snow,
    get_normal_season_wind,
)
from app.normal_year import get_normal_year
from app.observation import Observation, get_observation
from app.weather_code import get_weather_code, weather_code_fix


class Temp(TypedDict):
    value: Decimal
    kind: str


class Prcp(TypedDict):
    value: Decimal
    kind: str
    importance: float


class Maxprcp_1h(TypedDict):
    value: Decimal
    kind: str
    time: str
    importance: float


class Wind(TypedDict):
    value: Decimal
    kind: str
    dir: str
    importance: float


class Snow(TypedDict):
    value: Decimal
    kind: str
    importance: float


class TempDay(TypedDict):
    value: str
    importance: float


class TempAnomNormal(TypedDict):
    value: Decimal
    importance: float


class TempAnomPrev(TypedDict):
    value: Decimal
    importance: float


def main():
    try:
        point, ann = get_query()
        out = get_dict(point, ann)
        print("Content-type: application/json\n\n")
        print(json.dumps(out, ensure_ascii=False, ignore_nan=True))
    except Exception as e:
        import traceback

        tb = traceback.format_exc()
        out = {"ok": False, "error": tb}
        print("Content-type: application/json\n\n")
        print(json.dumps(out, ensure_ascii=False, ignore_nan=True))


def get_query() -> tuple[str, str]:
    # point = "A44132"
    point = "A31312"
    # ann = "2000-09-19_08"
    # ann = "latest"
    ann = "2023-12-17_08"
    if "QUERY_STRING" not in os.environ:
        return point, ann
    query = parse.parse_qs(os.environ["QUERY_STRING"])
    if "point" in query:
        point = query["point"][0]
    if "ann" in query:
        ann = query["ann"][0]
    return point, ann


def get_dict(point: str, ann: str):
    if point[0:1] not in ["A", "R"]:
        raise ValueError(f"Not an amedas point: {point}")
    amedas = int(point[1:])
    conn, cur = connect_amedas_tsdb()
    # moji_data = get_moji_data(point, ann)
    # print(moji_data)

    forecast_list, name, ann_time, ini_time, longitude, latitude = get_forecast(
        ann, point
    )
    date_list = list(map(lambda x: x["date"], forecast_list))
    latest_record_time = get_amedas_record_time(cur)
    observation_list = get_observation(
        cur, amedas, date_list, point, ann, latest_record_time
    )
    # print(date_list)
    # print(ann_time)
    print("longitude")
    print(longitude)
    print("latitude")
    print(latitude)

    normal_season_prcp_list = get_normal_season_prcp(cur, amedas, date_list)
    normal_season_maxprcp_1h_list = get_normal_season_maxprcp_1h(cur, amedas, date_list)
    normal_season_wind_list = get_normal_season_wind(cur, amedas, date_list)
    normal_season_snow_list = get_normal_season_snow(cur, amedas, date_list)

    mintemp_normal_day_list, maxtemp_normal_day_list = get_normal_day(
        cur, amedas, date_list
    )
    normal_month_list = get_normal_month(cur, amedas, date_list)
    normal_year = get_normal_year(cur, amedas)

    mintemp_list, maxtemp_list, prcp_list, wind_list, snow_list, maxprcp_1h_list = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for forecast, observation in zip(forecast_list, observation_list):
        mintemp, maxtemp = get_temp(forecast, observation)
        mintemp_list.append(mintemp)
        maxtemp_list.append(maxtemp)
        prcp = get_prcp(forecast, observation)
        prcp_list.append(prcp)
        maxprcp_1h = get_maxprcp_1h(forecast, observation)
        maxprcp_1h_list.append(maxprcp_1h)
        wind = get_wind(forecast, observation)
        wind_list.append(wind)
        snow = get_snow(forecast, observation)
        snow_list.append(snow)

    print("wind_list")
    print(wind_list)
    (
        mintemp_day_list,
        maxtemp_day_list,
        mintemp_anom_normal_list,
        maxtemp_anom_normal_list,
    ) = [], [], [], []
    for mintemp, normal_month in zip(mintemp_list, normal_month_list):
        mintemp_day = get_mintemp_day(mintemp, normal_month)
        mintemp_day_list.append(mintemp_day)
    for maxtemp, normal_month in zip(maxtemp_list, normal_month_list):
        maxtemp_day = get_maxtemp_day(maxtemp, normal_month)
        maxtemp_day_list.append(maxtemp_day)
    for mintemp, normal_day in zip(mintemp_list, mintemp_normal_day_list):
        mintemp_anom_normal = get_temp_anom_normal(mintemp, normal_day)
        mintemp_anom_normal_list.append(mintemp_anom_normal)
    for maxtemp, normal_day in zip(maxtemp_list, maxtemp_normal_day_list):
        maxtemp_anom_normal = get_temp_anom_normal(maxtemp, normal_day)
        maxtemp_anom_normal_list.append(maxtemp_anom_normal)

    mintemp_anom_prev_list, maxtemp_anom_prev_list = [None], [None]
    for mintemp_yesterday, mintemp_today in zip(mintemp_list[:-1], mintemp_list[1:]):
        mintemp_anom_prev = get_temp_anom_prev(
            mintemp_yesterday, mintemp_today, normal_year["mintemp_diff_sd"]
        )
        mintemp_anom_prev_list.append(mintemp_anom_prev)
    for maxtemp_yesterday, maxtemp_today in zip(maxtemp_list[:-1], maxtemp_list[1:]):
        maxtemp_anom_prev = get_temp_anom_prev(
            maxtemp_yesterday, maxtemp_today, normal_year["maxtemp_diff_sd"]
        )
        maxtemp_anom_prev_list.append(maxtemp_anom_prev)

    prcp_day_list = []
    for prcp, normal_season in zip(prcp_list, normal_season_prcp_list):
        prcp_day = get_prcp_day(prcp, normal_season)
        prcp_day["kind"] = prcp["kind"]
        prcp_day_list.append(prcp_day)

    maxprcp_1h_day_list = []
    for maxprcp_1h, normal_season in zip(
        maxprcp_1h_list, normal_season_maxprcp_1h_list
    ):
        maxprcp_1h_day = get_maxprcp_1h_day(maxprcp_1h, normal_season)
        maxprcp_1h_day["kind"] = maxprcp_1h["kind"]
        maxprcp_1h_day_list.append(maxprcp_1h_day)

    wind_day_list = []
    for wind, normal_season in zip(wind_list, normal_season_wind_list):
        wind_day = get_wind_day(wind, normal_season)
        wind_day["kind"] = wind["kind"]
        wind_day["dir"] = wind["dir"]
        wind_day_list.append(wind_day)

    snow_day_list = []
    for snow, normal_season in zip(snow_list, normal_season_snow_list):
        snow_day = get_snow_day(snow, normal_season)
        snow_day["kind"] = snow["kind"]
        snow_day_list.append(snow_day)

    point_place = name
    print("wind_day_list")
    print(wind_day_list)

    # 当日のデータに対して、予測値であると定義する
    if ann == "latest":
        prcp_day_list[1]["kind"] = "forecast"
        wind_day_list[1]["kind"] = "forecast"
        snow_day_list[1]["kind"] = "forecast"

    moji_data_list = []

    # 最初の日付の moji_data を取得
    first_date = date_list[0]
    ann_first = first_date + "_02"
    moji_data_first = get_moji_data(point, ann_first)
    if moji_data_first and "moji" in moji_data_first:
        moji_list = get_moji_list(moji_data_first["moji"])
        if len(moji_list) > 0:
            moji_data_list.append(moji_list[0])
        else:
            moji_data_list.append(None)
    else:
        moji_data_list.append(None)

    # 最新の moji_data を取得し、2日分を追加
    moji_data_latest = get_moji_data(point, ann)
    if moji_data_latest and "moji" in moji_data_latest:
        moji_list = get_moji_list(moji_data_latest["moji"])
        if len(moji_list) >= 2:
            moji_data_list.extend(moji_list[:2])  # 1つ目と2つ目を追加
        else:
            moji_data_list.extend(moji_list)
            moji_data_list.extend([None] * (2 - len(moji_list)))
    else:
        moji_data_list.extend([None, None])

    moji_data_list = get_weather_day(moji_data_list)

    # ピンポイント予報から「きのう」「きょう」「あした」の転記コードを取得する
    weather_codes, wd_values = get_weather_code(moji_data_list, first_date, point)

    # ピンポイント予報から撮ってきた「きのう」の過去の天気コードに対して、観測値を用いて更新する
    wd_values = weather_code_fix(weather_codes, wd_values)

    # 気圧配置判定のためのbatchSOMの読み込み（保存ファイルがなければ立ち上げ）
    (
        classify_new_data,
        scaler,
        pca,
        common_lat,
        common_lon,
        expected_shape,
        lat_range,
        lon_range,
    ) = train_som_classifier()

    # 気圧配置の判定
    (
        is_yesterday_pressure_pattern,
        is_today_pressure_pattern,
        is_tomorrow_pressure_pattern,
        node_row_yesterday,
        node_col_yesterday,
        node_row_today,
        node_col_today,
        node_row_tomorrow,
        node_col_tomorrow,
    ) = judgePressurePattern(
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
    )

    # 仮置き；判定結果の整数値を気圧配置の名称にマッピング
    ###
    pressure_pattern_mapping = {
        0: "該当なし",
        1: "冬型の気圧配置",
        2: "夏型の気圧配置",
        3: "梅雨型の気圧配置",
        4: "台風",
    }

    yesterday_Pressure_Pattern = pressure_pattern_mapping.get(
        is_yesterday_pressure_pattern, "該当なし"
    )
    today_Pressure_Pattern = pressure_pattern_mapping.get(
        is_today_pressure_pattern, "該当なし"
    )
    tomorrow_Pressure_Pattern = pressure_pattern_mapping.get(
        is_tomorrow_pressure_pattern, "該当なし"
    )
    ###

    # きのう、きょう、あしたの天気コードを取得
    wd_value_yesterday = wd_values[0]
    wd_value_today = wd_values[1]
    wd_value_tomorrow = wd_values[2]

   # check_winter_pattern 関数をきのうのデータで呼び出します
    if node_row_yesterday is None or node_col_yesterday is None:
        is_yesterday_winter_pressuer_pattern = 0
    else:
        is_yesterday_winter_pressuer_pattern = check_winter_pattern(
            longitude,
            latitude,
            wd_value_yesterday,
            node_row_yesterday,
            node_col_yesterday,
        )

    # check_winter_pattern 関数をきょうのデータで呼び出します
    if node_row_today is None or node_col_today is None:
        is_today_winter_pressuer_pattern = 0
    else:
        is_today_winter_pressuer_pattern = check_winter_pattern(
            longitude,
            latitude,
            wd_value_today,
            node_row_today,
            node_col_today,
        )

    # check_winter_pattern 関数をあしたのデータで呼び出します
    if node_row_tomorrow is None or node_col_tomorrow is None:
        is_tomorrow_winter_pressuer_pattern = 0
    else:
        is_tomorrow_winter_pressuer_pattern = check_winter_pattern(
            longitude,
            latitude,
            wd_value_tomorrow,
            node_row_tomorrow,
            node_col_tomorrow,
        )


    # 結果をリストにまとめます
    winter_pressuer_pattern_list = [
        is_yesterday_winter_pressuer_pattern,
        is_today_winter_pressuer_pattern,
        is_tomorrow_winter_pressuer_pattern,
    ]

    weather_data = {
        "info": {
            "date": date_list,
        },
        "mintemp": mintemp_list,
        "mintemp_anom_normal": mintemp_anom_normal_list,
        "mintemp_anom_prev": mintemp_anom_prev_list,
        "maxtemp": maxtemp_list,
        "maxtemp_anom_normal": maxtemp_anom_normal_list,
        "maxtemp_anom_prev": maxtemp_anom_prev_list,
        "prcp_day": prcp_day_list,
        "maxprcp_1h": maxprcp_1h_day_list,
        "wind": wind_day_list,
        "snow": snow_day_list,
        "weather": moji_data_list,
        "pressuer_pattern": winter_pressuer_pattern_list,
    }

    print(f"weather_data \n {weather_data}")

    weather_data_yesterday = extract_weather_data_for_day(weather_data, 0)
    weather_data_today = extract_weather_data_for_day(weather_data, 1)
    weather_data_tomorrow = extract_weather_data_for_day(weather_data, 2)

    yesterday_date = (
        weather_data_yesterday["info"]["date"] if weather_data_yesterday else ""
    )
    today_date = weather_data_today["info"]["date"] if weather_data_today else ""
    tomorrow_date = (
        weather_data_tomorrow["info"]["date"] if weather_data_tomorrow else ""
    )

    weather_comment = {}

    if weather_data_yesterday:
        keywords_yesterday = generate_keywords(weather_data_yesterday)
        weather_data_yesterday.update(keywords_yesterday)
        weather_text_yesterday = create_weather_text(weather_data_yesterday)
        comment_yesterday = generate_weather_comment(
            weather_text_yesterday, "きのう", yesterday_date, point, point_place
        )
        weather_comment["yesterday"] = comment_yesterday

    if weather_data_today:
        keywords_today = generate_keywords(weather_data_today)
        weather_data_today.update(keywords_today)
        weather_text_today = create_weather_text(weather_data_today)
        comment_today = generate_weather_comment(
            weather_text_today, "きょう", today_date, point, point_place
        )
        weather_comment["today"] = comment_today

    if weather_data_tomorrow:
        keywords_tomorrow = generate_keywords(weather_data_tomorrow)
        weather_data_tomorrow.update(keywords_tomorrow)
        weather_text_tomorrow = create_weather_text(weather_data_tomorrow)
        comment_tomorrow = generate_weather_comment(
            weather_text_tomorrow, "あした", tomorrow_date, point, point_place
        )
        weather_comment["tomorrow"] = comment_tomorrow

    cur.close()
    conn.close()

    return {
        "info": {
            "point": point,
            "name": name,
            "date": date_list,
            "ann_time": ann_time,
            "ini_time": ini_time,
            "amedas_time": latest_record_time,
        },
        "weather_comment": weather_comment,
        "yesterday_Pressure_Pattern": yesterday_Pressure_Pattern,
        "today_Pressure_Pattern": today_Pressure_Pattern,
        "tomorrow_Pressure_Pattern": tomorrow_Pressure_Pattern,
        "is_yesterday_winter_pressuer_pattern": is_yesterday_winter_pressuer_pattern,
        "is_today_winter_pressuer_pattern": is_today_winter_pressuer_pattern,
        "is_tomorrow_winter_pressuer_pattern": is_tomorrow_winter_pressuer_pattern,
        "mintemp": mintemp_list,
        "mintemp_day": mintemp_day_list,
        "mintemp_anom_normal": mintemp_anom_normal_list,
        "mintemp_anom_prev": mintemp_anom_prev_list,
        "maxtemp": maxtemp_list,
        "maxtemp_day": maxtemp_day_list,
        "maxtemp_anom_normal": maxtemp_anom_normal_list,
        "maxtemp_anom_prev": maxtemp_anom_prev_list,
        "prcp_day": prcp_day_list,
        "maxprcp_1h": maxprcp_1h_day_list,
        "wind": wind_day_list,
        "snow": snow_day_list,
        "weather": moji_data_list,
    }


def extract_weather_data_for_day(weather_data, index):
    if index >= len(weather_data["info"]["date"]):
        return None
    data = {
        "info": {"date": weather_data["info"]["date"][index]},
        "mintemp": weather_data["mintemp"][index],
        "mintemp_anom_normal": weather_data["mintemp_anom_normal"][index],
        "mintemp_anom_prev": weather_data["mintemp_anom_prev"][index]
        if index < len(weather_data["mintemp_anom_prev"])
        else None,
        "maxtemp": weather_data["maxtemp"][index],
        "maxtemp_anom_normal": weather_data["maxtemp_anom_normal"][index],
        "maxtemp_anom_prev": weather_data["maxtemp_anom_prev"][index]
        if index < len(weather_data["maxtemp_anom_prev"])
        else None,
        "prcp_day": weather_data["prcp_day"][index],
        "maxprcp_1h": weather_data["maxprcp_1h"][index],
        "wind": weather_data["wind"][index],
        "snow": weather_data["snow"][index],
        "weather": weather_data["weather"][index],
        "pressuer_pattern": weather_data["pressuer_pattern"][index],
    }
    return data


def get_temp(
    forecast: Forecast,
    observation: Observation,
) -> tuple[Temp, Temp]:
    if observation["mintemp"] is not None:
        mintemp_value = observation["mintemp"]
        mintemp_kind = "observation"
    elif forecast["mintemp"] is not None:
        mintemp_value = forecast["mintemp"]
        mintemp_kind = "forecast"
    else:
        raise ValueError("No mintemp data")

    if observation["maxtemp"] is not None:
        maxtemp_value = observation["maxtemp"]
        maxtemp_kind = "observation"
    elif forecast["maxtemp"] is not None:
        maxtemp_value = forecast["maxtemp"]
        maxtemp_kind = "forecast"
    else:
        raise ValueError("No maxtemp data")

    return (
        {"value": mintemp_value, "kind": mintemp_kind},
        {"value": maxtemp_value, "kind": maxtemp_kind},
    )


def get_prcp(
    forecast: Forecast,
    observation: Observation,
) -> Prcp:
    if observation["prcp"] is not None:
        prcp_value = observation["prcp"]
        prcp_kind = "observation"
    elif forecast["prcp"] is not None:
        prcp_value = forecast["prcp"]
        prcp_kind = "forecast"
    else:
        prcp_value = Decimal(0)
        prcp_kind = "observation"
    return {"value": prcp_value, "kind": prcp_kind}


def get_maxprcp_1h(
    forecast: Forecast,
    observation: Observation,
) -> Maxprcp_1h:
    maxprcp_1h_kind = observation.get("maxprcp_1h_kind", "none")
    if observation["maxprcp_1h"] is not None:
        maxprcp_1h_value = observation["maxprcp_1h"]
        time_maxprcp_1h = observation["time_maxprcp_1h"]
        maxprcp_1h_kind = observation["maxprcp_1h_kind"]
    elif forecast["maxprcp_1h"] is not None:
        maxprcp_1h_value = forecast["maxprcp_1h"]
        time_maxprcp_1h = forecast["time_maxprcp_1h"]
        maxprcp_1h_kind = "forecast"
    else:
        maxprcp_1h_value = Decimal(0)
        time_maxprcp_1h = "none"
        maxprcp_1h_kind = "observation"
    return {"value": maxprcp_1h_value, "time": time_maxprcp_1h, "kind": maxprcp_1h_kind}


def get_wind(forecast: Forecast, observation: Observation) -> Wind:
    wd16 = {
        0: "",
        1: "NNE",
        2: "NE",
        3: "ENE",
        4: "E",
        5: "ESE",
        6: "SE",
        7: "SSE",
        8: "S",
        9: "SSW",
        10: "SW",
        11: "WSW",
        12: "W",
        13: "WNW",
        14: "NW",
        15: "NNW",
        16: "N",
        -999: "",
    }
    if observation["wind"] is not None:
        wind_value = observation["wind"]
        wind_kind = "observation"
        wind_dir = observation.get("wind_dir", "none")
    elif forecast["wind"] is not None:
        wind_value = forecast["wind"]
        wind_kind = "forecast"
        wind_dir = forecast.get("wind_dir", "none")
    else:
        wind_value = Decimal(0)
        wind_kind = "observation"
        wind_dir = "none"
    if isinstance(wind_dir, int):
        wind_dir = wd16.get(wind_dir, "")
    return {"value": wind_value, "kind": wind_kind, "dir": wind_dir}


def get_snow(forecast: Forecast, observation: Observation) -> Wind:
    if observation["snow"] is not None:
        snow_value = observation["snow"]
        snow_kind = "observation"
    elif forecast["snow"] is not None:
        snow_value = forecast["snow"]
        snow_kind = "forecast"
    else:
        snow_value = Decimal(0)
        snow_kind = "observation"
    return {"value": snow_value, "kind": snow_kind}


def get_mintemp_day(mintemp: Temp, normal_month: NormalMonth) -> TempDay:
    t = mintemp["value"]
    if t < Decimal(0):
        p = normal_month["maxtemp_lt0_frequency"]
        information = -math.log10(max(p, 1 / normal_month["stat_years"]))
        impact = 0
        importance = information + impact
        return {"value": "冬日", "importance": importance}
    elif t >= Decimal(25):
        p = normal_month["maxtemp_ge25_frequency"]
        information = -math.log10(max(p, 1 / normal_month["stat_years"]))
        impact = 0
        importance = information + impact
        return {"value": "熱帯夜", "importance": importance}
    return {"value": "", "information": 0.0}


def get_maxtemp_day(maxtemp: Temp, normal_month: NormalMonth) -> TempDay:
    t = maxtemp["value"]
    if t >= Decimal(35):
        p = normal_month["maxtemp_ge35_frequency"]
        information = -math.log10(max(p, 1 / normal_month["stat_years"]))
        impact = 0
        importance = information + impact
        return {"value": "猛暑日", "importance": importance}
    elif t >= Decimal(30):
        p = normal_month["maxtemp_ge30_frequency"]
        information = -math.log10(max(p, 1 / normal_month["stat_years"]))
        impact = 0
        importance = information + impact
        return {"value": "真夏日", "importance": importance}
    elif t >= Decimal(25):
        p = normal_month["maxtemp_ge25_frequency"]
        information = -math.log10(max(p, 1 / normal_month["stat_years"]))
        impact = 0
        importance = information + impact
        return {"value": "夏日", "importance": importance}
    elif t < Decimal(0):
        p = normal_month["mintemp_lt0_frequency"]
        information = -math.log10(max(p, 1 / normal_month["stat_years"]))
        impact = 0
        importance = information + impact
        return {"value": "真冬日", "importance": importance}
    return {"value": "", "importance": 0.0}


def get_temp_anom_normal(temp: Temp, normal_day: NormalDay) -> TempAnomNormal:
    t = temp["value"]
    value = t - normal_day["nml"]
    p = get_temp_quantile(t, normal_day["quantile"], normal_day["stat_years"])
    p = max(min(p, 0.999), 0.001)
    lower_information = -math.log10(p)
    upper_information = -math.log10(1 - p)
    lower_information = min(lower_information, 2)
    upper_information = min(upper_information, 2)
    information = max(lower_information, upper_information)
    impact = 0
    importance = information + impact

    return {
        "value": value,
        "importance": importance,
    }


def get_temp_anom_prev(
    temp_yesterday: Temp, temp_today: Temp, sd: float
) -> TempAnomPrev:
    value = temp_today["value"] - temp_yesterday["value"]
    p = norm.cdf(-abs(float(value) / sd))
    information = -math.log10(p)
    impact = 0
    importance = information + impact

    return {"value": value, "importance": importance}


def get_temp_quantile(
    value: Decimal,
    quantile: tuple[Decimal, Decimal, Decimal, Decimal, Decimal, Decimal],
    stat_years: int,
) -> float:
    v = float(value)
    q000, q010, q033, q067, q090, q100 = [float(q) for q in quantile]
    p000, p010, p033, p067, p090, p100 = (
        1 / stat_years,
        1 / 10,
        1 / 3,
        1 - 1 / 3,
        1 - 1 / 10,
        1 - 1 / stat_years,
    )
    if v <= q000:
        return p000
    if v <= q010:
        fac = (v - q000) / (q010 - q000)
        n000 = float(norm.ppf(p000))
        n010 = float(norm.ppf(p010))
        n = n000 + fac * (n010 - n000)
        return float(norm.cdf(n))
    if v <= q033:
        fac = (v - q010) / (q033 - q010)
        n010 = float(norm.ppf(p010))
        n033 = float(norm.ppf(p033))
        n = n010 + fac * (n033 - n010)
        return float(norm.cdf(n))
    if v <= q067:
        fac = (v - q033) / (q067 - q033)
        n033 = float(norm.ppf(p033))
        n067 = float(norm.ppf(p067))
        n = n033 + fac * (n067 - n033)
        return float(norm.cdf(n))
    if v <= q090:
        fac = (v - q067) / (q090 - q067)
        n067 = float(norm.ppf(p067))
        n090 = float(norm.ppf(p090))
        n = n067 + fac * (n090 - n067)
        return float(norm.cdf(n))
    if v <= q100:
        return 0.9 + (v - q090) / (q100 - q090) * (1 - 0.9)
    return p100


def get_prcp_day(prcp: Prcp, normal_season: NormalSeason) -> dict:
    value = prcp["value"]
    if value <= Decimal(0):
        return {"value": value, "importance": 0.0}
    p = get_prcp_quantile(value, normal_season["quantile"], normal_season["stat_days"])
    information = -math.log10(1 - p)
    impact = 0
    importance = information + impact

    return {"value": value, "importance": importance}


def get_prcp_quantile(
    value: Decimal,
    quantile: tuple[Decimal, Decimal, Decimal, Decimal, Decimal, Decimal],
    stat_days: int,
) -> float:
    import numpy as np

    v = float(value)
    q000, q010, q033, q067, q090, q100 = [float(q) for q in quantile]
    if v <= 0.0:
        p = 0.0
        return p
    if v < q000:
        p = 1 / stat_days
        return p
    if v < q010:
        lnv = np.log(v + 1)
        lnq000 = np.log(q000 + 1)
        lnq010 = np.log(q010 + 1)
        denominator = lnq010 - lnq000
        if denominator == 0:
            fac = 0
        else:
            fac = (lnv - lnq000) / denominator
        p000 = 1 / stat_days
        p010 = 1 / 10
        p = p000 + fac * (p010 - p000)
        return p
    if v < q033:
        lnv = np.log(v + 1)
        lnq010 = np.log(q010 + 1)
        lnq033 = np.log(q033 + 1)
        denominator = lnq033 - lnq010
        if denominator == 0:
            fac = 0
        else:
            fac = (lnv - lnq010) / denominator
        p010 = 1 / 10
        p033 = 1 / 3
        p = p010 + fac * (p033 - p010)
        return p
    if v < q067:
        lnv = np.log(v + 1)
        lnq033 = np.log(q033 + 1)
        lnq067 = np.log(q067 + 1)
        denominator = lnq067 - lnq033
        if denominator == 0:
            fac = 0
        else:
            fac = (lnv - lnq033) / denominator
        p033 = 1 / 3
        p067 = 2 / 3
        p = p033 + fac * (p067 - p033)
        return p
    if v < q090:
        lnv = np.log(v + 1)
        lnq067 = np.log(q067 + 1)
        lnq090 = np.log(q090 + 1)
        denominator = lnq090 - lnq067
        if denominator == 0:
            fac = 0
        else:
            fac = (lnv - lnq067) / denominator
        p067 = 2 / 3
        p090 = 9 / 10
        p = p067 + fac * (p090 - p067)
        return p
    if v <= q100:
        denominator = q100 - q090
        if denominator == 0:
            p100 = 1 - 1 / stat_days
            return p100
        fac = (v - q090) / denominator
        p090 = 9 / 10
        p100 = 1 - 1 / stat_days
        p = p090 + fac * (p100 - p090)
        return p

    p = 1 - 1 / stat_days
    return p


def get_maxprcp_1h_day(maxprcp_1h: Maxprcp_1h, normal_season: NormalSeason) -> dict:
    value = maxprcp_1h["value"]
    time = maxprcp_1h.get("time", "none")
    if value <= Decimal(0):
        return {"value": value, "importance": 0.0, "time": "none"}
    p = get_maxprcp_1h_quantile(
        value, normal_season["quantile"], normal_season["stat_days"]
    )
    information = -math.log10(1 - p)
    impact = 0
    importance = information + impact

    return {"value": value, "importance": importance, "time": time}


def get_maxprcp_1h_quantile(
    value: Decimal,
    quantile: tuple[Decimal, Decimal, Decimal, Decimal, Decimal, Decimal],
    stat_days: int,
) -> float:
    import numpy as np

    v = float(value)
    q000, q010, q033, q067, q090, q100 = [float(q) for q in quantile]
    if v <= 0.0:
        p = 0.0
        return p
    if v < q000:
        p = 1 / stat_days
        return p
    if v < q010:
        lnv = np.log(v + 1)
        lnq000 = np.log(q000 + 1)
        lnq010 = np.log(q010 + 1)
        denominator = lnq010 - lnq000
        if denominator == 0:
            fac = 0
        else:
            fac = (lnv - lnq000) / denominator
        p000 = 1 / stat_days
        p010 = 1 / 10
        p = p000 + fac * (p010 - p000)
        return p
    if v < q033:
        lnv = np.log(v + 1)
        lnq010 = np.log(q010 + 1)
        lnq033 = np.log(q033 + 1)
        denominator = lnq033 - lnq010
        if denominator == 0:
            fac = 0
        else:
            fac = (lnv - lnq010) / denominator
        p010 = 1 / 10
        p033 = 1 / 3
        p = p010 + fac * (p033 - p010)
        return p
    if v < q067:
        lnv = np.log(v + 1)
        lnq033 = np.log(q033 + 1)
        lnq067 = np.log(q067 + 1)
        denominator = lnq067 - lnq033
        if denominator == 0:
            fac = 0
        else:
            fac = (lnv - lnq033) / denominator
        p033 = 1 / 3
        p067 = 2 / 3
        p = p033 + fac * (p067 - p033)
        return p
    if v < q090:
        lnv = np.log(v + 1)
        lnq067 = np.log(q067 + 1)
        lnq090 = np.log(q090 + 1)
        denominator = lnq090 - lnq067
        if denominator == 0:
            fac = 0
        else:
            fac = (lnv - lnq067) / denominator
        p067 = 2 / 3
        p090 = 9 / 10
        p = p067 + fac * (p090 - p067)
        return p
    if v <= q100:
        denominator = q100 - q090
        if denominator == 0:
            p100 = 1 - 1 / stat_days
            return p100
        fac = (v - q090) / denominator
        p090 = 9 / 10
        p100 = 1 - 1 / stat_days
        p = p090 + fac * (p100 - p090)
        return p

    p = 1 - 1 / stat_days
    return p


def get_wind_day(wind: Wind, normal_season: NormalSeason) -> dict:
    value = wind["value"]
    if value <= Decimal(0):
        return {"value": value, "importance": 0.0}
    p = get_wind_quantile(value, normal_season["quantile"], normal_season["stat_years"])
    information = -math.log10(1 - p)
    impact = 0
    importance = information + impact

    return {"value": value, "importance": importance}


def get_wind_quantile(
    value: Decimal,
    quantile: tuple[Decimal, Decimal, Decimal, Decimal, Decimal, Decimal],
    stat_years: int,
) -> float:
    import numpy as np

    v = float(value)
    q000, q010, q033, q067, q090, q100 = [float(q) for q in quantile]
    if v <= 0.0:
        p = 0.0
        return p
    if v < q000:
        p = 1 / stat_years
        return p
    if v < q010:
        lnv = np.log(v + 1)
        lnq000 = np.log(q000 + 1)
        lnq010 = np.log(q010 + 1)
        denominator = lnq010 - lnq000
        if denominator == 0:
            fac = 0
        else:
            fac = (lnv - lnq000) / denominator
        p000 = 1 / stat_years
        p010 = 1 / 10
        p = p000 + fac * (p010 - p000)
        return p
    if v < q033:
        lnv = np.log(v + 1)
        lnq010 = np.log(q010 + 1)
        lnq033 = np.log(q033 + 1)
        denominator = lnq033 - lnq010
        if denominator == 0:
            fac = 0
        else:
            fac = (lnv - lnq010) / denominator
        p010 = 1 / 10
        p033 = 1 / 3
        p = p010 + fac * (p033 - p010)
        return p
    if v < q067:
        lnv = np.log(v + 1)
        lnq033 = np.log(q033 + 1)
        lnq067 = np.log(q067 + 1)
        denominator = lnq067 - lnq033
        if denominator == 0:
            fac = 0
        else:
            fac = (lnv - lnq033) / denominator
        p033 = 1 / 3
        p067 = 2 / 3
        p = p033 + fac * (p067 - p033)
        return p
    if v < q090:
        lnv = np.log(v + 1)
        lnq067 = np.log(q067 + 1)
        lnq090 = np.log(q090 + 1)
        denominator = lnq090 - lnq067
        if denominator == 0:
            fac = 0
        else:
            fac = (lnv - lnq067) / denominator
        p067 = 2 / 3
        p090 = 9 / 10
        p = p067 + fac * (p090 - p067)
        return p
    if v <= q100:
        denominator = q100 - q090
        if denominator == 0:
            p100 = 1 - 1 / stat_years
            return p100
        fac = (v - q090) / denominator
        p090 = 9 / 10
        p100 = 1 - 1 / stat_years
        p = p090 + fac * (p100 - p090)
        return p

    p = 1 - 1 / stat_years
    return p


def get_snow_day(snow: Snow, normal_season: NormalSeason) -> dict:
    value = snow["value"]
    if value <= Decimal(0):
        return {"value": value, "importance": 0.0}
    p = get_snow_quantile(value, normal_season["quantile"], normal_season["stat_days"])
    information = -math.log10(1 - p)
    impact = 0
    importance = information + impact

    return {"value": value, "importance": importance}


def get_snow_quantile(
    value: Decimal,
    quantile: tuple[Decimal, Decimal, Decimal, Decimal, Decimal, Decimal],
    stat_days: int,
) -> float:
    import numpy as np

    v = float(value)
    q000, q010, q033, q067, q090, q100 = [float(q) for q in quantile]
    if v <= 0.0:
        p = 0.0
        return p
    if v < q000:
        p = 1 / stat_days
        return p
    if v < q010:
        lnv = np.log(v + 1)
        lnq000 = np.log(q000 + 1)
        lnq010 = np.log(q010 + 1)
        denominator = lnq010 - lnq000
        if denominator == 0:
            fac = 0
        else:
            fac = (lnv - lnq000) / denominator
        p000 = 1 / stat_days
        p010 = 1 / 10
        p = p000 + fac * (p010 - p000)
        return p
    if v < q033:
        lnv = np.log(v + 1)
        lnq010 = np.log(q010 + 1)
        lnq033 = np.log(q033 + 1)
        denominator = lnq033 - lnq010
        if denominator == 0:
            fac = 0
        else:
            fac = (lnv - lnq010) / denominator
        p010 = 1 / 10
        p033 = 1 / 3
        p = p010 + fac * (p033 - p010)
        return p
    if v < q067:
        lnv = np.log(v + 1)
        lnq033 = np.log(q033 + 1)
        lnq067 = np.log(q067 + 1)
        denominator = lnq067 - lnq033
        if denominator == 0:
            fac = 0
        else:
            fac = (lnv - lnq033) / denominator
        p033 = 1 / 3
        p067 = 2 / 3
        p = p033 + fac * (p067 - p033)
        return p
    if v < q090:
        lnv = np.log(v + 1)
        lnq067 = np.log(q067 + 1)
        lnq090 = np.log(q090 + 1)
        denominator = lnq090 - lnq067
        if denominator == 0:
            fac = 0
        else:
            fac = (lnv - lnq067) / denominator
        p067 = 2 / 3
        p090 = 9 / 10
        p = p067 + fac * (p090 - p067)
        return p
    if v <= q100:
        denominator = q100 - q090
        if denominator == 0:
            p100 = 1 - 1 / stat_days
            return p100
        fac = (v - q090) / denominator
        p090 = 9 / 10
        p100 = 1 - 1 / stat_days
        p = p090 + fac * (p100 - p090)
        return p

    p = 1 - 1 / stat_days
    return p


def get_amedas_record_time(cur: cursor) -> str:
    """
    v_obs_latestテーブルから最新のrecord_timeを取得する関数
    """
    sql = """
    SELECT record_time
    FROM v_obs_latest
    ORDER BY record_time DESC
    LIMIT 1;
    """
    cur.execute(sql)
    row = cur.fetchone()
    if row is None:
        raise ValueError("v_obs_latestテーブルにデータがありません。")
    record_time = row[0].strftime("%Y-%m-%d %H:%M:%S")
    return record_time


if __name__ == "__main__":
    main()
