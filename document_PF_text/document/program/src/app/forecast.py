import json
import os
import re
from datetime import datetime, timedelta
from decimal import Decimal
from typing import TypedDict

from . import tokyocabinet


class Forecast(TypedDict):
    date: str
    mintemp: Decimal | None
    maxtemp: Decimal | None
    prcp: Decimal | None
    wind: Decimal | None
    wind_dir: str | None
    snow: Decimal | None
    maxprcp_1h: Decimal | None
    time_maxprcp_1h: str | None


def get_forecast(ann: str, point: str) -> tuple[list[Forecast], str, str, str]:
    from .calculation import get_snow_forecast_from_netcdf

    pf = read_PF(ann, [point])
    current_date = datetime.now().strftime("%Y-%m-%d")
    next_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    start_time = f"{next_date} 00:00:00"
    latitude = round(float(pf[point]["location"]["lat"]), 2)
    longitude = round(float(pf[point]["location"]["lon"]), 2)

    today_maxprcp_1h, today_maxprcp_1h_time = get_max_hourly_prcp_forecast(
        pf, ann, point, current_date
    )
    nextday_maxprcp_1h, nextday_maxprcp_1h_time = get_max_hourly_prcp_forecast(
        pf, ann, point, next_date
    )

    daym1: Forecast = {
        "date": pf["info"]["time_def"]["timeD"]["D-1"],
        "mintemp": None,
        "maxtemp": None,
        "prcp": None,
        "wind": None,
        "wind_dir": None,
        "snow": None,
        "maxprcp_1h": None,
        "time_maxprcp_1h": None,
    }

    day0: Forecast = {
        "date": pf["info"]["time_def"]["timeD"]["D0"],
        "mintemp": format_PF_temp(pf[point]["F_daily_2"]["T_min"]["D0"]),
        "maxtemp": format_PF_temp(pf[point]["F_daily_2"]["T_max"]["D0"]),
        "prcp": format_PF_temp(pf[point]["F_daily_2"]["R_sum"]["D0"]),
        "wind": format_PF_temp(pf[point]["F_daily_2"]["W_spd"]["D0"]),
        "wind_dir": pf[point]["F_daily_2"]["W_dir"]["D0"],
        "snow": None,
        "maxprcp_1h": today_maxprcp_1h,
        "time_maxprcp_1h": today_maxprcp_1h_time,
    }

    day1: Forecast = {
        "date": pf["info"]["time_def"]["timeD"]["D1"],
        "mintemp": format_PF_temp(pf[point]["F_daily_2"]["T_min"]["D1"]),
        "maxtemp": format_PF_temp(pf[point]["F_daily_2"]["T_max"]["D1"]),
        "prcp": format_PF_temp(pf[point]["F_daily_2"]["R_sum"]["D1"]),
        "wind": format_PF_temp(pf[point]["F_daily_2"]["W_spd"]["D1"]),
        "wind_dir": pf[point]["F_daily_2"]["W_dir"]["D1"],
        "snow": get_snow_forecast_from_netcdf(
            date=current_date, ann=ann, point=point, start_time=start_time, hours=24
        ),
        "maxprcp_1h": nextday_maxprcp_1h,
        "time_maxprcp_1h": nextday_maxprcp_1h_time,
    }

    if "info" in pf:
        ann_time = pf["info"].get("ann_time", ann)
        ini_time = pf["info"].get("ini_time")
    else:
        ann_time = ann
        ini_time = None

    ann_time_cleaned = re.sub(r"時発表$", "", ann_time).strip()

    return (
        [daym1, day0, day1],
        pf[point]["name"],
        ann_time_cleaned,
        ini_time,
        longitude,
        latitude,
    )


def get_max_hourly_prcp_forecast(
    pf: dict, ann: str, point: str, date: str
) -> tuple[Decimal | None, str | None]:
    from datetime import datetime, timedelta

    today_date = date
    tomorrow_date = (
        datetime.strptime(today_date, "%Y-%m-%d") + timedelta(days=1)
    ).strftime("%Y-%m-%d")

    # 各時間スロットを作成
    time_slots = [
        f"{today_date} {str(hour).zfill(2)}:00:00" for hour in range(1, 24)
    ] + [f"{tomorrow_date} 00:00:00"]

    # PFデータから時間情報を取得
    pf_time_def = pf["info"]["time_def"]["timeE"]

    # 1時間降水量のデータを取得するための辞書
    prcp_data = {time_slot: None for time_slot in time_slots}

    # PFデータから各時間の降水量を取得
    for time_slot in prcp_data:
        prev_time = (
            datetime.strptime(time_slot, "%Y-%m-%d %H:%M:%S") - timedelta(hours=1)
        ).strftime("%Y-%m-%d %H:%M:%S")
        missing_key = f"{prev_time}_{time_slot}"
        for key, value in pf_time_def.items():
            if value == missing_key:
                rain_value = pf[point]["F_hourly"]["rain1h"].get(key, "")
                prcp_pf = format_PF_temp(rain_value)
                prcp_data[time_slot] = prcp_pf
                break

    # 最大の1時間降水量とその時間を取得
    max_pf_prcp = None
    max_pf_prcp_time = None

    for time_slot, prcp_value in prcp_data.items():
        if prcp_value is not None and (max_pf_prcp is None or prcp_value > max_pf_prcp):
            max_pf_prcp = prcp_value
            max_pf_prcp_time = time_slot

    return max_pf_prcp, max_pf_prcp_time


def read_PF(ann: str, points: list[str]):
    if ann == "latest":
        TC = tokyocabinet.TT("192.168.110.34", 17851)  # stb-sub
    else:
        if ann >= "2023-03-10_02":
            file = f"/mnt/isilon/archives/PF3.2/{ann[0:7]}/PF_{ann}.tch"
        else:
            file = f"/mnt/isilon/archives/PF3.1/{ann[0:7]}/PF_{ann}.tch"
        if not os.path.exists(file):
            raise FileNotFoundError(f"Data file {file} does not exist.")
        TC = tokyocabinet.TCH(file)
    dic = {}
    try:
        json_str = TC.get("info")
        if json_str is not None:
            dic["info"] = json.loads(json_str)
        for point in points:
            json_str = TC.get(point)
            if json_str is not None:
                dic[point] = json.loads(json_str)
    finally:
        TC.close()
    return dic


def format_PF_temp(s: str) -> Decimal | None:
    if s == "":
        return None
    else:
        return Decimal(s)
