import os
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Tuple
from decimal import ROUND_HALF_UP
import numpy as np
import xarray as xr
from psycopg2.extensions import cursor


def get_today_prcp(
    cur: cursor, amedas: int, point: str, ann: str, date: str
) -> Decimal | None:
    from .forecast import format_PF_temp, read_PF

    today_date = date
    tomorrow_date = (
        datetime.strptime(today_date, "%Y-%m-%d") + timedelta(days=1)
    ).strftime("%Y-%m-%d")

    time_slots = [
        f"{today_date} {str(hour).zfill(2)}:00:00" for hour in range(1, 24)
    ] + [f"{tomorrow_date} 00:00:00"]

    sql_obs = f"""
    SELECT record_time, prcp_1h
    FROM v_obs_prcp_10min
    WHERE amedas = %s
    AND record_time IN ({', '.join([f"'{slot}'" for slot in time_slots])});
    """
    cur.execute(sql_obs, (amedas,))
    obs_rows = cur.fetchall()

    prcp_data = {time_slot: None for time_slot in time_slots}

    for row in obs_rows:
        record_time, prcp_1h = row
        record_time_str = record_time.strftime("%Y-%m-%d %H:%M:%S")
        prcp_data[record_time_str] = prcp_1h

    pf = read_PF(ann, [point])
    pf_time_def = pf["info"]["time_def"]["timeE"]

    for time_slot in prcp_data:
        if prcp_data[time_slot] is None:
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

    total_prcp = sum(
        [value for value in prcp_data.values() if value is not None], Decimal(0)
    )

    return total_prcp if total_prcp > 0 else Decimal(0)


def get_today_wind(
    cur: cursor, amedas: int, point: str, ann: str, date: str
) -> Tuple[Decimal | None, str | None]:  # 返り値をタプルに変更
    from .forecast import format_PF_temp, read_PF

    today_date = date
    tomorrow_date = (
        datetime.strptime(today_date, "%Y-%m-%d") + timedelta(days=1)
    ).strftime("%Y-%m-%d")

    time_slots = [
        f"{today_date} {str(hour).zfill(2)}:00:00" for hour in range(1, 24)
    ] + [f"{tomorrow_date} 00:00:00"]

    sql_obs = f"""
    SELECT record_time, wind_speed, wind_speed_aqc, wind_dir16, wind_dir16_aqc
    FROM v_obs_wind_10min
    WHERE amedas = %s
    AND record_time IN ({', '.join([f"'{slot}'" for slot in time_slots])});
    """
    cur.execute(sql_obs, (amedas,))
    obs_rows = cur.fetchall()

    wind_data = {time_slot: None for time_slot in time_slots}
    dir_data = {time_slot: None for time_slot in time_slots}

    for row in obs_rows:
        record_time, wind_speed, wind_speed_aqc, wind_dir16, wind_dir16_aqc = row
        record_time_str = record_time.strftime("%Y-%m-%d %H:%M:%S")
        if wind_speed_aqc <= 1:
            wind_data[record_time_str] = wind_speed
        if wind_dir16_aqc <= 1:
            dir_data[record_time_str] = wind_dir16

    pf = read_PF(ann, [point])
    pf_time_def = pf["info"]["time_def"]["timeA"]

    for time_slot in wind_data:
        if wind_data[time_slot] is None:
            missing_key = f"{time_slot}"
            for key, value in pf_time_def.items():
                if value == missing_key:
                    wind_value = pf[point]["F_hourly"]["wind"].get(key, "")
                    wind_pf = format_PF_temp(wind_value)
                    wind_data[time_slot] = wind_pf
                    break

    for time_slot in dir_data:
        if dir_data[time_slot] is None:
            missing_key = f"{time_slot}"
            for key, value in pf_time_def.items():
                if value == missing_key:
                    wind_dir_value = pf[point]["F_hourly"]["wind_dir"].get(key, "")
                    dir_data[time_slot] = wind_dir_value
                    break

    valid_winds = [value for value in wind_data.values() if value is not None]
    valid_dirs = [value for value in dir_data.values() if value is not None]

    if valid_winds:
        average_wind = sum(valid_winds) / Decimal(len(valid_winds))
        average_wind = average_wind.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)
    else:
        average_wind = Decimal(0)

    # 最頻風向を取得
    from collections import Counter

    dir_counter = Counter(valid_dirs)
    if dir_counter:
        prevailing_wind_dir = dir_counter.most_common(1)[0][0]
    else:
        prevailing_wind_dir = None

    return average_wind, prevailing_wind_dir


def get_today_snow(
    cur: cursor, amedas: int, point: str, ann: str, date: str
) -> Optional[Decimal]:
    today_date = date
    tomorrow_date = (
        datetime.strptime(today_date, "%Y-%m-%d") + timedelta(days=1)
    ).strftime("%Y-%m-%d")
    time_slots = [
        f"{today_date} {str(hour).zfill(2)}:00:00" for hour in range(1, 24)
    ] + [f"{tomorrow_date} 00:00:00"]
    sql_obs = f"""
    SELECT record_time, snow_1h
    FROM v_obs_snow_hour
    WHERE amedas = %s
    AND record_time IN ({', '.join([f"'{slot}'" for slot in time_slots])});
    """
    cur.execute(sql_obs, (amedas,))
    obs_rows = cur.fetchall()
    snow_data = {time_slot: None for time_slot in time_slots}
    for row in obs_rows:
        record_time, snow_1h = row
        record_time_str = record_time.strftime("%Y-%m-%d %H:%M:%S")
        snow_data[record_time_str] = snow_1h

    missing_slots = [
        time_slot for time_slot, value in snow_data.items() if value is None
    ]
    total_snow_obs = sum(
        [value for value in snow_data.values() if value is not None], Decimal(0)
    )
    if missing_slots:
        total_snow_forecast = get_snow_forecast_from_netcdf(
            date=today_date,
            ann=ann,
            point=point,
            start_time=missing_slots[0],
            hours=len(missing_slots),
        )
    else:
        total_snow_forecast = Decimal(0)

    total_snow = total_snow_obs + total_snow_forecast
    return round(total_snow, 1) if total_snow > 0 else Decimal(0)


def get_snow_forecast_from_netcdf(
    date: str, point: str, ann: str, start_time: str, hours: int
) -> Decimal:
    from .forecast import read_PF
    # print(f"date: {date}")
    # print(f"ann{ann}")
    # print(f"point{point}")
    # print(f"start_time before adjustment (JST): {start_time}")
    # print(f"hours: {hours}")

    pf = read_PF(ann, [point])
    latitude = pf[point]["location"]["lat"]
    longitude = pf[point]["location"]["lon"]
    # print(f"longitude: {longitude}")
    # print(f"latitude: {latitude}")

    # start_timeの時刻部分を3の倍数に調整
    dt_start = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    adjusted_hour = (dt_start.hour // 3) * 3
    adjusted_start_time = dt_start.replace(hour=adjusted_hour, minute=0, second=0)

    # JSTからUTCに変換
    adjusted_start_time_utc = adjusted_start_time - timedelta(hours=9)
    t_start = np.datetime64(adjusted_start_time_utc.strftime("%Y-%m-%dT%H:%M:%S"))

    # print(f"start_time after adjustment (UTC): {adjusted_start_time_utc}")

    # NetCDFファイルのパスを作成
    year_month = date[:7].replace("-", "")

    # 降順でファイルパスを生成し、存在を確認するファイルの時間リスト
    file_times = [
        "210000",
        "180000",
        "150000",
        "120000",
        "090000",
        "060000",
        "030000",
        "000000",
    ]
    netcdf_file_path = None

    # 当日のファイル探索
    for file_time in file_times:
        file_path = f"/mnt/calc43/archives/GUID_grid_nc/MSM/{year_month}/MSM_GUID_Rjp_txy_snow_{date.replace('-', '')}{file_time}.nc"
        if os.path.exists(file_path):
            netcdf_file_path = file_path
            # print(f"Found NetCDF file for current day: {netcdf_file_path}")
            break

    # 当日のファイルがなければ前日のファイル探索
    if netcdf_file_path is None:
        previous_day = (
            datetime.strptime(date, "%Y-%m-%d") - timedelta(days=1)
        ).strftime("%Y%m%d")
        for file_time in file_times:
            file_path = f"/mnt/calc43/archives/GUID_grid_nc/MSM/{year_month}/MSM_GUID_Rjp_txy_snow_{previous_day}{file_time}.nc"
            if os.path.exists(file_path):
                netcdf_file_path = file_path
                # print(f"Found NetCDF file for previous day: {netcdf_file_path}")
                break

    # ファイルが見つからなければ終了
    if netcdf_file_path is None:
        print(f"No file found for both the current and previous day.")
        return Decimal(0)

    try:
        ds = xr.open_dataset(netcdf_file_path)

    except FileNotFoundError:
        print(f"File not found: {netcdf_file_path}")
        return Decimal(0)

    # 終了時間を計算（hours後）
    t_end = t_start + np.timedelta64(hours, "h")

    # 開始と終了時刻を確認
    # print(f"Start time (UTC): {t_start}, End time (UTC): {t_end}")

    # t_start, t_endをUNIXタイムスタンプに変換してNetCDFファイルのtime軸と一致させる
    t_start_unix = (adjusted_start_time_utc - datetime(1970, 1, 1)).total_seconds()
    t_end_unix = (
        adjusted_start_time_utc + timedelta(hours=hours) - datetime(1970, 1, 1)
    ).total_seconds()

    try:
        snow_forecast = (
            ds["snow"]
            .sel(longitude=longitude, latitude=latitude, method="nearest")
            .sel(time=slice(t_start_unix, t_end_unix))
            .sum()
            .values
        )

        if isinstance(snow_forecast, np.ndarray) and snow_forecast.size == 1:
            snow_forecast = snow_forecast.item()

    except KeyError as e:
        print(f"Time key error: {e}")
        snow_forecast = None

    ds.close()
    result = Decimal(snow_forecast * 100) if snow_forecast is not None else Decimal(0)
    # print(f"Snow forecast result: {result}")

    return result


def get_today_maxprcp_1h(
    cur: cursor, amedas: int, point: str, ann: str, date: str, latest_record_time: str
) -> tuple[Decimal | None, str | None, str]:
    from datetime import datetime, timedelta

    from .forecast import format_PF_temp, read_PF

    # 1. 観測データから最大の1時間降水量とその時間を取得
    start_date = f"{date} 00:00:00"
    end_date = (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime(
        "%Y-%m-%d 00:00:00"
    )

    sql_obs = """
    SELECT maxprcp_1h, maxprcp_1h_time
    FROM t_sta_prcp_10min
    WHERE amedas = %s
    AND record_time BETWEEN %s AND %s
    AND maxprcp_1h IS NOT NULL
    ORDER BY maxprcp_1h DESC
    LIMIT 1;
    """
    cur.execute(sql_obs, (amedas, start_date, end_date))
    obs_result = cur.fetchone()

    max_obs_prcp = None
    max_obs_prcp_time = None

    if obs_result:
        max_obs_prcp, max_obs_prcp_time = obs_result
        if max_obs_prcp_time:
            max_obs_prcp_time = max_obs_prcp_time.strftime("%Y-%m-%d %H:%M:%S")

    # 2. 予測データから最大の1時間降水量とその時間を取得
    today_date = date
    tomorrow_date = (
        datetime.strptime(today_date, "%Y-%m-%d") + timedelta(days=1)
    ).strftime("%Y-%m-%d")

    time_slots = [
        f"{today_date} {str(hour).zfill(2)}:00:00" for hour in range(1, 24)
    ] + [f"{tomorrow_date} 00:00:00"]

    pf = read_PF(ann, [point])
    pf_time_def = pf["info"]["time_def"]["timeE"]

    prcp_data = {}

    for time_slot in time_slots:
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

    # 最新の観測時間以降の予測データのみを対象とする
    latest_record_datetime = datetime.strptime(latest_record_time, "%Y-%m-%d %H:%M:%S")
    valid_pf_times = {
        time: value
        for time, value in prcp_data.items()
        if datetime.strptime(time, "%Y-%m-%d %H:%M:%S") > latest_record_datetime
    }

    # 予測データの最大の1時間降水量とその時間を取得
    max_pf_prcp = None
    max_pf_prcp_time = None

    for time_slot, prcp_value in valid_pf_times.items():
        if prcp_value is not None and (max_pf_prcp is None or prcp_value > max_pf_prcp):
            max_pf_prcp = prcp_value
            max_pf_prcp_time = time_slot

    # 3. 観測値と予測値を比較して高い方を選択
    if max_pf_prcp is not None and (max_obs_prcp is None or max_pf_prcp > max_obs_prcp):
        maxprcp_1h = max_pf_prcp
        maxprcp_1h_time = max_pf_prcp_time
        maxprcp_1h_kind = "forecast"
    else:
        maxprcp_1h = max_obs_prcp
        maxprcp_1h_time = max_obs_prcp_time
        maxprcp_1h_kind = "observation" if max_obs_prcp is not None else "none"

    # 結果を返す
    return maxprcp_1h, maxprcp_1h_time, maxprcp_1h_kind
