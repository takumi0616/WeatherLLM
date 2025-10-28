from datetime import datetime, timedelta
from decimal import (
    ROUND_HALF_UP,  # 追加
    Decimal,
)
from typing import Tuple, TypedDict

from psycopg2.extensions import cursor

from .calculation import (
    get_today_maxprcp_1h,
    get_today_prcp,
    get_today_snow,
    get_today_wind,
)


class Observation(TypedDict):
    mintemp: Decimal | None
    maxtemp: Decimal | None
    prcp: Decimal | None
    wind: Decimal | None
    wind_dir: str | None
    snow: Decimal | None
    maxprcp_1h: Decimal | None
    time_maxprcp_1h: str | None
    maxprcp_1h_kind: str | None


def get_observation(
    cur: cursor,
    amedas: int,
    date_list: list[str],
    point: str,
    ann: str,
    latest_record_time: str,
) -> tuple[list[Observation], str | None]:
    obs_list: list[Observation] = []
    current_date = datetime.now().strftime("%Y-%m-%d")

    for date in date_list:
        maxprcp_1h_obs = None
        maxprcp_1h_time = None
        maxprcp_1h_kind = None
        prcp_sta = None
        wind_sta = None
        snow_sta = None

        mintemp_obs = get_obs_mintemp(cur, amedas, date)
        maxtemp_obs = get_obs_maxtemp(cur, amedas, date)

        if date == current_date:
            maxprcp_1h_obs, maxprcp_1h_time, maxprcp_1h_kind = get_today_maxprcp_1h(
                cur, amedas, point, ann, date, latest_record_time
            )
            prcp_obs = get_today_prcp(cur, amedas, point, ann, date)
            wind_obs, wind_obs_dir = get_today_wind(cur, amedas, point, ann, date)
            snow_obs = get_today_snow(cur, amedas, point, ann, date)
        else:
            maxprcp_1h_obs, maxprcp_1h_time = get_obs_maxprcp_1h(cur, amedas, date)
            maxprcp_1h_kind = "observation"
            prcp_obs = get_obs_prcp(cur, amedas, date)
            wind_obs, wind_obs_dir = get_obs_wind(cur, amedas, date)
            snow_obs = get_obs_snow(cur, amedas, date)
            prcp_sta = get_sta_prcp(cur, amedas, date)
            wind_sta, wind_sta_dir = get_sta_wind(cur, amedas, date)
            snow_sta = get_sta_snow(cur, amedas, date)

        mintemp_sta, maxtemp_sta = get_sta_minmaxtemp(cur, amedas, date)

        obs: Observation = {
            "mintemp": mintemp_obs if mintemp_sta is None else mintemp_sta,
            "maxtemp": maxtemp_obs if maxtemp_sta is None else maxtemp_sta,
            "prcp": prcp_obs if prcp_sta is None else prcp_sta,
            "wind": wind_obs if wind_sta is None else wind_sta,
            "wind_dir": wind_obs_dir if wind_sta_dir is None else wind_sta_dir,
            "snow": snow_obs if snow_sta is None else snow_sta,
            "maxprcp_1h": maxprcp_1h_obs,
            "time_maxprcp_1h": maxprcp_1h_time,
            "maxprcp_1h_kind": maxprcp_1h_kind,
        }
        obs_list.append(obs)

    return obs_list


def get_obs_mintemp(cur: cursor, amedas: int, date: str) -> Decimal | None:
    time_07 = f"{date}T07:00:00"
    time_09 = f"{date}T09:00:00"
    sql = """
    SELECT
        mintemp_0009,
        mintemp_0009_aqc
    FROM
        t_obs_temp_10min_pp
    WHERE
        amedas = %s
        AND record_time >= %s
        AND record_time <= %s
    ORDER BY
        record_time DESC
    LIMIT 1
    ;
    """
    cur.execute(sql, (amedas, time_07, time_09))
    row = cur.fetchone()
    if row is None:
        return None
    mintemp, mintemp_flg = row
    if mintemp_flg <= 1:
        return mintemp
    else:
        return None


def get_obs_maxtemp(cur: cursor, amedas: int, date: str) -> Decimal | None:
    time_16 = f"{date}T16:00:00"
    time_18 = f"{date}T18:00:00"
    sql = """
    SELECT
        maxtemp_0918,
        maxtemp_0918_aqc
    FROM
        t_obs_temp_10min_pp
    WHERE
        amedas = %s
        AND record_time >= %s
        AND record_time <= %s
    ORDER BY
        record_time DESC
    LIMIT 1
    """
    cur.execute(sql, (amedas, time_16, time_18))
    row = cur.fetchone()
    if row is None:
        return None
    maxtemp, maxtemp_flg = row
    if maxtemp_flg <= 1:
        return maxtemp
    else:
        return None


def get_obs_prcp(cur: cursor, amedas: int, date: str) -> Decimal | None:
    date_obj = datetime.strptime(date, "%Y-%m-%d")
    next_day_obj = date_obj + timedelta(days=1)
    next_day = next_day_obj.strftime("%Y-%m-%dT00:00:00")

    sql = """
    SELECT
        prcp_24h,
        prcp_24h_aqc
    FROM
        v_obs_prcp_10min
    WHERE
        amedas = %s
        AND record_time = %s
    LIMIT 1
    ;
    """
    cur.execute(sql, (amedas, next_day))
    row = cur.fetchone()
    if row is None:
        return None
    prcp, prcp_flg = row
    if prcp_flg <= 1:
        return prcp
    else:
        return None


def get_obs_maxprcp_1h(
    cur: cursor, amedas: int, date: str
) -> Tuple[Decimal | None, str | None]:
    """
    指定された日付の最大の1時間降水量とその時間を取得する関数
    """
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

    return max_obs_prcp, max_obs_prcp_time


def get_obs_wind(
    cur: cursor, amedas: int, date: str
) -> Tuple[Decimal | None, str | None]:
    date_obj = datetime.strptime(date, "%Y-%m-%d")
    next_day_obj = date_obj + timedelta(days=1)
    sql = """
    SELECT
        wind_speed,
        wind_speed_aqc,
        wind_dir16,
        wind_dir16_aqc
    FROM
        v_obs_wind_10min
    WHERE
        amedas = %s
        AND record_time >= %s
        AND record_time < %s
    ;
    """
    cur.execute(sql, (amedas, date_obj, next_day_obj))
    rows = cur.fetchall()
    if not rows:
        return None, None

    valid_winds = []
    valid_dirs = []
    for wind_speed, wind_speed_aqc, wind_dir16, wind_dir16_aqc in rows:
        if wind_speed_aqc <= 1 and wind_dir16_aqc <= 1:
            valid_winds.append(wind_speed)
            valid_dirs.append(wind_dir16)

    if not valid_winds:
        return None, None

    average_wind = sum(valid_winds) / len(valid_winds)
    average_wind = average_wind.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    from collections import Counter

    dir_counter = Counter(valid_dirs)
    if dir_counter:
        prevailing_wind_dir = dir_counter.most_common(1)[0][0]
    else:
        prevailing_wind_dir = None

    return average_wind, prevailing_wind_dir


def get_obs_snow(cur: cursor, amedas: int, date: str) -> Decimal | None:
    date_obj = datetime.strptime(date, "%Y-%m-%d")
    next_day_obj = date_obj + timedelta(days=1)
    next_day = next_day_obj.strftime("%Y-%m-%dT00:00:00")
    sql = """
    SELECT
        snow_24h,
        snow_24h_aqc
    FROM
        v_obs_snow_hour
    WHERE
        amedas = %s
        AND record_time = %s
    LIMIT 1
    ;
    """
    cur.execute(sql, (amedas, next_day))
    row = cur.fetchone()
    if row is None:
        return None
    snow, snow_flg = row
    if snow_flg <= 1:
        return snow
    else:
        return None


def get_sta_minmaxtemp(
    cur: cursor, amedas: int, date: str
) -> tuple[Decimal | None, Decimal | None]:
    sql = """
    SELECT
        mintemp_0009,
        mintemp_0009_flg,
        maxtemp_0918,
        maxtemp_0918_flg
    FROM
        t_sta_temp_day_pp
    WHERE
        amedas = %s
        AND record_date = %s
    ;
    """
    cur.execute(sql, (amedas, date))
    row = cur.fetchone()
    if row is None:
        return (None, None)
    mintemp, mintemp_flg, maxtemp, maxtemp_flg = row
    return (
        mintemp if mintemp_flg < 16 else None,
        maxtemp if maxtemp_flg < 16 else None,
    )


def get_sta_prcp(
    cur: cursor, amedas: int, date: str
) -> tuple[Decimal | None, str | None]:
    sql_prcp = """
    SELECT
        prcp,
        prcp_flg
    FROM
        v_sta_prcp_day
    WHERE
        amedas = %s
        AND record_date = %s
    ;
    """
    cur.execute(sql_prcp, (amedas, date))
    row_prcp = cur.fetchone()
    if row_prcp is None:
        return None
    prcp, prcp_flg = row_prcp
    if prcp_flg > 1:
        return None
    return prcp


def get_sta_wind(
    cur: cursor, amedas: int, date: str
) -> Tuple[Decimal | None, str | None]:
    sql_wind = """
    SELECT
        meanwind,
        meanwind_flg,
        modedir16,
        modedir16_flg
    FROM
        v_sta_wind_day
    WHERE
        amedas = %s
        AND record_date = %s
    ;
    """
    cur.execute(sql_wind, (amedas, date))
    row_wind = cur.fetchone()
    if row_wind is None:
        return None, None
    wind, wind_flg, wind_dir16, wind_dir16_flg = row_wind
    if wind_flg > 1:
        wind = None
    if wind_dir16_flg > 1:
        wind_dir16 = None
    return wind, wind_dir16


def get_sta_snow(
    cur: cursor, amedas: int, date: str
) -> tuple[Decimal | None, str | None]:
    sql_snow = """
    SELECT
        snow,
        snow_flg
    FROM
        v_sta_snow_day
    WHERE
        amedas = %s
        AND record_date = %s
    ;
    """
    cur.execute(sql_snow, (amedas, date))
    row_snow = cur.fetchone()
    if row_snow is None:
        return None
    snow, snow_flg = row_snow
    if snow_flg > 1:
        return None
    return snow


def get_amedas_record_time(cur: cursor) -> str:
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
