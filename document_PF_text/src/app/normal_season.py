from decimal import Decimal
from typing import TypedDict

from psycopg2.extensions import cursor


class NormalSeason(TypedDict):
    quantile: tuple[Decimal, Decimal, Decimal, Decimal, Decimal, Decimal]
    stat_days: int
    stat_years: int


def get_normal_season_prcp(
    cur: cursor, amedas: int, date_list: list[str]
) -> list[NormalSeason]:
    normal_season_list: list[NormalSeason] = []
    for date in date_list:
        normal_season = _get_normal_season_prcp(cur, amedas, date)
        normal_season_list.append(normal_season)
    return normal_season_list


def _get_normal_season_prcp(cur: cursor, amedas: int, date: str) -> NormalSeason:
    _, mm, dd = date.split("-")
    month = int(mm)
    day = int(dd)
    record_day = day
    sql = """
    SELECT
        prcp_q000,
        prcp_q010,
        prcp_q033,
        prcp_q067,
        prcp_q090,
        prcp_q100,
        stat_days
    FROM
        m_nml_prcp_day_pp
    WHERE
        amedas = %s
        AND record_month = %s
        AND record_day = %s
    ;
    """
    cur.execute(sql, (amedas, month, record_day))
    row = cur.fetchone()
    if row is None or None in row[0:6]:
        quantile = (
            Decimal(0),
            Decimal(0),
            Decimal(0),
            Decimal(0),
            Decimal(0),
            Decimal(0),
        )
        stat_days = 1
    else:
        quantile = row[0:6]
        stat_days = row[6]
    normal_season: NormalSeason = {
        "quantile": quantile,
        "stat_days": stat_days,
    }
    return normal_season


def get_normal_season_wind(
    cur: cursor, amedas: int, date_list: list[str]
) -> list[NormalSeason]:
    normal_season_list: list[NormalSeason] = []
    for date in date_list:
        normal_season = _get_normal_season_wind(cur, amedas, date)
        normal_season_list.append(normal_season)
    return normal_season_list


def _get_normal_season_wind(cur: cursor, amedas: int, date: str) -> NormalSeason:
    _, mm, dd = date.split("-")
    month = int(mm)
    day = int(dd)
    if 1 <= day <= 10:
        record_day = 1
    elif 11 <= day <= 20:
        record_day = 11
    else:
        record_day = 21
    sql = """
    SELECT
        meanwind_q000,
        meanwind_q010,
        meanwind_q033,
        meanwind_q067,
        meanwind_q090,
        meanwind_q100,
        stat_years
    FROM
        m_nml_wind_mb10d
    WHERE
        amedas = %s
        AND record_month = %s
        AND record_day = %s
    ;
    """
    cur.execute(sql, (amedas, month, record_day))
    row = cur.fetchone()
    if row is None or None in row[0:6]:
        quantile = (
            Decimal(0),
            Decimal(0),
            Decimal(0),
            Decimal(0),
            Decimal(0),
            Decimal(0),
        )
        stat_years = 1
    else:
        quantile = row[0:6]
        stat_years = row[6]
    normal_season: NormalSeason = {
        "quantile": quantile,
        "stat_years": stat_years,
    }
    return normal_season


def get_normal_season_snow(
    cur: cursor, amedas: int, date_list: list[str]
) -> list[NormalSeason]:
    normal_season_list: list[NormalSeason] = []
    for date in date_list:
        normal_season = _get_normal_season_snow(cur, amedas, date)
        normal_season_list.append(normal_season)
    return normal_season_list


def _get_normal_season_snow(cur: cursor, amedas: int, date: str) -> NormalSeason:
    _, mm, dd = date.split("-")
    month = int(mm)
    day = int(dd)
    record_day = day
    sql = """
    SELECT
        snow_q000,
        snow_q010,
        snow_q033,
        snow_q067,
        snow_q090,
        snow_q100,
        stat_days
    FROM
        m_nml_snow_day_pp
    WHERE
        amedas = %s
        AND record_month = %s
        AND record_day = %s
    ;
    """
    cur.execute(sql, (amedas, month, record_day))
    row = cur.fetchone()
    if row is None or None in row[0:6]:
        quantile = (
            Decimal(0),
            Decimal(0),
            Decimal(0),
            Decimal(0),
            Decimal(0),
            Decimal(0),
        )
        stat_days = 1
    else:
        quantile = row[0:6]
        stat_days = row[6]
    normal_season: NormalSeason = {
        "quantile": quantile,
        "stat_days": stat_days,
    }
    return normal_season


def get_normal_season_maxprcp_1h(
    cur: cursor, amedas: int, date_list: list[str]
) -> list[NormalSeason]:
    normal_season_list: list[NormalSeason] = []
    for date in date_list:
        normal_season = _get_normal_season_maxprcp_1h(cur, amedas, date)
        normal_season_list.append(normal_season)
    return normal_season_list


def _get_normal_season_maxprcp_1h(cur: cursor, amedas: int, date: str) -> NormalSeason:
    _, mm, dd = date.split("-")
    month = int(mm)
    day = int(dd)
    record_day = day
    sql = """
    SELECT
        maxprcp_1h_q000,
        maxprcp_1h_q010,
        maxprcp_1h_q033,
        maxprcp_1h_q067,
        maxprcp_1h_q090,
        maxprcp_1h_q100,
        stat_days
    FROM
        m_nml_prcp_day_pp
    WHERE
        amedas = %s
        AND record_month = %s
        AND record_day = %s
    ;
    """
    cur.execute(sql, (amedas, month, record_day))
    row = cur.fetchone()
    if row is None or None in row[0:6]:
        quantile = (
            Decimal(0),
            Decimal(0),
            Decimal(0),
            Decimal(0),
            Decimal(0),
            Decimal(0),
        )
        stat_days = 1
    else:
        quantile = row[0:6]
        stat_days = row[6]
    normal_season: NormalSeason = {
        "quantile": quantile,
        "stat_days": stat_days,
    }
    return normal_season
