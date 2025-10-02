import calendar
from typing import TypedDict

from psycopg2.extensions import cursor


class NormalMonth(TypedDict):
    maxtemp_lt0_frequency: float
    maxtemp_ge25_frequency: float
    maxtemp_ge30_frequency: float
    maxtemp_ge35_frequency: float
    mintemp_lt0_frequency: float
    mintemp_ge25_frequency: float
    stat_years: int


def get_normal_month(
    cur: cursor, amedas: int, date_list: list[str]
) -> list[NormalMonth]:
    out: list[NormalMonth] = []
    for date in date_list:
        out.append(_get_normal_month(cur, amedas, date))
    return out


def _get_normal_month(
    cur: cursor,
    amedas: int,
    date: str,
) -> NormalMonth:
    _, mm, _ = date.split("-")
    month = int(mm)
    dom = calendar.monthrange(1991, month)[1]
    sql = """
    SELECT
        maxtemp_lt0,
        maxtemp_ge25,
        maxtemp_ge30,
        maxtemp_ge35,
        mintemp_lt0,
        mintemp_ge25,
        stat_years
    FROM
        m_nml_temp_month
    WHERE 
        amedas = %s
        AND record_month = %s
    ;
    """
    cur.execute(sql, (amedas, month))
    row = cur.fetchone()
    if row is None:
        raise ValueError(
            f"m_nml_temp_month に {amedas} の {date} のデータがありません。"
        )
    if row[0] is None:
        raise ValueError(
            f"m_nml_temp_month に {amedas} の {date} の真冬日日数がありません。"
        )
    if row[1] is None:
        raise ValueError(
            f"m_nml_temp_month に {amedas} の {date} の夏日日数がありません。"
        )
    if row[2] is None:
        raise ValueError(
            f"m_nml_temp_month に {amedas} の {date} の真夏日日数がありません。"
        )
    if row[3] is None:
        raise ValueError(
            f"m_nml_temp_month に {amedas} の {date} の猛暑日日数がありません。"
        )
    if row[4] is None:
        raise ValueError(
            f"m_nml_temp_month に {amedas} の {date} の冬日日数がありません。"
        )
    if row[5] is None:
        raise ValueError(
            f"m_nml_temp_month に {amedas} の {date} の熱帯夜日数がありません。"
        )
    normal_month: NormalMonth = {
        "maxtemp_lt0_frequency": row[0] / dom,
        "maxtemp_ge25_frequency": row[1] / dom,
        "maxtemp_ge30_frequency": row[2] / dom,
        "maxtemp_ge35_frequency": row[3] / dom,
        "mintemp_lt0_frequency": row[4] / dom,
        "mintemp_ge25_frequency": row[5] / dom,
        "stat_years": row[6],
    }
    return normal_month
