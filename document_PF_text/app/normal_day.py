from decimal import Decimal
from typing import TypedDict

from psycopg2.extensions import cursor


class NormalDay(TypedDict):
    nml: Decimal
    quantile: tuple[Decimal, Decimal, Decimal, Decimal, Decimal, Decimal]
    stat_years: int


def get_normal_day(
    cur: cursor, amedas: int, date_list: list[str]
) -> tuple[list[NormalDay], list[NormalDay]]:
    mintemp_nml_list: list[NormalDay] = []
    maxtemp_nml_list: list[NormalDay] = []
    for date in date_list:
        mintemp_nml, maxtemp_nml = _get_normal_day(cur, amedas, date)
        mintemp_nml_list.append(mintemp_nml)
        maxtemp_nml_list.append(maxtemp_nml)
    return mintemp_nml_list, maxtemp_nml_list


def _get_normal_day(
    cur: cursor,
    amedas: int,
    date: str,
) -> tuple[NormalDay, NormalDay]:
    _, mm, dd = date.split("-")
    month = int(mm)
    day = int(dd)
    sql = """
    SELECT
        mintemp_nml,
        maxtemp_nml,
        mintemp_q000,
        mintemp_q010,
        mintemp_q033,
        mintemp_q067,
        mintemp_q090,
        mintemp_q100,
        maxtemp_q000,
        maxtemp_q010,
        maxtemp_q033,
        maxtemp_q067,
        maxtemp_q090,
        maxtemp_q100,
        stat_years
    FROM
        m_nml_temp_day
    WHERE
        amedas = %s
        AND record_month = %s
        AND record_day = %s
    ;
    """
    cur.execute(sql, (amedas, month, day))
    row = cur.fetchone()
    if row is None:
        raise ValueError(f"m_nml_temp_day に {amedas} の {date} のデータがありません。")
    if row[0] is None:
        raise ValueError(
            f"m_nml_temp_day に {amedas} の {date} の最低気温の平年値がありません。"
        )
    if row[1] is None:
        raise ValueError(
            f"m_nml_temp_day に {amedas} の {date} の最高気温の平年値がありません。"
        )
    if None in row[2:8]:
        raise ValueError(
            f"m_nml_temp_day に {amedas} の {date} の最低気温の平年値の分位数がありません。"
        )
    if None in row[8:14]:
        raise ValueError(
            f"m_nml_temp_day に {amedas} の {date} の最高気温の平年値の分位数がありません。"
        )
    mintemp_normal: NormalDay = {
        "nml": row[0],
        "quantile": row[2:8],
        "stat_years": row[14],
    }
    maxtemp_normal: NormalDay = {
        "nml": row[1],
        "quantile": row[8:14],
        "stat_years": row[14],
    }
    return mintemp_normal, maxtemp_normal
