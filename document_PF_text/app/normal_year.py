from typing import TypedDict

from psycopg2.extensions import cursor


class NormalYear(TypedDict):
    mintemp_diff_sd: float
    maxtemp_diff_sd: float


def get_normal_year(cur: cursor, amedas: int) -> NormalYear:
    sql = """
    SELECT
        mintemp_0009_diff_sd,
        maxtemp_0918_diff_sd
    FROM
        m_nml_temp_year_pp
    WHERE 
        amedas = %s
    ;
    """
    cur.execute(sql, (amedas,))
    row = cur.fetchone()
    if row is None:
        raise ValueError(f"m_nml_temp_year_pp に {amedas} のデータがありません。")
    mintemp_diff_sd, maxtemp_diff_sd = row
    return NormalYear(
        mintemp_diff_sd=mintemp_diff_sd,
        maxtemp_diff_sd=maxtemp_diff_sd,
    )
