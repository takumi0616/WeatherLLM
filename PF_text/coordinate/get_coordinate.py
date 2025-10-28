#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
smaster.index（Shift-JIS/固定長146バイト）から
「管区・沖縄・地方気象台（= 観測所区分: 1,2,4）」の現行56地点の緯度・経度を抽出して出力します。

必要なライブラリ:
  - 追加インストール不要（標準ライブラリのみ使用）
    * argparse, csv, json, sys, pathlib, typing, dataclasses

使い方例:
  - CSVを標準出力:
      python get_coordinate.py
  - JSONを標準出力:
      python get_coordinate.py --format json
  - 入力ファイルパスを明示:
      python get_coordinate.py --index ./smaster.index
  - ファイル保存:
      python get_coordinate.py --format csv --out ./coordinates.csv

実装メモ（フォーマットの要点）:
  - 文字コード: Shift-JIS、レコード長: 146バイト + 改行
  - バイト位置（1始まり, []は長さ）:
      1[3]   : 地点番号
      15[1]  : 観測所区分（1:管区, 2:旧地台, 4:新地台 ほか）
      37[6]  : 緯度（ddmmdd : 分は2桁小数= mm.mm を小数点除去）
      43[7]  : 経度（dddmmdd : 同上）
      65[8]  : 観測期間_開始年月日 (YYYYMMDD)
      73[8]  : 観測期間_終了年月日 (YYYYMMDD) … 99999999 は継続中
      81[12] : 漢字地点名
      93[18] : 漢字官署名
      111[8] : 都道府県名（～振興局）
      119[5] : 標高（官署の高さ; 参照用）
  - 固定長は「バイト」単位なので、必ずバイナリ読み込みし、バイトスライスで切り出すこと。
  - 緯度/経度の数値化:
      例：緯度 "452490" = 45度24.90分 → 45 + (24.90 / 60) = 45.415
          経度 "139444" なら 139度44.40分 → 139 + (44.40 / 60)
    （分に2桁小数を含むため、後半4桁を100で割り、"mm.mm" として 60 で度に換算）

抽出ポリシー:
  - 観測所区分 in {"1","2","4"} の現行(終了年月日=99999999)のみ
    * 1: 管区気象台（沖縄気象台を含む）
    * 2: 旧地台（Local Observatory: 過去から継続して現行は "4" が多いが念のため）
    * 4: 新地台（Local Meteorological Observatory）
  - 想定件数は合計で 56 地点（管区6 + 地方気象台50）。実データに依存するため、
    実行時に件数を表示する（json/csv 出力後に stderr へ件数ログ）。

出力カラム:
  - CSV/JSON 共通:
    ["station_id", "division_code", "division_label", "prefecture", "office_name",
     "place_name", "lat", "lon", "start_date", "end_date", "elevation_m"]

注意:
  - smaster.index の Shift-JIS 内の日本語フィールドは bytes を各長さで取り出してから decode("shift_jis", errors="ignore").
  - 行末 CRLF/ LF は除去してから 146バイトを想定。足りない行はスキップ。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Iterable, Tuple


# 1始まりのバイト位置/長さ定義
B_STATION_ID = (1, 3)
B_DIVISION = (15, 1)      # 観測所区分
B_LAT = (37, 6)           # 緯度: dd mmdd (mm.dd の小数点除去で4桁)
B_LON = (43, 7)           # 経度: ddd mmdd
B_START = (65, 8)         # 観測期間 開始
B_END = (73, 8)           # 観測期間 終了
B_PLACE_NAME = (81, 12)   # 漢字地点名
B_OFFICE_NAME = (93, 18)  # 漢字官署名
B_PREF_NAME = (111, 8)    # 都道府県/振興局
B_ELEV = (119, 5)         # 標高（文字→数値化は任意）


DIVISION_LABEL = {
    "1": "管区気象台",
    "2": "旧地方気象台",
    "3": "海洋気象台（～2013/09/30）",
    "4": "地方気象台（新地台）",
    "5": "測候所",
    "6": "地球環境業務課",
    "7": "施設等機関",
    "8": "特別地域気象観測所",
}


@dataclass
class Station:
    station_id: str
    division_code: str
    division_label: str
    prefecture: str
    office_name: str
    place_name: str
    lat: float
    lon: float
    start_date: str
    end_date: str
    elevation_m: Optional[float]


def _bslice(data: bytes, start_1: int, length: int) -> bytes:
    s = start_1 - 1
    return data[s : s + length]


def _decode_sjis_str(b: bytes) -> str:
    return b.decode("shift_jis", errors="ignore").strip()


def _decode_ascii_str(b: bytes) -> str:
    # 数値項目/英数項目向け
    return b.decode("ascii", errors="ignore").strip()


def _parse_lat(lat_bytes: bytes) -> Optional[float]:
    """
    緯度: 6桁 "ddmmdd"（dd:度, mmdd: 分(小数2桁) を小数点を除いた4桁）
      例: "452490" = 45度24.90分 → 45 + 24.90/60
    """
    s = _decode_ascii_str(lat_bytes)
    if not s or len(s) < 4 or not s.isdigit():
        return None
    try:
        deg = int(s[:2])
        mmdd = int(s[2:])  # 4桁
        minutes = mmdd / 100.0  # 小数点復元
        return round(deg + (minutes / 60.0), 6)
    except Exception:
        return None


def _parse_lon(lon_bytes: bytes) -> Optional[float]:
    """
    経度: 7桁 "dddmmdd"（ddd:度, mmdd: 分(小数2桁)）
    """
    s = _decode_ascii_str(lon_bytes)
    if not s or len(s) < 5 or not s.isdigit():
        return None
    try:
        deg = int(s[:3])
        mmdd = int(s[3:])  # 4桁
        minutes = mmdd / 100.0
        return round(deg + (minutes / 60.0), 6)
    except Exception:
        return None


def _parse_float_ascii(b: bytes) -> Optional[float]:
    s = _decode_ascii_str(b)
    if not s:
        return None
    # 標高等: "01234" のようなゼロパディングが想定されるため int/floatへ
    try:
        return float(s)
    except Exception:
        # 非数値（空白）の場合 None
        return None


def parse_record(line_bytes: bytes) -> Optional[Station]:
    """
    1行(146バイト)を Station にパース。必要項目が欠ける場合は None。
    """
    # 改行除去
    rec = line_bytes.rstrip(b"\r\n")
    if len(rec) < 146:
        return None  # 想定外レコードはスキップ

    station_id = _decode_ascii_str(_bslice(rec, *B_STATION_ID))
    division_code = _decode_ascii_str(_bslice(rec, *B_DIVISION))
    division_label = DIVISION_LABEL.get(division_code, "")

    lat = _parse_lat(_bslice(rec, *B_LAT))
    lon = _parse_lon(_bslice(rec, *B_LON))

    start_date = _decode_ascii_str(_bslice(rec, *B_START))
    end_date = _decode_ascii_str(_bslice(rec, *B_END))

    place_name = _decode_sjis_str(_bslice(rec, *B_PLACE_NAME))
    office_name = _decode_sjis_str(_bslice(rec, *B_OFFICE_NAME))
    prefecture = _decode_sjis_str(_bslice(rec, *B_PREF_NAME))
    elevation_m = _parse_float_ascii(_bslice(rec, *B_ELEV))

    # 緯度経度が欠落していれば除外
    if lat is None or lon is None:
        return None

    return Station(
        station_id=station_id,
        division_code=division_code,
        division_label=division_label,
        prefecture=prefecture,
        office_name=office_name,
        place_name=place_name,
        lat=lat,
        lon=lon,
        start_date=start_date,
        end_date=end_date,
        elevation_m=elevation_m,
    )


def load_stations(index_path: Path) -> List[Station]:
    stations: List[Station] = []
    with index_path.open("rb") as f:
        for line in f:
            st = parse_record(line)
            if st:
                stations.append(st)
    return stations


def filter_kanku_okinawa_chihou(stations: Iterable[Station]) -> List[Station]:
    """
    観測所区分 in {1(管区),2(旧地台),4(新地台)} かつ 現行(end_date=99999999) を抽出。
    想定総数: 56地点（管区6 + 地方気象台50）
    """
    want_divs = {"1", "2", "4"}
    result: List[Station] = []
    for s in stations:
        if s.division_code in want_divs and s.end_date == "99999999":
            result.append(s)

    # 重複排除（同一地点番号が重複する場合に備え、end_date=99999999 なら原則重複しないはず）
    # 念のため station_id + division_code でユニーク化
    uniq: dict[Tuple[str, str], Station] = {}
    for s in result:
        uniq[(s.station_id, s.division_code)] = s
    return list(uniq.values())


def sort_for_presentation(stations: List[Station]) -> List[Station]:
    """
    表示順:
      1) 管区気象台 (division_code=1)
      2) 地方気象台（新地台=4, 旧地台=2）
      3) その他（該当しないはずだが保険）
    さらに都道府県→官署名で昇順。
    """
    def key(s: Station) -> Tuple[int, str, str]:
        priority = 3
        if s.division_code == "1":
            priority = 1
        elif s.division_code in {"4", "2"}:
            priority = 2
        return (priority, s.prefecture, s.office_name)
    return sorted(stations, key=key)


def write_csv(stations: List[Station], out_file: Optional[Path]) -> None:
    fieldnames = [
        "station_id", "division_code", "division_label", "prefecture",
        "office_name", "place_name", "lat", "lon", "start_date", "end_date", "elevation_m",
    ]
    if out_file:
        fp = out_file.open("w", newline="", encoding="utf-8")
        close = True
    else:
        fp = sys.stdout
        close = False
    try:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for s in stations:
            writer.writerow(asdict(s))
    finally:
        if close:
            fp.close()


def write_json(stations: List[Station], out_file: Optional[Path]) -> None:
    data = [asdict(s) for s in stations]
    text = json.dumps(data, ensure_ascii=False, indent=2)
    if out_file:
        out_file.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="smaster.index から『管区・沖縄・地方気象台』(現行)56地点の緯度経度を抽出して出力するツール"
    )
    parser.add_argument(
        "--index", type=Path, default=Path("smaster.index"),
        help="smaster.index へのパス (Shift-JIS / 固定長146バイト)"
    )
    parser.add_argument(
        "--format", choices=["csv", "json"], default="json",
        help="出力フォーマット（csv または json）"
    )
    parser.add_argument(
        "--out", type=Path, default=Path("src/WeatherLLM/kanku_chihou_56.json"),
        help="出力ファイルパス（未指定なら JSON を src/WeatherLLM/kanku_chihou_56.json に保存）"
    )
    args = parser.parse_args()

    if not args.index.exists():
        sys.stderr.write(f"[ERROR] index file not found: {args.index}\n")
        sys.exit(1)

    all_stations = load_stations(args.index)
    filtered = filter_kanku_okinawa_chihou(all_stations)
    ordered = sort_for_presentation(filtered)

    if args.format == "csv":
        write_csv(ordered, args.out)
    else:
        write_json(ordered, args.out)

    # 件数ログ
    total = len(ordered)
    # 区分別件数
    counts = {}
    for s in ordered:
        counts[s.division_label] = counts.get(s.division_label, 0) + 1
    sys.stderr.write(f"[INFO] extracted {total} stations\n")
    for k in sorted(counts):
        sys.stderr.write(f"[INFO]  {k}: {counts[k]}\n")


if __name__ == "__main__":
    main()
