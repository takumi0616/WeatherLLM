#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
気象庁「最新の気象データ」CSVダウンロード自動取得スクリプト

要件:
1) 現在時刻（日本時刻）を取得
2) 現在の年月日を確認
3) 現在の年月日の00時のCSVデータをダウンロード（全要素）
   - 1時間降水量, 3時間降水量, 6時間降水量, 12時間降水量,
     24時間降水量, 48時間降水量, 72時間降水量, 日降水量, 降水量全要素,
     最大風速, 最大瞬間風速, 最高気温, 最低気温
   - 以下の雪関連は休止中の可能性があるため、取得できなくてもエラーにしない
     （現在の積雪, 最深積雪, 3h/6h/12h/24h/48h/72h降雪量, 降雪量全要素, 累積降雪量）
4) 前日の年月日を確認
5) 前日の年月日の00時のCSVデータをダウンロード（3と同じ扱い）
6) 現在時刻を確認
7) 現在時刻が 00-07時 → 追加なし / 07-13時 → 当日06時のデータ取得 /
   13-19時 → 当日12時のデータ取得 / 19-24時 → 当日18時のデータ取得（3と同じ要素セット）

保存先: ./csv/{when_label}/ 以下に、日付_時刻_要素.csv で保存（例: csv/today_0000/2025-10-27_0000_pre1h.csv）

技術メモ:
- 文字コードはUTF-8で保存（JMA提供CSVはShift_JIS配布のため、取得後にUTF-8へ変換し、元のCSVは保持しない）
- 時刻指定ファイルは原則24時間のみ取得可能（JMA仕様）
  → 「00:00」指定が24時間を超える場合のフォールバック:
     前日24時（= 当日00:00）に相当する「各日24時ファイル（mmdd）」に自動的に切替を試行
     例) 2025-10-27 00:00 → 2025-10-26 の 24時ファイル: pre1h1026.csv
- 最新版URLは使用せず、要件に基づき時刻指定URLを構成して取得
- 観測所（地点）個別ではなく「alltable」（全国テーブル）を取得（仕様出典PDFの例に準拠）

出典（同梱PDF抜粋）:
https://www.data.jma.go.jp/stats/data/mdrr/pre_rct/alltable/pre1h00_YYYYMMDDHHMM.csv
https://www.data.jma.go.jp/stats/data/mdrr/pre_rct/alltable/pre1hMMDD.csv（各日24時）
他、3h/6h/12h/24h/48h/72h, predaily, preall, wind(最大風速/最大瞬間風速), tem(最高/最低)の同系パス

使用方法
- 前提:
  - Python 3.9+（標準ライブラリのみ使用）
  - ネットワーク接続が必要（JMAサイトからCSVを取得）

- 実行:
  - python get_csv_data.py
  - 引数は不要（現時点ではCLIオプションなし）

- 保存先:
  - ./csv/{when_label}/ にUTF-8 CSVとして保存（元のShift_JISは保持しない）
  - when_label は以下のいずれか:
    - today_0000（当日00:00）
    - yesterday_0000（前日00:00）
    - today_0600（現在時刻が07–13時帯のとき追加取得）
    - today_1200（現在時刻が13–19時帯のとき追加取得）
    - today_1800（現在時刻が19–24時帯のとき追加取得）
  - ファイル名: YYYY-MM-DD_HHMM_{要素キー}.csv
    - 例: csv/today_0000/2025-10-27_0000_pre1h.csv

- 取得する要素（必須扱い）:
  - pre1h: 1時間降水量
  - pre3h: 3時間降水量
  - pre6h: 6時間降水量
  - pre12h: 12時間降水量
  - pre24h: 24時間降水量
  - pre48h: 48時間降水量
  - pre72h: 72時間降水量
  - predaily: 日降水量
  - preall: 降水量全要素
  - mxwsp: 最大風速
  - gust: 最大瞬間風速
  - tmax: 最高気温
  - tmin: 最低気温
  - 各要素の意味は本文「要件」リスト参照

- 雪関連（任意・休止想定）:
  - 現状はURL未確定のため取得をスキップ
  - 今後URLが確定したら OPTIONAL_SNOW_ELEMENTS に追記して同様に取得可能
  - 任意要素は取得失敗でもエラーにしない

- 時刻判定ロジック:
  - 常に当日00:00と前日00:00を対象
  - 現在時刻に応じて以下を追加
    - 07–13時: 当日06:00
    - 13–19時: 当日12:00
    - 19–24時: 当日18:00
    - 00–07時: 追加なし

- 00:00のフォールバック:
  - JMAの時刻指定CSVは原則24時間保持
  - 当日00:00の時刻指定URLが取得不可の場合、前日24時（=当日00:00）の「MMDD形式」ファイルに自動フォールバック
  - 例: 2025-10-27 00:00 → 前日10/26の各要素 pre1h1026.csv など

- エンコード:
  - JMA配布CSVはShift_JIS想定
  - 本スクリプトはUTF-8に変換して保存（CRLF→LF）
  - pandas等からは encoding 指定なしで読み取り可能な想定
    - 例: pd.read_csv("csv/today_0000/2025-10-27_0000_pre1h.csv")

- ログと終了コード:
  - 取得結果はINFO/WARNINGログで出力
  - 任意要素の失敗はエラーとしない
  - 失敗が含まれても終了コードは0（仕様上未公開時間や更新待ちを考慮）

- 注意:
  - データの更新タイミングはJMA仕様に従う（毎時00分の観測は約50分後更新、前日〜7日前は1日4回更新など）
  - ネットワーク障害や一時的な404等は自動リトライ（指数バックオフ）を実施

- 出典:
  - 本文「出典（同梱PDF抜粋）」のURLパターンに従ってダウンロード
"""

from __future__ import annotations

import os
import sys
import time
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from urllib import request, error

# Python 3.9+ で zoneinfo が使える
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None  # type: ignore

JST = ZoneInfo("Asia/Tokyo") if ZoneInfo else timezone(timedelta(hours=9))

BASE_URL = "https://www.data.jma.go.jp/stats/data/mdrr"

# 取得対象（必須）: ドキュメントのURL命名規則に沿って構築
# time_prefix: 時刻指定用のファイル名先頭（末尾に _YYYYMMDDHHMM.csv が付く）
# daily_prefix: 各日24時のファイル名先頭（末尾に MMDD.csv が付く）
# path: pre_rct/wind_rct/tem_rct などのサブパス
REQUIRED_ELEMENTS: List[Dict[str, str]] = [
    # 降水量
    {"key": "pre1h", "desc": "1時間降水量", "path": "pre_rct/alltable", "time_prefix": "pre1h00", "daily_prefix": "pre1h"},
    {"key": "pre3h", "desc": "3時間降水量", "path": "pre_rct/alltable", "time_prefix": "pre3h00", "daily_prefix": "pre3h"},
    {"key": "pre6h", "desc": "6時間降水量", "path": "pre_rct/alltable", "time_prefix": "pre6h00", "daily_prefix": "pre6h"},
    {"key": "pre12h", "desc": "12時間降水量", "path": "pre_rct/alltable", "time_prefix": "pre12h00", "daily_prefix": "pre12h"},
    {"key": "pre24h", "desc": "24時間降水量", "path": "pre_rct/alltable", "time_prefix": "pre24h00", "daily_prefix": "pre24h"},
    {"key": "pre48h", "desc": "48時間降水量", "path": "pre_rct/alltable", "time_prefix": "pre48h00", "daily_prefix": "pre48h"},
    {"key": "pre72h", "desc": "72時間降水量", "path": "pre_rct/alltable", "time_prefix": "pre72h00", "daily_prefix": "pre72h"},
    {"key": "predaily", "desc": "日降水量", "path": "pre_rct/alltable", "time_prefix": "predaily00", "daily_prefix": "predaily"},
    {"key": "preall", "desc": "降水量全要素", "path": "pre_rct/alltable", "time_prefix": "preall00", "daily_prefix": "preall"},
    # 風
    {"key": "mxwsp", "desc": "最大風速", "path": "wind_rct/alltable", "time_prefix": "mxwsp00", "daily_prefix": "mxwsp"},
    {"key": "gust", "desc": "最大瞬間風速", "path": "wind_rct/alltable", "time_prefix": "gust00", "daily_prefix": "gust"},
    # 気温
    {"key": "tmax", "desc": "最高気温", "path": "tem_rct/alltable", "time_prefix": "mxtemsadext00", "daily_prefix": "mxtemsadext"},
    {"key": "tmin", "desc": "最低気温", "path": "tem_rct/alltable", "time_prefix": "mntemsadext00", "daily_prefix": "mntemsadext"},
]

# 取得対象（任意・休止想定）: URL仕様が明示されていないため、ここでは定義のみ（空）
# もし公式のCSV名規則が分かり次第、REQUIRED_ELEMENTS と同じ形式で追記して良い
OPTIONAL_SNOW_ELEMENTS: List[Dict[str, str]] = [
    # 例（名称のみ記録。URLが未確認のため実際のダウンロードはスキップ）
    # {"key": "snow_now", "desc": "現在の積雪", ...},
    # {"key": "snow_deep", "desc": "最深積雪", ...},
    # {"key": "snow3h", "desc": "3時間降雪量", ...},
    # ...
]


@dataclass
class DownloadResult:
    key: str
    desc: str
    when_label: str
    url: str
    path: str
    ok: bool
    status: Optional[int] = None
    error: Optional[str] = None
    note: Optional[str] = None


def jst_now() -> datetime:
    return datetime.now(JST)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def http_get_binary(url: str, timeout: float = 20.0, retries: int = 3, backoff: float = 1.5) -> Tuple[Optional[bytes], Optional[int], Optional[str]]:
    """
    urllibで単純GET（バイナリ）。HTTPエラー時はステータスコードとエラー文字列を返す。
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; JMA-CSV-Downloader/1.0; +local)",
        "Accept": "*/*",
        "Connection": "close",
    }
    req = request.Request(url, headers=headers, method="GET")

    for attempt in range(1, retries + 1):
        try:
            with request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
                status = getattr(resp, "status", 200)
                return data, status, None
        except error.HTTPError as e:
            # HTTPエラーはリトライしても改善しない可能性が高いが、仕様上一応リトライ
            if attempt >= retries:
                return None, e.code, f"HTTPError: {e}"
        except error.URLError as e:
            if attempt >= retries:
                return None, None, f"URLError: {e}"
        except Exception as e:
            if attempt >= retries:
                return None, None, f"Exception: {e}"
        time.sleep(backoff ** attempt)
    return None, None, "Unknown error"


def build_time_url(elem: Dict[str, str], dt: datetime) -> str:
    # 時刻指定（例） pre1h00_202310310940.csv
    return f"{BASE_URL}/{elem['path']}/{elem['time_prefix']}_{dt.strftime('%Y%m%d%H%M')}.csv"


def build_daily24_url(elem: Dict[str, str], dt: datetime) -> str:
    # 各日24時（例） pre1h1028.csv
    return f"{BASE_URL}/{elem['path']}/{elem['daily_prefix']}{dt.strftime('%m%d')}.csv"


def save_bytes(path: str, data: bytes) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "wb") as f:
        f.write(data)


def save_as_utf8(path: str, sjis_bytes: bytes) -> None:
    """
    取得したShift_JIS(CP932)想定のCSVをUTF-8として上書き保存する。
    元のShift_JISのファイルは保持しない（作成しない）。
    """
    try:
        text = sjis_bytes.decode("cp932", errors="strict")
    except UnicodeDecodeError:
        # 念のため shift_jis でも再試行し、一部不正なバイトは置換
        text = sjis_bytes.decode("shift_jis", errors="replace")
    ensure_dir(os.path.dirname(path))
    # 改行はLFで保存
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def cleanup_legacy_utf8_mirror(path: str) -> Optional[str]:
    """
    旧仕様で作成された *_utf8.csv ミラーが存在すれば削除する。
    削除した場合はそのパスを返す。存在しなければ None。
    """
    mirror = path[:-4] + "_utf8.csv" if path.lower().endswith(".csv") else path + ".utf8.csv"
    try:
        if os.path.exists(mirror):
            os.remove(mirror)
            return mirror
    except Exception:
        # 削除失敗は致命的ではないので無視
        pass
    return None


def download_for_time(dt_target: datetime, when_label: str, out_root: str = "csv", include_optional_snow: bool = True) -> List[DownloadResult]:
    """
    指定の日本時刻 dt_target の観測時刻CSVを全要素についてダウンロード。
    - 00:00 の場合、24h制限で取得できないときは前日24時(MMDD形式)へフォールバックを試行。
    """
    results: List[DownloadResult] = []
    label_dir = os.path.join(out_root, when_label)
    date_label = dt_target.strftime("%Y-%m-%d_%H%M")

    # 必須要素
    targets: List[Tuple[Dict[str, str], bool]] = [(e, True) for e in REQUIRED_ELEMENTS]

    # 任意（雪）要素（URL未確定のため、現状はスキップ設定）
    if include_optional_snow and OPTIONAL_SNOW_ELEMENTS:
        targets.extend((e, False) for e in OPTIONAL_SNOW_ELEMENTS)

    for elem, is_required in targets:
        # 1) 時刻指定URLで試行
        url_time = build_time_url(elem, dt_target)
        dest_path = os.path.join(label_dir, f"{dt_target.strftime('%Y-%m-%d')}_{dt_target.strftime('%H%M')}_{elem['key']}.csv")

        data, status, err = http_get_binary(url_time)
        if data and (status is None or 200 <= status < 300):
            # 直接UTF-8で保存（元のShift_JISファイルは保持しない）
            save_as_utf8(dest_path, data)
            deleted = cleanup_legacy_utf8_mirror(dest_path)
            note = f"旧ミラー削除: {deleted}" if deleted else None
            results.append(DownloadResult(key=elem["key"], desc=elem["desc"], when_label=when_label, url=url_time, path=dest_path, ok=True, status=status, note=note))
            continue

        # 2) フォールバック（00:00のみ）→ 前日24時 (= 当日00:00) を MMDD 形式で取得
        did_fallback = False
        fallback_note = None
        if dt_target.hour == 0 and dt_target.minute == 0:
            prev_day = (dt_target - timedelta(days=1))
            url_daily24 = build_daily24_url(elem, prev_day)
            data2, status2, err2 = http_get_binary(url_daily24)
            if data2 and (status2 is None or 200 <= status2 < 300):
                # 直接UTF-8で保存（元のShift_JISファイルは保持しない）
                save_as_utf8(dest_path, data2)
                did_fallback = True
                fallback_note = f"時刻指定の取得不可のためフォールバック（前日24時MMDD形式）: {prev_day.strftime('%m%d')}"
                deleted = cleanup_legacy_utf8_mirror(dest_path)
                notes = [fallback_note]
                if deleted:
                    notes.append(f"旧ミラー削除: {deleted}")
                results.append(DownloadResult(key=elem["key"], desc=elem["desc"], when_label=when_label, url=url_daily24, path=dest_path, ok=True, status=status2, note=" | ".join(notes)))
                continue
            else:
                # フォールバックも失敗
                err = err2 or err
                status = status2 if status2 is not None else status

        # 3) 失敗処理
        if is_required:
            results.append(DownloadResult(key=elem["key"], desc=elem["desc"], when_label=when_label, url=(url_daily24 if (dt_target.hour==0 and dt_target.minute==0 and did_fallback) else url_time), path=dest_path, ok=False, status=status, error=(err or "Failed to download")))
        else:
            # 任意要素は失敗してもエラー扱いにしない（ログのみ）
            results.append(DownloadResult(key=elem["key"], desc=elem["desc"], when_label=when_label, url=(url_daily24 if (dt_target.hour==0 and dt_target.minute==0 and did_fallback) else url_time), path=dest_path, ok=False, status=status, error=(err or "Optional element download failed")))
    return results


def plan_times_now(now_jst: datetime) -> List[Tuple[datetime, str]]:
    """
    ダウンロード対象の時刻を決める。
    - 当日00:00
    - 前日00:00
    - 現在時刻が:
        00-07 → 追加なし
        07-13 → 当日06:00
        13-19 → 当日12:00
        19-24 → 当日18:00
    """
    times: List[Tuple[datetime, str]] = []

    today00 = now_jst.replace(hour=0, minute=0, second=0, microsecond=0)
    yest00 = (today00 - timedelta(days=1))

    times.append((today00, "today_0000"))
    times.append((yest00, "yesterday_0000"))

    hour = now_jst.hour
    if 7 <= hour < 13:
        times.append((today00.replace(hour=6), "today_0600"))
    elif 13 <= hour < 19:
        times.append((today00.replace(hour=12), "today_1200"))
    elif 19 <= hour or hour < 0:  # hour < 0 は起こらないが表記上
        times.append((today00.replace(hour=18), "today_1800"))

    return times


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    now = jst_now()
    logging.info("現在（JST）: %s", now.strftime("%Y-%m-%d %H:%M:%S %Z"))

    targets = plan_times_now(now)
    all_results: List[DownloadResult] = []

    for dt, label in targets:
        logging.info("ダウンロード対象: %s (%s)", dt.strftime("%Y-%m-%d %H:%M"), label)
        results = download_for_time(dt, when_label=label, out_root="csv", include_optional_snow=True)
        for r in results:
            if r.ok:
                logging.info("OK  %-12s %-10s -> %s", r.key, label, r.path)
                if r.note:
                    logging.info("     note: %s", r.note)
            else:
                # 必須要素の失敗は WARNING、任意要素の失敗は INFO に留めたいが、
                # 現状は is_required を持っていないため、キーで判断は困難。
                # ここではメッセージに「(optional?)」を含めるかは省略し統一表示。
                logging.warning("NG  %-12s %-10s (status=%s) url=%s err=%s", r.key, label, str(r.status), r.url, (r.error or ""))
        all_results.extend(results)

    # 結果を要約
    total_ok = sum(1 for r in all_results if r.ok)
    total_ng = sum(1 for r in all_results if not r.ok)
    logging.info("完了: 成功=%d, 失敗=%d, 保存先ルート=./csv", total_ok, total_ng)

    # 失敗は含まれても終了コードは 0（雪要素・時刻指定24h超など仕様上やむなしケースがあるため）
    return 0


if __name__ == "__main__":
    sys.exit(main())
