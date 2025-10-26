#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
気象庁 JMAXML(防災情報XML) から警報・注意報などを汎用取得するスクリプト

目的:
- Qiita 記事「Pythonで気象警報・注意報を取得する【気象庁防災情報XML】」の実装を下敷きに、
  日本全国のどんな地域(市区町村コード/市区町村名)でも取得できるように汎用化。
- さらに「どんなデータでも持ってこれる」よう、任意の電文種別コード(例: VPWW53)を指定して
  Atomフィードから対象電文を取得・JSON化できる CLI を提供。

主な機能:
1) 最新の長期(随時)フィード(extra_l.xml)を取得し、エントリ(電文)一覧を処理
2) 電文種別コード(例: VPWW53=気象特別警報・警報・注意報)でフィルタ
3) 都道府県/市区町村名 または 地域コード(例: 金沢市=1720100)で該当の市町村の警報・注意報を抽出
4) 任意の電文を取得し、そのXMLを辞書(JSON)化して保存/標準出力に表示
5) VPWW53 の都道府県ごとの最新電文から配下エリア一覧(市町村名・コード)を列挙

使い方(例):
- 地域コード(1720100:金沢市)で指定
nohup python get_alarm.py warnings --area-code 1720100 > get_alarm.log 2>&1 &

- 電文コードだけ指定し、全エントリ(最新フィード内)の概要を一覧
nohup python get_alarm.py fetch --product VPWW53 --list > get_alarm_list.log 2>&1 &

注意:
- フィードは「長期(随時)」extra_l.xmlを既定で使用します。--feed でURLや別フィードも指定可。
- VPWW53 の警報抽出はJMAXML構造に依存します。将来のスキーマ変更時は調整が必要です。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import re
from datetime import datetime, timedelta, date, time as dtime
from typing import Any, Dict, Iterable, List, Optional, Union

# 依存パッケージの存在チェック
try:
    import requests
    import xmltodict
except Exception as e:
    print(
        "必要パッケージが不足しています。以下を実行してください:\n"
        "  pip install requests xmltodict",
        file=sys.stderr,
    )
    raise

# 既定のフィードURL(長期: 随時)
DEFAULT_FEED_URLS = {
    "extra_l": "https://www.data.jma.go.jp/developer/xml/feed/extra_l.xml",  # 長期(随時)
    "extra": "https://www.data.jma.go.jp/developer/xml/feed/extra.xml",      # 長期(標準)
}

USER_AGENT = "WeatherLLM-JMAXML-Fetcher/1.0 (+https://www.jma.go.jp/)"


# ========== 基本ユーティリティ ==========

def fetch_url(url: str, timeout: int = 20) -> bytes:
    """URL からXML(等)を取得して bytes を返す。HTTPエラーは例外。"""
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.content


def parse_xml_to_dict(xml_bytes: bytes) -> Dict[str, Any]:
    """XML(bytes) -> dict(xmltodict) に変換。BeautifulSoup に依存せず直接パース。"""
    # JMAのXMLはUTF-8が基本だが、念のためデコード例外を吸収
    if isinstance(xml_bytes, bytes):
        try:
            text = xml_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = xml_bytes.decode("utf-8", errors="ignore")
    else:
        text = str(xml_bytes)
    return xmltodict.parse(text)


def load_feed(feed: str) -> Dict[str, Any]:
    """
    フィードを取得して dict を返す。
    - feed に 'extra_l', 'extra' の名称 または フルURL を指定可能。
    """
    if feed.startswith("http://") or feed.startswith("https://"):
        url = feed
    else:
        url = DEFAULT_FEED_URLS.get(feed, DEFAULT_FEED_URLS["extra_l"])
    xml = fetch_url(url)
    return parse_xml_to_dict(xml)


def ensure_list(x: Union[None, Dict[str, Any], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """entryやItemなど、dict/配列/None を常に list[dict] に正規化。"""
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def iter_feed_entries(feed_dict: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """Atom feed の entry を列挙。"""
    entries = feed_dict.get("feed", {}).get("entry")
    for e in ensure_list(entries):
        yield e


def filter_entries(
    entries: Iterable[Dict[str, Any]],
    product_code: Optional[str] = None,
    author: Optional[str] = None,
    title_contains: Optional[str] = None,
) -> Iterable[Dict[str, Any]]:
    """条件で entry をフィルタ。product_code は entry['id'] に含まれるコードを想定。"""
    for e in entries:
        eid = e.get("id", "") or ""
        title = e.get("title", "") or ""
        author_name = (e.get("author") or {}).get("name", "") or ""
        if product_code and product_code not in eid:
            continue
        if author and author != author_name:
            continue
        if title_contains and title_contains not in title:
            continue
        yield e


def fetch_entry_dict(entry_or_url: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """エントリの id(URL) からXMLを取得して dict 化。"""
    url = entry_or_url if isinstance(entry_or_url, str) else entry_or_url.get("id", "")
    if not url:
        raise ValueError("entry が不正で id(URL) が得られません。")
    xml = fetch_url(url)
    return parse_xml_to_dict(xml)


# ========== VPWW53(気象特別警報・警報・注意報) 用の抽出 ==========

def get_head_title(report_dict: Dict[str, Any]) -> str:
    """jmx:Report/Head/Title を取得(なければ空文字)。"""
    try:
        return report_dict["jmx:Report"]["Head"]["Title"] or ""
    except Exception:
        return ""


def find_latest_vpww53_entry_for_prefecture(
    feed_dict: Dict[str, Any],
    prefecture: Optional[str] = None,
    author: Optional[str] = None,
    max_probe: int = 60,
) -> Optional[Dict[str, Any]]:
    """
    都道府県名 または 気象台(author) で、最新フィードから VPWW53 の該当電文エントリを探す。
    - author が指定されれば feed の author 完全一致で最優先フィルタ。
    - prefecture が与えられた場合は、候補VPWW53を新しい順で最大 max_probe 件まで順に取得し、
      jmx:Report/Head/Title に都道府県名が含まれるものを採用。
    - いずれも指定がなければ、最新の VPWW53 の先頭(最も新しい)を返す。
    """
    entries_all = list(iter_feed_entries(feed_dict))
    # 新しい順に並んでいる前提(Atom feedのupdated順)。念のため順序維持。
    candidates = list(filter_entries(entries_all, product_code="VPWW53", author=author))

    if author:
        # author で既に絞ってあるので最初のもの
        return candidates[0] if candidates else None

    if prefecture:
        # prefecture を Head.Title で検知
        for i, ent in enumerate(candidates[:max_probe]):
            try:
                rep = fetch_entry_dict(ent)
                title = get_head_title(rep)
                if prefecture in title:
                    return ent
            except Exception:
                # 失敗しても次を試す(通信/解析エラーの影響を狭める)
                continue
        return None

    # 何も指定なし→最新(先頭)
    return candidates[0] if candidates else None


def extract_city_warnings_from_vpww53(
    report_dict: Dict[str, Any],
    city_name: Optional[str] = None,
    area_code: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    VPWW53 の jmx:Report から、市町村ごとの警報・注意報を抽出。
    city_name または area_code のいずれか/両方でフィルタ可能。未指定なら全市町村分を返す。

    返却の各要素:
    {
      "area_name": 市町村名,
      "area_code": 市町村コード,
      "warnings": ["波浪警報","雷注意報", ...] (解除/なしは除外),
      "raw_kinds": [Kind辞書...],
      "change_status": "変化無" など
    }
    """
    res: List[Dict[str, Any]] = []

    report = report_dict.get("jmx:Report") or {}
    body = report.get("Body") or {}
    warnings = ensure_list(body.get("Warning"))

    # 「気象警報・注意報（市町村等）」の Item 群を探す
    items: List[Dict[str, Any]] = []
    for w in warnings:
        if str(w.get("@type", "")).startswith("気象警報・注意報（市町村"):
            items.extend(ensure_list(w.get("Item")))

    for item in items:
        area = item.get("Area") or {}
        name = area.get("Name") or ""
        code = area.get("Code") or ""

        if area_code and area_code != code:
            continue
        if city_name and city_name != name:
            continue

        kind = item.get("Kind")
        kinds: List[Dict[str, Any]] = []
        # Kind の形は dict(単発/なし) or list(複数)
        if isinstance(kind, dict):
            # 'Status'のみのケースは「発表警報・注意報はなし」
            if "Status" in kind and set(kind.keys()) == {"Status"}:
                kinds = []
            else:
                kinds = [kind]
        elif isinstance(kind, list):
            kinds = kind
        else:
            kinds = []

        names: List[str] = []
        details: List[Dict[str, Any]] = []
        for k in kinds:
            status = k.get("Status")
            if status == "解除":
                continue
            wname = k.get("Name") or ""
            if wname:
                names.append(wname)
                details.append(k)

        res.append(
            {
                "area_name": name,
                "area_code": code,
                "warnings": names,
                "raw_kinds": details,
                "change_status": item.get("ChangeStatus"),
            }
        )

    # city/area_code 未指定で全件欲しい場合、そのまま返す
    return res


def list_vpww53_areas(report_dict: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    VPWW53 電文から「市町村等」ブロックに現れる全エリア(市区町村名・コード)を列挙。
    """
    result: List[Dict[str, str]] = []
    report = report_dict.get("jmx:Report") or {}
    body = report.get("Body") or {}
    warnings = ensure_list(body.get("Warning"))
    for w in warnings:
        if str(w.get("@type", "")).startswith("気象警報・注意報（市町村"):
            for item in ensure_list(w.get("Item")):
                area = item.get("Area") or {}
                name = area.get("Name") or ""
                code = area.get("Code") or ""
                if name or code:
                    result.append({"area_name": name, "area_code": code})
    return result


def extract_prefecture_codes_from_vpww53(report_dict: Dict[str, Any]) -> List[str]:
    """
    VPWW53 電文から府県レベルの地域コード(府県予報区/都道府県等)を抽出して返す。
    複数見つかった場合は発見順で重複なしのリストを返す。見つからなければ空リスト。
    """
    result: List[str] = []
    report = report_dict.get("jmx:Report") or {}
    body = report.get("Body") or {}

    # Body/Warning のうち、府県レベルのブロックから抽出
    warnings = ensure_list(body.get("Warning"))
    for w in warnings:
        t = str(w.get("@type", ""))
        if ("府県予報区" in t) or ("都道府県" in t):
            for item in ensure_list(w.get("Item")):
                area = item.get("Area") or {}
                code = area.get("Code") or ""
                if code and code not in result:
                    result.append(code)

    # 見つからない場合は Head/Target に Area/Code があればフォールバックで取得
    if not result:
        head = report.get("Head") or {}
        target = head.get("Target")
        def gather_codes_from_target(tar) -> List[str]:
            codes: List[str] = []
            if isinstance(tar, dict):
                for a in ensure_list(tar.get("Area")):
                    code = (a or {}).get("Code") or ""
                    if code:
                        codes.append(code)
            elif isinstance(tar, list):
                for t1 in tar:
                    codes.extend(gather_codes_from_target(t1))
            return codes
        for c in gather_codes_from_target(target):
            if c and c not in result:
                result.append(c)

    return result


def extract_area_code_from_entry_id(entry: Dict[str, Any]) -> Optional[str]:
    """
    entry['id'] のURL末尾から VPWW53 の 6桁地域コードを抽出して返す。
    例: ..._VPWW53_170000.xml, ..._VPWW53_170000.xml?CACHE=1800 などに対応。
    """
    try:
        eid = entry.get("id", "") or ""
    except Exception:
        eid = ""

    # パターン1: もっとも厳密（末尾の .xml の直前に 6桁コード、クエリ文字列許容）
    m = re.search(r"_VPWW53_([0-9]{6})\.xml(?:\?.*)?$", eid, flags=re.IGNORECASE)
    if m:
        return m.group(1)

    # パターン2: ファイル名部分を '_' 区切りで解析し、VPWW53 の直後トークンを 6桁に正規化して取得
    fname = eid.rsplit("/", 1)[-1]
    parts = fname.split("_")
    for idx, part in enumerate(parts):
        if part.upper() == "VPWW53" and idx + 1 < len(parts):
            nxt = parts[idx + 1]
            # 末尾の .xml やクエリを除去
            nxt = re.sub(r"\.xml(?:\?.*)?$", "", nxt, flags=re.IGNORECASE)
            if re.fullmatch(r"\d{6}", nxt):
                return nxt

    # パターン3: 最後に .xml の直前にある 6桁を緩く拾う（誤検出を避けるため最後の .xml に限定）
    m2 = re.search(r"([0-9]{6})\.xml(?:\?.*)?$", eid, flags=re.IGNORECASE)
    if m2:
        return m2.group(1)

    return None


def parse_entry_datetime_local(entry: Dict[str, Any]) -> Optional[datetime]:
    """
    entry['id'] のURLに含まれる14桁時刻(YYYYMMDDhhmmss)をローカル時刻として datetime に変換。
    例: https://.../20251026063220_0_VPWW53_270000.xml → 2025-10-26 06:32:20
    取得できない場合は None。
    """
    try:
        eid = entry.get("id", "") or ""
    except Exception:
        eid = ""
    m = re.search(r"/(\d{14})_", eid)
    if not m:
        return None
    try:
        # 気象庁IDの14桁はJST相当で発番される前提。ここではローカル時刻(naive)として扱う。
        return datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
    except Exception:
        return None


def filter_entries_yesterday_to_now(entries: Iterable[Dict[str, Any]], now_dt: Optional[datetime] = None) -> List[Dict[str, Any]]:
    """
    昨日00:00:00 から 現在時刻(now) までに作成されたエントリのみ残す。
    """
    if now_dt is None:
        now_dt = datetime.now()
    start_dt = datetime.combine(now_dt.date() - timedelta(days=1), dtime(0, 0, 0))
    filtered: List[Dict[str, Any]] = []
    for e in entries:
        ts = parse_entry_datetime_local(e)
        if ts is None:
            continue
        if start_dt <= ts <= now_dt:
            filtered.append(e)
    return filtered


# ========== 汎用フェッチ(fetch) サブコマンド ==========

def cmd_fetch(args: argparse.Namespace) -> int:
    """
    任意の電文を Atom フィードから取得して JSON 出力。
    - --list 指定: 条件に合う entry の概要(タイトル/著者/ID)を一覧表示(ダウンロードなし)
    - --json-out 指定: ダウンロードしたXML辞書をJSON保存
    - いずれも未指定なら dict を標準出力(JSON)へ出力
    """
    feed = load_feed(args.feed)
    entries = list(
        filter_entries(
            iter_feed_entries(feed),
            product_code=args.product,
            author=args.author,
            title_contains=args.title_contains,
        )
    )

    if args.list:
        # 取得範囲: 「前日(00:00:00) 〜 現在」だけに限定
        now_dt = datetime.now()
        start_dt = datetime.combine(now_dt.date() - timedelta(days=1), dtime(0, 0, 0))
        entries_in_range = filter_entries_yesterday_to_now(entries, now_dt=now_dt)

        print(f"# Entries for product={args.product or '*'} author={args.author or '*'} title~={args.title_contains or '*'}")
        print(f"# 現在日時: {now_dt:%Y-%m-%d %H:%M:%S} (ローカル)")
        print(f"# 取得範囲: {start_dt:%Y-%m-%d %H:%M:%S} 〜 {now_dt:%Y-%m-%d %H:%M:%S} (ローカル)")

        for i, e in enumerate(entries_in_range):
            region_code = ""
            if (args.product or "").upper() == "VPWW53":
                # まず entry の id(URL) 末尾から地域コードを抽出（高速・確実）
                code_from_id = extract_area_code_from_entry_id(e)
                if code_from_id:
                    region_code = code_from_id
                else:
                    # フォールバックとして XML を取得し、府県予報区/都道府県 等から抽出
                    try:
                        rep = fetch_entry_dict(e)
                        codes = extract_prefecture_codes_from_vpww53(rep)
                        region_code = ",".join(codes) if codes else "-"
                    except Exception:
                        region_code = "?"
            print(
                f"- [{i}] code={region_code} | title={e.get('title','')} | author={(e.get('author') or {}).get('name','')} | id={e.get('id','')}"
            )
        print(f"# 処理完了: {len(entries_in_range)}件")
        return 0

    if not entries:
        print("該当エントリが見つかりませんでした。条件(電文コード/author/title)を見直してください。", file=sys.stderr)
        return 2

    # 先頭(最新)を採用
    entry = entries[0]
    data = fetch_entry_dict(entry)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"JSON を保存しました: {args.json_out}")
        print("# 処理完了")
        return 0

    print(json.dumps(data, ensure_ascii=False, indent=2))
    print("# 処理完了")
    return 0


# ========== warnings サブコマンド(VPWW53専用) ==========

def cmd_warnings(args: argparse.Namespace) -> int:
    """
    VPWW53 の警報・注意報を、市町村名(--city) あるいは 地域コード(--area-code) で抽出・表示。
    どちらも未指定なら、対象都道府県の全市町村を一覧表示。
    """
    feed = load_feed(args.feed)

    entry = find_latest_vpww53_entry_for_prefecture(
        feed_dict=feed,
        prefecture=args.prefecture,
        author=args.author,
        max_probe=args.max_probe,
    )
    if not entry:
        print("VPWW53 の該当エントリが見つかりませんでした。--prefecture または --author を調整してください。", file=sys.stderr)
        return 2

    report = fetch_entry_dict(entry)
    head_title = get_head_title(report)

    results = extract_city_warnings_from_vpww53(
        report,
        city_name=args.city,
        area_code=args.area_code,
    )

    # 出力
    print(f"# 電文: {entry.get('title','')} | author={(entry.get('author') or {}).get('name','')} | id={entry.get('id','')}")
    if head_title:
        print(f"# ヘッドタイトル: {head_title}")

    # JSON保存オプション
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"結果JSON を保存しました: {args.json_out}")
        print("# 処理完了")
        return 0

    # 人間可読表示
    if args.city or args.area_code:
        if not results:
            print("該当市町村の発表警報・注意報はなし もしくは 見つかりませんでした。")
            print("# 処理完了")
            return 0
        for r in results:
            aname = r["area_name"]
            acode = r["area_code"]
            warnings_list = r["warnings"]
            if warnings_list:
                print(f"[{aname} ({acode})] 発表中: " + " / ".join(warnings_list))
            else:
                print(f"[{aname} ({acode})] 発表警報・注意報はなし")
        print("# 処理完了")
        return 0

    # 全市町村一覧
    for r in results:
        aname = r["area_name"]
        acode = r["area_code"]
        w = r["warnings"]
        if w:
            print(f"[{aname} ({acode})] 発表中: " + " / ".join(w))
        else:
            print(f"[{aname} ({acode})] 発表警報・注意報はなし")

    print(f"# 処理完了: {len(results)}件")
    return 0


# ========== list-areas サブコマンド(VPWW53専用) ==========

def cmd_list_areas(args: argparse.Namespace) -> int:
    """
    指定都道府県(または author)の最新VPWW53電文から、「市町村等」エリア一覧を表示。
    """
    feed = load_feed(args.feed)
    entry = find_latest_vpww53_entry_for_prefecture(
        feed_dict=feed, prefecture=args.prefecture, author=args.author, max_probe=args.max_probe
    )
    if not entry:
        print("VPWW53 の該当エントリが見つかりませんでした。--prefecture または --author を調整してください。", file=sys.stderr)
        return 2

    report = fetch_entry_dict(entry)
    areas = list_vpww53_areas(report)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(areas, f, ensure_ascii=False, indent=2)
        print(f"エリア一覧を保存しました: {args.json_out}")
        print("# 処理完了")
        return 0

    print(f"# 電文: {entry.get('title','')} | author={(entry.get('author') or {}).get('name','')} | id={entry.get('id','')}")
    for a in areas:
        print(f"- {a['area_name']} ({a['area_code']})")
    print(f"# 処理完了: {len(areas)}件")
    return 0


# ========== CLI 引数 ==========

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="JMAXML 汎用フェッチャ: 警報・注意報など日本全国のデータ取得",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # 共通オプション
    def add_common_feed_opts(sp: argparse.ArgumentParser):
        sp.add_argument(
            "--feed",
            default="extra_l",
            help="フィード指定。'extra_l'|'extra' または フィードURL(既定: extra_l)",
        )
        sp.add_argument(
            "--max-probe",
            type=int,
            default=60,
            help="都道府県で Head.Title を照合するときに取得して試す最大件数(既定: 60)",
        )

    # fetch: 任意電文を取得
    sp_fetch = sub.add_parser("fetch", help="任意の電文を取得(JSON表示/保存)")
    add_common_feed_opts(sp_fetch)
    sp_fetch.add_argument("--product", help="電文コード(例: VPWW53)。未指定なら全件対象(一覧向け)", default=None)
    sp_fetch.add_argument("--author", help="author(気象台名など)完全一致で絞り込み", default=None)
    sp_fetch.add_argument("--title-contains", help="title に含まれる文字列で絞り込み", default=None)
    sp_fetch.add_argument("--list", action="store_true", help="ダウンロードせず該当エントリの概要を一覧表示")
    sp_fetch.add_argument("--json-out", help="取得したXML辞書をJSON保存", default=None)
    sp_fetch.set_defaults(func=cmd_fetch)

    # warnings: VPWW53 専用(市町村の警報・注意報抽出)
    sp_warn = sub.add_parser("warnings", help="VPWW53(警報・注意報)から市町村の発表状況を抽出")
    add_common_feed_opts(sp_warn)
    sp_warn.add_argument("--prefecture", help="都道府県名(例: 石川県)。authorが未指定のときの探索条件", default=None)
    sp_warn.add_argument("--author", help="author(気象台名)を直接指定(例: 金沢地方気象台)", default=None)
    sp_warn.add_argument("--city", help="市区町村名でフィルタ(例: 金沢市)", default=None)
    sp_warn.add_argument("--area-code", help="市区町村コードでフィルタ(例: 1720100)", default=None)
    sp_warn.add_argument("--json-out", help="抽出結果をJSON保存", default=None)
    sp_warn.set_defaults(func=cmd_warnings)

    # list-areas: VPWW53 の市町村一覧
    sp_list = sub.add_parser("list-areas", help="VPWW53の『市町村等』に現れる市区町村名・コードを一覧")
    add_common_feed_opts(sp_list)
    sp_list.add_argument("--prefecture", help="都道府県名(例: 石川県)。authorが未指定のときの探索条件", default=None)
    sp_list.add_argument("--author", help="author(気象台名)を直接指定(例: 金沢地方気象台)", default=None)
    sp_list.add_argument("--json-out", help="一覧をJSON保存", default=None)
    sp_list.set_defaults(func=cmd_list_areas)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except requests.HTTPError as he:
        print(f"HTTPエラー: {he}", file=sys.stderr)
        return 3
    except Exception as e:
        print(f"エラー: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
