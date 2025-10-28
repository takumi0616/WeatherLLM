#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kanku_chihou_56.json の各オブジェクトに対し、place_name と
CSV(2025-10-27_0000_gust.csv) の「地点」を突き合わせて
「観測所番号」を observatory_number に補完するユーティリティ。

- 既に observatory_number がある要素は上書きしません（尊重）。
- 一致ロジック:
  1) CSVの「地点」の漢字部分(最初の全角/半角カッコの前)と JSONのplace_name の厳密一致
  2) 括弧内情報をすべて削ったCSV地点文字列に対する厳密一致
  3) それでも無い場合、括弧を除去したCSV地点文字列に place_name が部分一致する最初の候補を採用
- JSON はバックアップ(.bak)作成後に上書きします。
"""

import csv
import json
import os
import re
import sys
from typing import Dict, List, Tuple

# パス（必要に応じて引数化可能）
CSV_PATH = "./csv/today_0000/2025-10-27_0000_gust.csv"
JSON_PATH = "./kanku_chihou_56.json"
BACKUP_PATH = JSON_PATH + ".bak"


def strip_brackets(s: str) -> str:
    """全角/半角の括弧とその中身をすべて除去する（ネストや複数括弧にも対応）"""
    if s is None:
        return ""
    s2 = s
    # ネスト対応: 除去できなくなるまで繰り返す（全角）
    prev = None
    while prev != s2:
        prev = s2
        s2 = re.sub(r"（[^（）]*）", "", s2)
    # ネスト対応: 除去できなくなるまで繰り返す（半角）
    prev = None
    while prev != s2:
        prev = s2
        s2 = re.sub(r"\([^()]*\)", "", s2)
    # 残存する片側括弧文字の掃除
    s2 = s2.replace("（", "").replace("）", "").replace("(", "").replace(")", "")
    return s2.strip()


def base_kanji(s: str) -> str:
    """最初に現れる全角/半角カッコの直前までを抽出（漢字ベース）"""
    if s is None:
        return ""
    # 最初に出るカッコの位置を取得（全角/半角の小さい方）
    idx1 = s.find("（")
    idx2 = s.find("(")
    idxs = [i for i in [idx1, idx2] if i >= 0]
    if idxs:
        cut = min(idxs)
        return s[:cut].strip()
    return s.strip()


def load_csv_mapping(csv_path: str) -> Tuple[Dict[str, str], List[Tuple[str, str]]]:
    """
    CSVから以下のマップを作成:
      - key: base_kanji(地点), value: 観測所番号
      - 併せて検索用に (観測所番号, 括弧除去済み地点文字列) のリストも返す
    """
    mapping: Dict[str, str] = {}
    search_space: List[Tuple[str, str]] = []

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)

        try:
            idx_code = header.index("観測所番号")
            idx_place = header.index("地点")
        except ValueError as e:
            raise RuntimeError(
                "CSVヘッダに『観測所番号』または『地点』が見つかりません。"
            ) from e

        for row in reader:
            if not row or len(row) <= max(idx_code, idx_place):
                continue
            code = str(row[idx_code]).strip()
            place = str(row[idx_place]).strip()
            if not code or not place:
                continue

            base = base_kanji(place)
            no_brackets = strip_brackets(place)

            # 厳密一致用に base を優先キーとして登録（既存は保持）
            mapping.setdefault(base, code)
            # 検索用に括弧除去済み文字列も併用
            mapping.setdefault(no_brackets, code)

            # 部分一致探索用に追加
            search_space.append((code, no_brackets))

    return mapping, search_space


def load_json(json_path: str) -> List[dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(json_path: str, data: List[dict]) -> None:
    # ensure_ascii=False で日本語をそのまま出力、インデントは2に統一
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def backup(src: str, dst: str) -> None:
    # 既にバックアップがある場合でも上書きして最新状態にする
    with open(src, "r", encoding="utf-8") as rf, open(dst, "w", encoding="utf-8") as wf:
        wf.write(rf.read())


def resolve_code_for_place(place_name: str, mapping: Dict[str, str], search_space: List[Tuple[str, str]]) -> str:
    """
    place_name に対する観測所番号を決定する。
    1) mapping の厳密一致 (place_name)
    2) mapping の厳密一致 (base_kanji(place_name)) - 念のため
    3) search_space の括弧除去済みCSV地点に部分一致
    見つからなければ空文字を返す。
    """
    # まずはそのまま一致（JSONのplace_nameは括弧を含まない想定）
    if place_name in mapping:
        return mapping[place_name]

    # 念のため、place_name 側に括弧が含まれていた場合の対策
    place_base = base_kanji(place_name)
    if place_base in mapping:
        return mapping[place_base]

    # 部分一致: JSONとCSVのいずれかが他方を含む場合を許容
    # 例: CSV「南大東」 vs JSON「南大東島」など表記ゆれを吸収
    candidates = []
    for code, csv_no_br in search_space:
        if place_name and (place_name in csv_no_br or csv_no_br in place_name):
            candidates.append((code, csv_no_br))

    if candidates:
        # ヒューリスティック: 一番短い一致文字列(=より固有)を優先
        candidates.sort(key=lambda x: len(x[1]))
        return candidates[0][0]

    return ""


def main():
    # パス存在チェック
    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] CSV not found: {CSV_PATH}")
        sys.exit(1)
    if not os.path.exists(JSON_PATH):
        print(f"[ERROR] JSON not found: {JSON_PATH}")
        sys.exit(1)

    mapping, search_space = load_csv_mapping(CSV_PATH)
    data = load_json(JSON_PATH)

    updated = 0
    skipped_existing = 0
    unresolved: List[str] = []

    for obj in data:
        place_name = str(obj.get("place_name", "")).strip()
        # すでに値があれば尊重
        if "observatory_number" in obj and str(obj["observatory_number"]).strip():
            skipped_existing += 1
            continue

        code = resolve_code_for_place(place_name, mapping, search_space)
        if code:
            obj["observatory_number"] = str(code)
            updated += 1
        else:
            unresolved.append(place_name)

    # 変更があればバックアップを取って保存
    if updated > 0:
        backup(JSON_PATH, BACKUP_PATH)
        save_json(JSON_PATH, data)

    print("=== 完了サマリ ===")
    print(f"既存を尊重して未変更: {skipped_existing} 件")
    print(f"新規に設定            : {updated} 件")
    print(f"未解決                : {len(unresolved)} 件")
    if unresolved:
        print("未解決の place_name 一覧:")
        for name in unresolved:
            print(f" - {name}")


if __name__ == "__main__":
    main()
