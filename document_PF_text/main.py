#!/home/devel/miniconda3/envs/weather310/bin/python

import gzip
import json
import math
import os
import sys
from datetime import datetime, timedelta
from decimal import Decimal

sys.path.append(os.path.dirname(__file__))

from get import get_dict


def replace_nan(obj):
    if isinstance(obj, float):
        if math.isnan(obj):
            return None
        else:
            return obj
    elif isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: replace_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_nan(v) for v in obj]
    else:
        return obj


def adjust_time(date_part, time_part):
    valid_times = [2, 8, 14, 20]
    hour = int(time_part)
    adjusted_hour = None

    for t in sorted(valid_times):
        if t <= hour:
            adjusted_hour = t
        else:
            break

    if adjusted_hour is None:
        # Need to go back to previous day
        date_obj = datetime.strptime(date_part, "%Y-%m-%d") - timedelta(days=1)
        date_part = date_obj.strftime("%Y-%m-%d")
        adjusted_hour = 20
    else:
        date_obj = datetime.strptime(date_part, "%Y-%m-%d")

    adjusted_time_part = f"{adjusted_hour:02d}"

    return date_part, adjusted_time_part


def main():
    # 対象となる地点のリストを定義
    points = [
        ["A11016", "稚内"],
        ["A14163", "札幌"],
        ["A12442", "旭川"],
        ["A19432", "釧路"],
        ["A23232", "函館"],
        ["A31312", "青森"],
        ["A33431", "盛岡"],
        ["A34392", "仙台"],
        ["A43056", "熊谷"],
        ["A44132", "東京"],
        ["A45212", "千葉"],
        ["A46106", "横浜"],
        ["A48156", "長野"],
        ["A54232", "新潟"],
        ["A56227", "金沢"],
        ["A51106", "名古屋"],
        ["A62078", "大阪"],
        ["A67437", "広島"],
        ["A68132", "松江"],
        ["A74182", "高知"],
        ["A82182", "福岡"],
        ["A88317", "鹿児島"],
        ["A88836", "名瀬"],
        ["A91197", "那覇"],
        ["A94081", "石垣島"],
    ]

    # ann_time 引数を解析
    if len(sys.argv) < 2:
        # print("Usage: python main.py <ann_time>")
        # print("Example: python main.py 2024-10-11_15 or python main.py latest")
        sys.exit(1)

    ann_arg = sys.argv[1]

    # ルートパスを定義
    root_path_json = "/home/devel/mnt/optn/archives/PF_text/v3/json"
    root_path_json_gz = "/home/devel/mnt/optn/archives/PF_text/v3/"

    for point in points:
        point_number = point[0]
        point_name = point[1]

        # データを取得
        try:
            out = get_dict(point_number, ann_arg)

            # 出力の 'info' 内の 'point' が期待通りか確認
            if "info" in out and out["info"].get("point") != point_number:
                # print(f"Error: Point number mismatch for {point_number}")
                continue

            if ann_arg == "latest":
                ann_time = out["info"]["ann_time"]
            else:
                ann_time = ann_arg

            # ann_time を処理し、余分なテキスト（例："時発表"）を削除
            if ann_time is not None:
                ann_time = ann_time.replace("時発表", "").strip()
            else:
                # print(f"Error: ann_time is None for point {point_number}")
                continue

            # ann_time を分割して日付と時間を取得
            date_time_parts = ann_time.split("_")
            if len(date_time_parts) != 2:
                # print(f"Error: Invalid ann_time format for point {point_number}")
                continue

            date_part = date_time_parts[0]  # 'YYYY-MM-DD'
            time_part = date_time_parts[1]  # 'HH'

            # 時刻を調整
            date_part, time_part = adjust_time(date_part, time_part)

            # 調整後の日付を再度分割
            date_components = date_part.split("-")
            if len(date_components) != 3:
                print(
                    f"Error: Invalid date format in ann_time for point {point_number}"
                )
                continue

            year, month, day = date_components  # ['YYYY', 'MM', 'DD']

            # 調整後の ann_time を再構築
            adjusted_ann_time = f"{date_part}_{time_part}"

            # フォルダパスを構築
            folder_path_json = os.path.join(
                root_path_json, f"{year}-{month}", f"{day}_{time_part}"
            )
            folder_path_json_gz = os.path.join(
                root_path_json_gz, f"{year}-{month}", f"{day}_{time_part}"
            )

            # フォルダが存在しない場合は作成
            os.makedirs(folder_path_json, exist_ok=True)
            os.makedirs(folder_path_json_gz, exist_ok=True)

            # ファイル名を構築
            filename = f"PF-text_{adjusted_ann_time}_{point_number}.json"
            filename_gz = f"{filename}.gz"

            full_path_json = os.path.join(folder_path_json, filename)
            full_path_json_gz = os.path.join(folder_path_json_gz, filename_gz)

            # NaN 値と Decimal を置換
            out_serializable = replace_nan(out)

            # JSON ファイルを保存
            with open(full_path_json, "w", encoding="utf-8") as f_json:
                json.dump(out_serializable, f_json, ensure_ascii=False)

            # JSON.GZ ファイルを保存
            with gzip.open(full_path_json_gz, "wt", encoding="utf-8") as f_gz:
                json.dump(out_serializable, f_gz, ensure_ascii=False)

            print(f"Saved JSON data for point {point_number} at {full_path_json}")
            print(f"Saved JSON.GZ data for point {point_number} at {full_path_json_gz}")

        except Exception as e:
            print(f"Error processing point {point_number}: {e}")


if __name__ == "__main__":
    main()
