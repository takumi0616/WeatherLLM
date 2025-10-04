# -*- coding: utf-8 -*-
"""
日々の天気図PDFダウンロードスクリプト（改善版）

https://www.data.jma.go.jp/yoho/hibiten/index.html からのPDF一括取得
"""

import requests
import os
import time
from pathlib import Path
from typing import List
import argparse
from datetime import datetime


def create_pdf_urls(start_year: int, end_year: int) -> List[str]:
    """指定された年範囲のPDF URLリストを生成
    
    Args:
        start_year: 開始年
        end_year: 終了年（この年を含む）
    
    Returns:
        PDF URLのリスト
    """
    root_url = 'https://www.data.jma.go.jp/fcd/yoho/data/hibiten/'
    pdf_urls = []
    
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            year_str = str(year)
            month_str = str(month).zfill(2)
            # 例: https://www.data.jma.go.jp/fcd/yoho/data/hibiten/2022/2201.pdf
            url = f"{root_url}{year_str}/{year_str[-2:]}{month_str}.pdf"
            pdf_urls.append(url)
    
    return pdf_urls


def download_pdf(url: str, output_dir: Path, max_retries: int = 3, skip_existing: bool = True) -> bool:
    """PDFファイルをダウンロード
    
    Args:
        url: ダウンロードするPDFのURL
        output_dir: 保存先ディレクトリ
        max_retries: 最大リトライ回数
        skip_existing: 既存ファイルをスキップするか
    
    Returns:
        ダウンロード成功時True、それ以外False
    """
    filename = output_dir / url.split("/")[-1]
    
    # 既存ファイルのスキップ
    if skip_existing and filename.exists():
        print(f"[スキップ] {filename.name} は既に存在します")
        return True
    
    # リトライ機能付きダウンロード
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                with open(filename, 'wb') as f:
                    f.write(response.content)
                print(f"[成功] {filename.name} をダウンロードしました")
                return True
            elif response.status_code == 404:
                print(f"[404] {filename.name} は存在しません")
                return False
            else:
                print(f"[エラー] {filename.name}: ステータスコード {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 指数バックオフ
                print(f"[リトライ] {filename.name}: {e} - {wait_time}秒後に再試行...")
                time.sleep(wait_time)
            else:
                print(f"[失敗] {filename.name}: {e}")
                return False
    
    return False


def main():
    """メイン処理"""
    # コマンドライン引数のパース
    parser = argparse.ArgumentParser(
        description='気象庁の日々の天気図PDFをダウンロード'
    )
    parser.add_argument(
        '--start-year',
        type=int,
        default=2002,
        help='開始年（デフォルト: 2002）'
    )
    parser.add_argument(
        '--end-year',
        type=int,
        default=datetime.now().year - 1,
        help='終了年（デフォルト: 昨年）'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./pdf',
        help='保存先ディレクトリ（デフォルト: ./pdf）'
    )
    parser.add_argument(
        '--no-skip-existing',
        action='store_true',
        help='既存ファイルを上書きする'
    )
    parser.add_argument(
        '--max-retries',
        type=int,
        default=3,
        help='最大リトライ回数（デフォルト: 3）'
    )
    
    args = parser.parse_args()
    
    # 出力ディレクトリの作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"出力ディレクトリ: {output_dir.absolute()}")
    
    # URLリストの生成
    pdf_urls = create_pdf_urls(args.start_year, args.end_year)
    total_files = len(pdf_urls)
    print(f"\n対象ファイル数: {total_files}件 ({args.start_year}年～{args.end_year}年)")
    print("-" * 60)
    
    # ダウンロード実行
    success_count = 0
    failed_count = 0
    skipped_count = 0
    
    for i, url in enumerate(pdf_urls, 1):
        print(f"\n[{i}/{total_files}]", end=" ")
        
        if download_pdf(
            url,
            output_dir,
            max_retries=args.max_retries,
            skip_existing=not args.no_skip_existing
        ):
            if not args.no_skip_existing and (output_dir / url.split("/")[-1]).exists():
                # ファイルが既に存在していた場合
                if "[スキップ]" in str(output_dir / url.split("/")[-1]):
                    skipped_count += 1
            success_count += 1
        else:
            failed_count += 1
        
        # サーバーへの負荷軽減のため少し待機
        if i < total_files:
            time.sleep(0.5)
    
    # 結果サマリー
    print("\n" + "=" * 60)
    print("ダウンロード完了")
    print(f"成功: {success_count}件")
    print(f"失敗: {failed_count}件")
    print("=" * 60)


if __name__ == "__main__":
    main()
