#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monthly Text to Daily Text Converter
月ごとのテキストファイルを日ごとのテキストファイルに分割します
"""

import os
import glob
import argparse


def split_monthly_to_daily(input_file, output_dir, full_text_lines):
    """
    月ごとのテキストファイルを日ごとのファイルに分割する
    
    Args:
        input_file: 入力ファイルパス（例：pdf-to-text/0201.txt）
        output_dir: 出力ディレクトリパス
        full_text_lines: 全テキストを格納するリスト（参照渡し）
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return
    
    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.basename(input_file)
    print(f"Processing: {filename}")
    
    processed_count = 0
    
    # ファイルを読み込んで各行を処理
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # 行の最初の8文字が日付（YYYYMMDD形式）
            if len(line) >= 8 and line[:8].isdigit():
                date_str = line[:8]
                
                # 日付をファイル名として使用（例：20020301.txt）
                output_filename = f"{date_str}.txt"
                output_path = os.path.join(output_dir, output_filename)
                
                # 該当日のデータを書き込む
                with open(output_path, 'w', encoding='utf-8') as out_f:
                    out_f.write(line)
                
                # 全テキスト用のリストに追加
                full_text_lines.append(line)
                
                processed_count += 1
            else:
                print(f"Warning: Skipping invalid line format: {line[:50]}...")
    
    print(f"  Created {processed_count} daily files")


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description="Monthly Text to Daily Text Converter")
    parser.add_argument('--input-dir', default='./pdf-to-text', 
                       help='入力テキストファイルのディレクトリ（月ごとのファイルがある場所）')
    parser.add_argument('--output-dir', default='./daily-text', 
                       help='出力ディレクトリ（日ごとのファイルを保存する場所）')
    parser.add_argument('--pattern', default='*.txt', 
                       help='入力ディレクトリ内のglobパターン（例：02*.txt）')
    parser.add_argument('--files', nargs='*', 
                       help='処理するテキストファイルの明示指定')
    args = parser.parse_args()
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    
    # 入力ファイルを取得
    if args.files:
        files = []
        for f in args.files:
            p = f if os.path.isabs(f) else os.path.join(input_dir, f)
            files.append(p)
    else:
        pattern = os.path.join(input_dir, args.pattern)
        files = glob.glob(pattern)
        files.sort()
    
    if not files:
        print(f"No text files found (input_dir={input_dir}, pattern={args.pattern})")
        return
    
    print(f"Found {len(files)} text file(s)")
    print(f"Output directory: {output_dir}")
    
    # 全テキストを格納するリスト
    full_text_lines = []
    
    # 各ファイルを処理
    total_daily_files = 0
    for input_file in files:
        try:
            count_before = len(glob.glob(os.path.join(output_dir, '*.txt')))
            split_monthly_to_daily(input_file, output_dir, full_text_lines)
            count_after = len(glob.glob(os.path.join(output_dir, '*.txt')))
            total_daily_files += (count_after - count_before)
        except Exception as e:
            print(f"Error processing {input_file}: {e}")
    
    # 全テキストを1つのファイルに結合して保存
    if full_text_lines:
        full_text_path = os.path.join(os.path.dirname(output_dir), 'full_text.txt')
        with open(full_text_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(full_text_lines))
        print(f"\nCreated combined file: {full_text_path}")
        print(f"Total lines in combined file: {len(full_text_lines)}")
    
    print(f"\nProcessing complete!")
    print(f"Total daily files created: {total_daily_files}")


if __name__ == '__main__':
    main()
