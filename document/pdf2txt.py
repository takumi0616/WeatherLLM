#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF to Text Converter
PDFファイルからテキストを抽出してテキストファイルに変換します
"""

import glob
import os
from PyPDF2 import PdfReader


def extract_text_from_pdf_with_pypdf2(pdf_path):
    """PDFからテキストを抽出する"""
    text = ''
    with open(pdf_path, 'rb') as file:
        pdf = PdfReader(file)
        for i in range(len(pdf.pages)):
            page = pdf.pages[i]
            text += page.extract_text()
    return text


def text_conv(txt, ym):
    """テキストを変換する"""
    txt_output = ym
    for i, t in enumerate(txt.split('\n')):
        if i == 0:
            txt_output += t.split('日')[0].zfill(2) + ','
            s = t.split(')')
            txt_output += s[0] + '),' + s[1]
            txt_output += ','
        else:
            txt_output += t
    return txt_output


def text_write(pdf_path, txt, output_dir):
    """テキストをテキストファイルに書き込む"""
    # ファイル名から年月を取得
    filename = os.path.basename(pdf_path)
    ym = '20' + filename.replace('.pdf', '')
    
    # 出力ファイルパスを生成
    output_filename = filename.replace('.pdf', '.txt')
    output_path = os.path.join(output_dir, output_filename)
    
    f = open(output_path, 'w')
    txts = []
    i = txt.find('日(')
    start = 0
    
    while i > 0:
        for j in range(i - 1, 0, -1):
            if txt[j].isdigit() == False:
                end = j + 1
                break
        if start != 0:
            f.write(text_conv(txt[start:end], ym) + '\n')
        start = end
        i += 1
        i = txt.find('日(', i)
    
    f.write(text_conv(txt[start:], ym))
    f.close()
    print(f"Saved: {output_path}")


def main():
    """メイン処理"""
    # 入力と出力のディレクトリを設定
    input_dir = '/Users/takumi0616/Develop/docker_miniconda/src/WeatherLLM/document/pdf'
    output_dir = '/Users/takumi0616/Develop/docker_miniconda/src/WeatherLLM/document/pdf-to-text'
    
    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # PDFファイルを取得
    pdf_pattern = os.path.join(input_dir, '*.pdf')
    files = glob.glob(pdf_pattern)
    files.sort()
    
    if not files:
        print(f"No PDF files found in {input_dir}")
        return
    
    print(f"Found {len(files)} PDF file(s)")
    
    # 各PDFファイルを処理
    for pdf_path in files:
        print(f"Processing: {pdf_path}")
        try:
            txt = extract_text_from_pdf_with_pypdf2(pdf_path)
            # 全角括弧を半角に変換
            txt = txt.replace('（', '(').replace('）', ')')
            text_write(pdf_path, txt, output_dir)
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
    
    print("Processing complete!")


if __name__ == '__main__':
    main()
