#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF to Text Converter
PDFファイルからテキストを抽出してテキストファイルに変換します
"""

import glob
import os
import argparse
import unicodedata
from PyPDF2 import PdfReader


def extract_text_from_pdf_with_pypdf2(pdf_path):
    """PDFからテキストを抽出する (フォールバック付き: PyMuPDF -> pdfminer.six -> PyPDF2)"""
    # 1) PyMuPDF (高速・日本語対応が良いことが多い)
    try:
        import fitz  # PyMuPDF
        text_parts = []
        with fitz.open(pdf_path) as doc:
            for i, page in enumerate(doc):
                try:
                    t = page.get_text("text") or ""
                    text_parts.append(t)
                except Exception as e:
                    print(f"  [warn] PyMuPDF skip page {i+1}: {e}")
                    continue
        joined = "".join(text_parts)
        if joined.strip():
            print("  [info] used extractor: PyMuPDF")
            return joined
    except Exception as e:
        print(f"  [info] PyMuPDF unavailable or failed: {e}")

    # 2) pdfminer.six（CJKに比較的強い）
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract_text
        t = pdfminer_extract_text(pdf_path) or ""
        if t.strip():
            print("  [info] used extractor: pdfminer.six")
            return t
    except Exception as e:
        print(f"  [info] pdfminer.six unavailable or failed: {e}")

    # 3) 最後に PyPDF2 での抽出（既存の堅牢化ロジック）
    text = ''
    with open(pdf_path, 'rb') as file:
        pdf = PdfReader(file)
        for i in range(len(pdf.pages)):
            try:
                page = pdf.pages[i]
                t = page.extract_text()
                if t:
                    text += t
            except Exception as e:
                print(f"  [warn] PyPDF2 skip page {i+1}: {e}")
                continue
    print("  [info] used extractor: PyPDF2 (fallback)")
    return text


def text_conv(txt, ym):
    """テキストを変換する（パターンが崩れていても落ちないように防御的に処理）"""
    txt_output = ym
    for i, t in enumerate(txt.split('\n')):
        if i == 0:
            # 例: "12日(月)..." を想定。存在しない場合は保守的に処理。
            day_part, sep_day, rest_after_day = t.partition('日')

            # 日付（数字部分）
            day_digits = ''
            if sep_day:
                # day_part の末尾の連続数字を抽出
                for ch in reversed(day_part):
                    if ch.isdigit():
                        day_digits = ch + day_digits
                    else:
                        break
            txt_output += (day_digits.zfill(2) if day_digits else '00') + ','

            # 括弧と残り
            before_paren, sep_paren, after_paren = rest_after_day.partition(')')
            if sep_paren:
                txt_output += before_paren + '),' + after_paren
            else:
                # 括弧が見つからない場合は元の文字列をそのまま
                txt_output += t
            txt_output += ','
        else:
            txt_output += t
    return txt_output


def convert_fullwidth_to_halfwidth(text):
    """全角文字を半角文字に変換する
    
    全角数字、全角英字、全角記号、全角スペースを半角に変換します。
    日本語（ひらがな、カタカナ、漢字）は変換しません。
    """
    if not text:
        return text
    
    # NFKCで正規化することで、全角英数字を半角に変換
    # ただし、カタカナも半角になってしまうため、選択的に処理
    result = []
    for char in text:
        # 全角数字（０-９）→ 半角数字（0-9）
        if '\uff10' <= char <= '\uff19':  # ０-９
            result.append(chr(ord(char) - 0xfee0))
        # 全角英大文字（Ａ-Ｚ）→ 半角英大文字（A-Z）
        elif '\uff21' <= char <= '\uff3a':  # Ａ-Ｚ
            result.append(chr(ord(char) - 0xfee0))
        # 全角英小文字（ａ-ｚ）→ 半角英小文字（a-z）
        elif '\uff41' <= char <= '\uff5a':  # ａ-ｚ
            result.append(chr(ord(char) - 0xfee0))
        # 全角スペース → 半角スペース
        elif char == '\u3000':
            result.append(' ')
        # 全角記号の変換（主要なもの）
        elif char == '！':
            result.append('!')
        elif char == '"':
            result.append('"')
        elif char == '＃':
            result.append('#')
        elif char == '＄':
            result.append('$')
        elif char == '％':
            result.append('%')
        elif char == '＆':
            result.append('&')
        elif char == '\u2019' or char == '\uff07':  # 全角シングルクォート
            result.append("'")
        elif char == '（':
            result.append('(')
        elif char == '）':
            result.append(')')
        elif char == '＊':
            result.append('*')
        elif char == '＋':
            result.append('+')
        elif char == '，':
            result.append(',')
        elif char == '－':
            result.append('-')
        elif char == '．':
            result.append('.')
        elif char == '／':
            result.append('/')
        elif char == '：':
            result.append(':')
        elif char == '；':
            result.append(';')
        elif char == '＜':
            result.append('<')
        elif char == '＝':
            result.append('=')
        elif char == '＞':
            result.append('>')
        elif char == '？':
            result.append('?')
        elif char == '＠':
            result.append('@')
        elif char == '［':
            result.append('[')
        elif char == '＼':
            result.append('\\')
        elif char == '］':
            result.append(']')
        elif char == '＾':
            result.append('^')
        elif char == '＿':
            result.append('_')
        elif char == '｀':
            result.append('`')
        elif char == '｛':
            result.append('{')
        elif char == '｜':
            result.append('|')
        elif char == '｝':
            result.append('}')
        elif char == '～':
            result.append('~')
        else:
            # その他の文字はそのまま
            result.append(char)
    
    return ''.join(result)


def maybe_fix_mojibake(s):
    """PyPDF2等で発生する日本語の文字化けに対して簡易修復を試みる"""
    if not s:
        return s
    # すでに日本語（かな・カナ・CJK）や目標パターンが見えていれば何もしない
    if '日(' in s or '（' in s or '）' in s or any('\u3040' <= ch <= '\u30ff' or '\u4e00' <= ch <= '\u9fff' for ch in s):
        return s
    # C1制御領域(0x80-0x9F)など怪しい文字が多いときは再デコードを試す
    suspicious = sum(1 for ch in s if 0x80 <= ord(ch) <= 0x9F)
    if suspicious < 10:
        return s
    for enc in ('latin-1', 'cp1252'):
        try:
            fixed = s.encode(enc, errors='ignore').decode('cp932', errors='ignore')
            if fixed and ('日' in fixed or any('\u3040' <= ch <= '\u30ff' or '\u4e00' <= ch <= '\u9fff' for ch in fixed)):
                print(f"  [info] mojibake fixed via {enc}->cp932")
                return fixed
        except Exception:
            continue
    return s


def text_write(pdf_path, txt, output_dir):
    """テキストをテキストファイルに書き込む"""
    # ファイル名から年月を取得
    filename = os.path.basename(pdf_path)
    ym = '20' + filename.replace('.pdf', '')
    
    # 出力ファイルパスを生成
    output_filename = filename.replace('.pdf', '.txt')
    output_path = os.path.join(output_dir, output_filename)

    # 想定パターンが全く見つからない場合は、生テキストを書き出して終了（不正な"20020300,"等を避ける）
    if '日(' not in txt:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(txt)
        print(f"Saved (raw): {output_path}")
        return

    with open(output_path, 'w', encoding='utf-8') as f:
        i = txt.find('日(')
        start = 0

        while i > 0:
            # 「日(」の直前に連続する数字ブロックの開始位置を探す
            end = None
            for j in range(i - 1, -1, -1):
                if not txt[j].isdigit():
                    end = j + 1
                    break
            if end is None:
                end = 0

            if start != 0:
                f.write(text_conv(txt[start:end], ym) + '\n')

            start = end
            i = txt.find('日(', i + 1)

        # 最後のチャンク
        f.write(text_conv(txt[start:], ym))
    print(f"Saved: {output_path}")


def main():
    """メイン処理（引数対応）"""
    parser = argparse.ArgumentParser(description="PDF to Text Converter")
    parser.add_argument('--input-dir', default='./pdf', help='入力PDFディレクトリ')
    parser.add_argument('--output-dir', default='./pdf-to-text', help='出力テキストディレクトリ')
    parser.add_argument('--pattern', default='*.pdf', help='入力ディレクトリ内のglobパターン (例: 020[1-4].pdf)')
    parser.add_argument('--files', nargs='*', help='処理するPDFファイルの明示指定（絶対パスまたは input-dir からの相対/ベース名）')
    parser.add_argument('--limit', type=int, help='先頭から最大N件のみ処理')
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # PDFファイルを取得
    if args.files:
        files = []
        for f in args.files:
            p = f if os.path.isabs(f) else os.path.join(input_dir, f)
            files.append(p)
    else:
        pdf_pattern = os.path.join(input_dir, args.pattern)
        files = glob.glob(pdf_pattern)
        files.sort()

    if args.limit is not None:
        files = files[:args.limit]
    
    if not files:
        print(f"No PDF files found (input_dir={input_dir}, pattern={args.pattern}, files={args.files})")
        return
    
    print(f"Found {len(files)} PDF file(s)")
    
    # 各PDFファイルを処理
    for pdf_path in files:
        print(f"Processing: {pdf_path}")
        try:
            txt = extract_text_from_pdf_with_pypdf2(pdf_path)
            # 文字化けの簡易修復を試行
            txt = maybe_fix_mojibake(txt)
            # 全角文字を半角に変換（数字、英字、記号、スペース）
            txt = convert_fullwidth_to_halfwidth(txt)
            text_write(pdf_path, txt, output_dir)
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
    
    print("Processing complete!")


if __name__ == '__main__':
    main()
