# -*- coding: utf-8 -*-
"""
現在のライブラリバージョン確認用スクリプト

表示内容:
- Transformers のバージョン
- Sentence-Transformers のバージョン

実行例:
python src/WeatherLLM/Embedding/v_check.py
"""

import sys

def main() -> None:
    # Transformers
    try:
        import transformers  # type: ignore
        tf_ver = getattr(transformers, "__version__", "unknown")
    except Exception as e:
        tf_ver = f"not installed ({e.__class__.__name__}: {e})"

    # Sentence-Transformers
    try:
        import sentence_transformers as st  # type: ignore
        st_ver = getattr(st, "__version__", "unknown")
    except Exception as e:
        st_ver = f"not installed ({e.__class__.__name__}: {e})"

    print(f"Transformers: {tf_ver}")
    print(f"Sentence-Transformers: {st_ver}")

if __name__ == "__main__":
    main()
