実行コマンド (Quick Start)
```
# v1: 画像 + v1_instruction.txt
notify-run via-tml2 -- nohup python v1.py --date 20220101 > v1.out 2>&1 &

# v2: 画像 + v2_instruction.txt
notify-run via-tml2 -- nohup python v2.py --date 20220101 > v2.out 2>&1 &

# v3: 画像 + 数値データ + v3_instruction.txt
notify-run via-tml2 -- nohup python v3.py --date 20220101 > v3.out 2>&1 &

# v4: 画像 + 数値データ + v4_instruction.txt
notify-run via-tml2 -- nohup python v4.py --date 20220101 > v4.out 2>&1 &
```

# WeatherLLM/program 使い方ガイド

本ディレクトリには、天気図画像とプロンプト/気象数値データを用いてコメント生成を行う Python スクリプトが含まれます。Google Colab 依存を排除し、ローカルの Python 環境で実行できるように整備されています。

- 対応スクリプト: v1.py / v2.py / v3.py / v4.py
- 結果出力先: program/results/ 配下に自動保存
- ルート/パス: program ディレクトリを基準にデータ/出力を扱い、.env は上位ディレクトリを遡って探索（見つからなければ program/.env を使用）


## 1. ディレクトリ構成

```
src/WeatherLLM/program/
├── v1.py
├── v2.py
├── v3.py
├── v4.py
├── data/
│   ├── png/                          # 天気図画像 (YYYYMMDD.png)
│   ├── Numerical_weather_data/       # 気象数値データ (Y-M-D / Y-MM-DD など混在可)
│   └── prompt_gpt/                   # 各バージョン用インストラクション
│       ├── v1_instruction.txt
│       ├── v2_instruction.txt
│       ├── v3_instruction.txt
│       └── v4_instruction.txt
└── results/                          # 実行時に自動生成。出力はここに保存
```

本スクリプト群は program ディレクトリを基準に入出力を行います。`/home/xxx` のような固定パスに依存しません。.env は program から親ディレクトリを上向きに探索し、最初に見つかったものを使用します（無い場合は program/.env を利用可能）。


## 2. 必要条件

- Python 3.9+ 推奨
- インターネット接続
- Python パッケージ
  - requests（必須）
  - python-dotenv（任意、.env を読みやすくするため）
- OpenAI API キー（Vision対応モデルが利用可能なキー）


## 3. API キーの設定（OpenAI）

各スクリプトは以下の優先順位で API キーを取得します（環境変数名は OpenAI_KEY_TOKEN）。

1) 環境変数: `OpenAI_KEY_TOKEN`
2) `.env` 読み込み（python-dotenv がインストールされている場合）
3) `.env` の手動パース

`.env` は program から上位にある任意の場所（通常はリポジトリ直下など）に置けます。スクリプトは program から親ディレクトリを遡って最初に見つかった .env を使用します。見つからなければ program/.env を読み込みます。

例: docker_miniconda/.env
```
OpenAI_KEY_TOKEN=sk-xxxxxxxxxxxxxxxxxxxxxxxx
```

環境変数を直接設定する例（bash）:
```
export OpenAI_KEY_TOKEN=sk-xxxxxxxxxxxxxxxxxxxxxxxx
```


## 4. インストール（必要に応じて）

pip 例:
```
pip install requests python-dotenv
```

conda 例:
```
conda install -c conda-forge requests python-dotenv
```


## 5. 実行方法

共通オプション:
- `--date YYYYMMDD`（例: 20220106）
  - 省略時は `data/png` 内で最も古い日付のファイルを自動選択します。

実行例:
```
# v1: 画像 + v1_instruction.txt
python3 src/WeatherLLM/program/v1.py --date 20220101

# v2: 画像 + v2_instruction.txt
python3 src/WeatherLLM/program/v2.py --date 20220101

# v3: 画像 + 数値データ + v3_instruction.txt
python3 src/WeatherLLM/program/v3.py --date 20220106

# v4: 画像 + 数値データ + v4_instruction.txt
python3 src/WeatherLLM/program/v4.py --date 20220106
```

注意:
- `data/png/` に `YYYYMMDD.png` 形式のファイルが必要です。
- v3/v4 は `data/Numerical_weather_data/` に気象数値データのテキストが必要です。


## 6. モデルの設定

各スクリプト冒頭付近の定数:
```
MODEL_NAME = "gpt-4o"
```

- 既定値は Vision 対応の `gpt-4o` です。
- 利用可能なモデル/権限に応じて `"gpt-4o-mini"` などへ変更してください。
- エンドポイントは Chat Completions API（`https://api.openai.com/v1/chat/completions`）を使用しています。


## 7. 入出力仕様

入力（必須/想定）:
- 画像: `data/png/YYYYMMDD.png`
- 数値データ（v3・v4 のみ）: `data/Numerical_weather_data/` に下記いずれかのファイル名で存在
  - `YYYY-M-D.txt`
  - `YYYY-MM-DD.txt`
  - `YYYY-MM-D.txt`
  - `YYYY-M-DD.txt`

出力（自動保存、各スクリプト共通）:
- `program/results/vN_YYYYMMDD_result.txt`（生成された本文）
- `program/results/vN_YYYYMMDD_response.json`（OpenAI API の応答全文 JSON）
  - 実行時に `program/results/` が自動生成されます。


## 8. ポータビリティ（他サーバーでも動作）

- スクリプトは program ディレクトリを基準に動作します。特定のディレクトリ名は不要です。
- そのため、`/home/xxxx` 固定などに依存せず、コンテナ内の `/app/...` のようなパスでも動作します。
- .env は program から上位に向かって探索し、最初に見つかったものを使用します（無い場合は program/.env を使用可能）。


## 9. バッチ実行の例

`data/png` に存在する全日付に対して v4 を順次実行する bash 例（実行時間・APIコスト注意）:
```bash
cd /path/to/docker_miniconda  # ルート位置に合わせて変更
for img in src/WeatherLLM/program/data/png/*.png; do
  base=$(basename "$img")
  date=${base%.png}
  python3 src/WeatherLLM/program/v4.py --date "$date"
done
```

必要に応じて v1/v2/v3 に置き換えることで同様に処理できます。


## 10. トラブルシューティング

- API キー関連
  - `OpenAI API key not found...`:
    - 環境変数 `OpenAI_KEY_TOKEN` が設定されているか確認
    - ルート直下（`docker_miniconda/.env`）に `.env` があり、`OpenAI_KEY_TOKEN=...` 形式で記載されているか確認
- モデル/権限関連
  - 401/403/404 などのエラー:
    - `MODEL_NAME` の切り替え（例: `gpt-4o-mini`）
    - API キーの権限・有効性を確認
- ネットワーク関連
  - タイムアウト・SSL エラー:
    - ネットワーク/プロキシ設定の見直し
    - 後ほど再試行
- 入力ファイル関連
  - `FileNotFoundError: 画像ファイルが存在しません`:
    - `data/png/YYYYMMDD.png` が存在するか確認
  - `気象数値データファイルが見つかりませんでした`:
    - `data/Numerical_weather_data/` 内に、対象日のテキストファイルが上記 4 パターンのいずれかで存在するか確認
- .env 探索関連
  - `.env` が見つからない:
    - 環境変数 `OpenAI_KEY_TOKEN` を設定するか、program から上位に `.env` を配置（無い場合は `program/.env` を作成）


## 11. よくある変更点

- モデル変更:
  - 各スクリプト先頭の `MODEL_NAME` を編集
- 出力先変更:
  - `PROGRAM_DIR / "results"` を変更することで出力ディレクトリを変更可能
- 画像/数値データの保管場所変更:
  - `PROGRAM_DIR / "data" / "..."` 部分を任意の場所に変更可能（相対/絶対どちらでも可）


## 12. 連絡先/メモ

- 本 README の内容は、`src/WeatherLLM/program` 配下のスクリプト群（v1〜v4）の仕様に基づいています。
- 運用時は API コストおよび実行回数にご留意ください。
