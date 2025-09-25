実行コマンド (Quick Start)

```
# 単一日付を v4 で実行（推奨・最小例）
python -u main_v2.py --pipeline v4 --date 20220106

# v1〜v4 を data/png の先頭3日分で一括実行（APIコスト注意）
python -u main_v2.py --pipeline all --auto --limit 3 --env-file /home/s233319/docker_miniconda/.env

# v1 をAPIキー明示指定（.env不要）
python -u main_v2.py --pipeline v1 --date 20220101 --api-key sk-xxxxxxxxxxxxxxxx
```

# WeatherLLM/program 使い方ガイド（統合版）

本ディレクトリは、v1〜v4 の処理を `main_v2.py` に統合しました（config.py で OpenAI/LLaVA 切替対応）。  
OpenAI Responses API（gpt-4.1）を用いて、天気図画像（＋必要に応じて気象数値データ）からコメント生成を行い、`original_comment` と比較して Embedding 類似度/BLEU/ROUGE-1 を算出・出力します。

- 実行スクリプト: `main_v2.py`（v1〜v4 を一元化、config.py で OpenAI/LLaVA 切替対応）
- 設定: `config.py`（任意。CLI 引数で上書き可能）
- 結果出力先: `program/results/` 配下に自動保存
- ルート/パス: `program` ディレクトリを基準にデータ/出力を扱い、`.env` は上位ディレクトリを遡って探索（見つからなければ `program/.env` も可）

## 1. ディレクトリ構成

```

├── main_v2.py                    # v1〜v4 処理の統合スクリプト
├── config.py                     # 既定設定（任意・CLIで上書き可能）
├── data/
│   ├── png/                      # 天気図画像 (YYYYMMDD.png)
│   ├── Numerical_weather_data/   # 気象数値データ (Y-M-D / Y-MM-DD など混在可)
│   ├── original_comment/         # 既存のオリジナルコメント (YYYY_MM_DD_original.txt)
│   └── prompt_gpt/               # 各バージョン用インストラクション
│       ├── v1_instruction.txt
│       ├── v2_instruction.txt
│       ├── v3_instruction.txt
│       └── v4_instruction.txt
└── results/                      # 実行時に自動生成。出力はここに保存
```

本スクリプト群は `program` ディレクトリを基準に入出力を行います。`/home/xxx` のような固定パスに依存しません。`.env` は `program` から親ディレクトリを上向きに探索し、最初に見つかったものを使用します（無い場合は `program/.env` も利用可能）。

## 2. 必要条件

- Python 3.9+ 推奨（llm_env は 3.11）
- インターネット接続
- Python パッケージ
  - 必須: `openai`
  - 推奨: `sacrebleu`, `rouge-score`, `fugashi[unidic-lite]` または `janome`
    - これらが未導入でもフォールバック計算は動作しますが、スコアの信頼性が向上します

インストール例:

```
pip install openai sacrebleu rouge-score fugashi[unidic-lite] janome
```

## 3. API キーの設定（OpenAI）

`main_v2.py` は以下の優先順位で API キーを取得します（環境変数名は `OpenAI_KEY_TOKEN`）。

1. 環境変数: `OpenAI_KEY_TOKEN`
2. `.env` 読み込み（`--env-file` を指定すれば最優先、指定が無ければ `program` から親を遡って探索）
3. `.env` の手動パース

`.env` の例:

```
OpenAI_KEY_TOKEN=sk-xxxxxxxxxxxxxxxxxxxxxxxx
```

## 4. 実行方法

共通オプション:

- `--pipeline` v1 | v2 | v3 | v4 | all（既定: v4）
- `--date YYYYMMDD`（繰り返し指定可）
- `--auto`（`--date` 省略時に `data/png` から自動発見）
- `--limit N`（自動発見した日付を先頭から N 件に制限）
- `--env-file /path/to/.env`（`.env` の絶対パスを明示）
- `--api-key sk-...`（API キー明示、`.env` 不要）

例:

```
# v4: 画像 + v4_instruction + 気象数値データ
python -u main_v2.py --pipeline v4 --date 20220106 --env-file /home/s233319/docker_miniconda/.env

# v1〜v4 をまとめて（APIコスト注意）
python -u main_v2.py --pipeline all --auto --limit 3 --env-file /home/s233319/docker_miniconda/.env

# v1: 画像 + v1_instruction
python -u main_v2.py --pipeline v1 --date 20220101 --api-key sk-xxxxxxxxxxxxxxxx
```

注意:

- `data/png/` に `YYYYMMDD.png` が必要です。
- v3/v4 は `data/Numerical_weather_data/` に気象数値データ（`YYYY-M-D.txt` など複数パターン対応）が必要です。
- `original_comment` が存在すれば評価（比較）を行います（`YYYY_MM_DD_original.txt`）。

## 5. モデルと API

- バックエンドは `program/config.py` の `MODEL_BACKEND` で選択可能（"openai" | "llava"）
  - "openai": OpenAI Responses API（既定: `gpt-4.1`）
    - 入力: `input_text` + `input_image`（Base64 PNG）
    - 出力: `response.output_text`（フォールバックあり）
  - "llava": ローカル LLaVA（`src/WeatherLLM/llava.py` のローダを使用）
    - 入力: 画像 + テキスト（chat_template 経由で `<image>` を自動挿入）
    - 出力: デコード済みテキスト

### Backend 切替（config.py）

例 1) LLaVA を使う（必要に応じて 4bit、省メモリ）

```python
# program/config.py
MODEL_BACKEND = "llava"

# LLaVA 関連
LLAVA_MODEL_ID = "llava-hf/llava-1.5-7b-hf"  # 13B 等へ変更可
LLAVA_LOCAL_DIR = None        # 既にローカルへ保存済みならパスを指定可
LLAVA_DEVICE = "auto"         # "auto" | "cpu" | "cuda"
LLAVA_USE_4BIT_INFERENCE = False  # True にすると bitsandbytes 必要
LLAVA_MAX_NEW_TOKENS = 256
LLAVA_TEMPERATURE = 0.7
LLAVA_TOP_P = 0.95
LLAVA_TOP_K = 50

# 評価（オプション）
ENABLE_EVALUATION = True
EVAL_EMBEDDINGS = "openai"    # "openai" or None（OpenAI 埋め込みで類似度を取りたい場合は "openai"）
```

例 2) OpenAI を使う（既定）

```python
# program/config.py
MODEL_BACKEND = "openai"
OPENAI_MODEL = "gpt-4.1"
# 評価用の埋め込みモデル（変更可）
OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
```

注意:

- LLaVA を選択した場合でも、`EVAL_EMBEDDINGS="openai"` のままだと評価(Embedding cosine)で OpenAI API キーが必要です。
  - キー不要にするには `EVAL_EMBEDDINGS = None` に設定してください。
- LLaVA のローカル推論には以下のパッケージ/環境が推奨です:
  - `transformers`, `huggingface_hub`, `Pillow`, `torch`（GPU 推奨）
  - 4bit 量子化利用時は `bitsandbytes` が必要
  - 例: `pip install transformers huggingface_hub pillow`（環境に応じて `torch`/`bitsandbytes` を追加）

### Python 設定（config.py を編集・推奨）

- 設定ファイル: `config.py`
- `CFG` という辞書を直接編集します。わかりやすい階層構造で、主要項目は以下です:

  - 実行: `CFG["PIPELINE"]`, `CFG["DATES"]`, `CFG["AUTO_FROM_PNG"]`, `CFG["AUTO_LIMIT"]`
  - 認証: `CFG["ENV_FILE"]`, `CFG["API_KEY"]`
  - バックエンド: `CFG["BACKEND"]["TYPE"]` を `"openai"` または `"llava"` に切替
    - OpenAI 詳細: `CFG["BACKEND"]["OPENAI"]["MODEL"]`, `...["EMBEDDING_MODEL"]`
    - LLaVA 詳細: `CFG["BACKEND"]["LLAVA"]["MODEL_ID"]`, `...["DEVICE"]`, `...["USE_4BIT_INFERENCE"]` など
  - 評価: `CFG["EVALUATION"]["ENABLE"]`, `CFG["EVALUATION"]["EMBEDDINGS"]`（"openai" or None）

- 最小例 1（LLaVA を使う、評価で OpenAI を使わない）

```python
# config.py
CFG["BACKEND"]["TYPE"] = "llava"
CFG["BACKEND"]["LLAVA"].update({
    "MODEL_ID": "llava-hf/llava-1.5-7b-hf",
    "DEVICE": "auto",
    "USE_4BIT_INFERENCE": False,
    "MAX_NEW_TOKENS": 256,
    "TEMPERATURE": 0.7,
    "TOP_P": 0.95,
    "TOP_K": 50,
    "SEED": 0,
})
CFG["EVALUATION"]["ENABLE"] = True
CFG["EVALUATION"]["EMBEDDINGS"] = None  # OpenAI 埋め込みを使わない
```

- 最小例 2（OpenAI を使う）

```python
# config.py
CFG["BACKEND"]["TYPE"] = "openai"
CFG["BACKEND"]["OPENAI"]["MODEL"] = "gpt-4.1"
CFG["EVALUATION"]["ENABLE"] = True
CFG["EVALUATION"]["EMBEDDINGS"] = "openai"
# 必要なら ENV_FILE か API_KEY を設定
# CFG["ENV_FILE"] = "/home/s233319/docker_miniconda/.env"
# CFG["API_KEY"] = "sk-xxxx"
```

- 実行時の確認:

  - 起動時に `[config] Python: ...` として使用中の `config.py` パス、`[config] Backend: ...` としてバックエンド概要が表示されます。

- 優先順位:
  - CLI（`--env-file`/`--api-key` など） > Python 設定(config.py) > 既定値
  - 特に `--env-file`/`--api-key` は CLI が最優先です。

## 6. 評価指標（自動表示＋保存）

生成したコメントと `original_comment` を比較し、以下を標準出力へ表示、JSON に保存します。

- Embedding 類似度（`text-embedding-3-large` のコサイン類似度）
- BLEU（`sacrebleu`）
- ROUGE-1（`rouge-score`。未導入時はフォールバックのユニグラム F1）
  - さらに ROUGE-1 が 0 の場合、文字レベル F1（char-level F1）を代替値として採用

比較表示:

- `--- 比較対象 ---` として `[original_comment]` と `[generated]` を併記
- 続いて `--- 評価結果 ---` として各スコアを表示

正規化:

- 句読点・記号除去、空白削減、小文字化で、表記ゆれの影響を低減

## 7. 出力（結果の保存）

`program/results/` に以下を保存します（パイプライン= vN, 日付= YYYYMMDD）:

- `vN_YYYYMMDD_result.txt`（生成本文）
- `vN_YYYYMMDD_response.json`（OpenAI 応答＋ metrics）

## 8. ポータビリティ（他サーバーでも動作）

- スクリプトは `program` ディレクトリを基準に動作します。特定のディレクトリ名は不要です。
- `.env` は `program` から上位に向かって探索し、最初に見つかったものを使用します（無い場合は `program/.env`）。
- Compose/DevContainer/`/app` などのパスでも動作するよう探索を強化しています。

## 9. バッチ実行の例（all）

`data/png` に存在する先頭 5 日分に対して v4 を順次実行:

```
python -u main_v2.py --pipeline v4 --auto --limit 5 --env-file /home/s233319/docker_miniconda/.env
```

## 10. トラブルシューティング

- API キー関連
  - `OpenAI API key not found...`:
    - 環境変数 `OpenAI_KEY_TOKEN` が設定されているか確認
    - `--env-file` で `.env` を明示
    - もしくは `program` から親に `.env` を配置
- モデル/権限関連
  - 401/403/404:
    - API キーの権限と有効性を確認
- ネットワーク関連
  - タイムアウト・SSL エラー:
    - ネットワーク/プロキシ設定の見直し
- 入力ファイル関連
  - `画像ファイルが存在しません`:
    - `data/png/YYYYMMDD.png` の存在と日付指定を確認
  - `気象数値データファイルが見つかりませんでした`（v3/v4）:
    - `data/Numerical_weather_data/` 内に `YYYY-M-D.txt` 等が存在するか確認
- 評価が全て 0 付近
  - 句読点・記号・空白や表記ゆれの影響が強い場合がありますが、正規化・フォールバック（ユニグラム F1/文字 F1）を実装済み
  - `sacrebleu`・`rouge-score` を導入すると安定性が向上

## 11. 変更履歴（今回の主な更新）

- v1〜v4 の処理を `main_v2.py` に完全統合
- 従来の `v1.py / v2.py / v3.py / v4.py` を削除
- `original_comment` 比較の事前表示（原文/生成文）を追加
- ROUGE-1 のフォールバック（ユニグラム F1、さらに 0 時は文字 F1）を追加
- `.env` 探索の堅牢化（compose/.docker、`/app`、`/workspace` など）

以上で、最新構成に沿った README へ更新しました。
