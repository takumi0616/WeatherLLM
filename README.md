# WeatherLLM

本ディレクトリは、以下の研究目的のために各種 LLM（大規模言語モデル）を探索・実行・調整（微調整）するためのコードと手順をまとめたものです。

目的（研究テーマ）
- 「気象予報士の代替をAIに任せるために、気象知識、日本語応答性能において性能の良いLLMを探索、実行、調整する」
- 具体的には、気象に関する知識（専門用語、観測/解析/予測の文脈）、日本語での指示理解・応答品質、計算資源効率（GPUメモリ、推論速度）を総合的に評価し、実務利用可能なモデル・設定・ワークフローを確立することを目指します。

本リポジトリ（docker_miniconda）全体と連携しており、Docker + Conda による再現性の高い GPU 環境（llm_env）で実行します。

---

## ディレクトリ構成（WeatherLLM）

- llm_utils.py
  - 共通ユーティリティ（ログ、dtype選択、device_map解決、TF32有効化、HF_TOKEN取得、チャットテンプレート適用、出力/保存ヘルパ）
- borea.py
  - HODACHI/Borea-Phi-3.5-mini-Instruct-Jp を対象に、生成＋最小学習（--train）まで行えるスクリプト
  - phi3系モデルでの DynamicCache 問題に対する暫定パッチも実装
- mixtral_8x7b_v0_1.py
  - mistralai/Mixtral-8x7B-Instruct-v0.1 を対象に、既定で 4bit 量子化（bitsandbytes）を有効化して省メモリ推論
- llama3_8b.py
  - meta-llama/Meta-Llama-3-8B-Instruct を対象に、標準は非量子化、--use-4bit で 4bit 量子化をオン
- gemma3_4b.py
  - google/gemma-2-2b-jpn-it（Gemma-2）を既定とし、--model が gemma-3-* の場合は Gemma-3 経路へ自動分岐（本スクリプトはテキストのみ対応）
- outputs/（自動生成）
  - 各スクリプトの実行成果物（プロンプト、生成テキスト、実行引数）を「outputs/<model_id_安全化>/<timestamp>/」に保存します
- .gitignore
  - outputs/、fine_tuned-model/、logs/、巨大モデルファイル（*.safetensors 等）をGit管理外にしています

---

## 実行環境

- コンテナ起動時のデフォルト Conda 環境は llm_env（GPU向け）に設定済み
  - compose.gpu.yml → build.args → DEFAULT_CONDA_ENV=llm_env
- llm_env（/environments_gpu/llm_env.yml）には、LLM 実行・評価・RAG・可視化・軽量サービングに有用なライブラリを幅広く追加済み
  - PyTorch + CUDA、科学計算（numpy/pandas/scipy/pyarrow/protobuf/xarray/netcdf4 等）
  - Hugging Face/LLM（transformers@main、accelerate、datasets、tokenizers、sentencepiece、safetensors ほか）
  - 量子化/最適化（bitsandbytes、optimum、peft）
  - 評価（sacrebleu、rouge-score、bert-score）
  - 日本語処理（fugashi[unidic-lite]）
  - RAG/検索（sentence-transformers、faiss-cpu、langchain、chromadb）
  - 可視化（matplotlib、seaborn）、ログ・設定（rich、omegaconf）、トラッキング（wandb）
  - 簡易サービング（fastapi、uvicorn）
- Hugging Face トークン
  - リポジトリ直下の .env に HF_TOKEN=... を設定済み
  - 各スクリプトは自動で os.getenv("HF_TOKEN") を参照し、from_pretrained(token=...) に渡します
  - ダウンロード高速化のため、HF_HUB_ENABLE_HF_TRANSFER=1 を自動設定

> 注意: Llama 等のゲート付きモデルは、Hugging Face 側で「利用規約を承認」済みである必要があります。

---

## 使い方（コンテナ内、llm_env 有効）

以下は実行例です（出力ディレクトリは自動作成、実行は必要に応じて行ってください）。

1) Borea（生成のみ）
- 例
  - python src/WeatherLLM/borea.py
- 主なオプション
  - --model HODACHI/Borea-Phi-3.5-mini-Instruct-Jp
  - --max-new-tokens, --temperature, --top-p
  - --out-base, --tag 出力パス調整
- 生成＋最小学習
  - python src/WeatherLLM/borea.py --train --epochs 1 --batch-size 1 --output-dir src/WeatherLLM/fine_tuned-model

2) Mixtral（4bit 量子化が既定）
- 例
  - python src/WeatherLLM/mixtral_8x7b_v0_1.py --prompt "Who is the cutest in Madoka Magica?"
- 量子化無効化
  - --no-4bit
- dtype 指定
  - --dtype bfloat16 / float16 / float32

3) Llama 3 8B
- 例
  - python src/WeatherLLM/llama3_8b.py --prompt "短い自己紹介を日本語で"
- 4bit 量子化
  - --use-4bit

4) Gemma（Gemma-2/3 自動分岐・テキストのみ）
- 例（Gemma-2 日本語ITモデル。既定）
  - python src/WeatherLLM/gemma3_4b.py --model google/gemma-2-2b-jpn-it --prompt "あなたの好きな食べ物は？"
- 例（Gemma-3 系）
  - python src/WeatherLLM/gemma3_4b.py --model google/gemma-3-4b-it --prompt "あなたの好きな食べ物は？"

---

## 出力（成果物）の整理

- 各スクリプトは以下を自動保存します
  - outputs/<model_id_安全化>/<timestamp>/run_args.json（実行引数）
  - outputs/<...>/messages.json（使用したメッセージ）
  - outputs/<...>/generation.txt（生成テキスト）
  - （borea の学習時）fine_tuned-model/ に学習済みモデルを保存
- src/WeatherLLM/.gitignore で outputs/、fine_tuned-model/、logs/ を Git 管理外に設定

---

## 評価（自動評価のためのライブラリ）

- sacrebleu、rouge-score、bert-score を llm_env に追加済み
  - 将来的に評価スクリプト（例：evaluate_llm.py）を追加し、気象 QA、要約、指示応答に対する自動スコアリングを統一的に実行する計画
- 日本語前処理
  - fugashi[unidic-lite] を追加済み（形態素解析）
- 実験管理
  - wandb を追加済み（任意で API キーを環境変数などで設定）

---

## RAG（検索拡張生成）/ 知識統合

- sentence-transformers、faiss-cpu、langchain、chromadb を llm_env に追加済み
  - 気象ドメインの PDF、手順書、ガイドライン、ブログ等からベクタDBを構築し、LLM への知識供給を強化可能
  - 例）観測/解析手順や警報発令基準などを検索し、応答に参照を付ける

---

## 気象分野向けのデータ/可視化

- xarray、netcdf4 を llm_env に追加済み（NetCDF データの取り扱い）
- 可視化：matplotlib, seaborn
  - cartopy は依存が重めのため llm_env では未追加（必要なら別環境に追加、もしくは要望に応じて拡張可能）
- 既存の weather_env など他 Conda 環境と用途を分離し、LLM 実験は llm_env に集約

---

## ベストプラクティス / メモリ・精度のトレードオフ

- 大きなモデルは 4bit 量子化（Mixtral 既定・Llama は --use-4bit）で GPU メモリ削減
- dtype は bf16/float16/float32 をモデル/デバイスに合わせて選択
- TF32 を有効にすることで、Ampere 以降の GPU でスループット改善（精度要件に注意）
- 生成長（max_new_tokens）やバッチサイズ（学習時）を適切に調整し、OOM を回避
- HF_TOKEN が必要なゲート付きモデルは、Hugging Face で利用規約を事前承認

---

## よくあるトラブル

- モデル取得に失敗
  - HF_TOKEN の設定やアクセストークンの権限、モデルの利用承認を確認
- CUDA バージョン不整合
  - compose.gpu.yml の PYTORCH_CUDA_VERSION と llm_env.yml の pytorch-cuda が Dockerfile により自動整合化される想定（ビルド時に sed で置換）。CUDA を変更する際は両方の設定に気をつけて再ビルド
- 量子化のビルドエラー
  - bitsandbytes は GPU/ドライバとの相性があり、ビルド不可なケースもあるため、その場合は量子化をオフにして実行

---

## 追加・拡張の方針

- 新しい LLM を試す場合は、既存スクリプト（llm_utils, mixtral, llama3, gemma, borea）を参考に最小の CLI スクリプトを追加
- 評価・RAG のフローは llm_env 上で統一
- 必要なライブラリがあれば environments_gpu/llm_env.yml に追加し、GPU 用イメージを再ビルド

---

## 参考：コンテナの起動（実行は不要、参考情報）

- GPU 環境
  - compose.gpu.yml のベースイメージ（CUDA）と PYTORCH_CUDA_VERSION を揃えてビルドし、DEFAULT_CONDA_ENV=llm_env で起動
- コンテナに入る
  - sudo docker compose exec app bash
  - プロンプトで llm_env が自動有効化される

---

この README とスクリプト群により、気象分野に特化した LLM の探索・実行・調整を再現性高く、かつ整理された成果物管理で進められます。必要な追加要望（評価スクリプト、RAG サンプル、可視化ノートなど）があれば、順次拡張していきます。
