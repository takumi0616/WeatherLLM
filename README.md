# WeatherLLM

本ディレクトリは、以下の研究目的のために各種 LLM（大規模言語モデル）を探索・実行・調整（微調整）するためのコードと手順をまとめたものです。

目的（研究テーマ）
- 「気象予報士の代替をAIに任せるために、気象知識、日本語応答性能において性能の良いLLMを探索、実行、調整する」
- 具体的には、気象に関する知識（専門用語、観測/解析/予測の文脈）、日本語での指示理解・応答品質、計算資源効率（GPUメモリ、推論速度）を総合的に評価し、実務利用可能なモデル・設定・ワークフローを確立することを目指します。

本リポジトリ（docker_miniconda）全体と連携しており、Docker + Conda による再現性の高い GPU 環境（llm_env）で実行します。

---

## 重要: 本リポジトリの「単一コマンド実行（run-all）」方針

各モデルスクリプト（borea / llama3_8b / mixtral_8x7b_v0_1 / gemma3_4b）に、次の一連処理を1コマンドで実行する `--run-all` を実装しています。

実行フロー（--run-all）
1) モデルをローカルへ保存（未保存ならダウンロード）
2) ローカル保存モデルで軽い推論（動作確認）
3) LoRA/QLoRA で最小学習（未保存なら学習）
   - Gemma-2: LoRA 学習を実施
   - Gemma-3: 本スクリプトでは学習スキップ（推論のみ）
   - Llama3: 既定 QLoRA（4bit）
   - Mixtral: 既定 QLoRA（4bit）
   - Borea: LoRA（4bit 量子化ではなく16bit系での簡易LoRA。動作はborea.pyに準拠）
4) 学習済みアダプタ（LoRA）を適用して推論（最終確認）
5) 成果物（生成テキスト/プロンプト/実行引数）を outputs/ 以下に保存

スキップ動作（再実行時）
- 既に保存されている場合は各フェーズを自動スキップ
  - ベースモデル: `models/<safe_id>/` に `config.json` と `.safetensors`/`.bin` があればダウンロードをスキップ
  - LoRA アダプタ: `models/<safe_id>-lora/` に `adapter_config.json` と `adapter_model*.safetensors` があれば学習をスキップ

保存パス規約
- ベースモデル: `models/<safe_id>/`
- LoRA アダプタ: `models/<safe_id>-lora/`
- 生成物／ログ等: `outputs/<safe_id>/<timestamp>/<tag or run-all>/`
  - run_args.json（実行引数）、messages/chat JSON（プロンプト）、generation_*.txt（生成結果）

注記
- peft が必要（llm_env に同梱済み想定）
- flash-attn 未導入環境でも動作するよう安定化（必要に応じて `attn_implementation='eager'`）
- Llama などのゲート付きモデルは Hugging Face 側で利用承認が必要
- 環境変数 `HF_TOKEN` を .env に設定しておくことを推奨

---

## ディレクトリ構成（WeatherLLM）

- llm_utils.py
  - 共通ユーティリティ（ログ、dtype選択、device_map解決、TF32有効化、HF_TOKEN取得、チャットテンプレート適用、出力/保存ヘルパ、model_idの安全化）
- borea.py
  - HODACHI/Borea-Phi-3.5-mini-Instruct-Jp を対象。`--run-all` でダウンロード→軽い推論→LoRA微調整→推論まで一括実行
  - Phi-3 系 DynamicCache/API 非互換に対する安定化パッチ込み
- mixtral_8x7b_v0_1.py
  - mistralai/Mixtral-8x7B-Instruct-v0.1。既定で 4bit 量子化（bitsandbytes）を使用
  - `--run-all` でダウンロード→軽い推論→QLoRA→推論
- llama3_8b.py
  - meta-llama/Meta-Llama-3-8B-Instruct。`--run-all` でダウンロード→軽い推論→QLoRA→推論
- gemma3_4b.py
  - google/gemma-2-2b-jpn-it（Gemma-2）を既定。`--run-all` は Gemma-2 で LoRA 学習、Gemma-3 は推論のみ（テキストのみ）
- outputs/（自動生成）
  - 各スクリプトの実行成果物を保存
- models/（自動生成）
  - ベースモデルおよび LoRA アダプタの保存先
- .gitignore
  - outputs/、models/ の巨大ファイル類（モデル等）は Git 管理外

---

## 実行環境

- コンテナのデフォルト Conda 環境は llm_env（GPU向け）
  - compose.gpu.yml → DEFAULT_CONDA_ENV=llm_env
- llm_env（/environments_gpu/llm_env.yml）
  - PyTorch + CUDA、transformers, accelerate, datasets, tokenizers, sentencepiece, safetensors
  - 量子化/最適化: bitsandbytes, peft, optimum
  - 評価: sacrebleu, rouge-score, bert-score
  - 日本語処理: fugashi[unidic-lite]
  - RAG: sentence-transformers, faiss-cpu, langchain, chromadb
  - 可視化: matplotlib, seaborn、設定/ログ: rich, omegaconf
- Hugging Face トークン
  - .env に `HF_TOKEN=...` を設定。`HF_HUB_ENABLE_HF_TRANSFER=1` は自動設定

> Llama 等のゲート付きモデルは事前に Hugging Face の利用承認が必要。

---

## 使い方（コンテナ内、llm_env 有効、作業ディレクトリは src/WeatherLLM）

以下のコマンドは全て「この WeatherLLM ディレクトリ内（src/WeatherLLM）」で実行します。

### 1) Borea: ダウンロード→軽い推論→LoRA→推論（1コマンド）
```shell
notify-run wsl-ubuntu -- nohup python borea.py --run-all --device cuda --max-new-tokens 64 --epochs 1 --batch-size 1 --tag lora_all > borea.log 2>&1 &
# ログ監視
tail -f borea.log
```
- ベースモデル保存: `models/HODACHI_Borea-Phi-3.5-mini-Instruct-Jp/`
- LoRA 保存: `models/HODACHI_Borea-Phi-3.5-mini-Instruct-Jp-lora/`
- 生成物例: `outputs/HODACHI_Borea-Phi-3.5-mini-Instruct-Jp/<timestamp>/lora_all/`
  - generation_base_initial.txt, generation_lora_after.txt, prompt_*.json, run_args.json

### 2) Llama 3 8B: ダウンロード→軽い推論→QLoRA→推論（1コマンド）
```shell
notify-run wsl-ubuntu -- nohup python llama3_8b.py --run-all --device cuda --max-new-tokens 64 --epochs 1 --batch-size 1 --tag lora_all > llama3_8b.log 2>&1 &
tail -f llama3_8b.log
```
- ベースモデル保存: `models/meta-llama_Meta-Llama-3-8B-Instruct/`
- LoRA 保存: `models/meta-llama_Meta-Llama-3-8B-Instruct-lora/`
- 生成物: `outputs/meta-llama_Meta-Llama-3-8B-Instruct/<timestamp>/lora_all/`
  - generation_base.txt, generation_lora.txt, messages_*.json, run_args.json
- 備考: 既定で QLoRA（4bit）で学習

### 3) Mixtral 8x7B: ダウンロード→軽い推論→QLoRA→推論（1コマンド）
```shell
notify-run wsl-ubuntu -- nohup python mixtral_8x7b_v0_1.py --run-all --device cuda --max-new-tokens 64 --epochs 1 --batch-size 1 --tag lora_all > mixtral_8x7b.log 2>&1 &
tail -f mixtral_8x7b.log
```
- ベースモデル保存: `models/mistralai_Mixtral-8x7B-Instruct-v0_1/`（safe_id 変換）
- LoRA 保存: `models/mistralai_Mixtral-8x7B-Instruct-v0_1-lora/`
- 生成物: `outputs/mistralai_Mixtral-8x7B-Instruct-v0_1/<timestamp>/lora_all/`
  - generation_base.txt, generation_lora.txt, messages_*.json, run_args.json
- 備考: 既定で 4bit 量子化（bitsandbytes）。`--no-4bit` で無効化可能

### 4) Gemma（Gemma-2/3）: ダウンロード→軽い推論→（Gemma-2のみLoRA）→推論（1コマンド）
```shell
# Gemma-2 日本語IT（既定）
notify-run wsl-ubuntu -- nohup python gemma3_4b.py --run-all --device cuda --max-new-tokens 64 --epochs 1 --batch-size 1 --tag lora_all > gemma3_4b.log 2>&1 &
tail -f gemma3_4b.log

# Gemma-3 系（テキストのみ、LoRA はスキップ）
notify-run wsl-ubuntu -- nohup python gemma3_4b.py --run-all --device cuda --model google/gemma-3-4b-it --max-new-tokens 64 --tag g3_run > gemma3_4b.log 2>&1 &
tail -f gemma3_4b.log
```
- Gemma-2
  - ベース保存: `models/google_gemma-2-2b-jpn-it/`
  - LoRA 保存: `models/google_gemma-2-2b-jpn-it-lora/`
  - 生成物: `outputs/google_gemma-2-2b-jpn-it/<timestamp>/lora_all/`
- Gemma-3
  - ベース保存: `models/google_gemma-3-4b-it/`
  - 生成物: `outputs/google_gemma-3-4b-it/<timestamp>/g3_run/`
  - 備考: 本スクリプトでは Gemma-3 は推論のみ（LoRA 学習は対象外）

---

## 生成物（成果物）の構成

各 run-all 実行で、以下の成果物が保存されます。

- `outputs/<safe_id>/<timestamp>/<tag>/`
  - `run_args.json`（実行引数）
  - `messages_*.json` もしくは `chat_*.json`（使用したプロンプト）
  - `generation_base*.txt`（ベースモデルの生成結果）
  - `generation_lora*.txt`（LoRA 適用後の生成結果、対象モデルのみ）
- `models/<safe_id>/`（ベースモデル、config/重み/tokenizer 等）
- `models/<safe_id>-lora/`（LoRA アダプタ、adapter_config/adapter_model 等）

safe_id の例
- `HODACHI/Borea-Phi-3.5-mini-Instruct-Jp` → `HODACHI_Borea-Phi-3.5-mini-Instruct-Jp`
- `mistralai/Mixtral-8x7B-Instruct-v0.1` → `mistralai_Mixtral-8x7B-Instruct-v0_1`

---

## 仕組み（何をしているか）

- ダウンロード
  - `huggingface_hub.snapshot_download` でリポジトリ全体を保存（未インストール時は `from_pretrained`→`save_pretrained` のフォールバック）
- 軽い推論
  - 保存済みローカルモデルを `from_pretrained(local_dir=...)` でロードし、サンプルプロンプトで短い生成（動作確認）
- LoRA/QLoRA 学習
  - `peft` により、一般的なアテンション/MLP 投影層をターゲットに LoRA を適用
  - Llama/Mixtral は省メモリのため QLoRA（4bit）で学習
  - 学習済みアダプタは `models/<safe_id>-lora/` に保存
- LoRA 適用推論
  - ベースモデルに `PeftModel.from_pretrained` でアダプタを適用し、短い生成（確認）
- スキップ制御
  - 既存の保存物を検出し、ダウンロード/学習の重複を回避
- 安定化
  - bf16/fp16/float32 の自動選択、TF32 有効化、4bit 量子化（対応スクリプト）
  - Borea(Phi-3系) は DynamicCache 非互換のガードを実装し、安定動作を優先

---

## パラメータ解説（--max-new-tokens / --epochs / --batch-size）

本リポジトリで推奨する「単一コマンド実行（run-all）」で頻出する3つのパラメータについて、意味・影響・目安値を整理します。

- --max-new-tokens
  - 意味: 生成フェーズで新たに生成する最大トークン数（プロンプト長は含まず、生成分のみの上限）
  - 影響:
    - 出力の長さ（冗長さ・情報量）に直結。大きいほど長文を生成
    - 推論時間はほぼ生成トークン数に比例して増加（1トークンずつ生成するため）
    - KVキャッシュ（注意機構のための過去状態）が増えるため、(プロンプト長 + 生成長) に応じてGPUメモリ使用量も増える
  - 目安（用途別・クイックチェック想定）:
    - 32〜64: 動作確認・小さな応答（高速）
    - 128〜256: 一般的な説明・短めの要約
    - 512〜1024: 詳細説明・物語・長めの要約（GPUメモリに余裕がある場合）
  - 実務的な指針:
    - まず64〜128で動作確認 → 必要に応じて段階的に増やす
    - OOM（メモリ不足）が出る場合は max-new-tokens を下げるか、プロンプトを短くする

- --epochs
  - 意味: 学習データ（train split）全体を何周するか（エポック回数）
  - 影響:
    - 小さいほど学習は速いが、十分に学習できない可能性
    - 大きいほど過学習のリスク・時間増大。特にデモ用の極小データでは過学習しやすい
  - 目安（本リポジトリの「最小デモ学習」前提）:
    - 1〜2: デモ・動作確認（最推奨）
    - 3〜5: もう少し効果を見たい場合（データが数百〜数千例以上ある前提）
    - 10以上: しっかり学習する用途だが、学習データ・評価・学習率スケジュール等のチューニングが必須
  - 実務的な指針:
    - まず1エポックで pipelineや保存形式が正しく動くか検証
    - 効果を見ながら段階的に増やす（学習損失や検証指標が悪化し始めたら過学習の兆候）

- --batch-size（= per_device_train_batch_size）
  - 意味: 1GPU（1デバイス）あたりのミニバッチサイズ
  - 影響:
    - 大きいほど統計的な安定性・スループット向上の可能性。ただし学習メモリ使用量はほぼバッチサイズに比例して増える
    - 量子化（QLoRAの4bit）は主に「重み」メモリを圧縮するが、「活性・勾配」メモリは依然ボトルネックになりうる
  - 目安（推奨初期値）:
    - 8〜12GB級GPU: 1（まずは確実に動かす）
    - 16〜24GB級GPU: 1〜2（seq長やモデルにも依存）
    - 24GB超: 2〜4（状況に応じて）
  - 実務的な指針:
    - OOMが出たら batch-size を下げる、学習系列長（--max-length-train）を下げる
    - さらに改善したい場合は「勾配蓄積（gradient_accumulation）」を導入（現行スクリプトにはCLI未露出。必要なら拡張可）

補足（相互作用とチューニングのコツ）
- 生成長とメモリ: 生成中はプロンプト長 + 生成済みトークン分のKVキャッシュが蓄積。max-new-tokens を大きくするとメモリも伸びる
- 学習系列長（--max-length-train）: 長いほど表現力が増すが、メモリ・時間コストが上がる。まずは1024程度で開始し、OOMなら短く
- 量子化とLoRA: QLoRA（4bit）は重みを圧縮し、LoRAは微小なアダプタのみ更新するため、フル微調整よりメモリ節約
- 最初は保守的に:
  - 推論: --max-new-tokens 64〜128
  - 学習: --epochs 1 / --batch-size 1
  - 動作確認後に段階的に増やすのが安全

## よくあるトラブルと対処

- モデル取得に失敗
  - HF_TOKEN の設定（.env）、モデルのアクセス権限・利用承認、ネットワークを確認
- CUDA/ドライバ不整合
  - compose.gpu.yml と llm_env.yml の CUDA/PyTorch バージョン整合を確認
- bitsandbytes（4bit）が使えない
  - `--no-4bit` で量子化を無効化して実行
- オフライン検証
  - 事前に run-all で保存後、`HF_HUB_OFFLINE=1` を環境に設定して再実行（ローカルのみで動くか検証）

---

## 既存スクリプトの単発モード（互換維持）

run-all を使わない従来モードも残しています。例えば:

- Borea 生成のみ
  - `python borea.py`
- Llama 生成のみ
  - `python llama3_8b.py --prompt "短い自己紹介を日本語で"`
- Mixtral 生成のみ
  - `python mixtral_8x7b_v0_1.py --prompt "Who is the cutest in Madoka Magica?"`
- Gemma 生成のみ
  - Gemma-2: `python gemma3_4b.py --model google/gemma-2-2b-jpn-it --prompt "あなたの好きな食べ物は？"`
  - Gemma-3: `python gemma3_4b.py --model google/gemma-3-4b-it --prompt "あなたの好きな食べ物は？"`

---

## RAG / 評価 / 可視化（拡張余地）

- RAG: sentence-transformers, faiss-cpu, langchain, chromadb
- 評価: sacrebleu、rouge-score、bert-score（将来的に統一評価スクリプトを追加予定）
- 可視化・NetCDF: matplotlib, seaborn, xarray, netcdf4

---

## コンテナ補足

- GPU 環境
  - compose.gpu.yml のベースイメージ（CUDA）と PYTORCH_CUDA_VERSION をそろえてビルド
- コンテナに入る
  - `sudo docker compose exec app bash`
  - プロンプト時に llm_env が自動有効化

---

本 README とスクリプト群により、「ダウンロード→軽い推論→LoRA/QLoRA 微調整→推論」の再現性の高いフローを確立しています。  
生成物は outputs/、保存物は models/ に集約し、再実行時は既存成果物に応じて自動スキップされます。必要に応じて run-all を活用してください。
