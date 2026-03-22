# criticism_bot (Anthropic first step)

Anthropic API を使った対話型の美術批評 Chat Agent です。

対応機能:
- テキスト対話
- ブラウザUI（読みやすいチャット表示）
- 作品画像の入力（ローカルファイル / 画像URL）
- WebサイトURLの本文抽出と参照
- 5段階の星レビュー（★1〜★5）付き批評
- `max_tokens` 到達時の自動継続（途中で切れにくい）
- SQLite への会話履歴永続化
- 再起動時の前回セッション自動復帰（同一DB利用時）
- 非表示の関連過去対話参照（RAG風メモリ）
- 長期対話の圧縮メモリ（古い履歴をセッション要約へ集約）

## 1. Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. API key

`.env` ファイルで管理するのを推奨します（`.env` は `.gitignore` 済み）。

```bash
cp .env.example .env
# .env を編集してキーを設定
```

`.env` を使わずにシェル環境変数で渡す場合:

```bash
export ANTHROPIC_API_KEY="your_api_key"
```

## 3. Run

```bash
python anthropic_art_critic_chat.py
```

デフォルトでは、同じ `--db-path` を使う限り前回セッションを自動で再開します。
毎回新規セッションにしたい場合:

```bash
python anthropic_art_critic_chat.py --new-session
```

オプション例:

```bash
python anthropic_art_critic_chat.py \
  --model claude-sonnet-4-6 \
  --max-tokens 1600 \
  --db-path chat_memory.sqlite3 \
  --memory-top-k 3
```

特定セッションを再開する場合:

```bash
python anthropic_art_critic_chat.py --session-id 20260227-120000-abcd1234
```

SSL証明書エラーが出る環境では、明示的に以下を指定できます（通常は不要）:

```bash
python anthropic_art_critic_chat.py --allow-insecure-ssl
```

## 4. Interactive commands

- `/help` ヘルプ表示
- `/reset` 実行中の会話コンテキストのみクリア（DBは保持）
- `/model <model_id>` 次ターンからモデル変更
- `/max_tokens <number>` 次ターンから max_tokens 変更
- `/image <path_or_url>` 次ターンに画像を添付
- `/url <url>` 次ターンにWeb URL本文を添付
- `/clear_inputs` 添付キューをクリア
- `/session` 現在の session_id 表示
- `/session new` 新しい session_id で開始
- `/session <session_id>` セッション切り替え + 履歴読み込み
- `/show` 現在の設定確認
- `/exit` 終了

## 4.5 Web UI

ターミナルではなくブラウザで会話したい場合:

```bash
streamlit run chat_ui.py
```

- 既定で `chat_memory.sqlite3` を利用
- 前回セッションを自動復元
- 画像アップロード、テキスト系ファイルのドラッグ&ドロップ添付、画像URL、Web URL参照に対応
- 入力トークンを送信前に自動計測し、予算超過時は添付文脈/履歴を自動で圧縮
- 過去ターン編集とロールバック再生成（プルーフカット）

常駐起動（ターミナルを閉じても継続したい場合）:

```bash
./run_ui.sh
./status_ui.sh
./stop_ui.sh
```

## 5. URL / 画像入力の仕様

- ユーザー入力テキスト中の URL は自動検出します
- 画像拡張子の URL（`.jpg/.jpeg/.png/.gif/.webp`）は画像入力として扱います
- それ以外の URL はページ本文を取得し、批評の参照文脈として投入します
- JavaScriptで本文が後から描画されるサイトは十分に取得できない場合があります

## 6. 永続化と非表示メモリ

- 各ターンは SQLite (`chat_memory.sqlite3`) に保存されます
- 新しい入力ごとに、過去対話から関連性の高いものを検索し、内部参照として system prompt に注入します
- 参照メモリはユーザー表示用ではなく、応答品質を上げるための補助コンテキストです
- 同一セッションの古い履歴は自動で圧縮要約され、最新ターンのみ生ログとして保持されます（コンテキスト飽和対策）
- 圧縮要約は `session_summaries` テーブルに保存されます

## 7. ログをMarkdownで出力

ターミナル表示が読みにくい場合は、SQLiteログを `.md` に書き出せます。

```bash
python export_chat_logs_to_md.py
```

- `--session-id <id>`: 指定セッションを書き出す
- `--limit <N>`: 最新Nターンだけ書き出す
- `--output <path>`: 出力先を指定（未指定時は `exports/` 配下に自動作成）

## 8. TRM Pipeline

TRM-A / TRM-B の初期実装を `trm_pipeline/` に追加しています。
本段階では以下までを対象にします。

- Lenia rollout recorder
- TRM-A 学習と 8-step rollout 評価
- frozen TRM-A inference pass
- TRM-B 学習

データ生成:

```bash
python -m trm_pipeline.lenia_data \
  --seed-catalog data/lenia_official/animals2d_seeds.json \
  --output-root data/trm_rollouts \
  --num-seeds 200
```

TRM-A 学習:

```bash
python -m trm_pipeline.train_trm_a \
  --manifest data/trm_rollouts/manifest.jsonl \
  --output-dir artifacts/trm_a \
  --objective variational \
  --z-dim 32 \
  --device cuda \
  --grad-clip 1.0 \
  --amp \
  --log-interval 50
```

`TRM-A` は 3 段階の objective を持ちます。

- `deterministic`: 既存の MSE baseline
- `gaussian_nll`: 平均 + 分散の確率的予測
- `variational`: Gaussian NLL + KL + latent prior/posterior

保存物には `parameter_count`、`val_nll`、`mean_kl`、`mean_pred_var`、`coverage_1sigma_*` が含まれます。`TRM-A` は初期段階で `7M` params 以下に制限しています。

TRM-B 用の frozen 前処理:

```bash
python -m trm_pipeline.prepare_trm_b_data \
  --manifest data/trm_rollouts/manifest.jsonl \
  --checkpoint artifacts/trm_a/trm_a.pt \
  --output-root data/trm_b_cache
```

この cache には既存の `error_map` に加えて、将来拡張用の `pred_logvar`、`pred_var`、`precision_map`、`surprise_map` も保存されます。

TRM-B 学習:

```bash
python -m trm_pipeline.train_trm_b \
  --manifest data/trm_b_cache/manifest.jsonl \
  --output-dir artifacts/trm_b \
  --device cuda \
  --grad-clip 1.0 \
  --amp \
  --log-interval 50
```

注意:

- `torch` が必要です
- 現在の `Lenia_official` は単一チャネル seed を使うため、5チャネル Lenia 状態は recorder 側で導出しています
- rollout manifest には `regime` と `regime_stats` を保存し、`1σ` 被覆率を stable / chaotic で分けて読めるようにしています
- GNW 統合、TRM-C、TRM-D、ERIE 実行ループはまだ対象外です

## 8.5 Minimal ERIE Runtime

Lenia 上で最小の ERIE 自己維持ループを試すための runtime を `trm_pipeline/erie_runtime.py` に追加しています。
この段階では次を持ちます。

- `resource / hazard / shelter` 環境場
- `occupancy / boundary / permeability` による空間 body
- boundary interface 越しの観測
- precision-weighted belief update
- `Risk + Ambiguity - Epistemic` による policy scoring
- `G_t / B_t` viability と death criterion

実行例:

```bash
python -m trm_pipeline.erie_runtime \
  --seed-catalog data/lenia_official/animals2d_seeds.json \
  --output-root artifacts/erie_runtime \
  --steps 128 \
  --lookahead-horizon 3 \
  --lookahead-discount 0.9
```

比較用に `policy mode` を切り替えられます。

```bash
python -m trm_pipeline.erie_runtime --policy-mode closed_loop
python -m trm_pipeline.erie_runtime --policy-mode random
python -m trm_pipeline.erie_runtime --policy-mode no_action
```

既存 checkpoint を使う場合:

```bash
python -m trm_pipeline.erie_runtime \
  --trm-a-checkpoint artifacts/trm_a_variational_smoke/trm_a.pt \
  --trm-b-checkpoint artifacts/trm_b_smoke/trm_b.pt
```

3 モード比較:

```bash
python -m trm_pipeline.compare_erie_runtime \
  --output-root artifacts/erie_runtime_compare \
  --steps 64 \
  --lookahead-horizon 3 \
  --lookahead-discount 0.9
```

主要ノブ:

- `--lookahead-horizon`, `--lookahead-discount`
  policy score の先読み深さと割引率
- `--resource-patches`, `--hazard-patches`, `--shelter-patches`
  Lenia 環境場の粗い難度設定

出力:

- `*_summary.json`: 生存結果と action 集計
- `*_history.json`: step ごとの belief / policy / viability ログ
- `*.npz`: occupancy, boundary, permeability, env_channels などの時系列

## 8.6 TRM-Vm / TRM-As Bootstrap Pipeline

`TRM-Vm` と `TRM-As` は、最初の段階では analytic runtime を教師にして bootstrap します。
この段階では

- `TRM-Vm`: viability monitoring
- `TRM-As`: action scoring

を対象にします。

bootstrap cache 生成:

```bash
python -m trm_pipeline.prepare_trm_va_data \
  --seed-catalog data/lenia_official/animals2d_seeds.json \
  --output-root data/trm_va_cache \
  --episodes 16 \
  --steps 32
```

`TRM-Vm` 学習:

```bash
python -m trm_pipeline.train_trm_vm \
  --manifest data/trm_va_cache/manifest.jsonl \
  --output-dir artifacts/trm_vm \
  --device cuda \
  --grad-clip 1.0 \
  --amp \
  --log-interval 50
```

`TRM-As` 学習:

```bash
python -m trm_pipeline.train_trm_as \
  --manifest data/trm_va_cache/manifest.jsonl \
  --output-dir artifacts/trm_as \
  --device cuda \
  --grad-clip 1.0 \
  --amp \
  --log-interval 50
```

この bootstrap は「最終解」ではなく、まず

- analytic runtime の viability 監視を学べること
- analytic policy score を模倣できること

を確認する初期段階です。`TRM-Vm / TRM-As` の受け入れ基準は
[TRM_GRANULAR_ROLE_SPEC.md](/Users/yamaguchimitsuyuki/criticism_bot/TRM_GRANULAR_ROLE_SPEC.md)
を参照してください。

`TRM-Vm / TRM-As` の統合モード比較:

```bash
python -m trm_pipeline.compare_trm_va_modes \
  --seed-catalog data/lenia_official/animals2d_seeds.json \
  --module-manifest artifacts/modules_vm_as.json \
  --output-root artifacts/trm_va_mode_compare
```

これは `viability_mode` と `action_mode` の
`analytic / assistive / module_primary`
を総当たりで比較し、`comparison_summary.json` に
`best_mode_by_final_homeostasis` と
`best_mode_by_mean_homeostasis` を保存します。

複数 seed 集計:

```bash
python -m trm_pipeline.sweep_trm_va_modes \
  --seed-catalog data/lenia_official/animals2d_seeds.json \
  --module-manifest artifacts/modules_vm_as_run2.json \
  --output-root artifacts/trm_va_mode_sweep_run3 \
  --seed-start 20260318 \
  --num-seeds 3 \
  --steps 12 \
  --warmup-steps 4
```

これは seed ごとの最良 mode を集計し、`aggregate_summary.json` に
`counts_by_best_final_homeostasis` と
`counts_by_best_mean_homeostasis` を保存します。

## 9. Tests

まず `pytest` ベースの基盤を入れています。現時点の重点対象は `trm_pipeline/erie_runtime.py` と `trm_pipeline/compare_erie_runtime.py` です。

```bash
pytest
```

カバレッジ確認:

```bash
pytest --cov=trm_pipeline --cov-report=term-missing
```

## 10. VastAI / GPU Training

学習 script は `device / resume / grad-clip / AMP / epoch jsonl logging / best checkpoint` を持っています。

- latest checkpoint:
  - `trm_a.pt`
  - `trm_b.pt`
  - `trm_vm.pt`
  - `trm_as.pt`
- best checkpoint:
  - `trm_a_best.pt`
  - `trm_b_best.pt`
  - `trm_vm_best.pt`
  - `trm_as_best.pt`

例:

```bash
python -m trm_pipeline.train_trm_a \
  --manifest data/trm_rollouts/manifest.jsonl \
  --output-dir artifacts/trm_a \
  --objective variational \
  --device cuda \
  --grad-clip 1.0 \
  --amp \
  --log-interval 50
```

resume:

```bash
python -m trm_pipeline.train_trm_a \
  --manifest data/trm_rollouts/manifest.jsonl \
  --output-dir artifacts/trm_a \
  --objective variational \
  --device cuda \
  --resume artifacts/trm_a/trm_a.pt \
  --amp \
  --log-interval 50
```

長時間学習では、各 script が `*_epoch_log.jsonl` を逐次追記します。途中停止しても epoch 単位の評価値は残ります。

Docker build:

```bash
docker build -t criticism-bot-trm .
```

VastAI での一括実行:

```bash
./run_vast_pretrain.sh all
```

段階実行もできます。

```bash
./run_vast_pretrain.sh prepare_rollouts
./run_vast_pretrain.sh train_trm_a
./run_vast_pretrain.sh prepare_trm_b
./run_vast_pretrain.sh train_trm_b
./run_vast_pretrain.sh prepare_trm_va
./run_vast_pretrain.sh train_trm_vm
./run_vast_pretrain.sh train_trm_as
./run_vast_pretrain.sh write_manifest
./run_vast_pretrain.sh compare_modes
```

## 11. openFrameworks GLSL Visualization

ERIE runtime の `.npz` を openFrameworks 用の texture sequence に変換できます。

```bash
python3 -m trm_pipeline.export_erie_openframeworks_frames \
  --npz artifacts/lenia_runtime_check/erie_20260318_seed_000510.npz \
  --output-root artifacts/of_viewer_export_smoke
```

出力:

- `manifest.json`
- `frames/life_*.png`
- `frames/field_*.png`
- `frames/body_*.png`
- `frames/aura_*.png`

openFrameworks 側の viewer skeleton と GLSL shader は
[openframeworks/erie_life_viewer/README.md](/Users/yamaguchimitsuyuki/criticism_bot/openframeworks/erie_life_viewer/README.md)
を参照してください。

## 参考（公式）

- API overview: https://platform.claude.com/docs/en/api/overview
- Vision (image): https://docs.anthropic.com/en/docs/build-with-claude/vision
- Messages examples: https://docs.anthropic.com/en/api/messages-examples
- Python SDK: https://github.com/anthropics/anthropic-sdk-python
