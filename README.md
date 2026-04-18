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
./scripts/bootstrap_env.sh
```

手動で整える場合:

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
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
./.venv/bin/python anthropic_art_critic_chat.py
```

デフォルトでは、同じ `--db-path` を使う限り前回セッションを自動で再開します。
毎回新規セッションにしたい場合:

```bash
./.venv/bin/python anthropic_art_critic_chat.py --new-session
```

オプション例:

```bash
./.venv/bin/python anthropic_art_critic_chat.py \
  --model claude-sonnet-4-6 \
  --max-tokens 1600 \
  --db-path chat_memory.sqlite3 \
  --memory-top-k 3
```

特定セッションを再開する場合:

```bash
./.venv/bin/python anthropic_art_critic_chat.py --session-id 20260227-120000-abcd1234
```

SSL証明書エラーが出る環境では、明示的に以下を指定できます（通常は不要）:

```bash
./.venv/bin/python anthropic_art_critic_chat.py --allow-insecure-ssl
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
./.venv/bin/python export_chat_logs_to_md.py
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
./.venv/bin/python -m trm_pipeline.lenia_data \
  --seed-catalog data/lenia_official/animals2d_seeds.json \
  --output-root data/trm_rollouts \
  --num-seeds 200
```

TRM-A 学習:

```bash
./.venv/bin/python -m trm_pipeline.train_trm_a \
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
./.venv/bin/python -m trm_pipeline.prepare_trm_b_data \
  --manifest data/trm_rollouts/manifest.jsonl \
  --checkpoint artifacts/trm_a/trm_a.pt \
  --output-root data/trm_b_cache
```

この cache には既存の `error_map` に加えて、将来拡張用の `pred_logvar`、`pred_var`、`precision_map`、`surprise_map` も保存されます。

TRM-B 学習:

```bash
./.venv/bin/python -m trm_pipeline.train_trm_b \
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

## 8.4 Dataset Harness

データ収集自体を contract 付きで回すためのハーネスを
`trm_pipeline/dataset_harness.py` に追加しています。

受動的な `TRM-A / TRM-B` 用 Lenia rollout dataset:

```bash
./.venv/bin/python -m trm_pipeline.dataset_harness plan \
  --output-root artifacts/dataset_passive_stage1 \
  --dataset-name passive_stage1 \
  --dataset-kind passive_lenia_pretrain

./.venv/bin/python -m trm_pipeline.dataset_harness run \
  --contract artifacts/dataset_passive_stage1/contract.json
```

人工主体らしい振る舞いを持つ `TRM-Vm / TRM-As` bootstrap dataset:

```bash
./.venv/bin/python -m trm_pipeline.dataset_harness plan \
  --output-root artifacts/dataset_agentic_stage1 \
  --dataset-name agentic_stage1 \
  --dataset-kind agentic_bootstrap

./.venv/bin/python -m trm_pipeline.dataset_harness run \
  --contract artifacts/dataset_agentic_stage1/contract.json
```

canonical contract を下書きする場合:

```bash
./.venv/bin/python -m trm_pipeline.dataset_harness plan \
  --output-root artifacts/dataset_passive_canonical \
  --dataset-name passive_canonical \
  --dataset-kind passive_lenia_pretrain \
  --preset passive_canonical

./.venv/bin/python -m trm_pipeline.dataset_harness plan \
  --output-root artifacts/dataset_agentic_canonical \
  --dataset-name agentic_canonical \
  --dataset-kind agentic_bootstrap \
  --preset agentic_canonical

./.venv/bin/python -m trm_pipeline.dataset_harness plan \
  --output-root artifacts/dataset_passive_production \
  --dataset-name passive_production \
  --dataset-kind passive_lenia_pretrain \
  --preset passive_production

./.venv/bin/python -m trm_pipeline.dataset_harness plan \
  --output-root artifacts/dataset_agentic_production \
  --dataset-name agentic_production \
  --dataset-kind agentic_bootstrap \
  --preset agentic_production
```

主な生成物:

- `dataset/manifest.jsonl`: 収集した episode 一覧
- `dataset/summary.json`: 収集結果の集計
- `dataset_eval_report.json`: coverage / entropy / family coverage の gate 判定
- `collection_decision.json`: この collection を下流学習に回してよいかの最終判断
- `training_plan.json`: gate を通ったときに次へ進める学習コマンド列
- `training_run_report.json`: handoff 実行結果
- `model_eval_report.json`: 学習後メトリクスに基づく module gate
- `promotion_decision.json`: 学習済み module を次段へ昇格させるかの判断
- `revision_search_report.json`: failed collection に対して試した次回 contract 候補と採用理由
- `revised_contract.json`: failed collection から導いた次回 contract
- `artifacts/dataset_registry.jsonl`: collection lineage の追跡ログ
- `artifacts/model_registry.jsonl`: dataset から生まれた各 checkpoint の lineage
- `gpu_handoff_report.json`: 外部 GPU 実行に渡す training plan の handoff artifact
- `run_external_gpu.sh`: 外部 GPU host 上でそのまま叩ける学習コマンド
- `external_finalize_report.json`: 外部 GPU 実行後に synced-back artifact から promotion と registry を確定した結果
- `next_steps.json`: 次回の収集で直すべき点

閉ループで回す場合:

```bash
./.venv/bin/python -m trm_pipeline.dataset_harness campaign \
  --contract artifacts/dataset_agentic_production/contract.json \
  --auto-handoff

./.venv/bin/python -m trm_pipeline.dataset_harness revise \
  --contract artifacts/dataset_passive_production/contract.json

./.venv/bin/python -m trm_pipeline.dataset_harness handoff \
  --contract artifacts/dataset_agentic_production/contract.json

./.venv/bin/python -m trm_pipeline.dataset_harness post-train-eval \
  --contract artifacts/dataset_agentic_production/contract.json

./.venv/bin/python -m trm_pipeline.dataset_harness gpu-handoff \
  --contract artifacts/dataset_agentic_production/contract.json \
  --provider vastai \
  --remote-root /workspace/criticism_bot

./.venv/bin/python -m trm_pipeline.dataset_harness finalize-external \
  --contract artifacts/dataset_agentic_production/contract.json \
  --status auto
```

production 向け agentic dataset では、`closed_loop / random / no_action` の mode mix、
success/failure mix、all-actions coverage、dead trajectory の dominant action diversity
まで dataset gate に含めています。さらに handoff 後は
`*_metrics_latest.json` を読んで `model_eval_report.json` と
`promotion_decision.json` を出します。`revise` は単一 heuristic ではなく、
複数候補の `revision_search_report.json` を残しながら次の contract を選びます。

理想的なデータの定義そのものは
[IDEAL_DATA_CRITERIA.md](/Users/yamaguchimitsuyuki/criticism_bot/IDEAL_DATA_CRITERIA.md)
を参照してください。ここでは harness と training の運用手順を中心に扱います。

production preset を doctor / preflight 付きで回す専用 runner:

```bash
./.venv/bin/python -m trm_pipeline.production_runner plan \
  --preset agentic_production \
  --output-root artifacts/production_agentic_plan

./.venv/bin/python -m trm_pipeline.production_runner run \
  --preset agentic_production \
  --output-root artifacts/production_agentic_run \
  --execution-target gpu-handoff \
  --provider vastai

./.venv/bin/python -m trm_pipeline.production_runner finalize \
  --output-root artifacts/production_agentic_run

make production-campaign
make production-finalize
```

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
./.venv/bin/python -m trm_pipeline.erie_runtime \
  --seed-catalog data/lenia_official/animals2d_seeds.json \
  --output-root artifacts/erie_runtime \
  --steps 128 \
  --lookahead-horizon 3 \
  --lookahead-discount 0.9
```

比較用に `policy mode` を切り替えられます。

```bash
./.venv/bin/python -m trm_pipeline.erie_runtime --policy-mode closed_loop
./.venv/bin/python -m trm_pipeline.erie_runtime --policy-mode random
./.venv/bin/python -m trm_pipeline.erie_runtime --policy-mode no_action
```

既存 checkpoint を使う場合:

```bash
./.venv/bin/python -m trm_pipeline.erie_runtime \
  --trm-a-checkpoint artifacts/trm_a_variational_smoke/trm_a.pt \
  --trm-b-checkpoint artifacts/trm_b_smoke/trm_b.pt
```

3 モード比較:

```bash
./.venv/bin/python -m trm_pipeline.compare_erie_runtime \
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
./.venv/bin/python -m trm_pipeline.prepare_trm_va_data \
  --seed-catalog data/lenia_official/animals2d_seeds.json \
  --output-root data/trm_va_cache \
  --episodes 16 \
  --steps 32
```

この canonical cache は同時に role ごとの view manifest も出します。

- `data/trm_va_cache/views/trm_wp.jsonl`
- `data/trm_va_cache/views/trm_bd.jsonl`
- `data/trm_va_cache/views/trm_bp.jsonl`
- `data/trm_va_cache/views/trm_vm.jsonl`
- `data/trm_va_cache/views/trm_as.jsonl`

対応関係は `data/trm_va_cache/views/summary.json` と `data/trm_va_cache/summary.json` の
`role_view_manifests` に入ります。

`dataset_harness` / `production_runner` はこの `role_view_manifests` を検出すると、
`TRM-Wp / Bd / Bp / Vm / As` の学習計画に自動で反映します。view manifest がない dataset は従来どおり canonical manifest を使います。

`TRM-Wp` の multispecies world-prediction 学習:

```bash
./.venv/bin/python -m trm_pipeline.train_trm_a \
  --manifest data/trm_va_cache/views/trm_wp.jsonl \
  --output-dir artifacts/trm_a_wp \
  --device cuda \
  --in-channels 18 \
  --out-channels 11 \
  --input-key wp_input_view \
  --target-key wp_target_observation \
  --baseline-key wp_observation
```

`TRM-Bd` の multispecies boundary-detection 学習:

```bash
./.venv/bin/python -m trm_pipeline.train_trm_b \
  --manifest data/trm_va_cache/views/trm_bd.jsonl \
  --output-dir artifacts/trm_b_bd \
  --device cuda \
  --boundary-in-channels-total 34 \
  --state-key bd_observation \
  --delta-key bd_delta_observation \
  --error-key bd_world_error \
  --sensor-gate-key bd_sensor_gate \
  --boundary-target-key bd_boundary_target \
  --permeability-target-key bd_permeability_target
```

`TRM-Vm` 学習:

```bash
./.venv/bin/python -m trm_pipeline.train_trm_vm \
  --manifest data/trm_va_cache/manifest.jsonl \
  --output-dir artifacts/trm_vm \
  --device cuda \
  --grad-clip 1.0 \
  --amp \
  --log-interval 50
```

`TRM-Bp` 学習:

```bash
./.venv/bin/python -m trm_pipeline.train_trm_bp \
  --manifest data/trm_va_cache/views/trm_bp.jsonl \
  --output-dir artifacts/trm_bp \
  --device cuda \
  --grad-clip 1.0 \
  --amp \
  --log-interval 50
```

`TRM-As` 学習:

```bash
./.venv/bin/python -m trm_pipeline.train_trm_as \
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
./.venv/bin/python -m trm_pipeline.compare_trm_va_modes \
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
./.venv/bin/python -m trm_pipeline.sweep_trm_va_modes \
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

2026-03-25 時点の実験メモ:

- `TRM-As` の family-diversified baseline
  [artifacts/trm_va_mode_sweep_family_run1/aggregate_summary.json](/Users/yamaguchimitsuyuki/criticism_bot/artifacts/trm_va_mode_sweep_family_run1/aggregate_summary.json)
  では、`analytic__module_primary` が 5 seed 中 4 seed で最良でした。
- `stage 3`
  (`--target-band-weight 0.75 --target-g-overshoot-weight 1.0 --defensive-family-bias 2.0`)
  は問題 seed `20260326` の補正には有効でしたが、
  [artifacts/trm_va_mode_sweep_family_stage3/aggregate_summary.json](/Users/yamaguchimitsuyuki/criticism_bot/artifacts/trm_va_mode_sweep_family_stage3/aggregate_summary.json)
  では 5 seed 集計で baseline を上回りませんでした。
- `stage 2.5`
  (`--target-band-weight 0.75 --target-g-overshoot-weight 1.0 --defensive-family-bias 1.0`)
  は
  [artifacts/trm_va_mode_sweep_family_stage2p5/aggregate_summary.json](/Users/yamaguchimitsuyuki/criticism_bot/artifacts/trm_va_mode_sweep_family_stage2p5/aggregate_summary.json)
  で baseline と同じ `4/5` seed で `analytic__module_primary` を維持しつつ、
  `analytic__module_primary` の平均 `final_homeostatic_error` を baseline より下げました。
  現時点では、`TRM-As` の次の有力候補はこの `stage 2.5` です。
- quality gate を含む 3 回の最適化探索
  [artifacts/trm_as_gate_search_rounds.json](/Users/yamaguchimitsuyuki/criticism_bot/artifacts/trm_as_gate_search_rounds.json)
  では、
  `gate_r1` (`samples=12, distinct=2, dominant=0.85, entropy=1.00`),
  `gate_r2` (`samples=12, distinct=3, dominant=0.85, entropy=1.00`),
  `gate_r3` (`samples=14, distinct=2, dominant=0.84, entropy=1.02`)
  を比較しました。
  この中では `gate_r3` が最も良く、
  `analytic__module_primary` の平均 `final_homeostatic_error` は `0.2943`,
  平均 `mean_homeostatic_error` は `0.2394` でした。
  ただし、`stage 2.5` の `0.2784 / 0.2336` には届かず、
  最良 seed 数も `3/5` までに留まりました。
  したがって、quality gate 付き探索は有益でしたが、
  現時点では `stage 2.5` を置き換えるほどではありません。
- したがって、現時点では `TRM-As` の標準教師設定は family-diversified baseline を維持し、
  `stage 3` は問題 seed 向けの改善候補として保留します。
- 次に標準昇格を再検討するなら、第一候補は `stage 2.5`、
  高品質データ寄りの次点は `gate_r3` とします。
- `TRM-Vm` は同じ stage 3 分布で再学習しても
  [artifacts/trm_vm_family_stage3/trm_vm_metrics_latest.json](/Users/yamaguchimitsuyuki/criticism_bot/artifacts/trm_vm_family_stage3/trm_vm_metrics_latest.json)
  の通り MAE は改善しましたが、`val_viability_risk_auroc = 0.5` のため、
  まだ標準設定の昇格材料にはしません。

## 9. Tests

まず `pytest` ベースの基盤を入れています。現時点の重点対象は `trm_pipeline/erie_runtime.py` と `trm_pipeline/compare_erie_runtime.py` です。

```bash
make test
```

カバレッジ確認:

```bash
make test-cov
```

最小の end-to-end 実験を 1 本だけ流して artifact まで確認する場合:

```bash
make harness-smoke
```

データ収集側の smoke run を回す場合:

```bash
make dataset-smoke
```

## 9.5 Experiment Harness

長時間の反復実験を file-based artifact で回すための最小ハーネスを
`trm_pipeline/experiment_harness.py` に追加しています。

まず contract を作成します。

```bash
./.venv/bin/python -m trm_pipeline.experiment_harness plan \
  --output-root artifacts/harness_stage1 \
  --experiment-name vm_as_stage1
```

特定 family だけで contract を作る場合:

```bash
./.venv/bin/python -m trm_pipeline.experiment_harness plan \
  --output-root artifacts/harness_toxic_fragile \
  --experiment-name vm_as_toxic_fragile \
  --families toxic_band fragile_boundary
```

環境チェックだけ先に走らせる場合:

```bash
make doctor
```

contract に従って sweep と gate 判定まで実行する場合:

```bash
./.venv/bin/python -m trm_pipeline.experiment_harness run \
  --contract artifacts/harness_stage1/contract.json
```

gate fail 時に bounded な自動微調整ラウンドを回す場合:

```bash
./.venv/bin/python -m trm_pipeline.experiment_harness tune \
  --contract artifacts/harness_stage1/contract.json \
  --max-rounds 3
```

関連仕様:

- [TRM_AG_TUNING_REQUIREMENTS.md](/Users/yamaguchimitsuyuki/criticism_bot/TRM_AG_TUNING_REQUIREMENTS.md)
- [TRM_AG_TUNING_SPEC.md](/Users/yamaguchimitsuyuki/criticism_bot/TRM_AG_TUNING_SPEC.md)

主な生成物:

- `doctor_report.json`: Python / NumPy / PyTorch / import 経路の健全性
- `compare/<family>/aggregate_summary.json`: family 別 multi-seed sweep 集計
- `compare/aggregate_summary.json`: family index と promotion target
- `eval_report.json`: family-aware gate 判定と promotion 可否
- `promotion_decision.json`: 昇格判断を family ごとに潰した最終判定
- `next_steps.json`: 次ラウンドで何を直すかの短い handoff

CI でも `bootstrap -> doctor -> test -> harness-smoke` を回すので、
ローカルと同じ入口で実行環境と最小ハーネスの両方を壊さないようにしています。

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
./.venv/bin/python -m trm_pipeline.train_trm_a \
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
./.venv/bin/python -m trm_pipeline.train_trm_a \
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
./.venv/bin/python -m trm_pipeline.export_erie_openframeworks_frames \
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

## 12. Harness Retry Loop

dataset harness は、collection が `revise` のときに改訂 contract へ自動で乗り換えながら再試行できます。

```bash
./.venv/bin/python -m trm_pipeline.dataset_harness campaign-until-pass \
  --contract artifacts/dataset_agentic_production/contract.json \
  --max-rounds 3
```

成功条件:

- dataset only: `status=collect`
- `--auto-handoff`: `promotion_status=promote`
- `--external-gpu-provider`: `gpu_handoff_status=ready`

production runner 側でも同じ反復を使えます。

```bash
./.venv/bin/python -m trm_pipeline.production_runner run \
  --preset agentic_production \
  --output-root artifacts/production_agentic_run \
  --max-rounds 3
```

## 参考（公式）

- API overview: https://platform.claude.com/docs/en/api/overview
- Vision (image): https://docs.anthropic.com/en/docs/build-with-claude/vision
- Messages examples: https://docs.anthropic.com/en/api/messages-examples
- Python SDK: https://github.com/anthropics/anthropic-sdk-python
