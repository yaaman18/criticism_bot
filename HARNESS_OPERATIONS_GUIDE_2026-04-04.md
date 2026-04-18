# Harness Operations Guide 2026-04-04

## 1. 目的

この文書は、別セッションや別担当者がこのリポジトリの TRM ハーネスをそのまま運用するための実務ガイドである。

対象は次の 2 系統。

- dataset harness
  Lenia 由来 dataset の収集、gate、改訂、学習 handoff、学習後判定
- production runner
  production preset を使った preflight 付き実行と external GPU finalize

設計判断そのものは `HARNESS_DESIGN_SPEC_2026-03-31.md` を参照すること。
この文書は「何をどう叩けばよいか」に絞る。

## 2. 前提

作業ディレクトリ:

```bash
cd /Users/yamaguchimitsuyuki/criticism_bot
```

標準実行環境:

- repo 内 `.venv`
- `make` または `./.venv/bin/python`

最初に必ず実行すること:

```bash
./scripts/bootstrap_env.sh
make doctor
make test
```

期待値:

- `make doctor` が `ok`
- `make test` が全件 pass

`doctor` が落ちている環境では collection や training を信用しない。

## 3. 入口の整理

日常運用で使う入口は次の 4 つだけ覚えればよい。

1. `trm_pipeline.dataset_harness`
2. `trm_pipeline.production_runner`
3. `make dataset-smoke`
4. `make production-campaign` / `make production-finalize`

用途の切り分け:

- 試験的な contract や個別 dataset を回したい:
  `dataset_harness`
- production preset を前提に回したい:
  `production_runner`
- まず環境と最小フローを確認したい:
  `make dataset-smoke`

## 4. Dataset Harness の基本フロー

### 4.1 contract を作る

passive dataset:

```bash
./.venv/bin/python -m trm_pipeline.dataset_harness plan \
  --output-root artifacts/dataset_passive_stage1 \
  --dataset-name passive_stage1 \
  --dataset-kind passive_lenia_pretrain
```

agentic dataset:

```bash
./.venv/bin/python -m trm_pipeline.dataset_harness plan \
  --output-root artifacts/dataset_agentic_stage1 \
  --dataset-name agentic_stage1 \
  --dataset-kind agentic_bootstrap
```

production preset を使う場合:

```bash
./.venv/bin/python -m trm_pipeline.dataset_harness plan \
  --output-root artifacts/dataset_agentic_production \
  --dataset-name agentic_production \
  --dataset-kind agentic_bootstrap \
  --preset agentic_production
```

この時点で `contract.json` が作られる。

### 4.2 collection を実行する

```bash
./.venv/bin/python -m trm_pipeline.dataset_harness run \
  --contract artifacts/dataset_agentic_stage1/contract.json
```

出力確認:

- `dataset/manifest.jsonl`
- `dataset/summary.json`
- `dataset_eval_report.json`
- `collection_decision.json`
- `training_plan.json`

### 4.3 評価だけやり直す

```bash
./.venv/bin/python -m trm_pipeline.dataset_harness evaluate \
  --contract artifacts/dataset_agentic_stage1/contract.json
```

### 4.4 fail した contract を改訂する

```bash
./.venv/bin/python -m trm_pipeline.dataset_harness revise \
  --contract artifacts/dataset_agentic_stage1/contract.json
```

このコマンドは次を出す。

- `revision_search_report.json`
- `revised_contract.json`

`revision_search_report.json` に候補比較が出る。
次回 run の入力は通常 `revised_contract.json` を使う。

## 5. Campaign の使い方

`campaign` は collection 後の分岐までまとめて処理する。

### 5.1 local training まで一気に回す

```bash
./.venv/bin/python -m trm_pipeline.dataset_harness campaign \
  --contract artifacts/dataset_agentic_stage1/contract.json \
  --auto-handoff
```

分岐:

- dataset gate が pass:
  `training_plan -> training_run_report -> model_eval_report -> promotion_decision`
- dataset gate が fail:
  `revised_contract.json`

### 5.2 external GPU handoff まで作る

```bash
./.venv/bin/python -m trm_pipeline.dataset_harness campaign \
  --contract artifacts/dataset_agentic_production/contract.json \
  --external-gpu-provider vastai \
  --external-gpu-remote-root /workspace/criticism_bot
```

この場合は local training は走らず、次が出る。

- `gpu_handoff_report.json`
- `run_external_gpu.sh`

### 5.3 pass するまで反復する

1 回で `collect` / `promote` / `handoff ready` まで届かない前提なら、`campaign-until-pass` を使う。

```bash
./.venv/bin/python -m trm_pipeline.dataset_harness campaign-until-pass \
  --contract artifacts/dataset_agentic_production/contract.json \
  --max-rounds 3
```

成功条件:

- dataset only:
  `status=collect`
- `--auto-handoff`:
  `promotion_status=promote`
- `--external-gpu-provider`:
  `gpu_handoff_status=ready`

出力:

- `campaign_until_report.json`

この report には各 round の contract, status, revised_contract が残る。

## 6. Local Training の個別運用

collection 済み dataset に対して training を個別実行したい場合:

```bash
./.venv/bin/python -m trm_pipeline.dataset_harness handoff \
  --contract artifacts/dataset_agentic_stage1/contract.json
```

学習後判定だけやり直したい場合:

```bash
./.venv/bin/python -m trm_pipeline.dataset_harness post-train-eval \
  --contract artifacts/dataset_agentic_stage1/contract.json
```

この経路では `model_registry.jsonl` に checkpoint lineage が追記される。

## 7. External GPU 運用

### 7.1 handoff artifact を作る

```bash
./.venv/bin/python -m trm_pipeline.dataset_harness gpu-handoff \
  --contract artifacts/dataset_agentic_production/contract.json \
  --provider vastai \
  --remote-root /workspace/criticism_bot
```

確認するファイル:

- `gpu_handoff_report.json`
- `run_external_gpu.sh`

`gpu_handoff_report.json` には次が入る。

- remote で叩く training command
- sync するファイル一覧
- pull back する metrics / summary 一覧
- local に戻ってから実行する finalize command

### 7.2 remote host で学習を走らせる

想定:

- repo が remote 側にも同じ相対パス構造で置かれている
- contract と dataset artifact が同期済み

実行:

```bash
bash artifacts/dataset_agentic_production/run_external_gpu.sh
```

### 7.3 metrics をローカルへ戻す

最低限戻すもの:

- 各 `*_metrics_latest.json`
- 必要なら `summary.json`
- `training_plan.json`
- `contract.json`

戻す対象は `gpu_handoff_report.json` の `files_to_pull` に入っている。

### 7.4 finalize する

```bash
./.venv/bin/python -m trm_pipeline.dataset_harness finalize-external \
  --contract artifacts/dataset_agentic_production/contract.json \
  --status auto
```

`auto` の意味:

- 必要な metrics が揃っていれば `passed`
- 不足があれば `failed`
- training plan 自体が blocked なら `blocked`

出力:

- `external_finalize_report.json`
- `training_run_report.json`
- `model_eval_report.json`
- `promotion_decision.json`
- `artifacts/model_registry.jsonl`

## 8. Production Runner の使い方

production runner は production preset 用の薄い orchestration 層である。

### 8.1 production contract を切る

```bash
./.venv/bin/python -m trm_pipeline.production_runner plan \
  --preset agentic_production \
  --output-root artifacts/production_agentic_run
```

### 8.2 local で production campaign を回す

```bash
./.venv/bin/python -m trm_pipeline.production_runner run \
  --preset agentic_production \
  --output-root artifacts/production_agentic_run \
  --execution-target local \
  --auto-handoff
```

反復つきで回す場合:

```bash
./.venv/bin/python -m trm_pipeline.production_runner run \
  --preset agentic_production \
  --output-root artifacts/production_agentic_run \
  --execution-target local \
  --auto-handoff \
  --max-rounds 3
```

### 8.3 external GPU 用に production campaign を回す

```bash
./.venv/bin/python -m trm_pipeline.production_runner run \
  --preset agentic_production \
  --output-root artifacts/production_agentic_run \
  --execution-target gpu-handoff \
  --provider vastai \
  --remote-root /workspace/criticism_bot
```

wrapper:

```bash
make production-campaign
```

### 8.4 external GPU 後に finalize する

```bash
./.venv/bin/python -m trm_pipeline.production_runner finalize \
  --output-root artifacts/production_agentic_run
```

wrapper:

```bash
make production-finalize
```

## 9. 判定の見方

dataset 段階の主判定:

- `collection_decision.json`
  - `collect`
  - `revise`
  - `blocked`

model 段階の主判定:

- `promotion_decision.json`
  - `promote`
  - `hold`
  - `blocked`

解釈:

- `collect`:
  dataset は下流学習へ進めてよい
- `revise`:
  dataset はまだ canonical 扱いしない
- `promote`:
  学習済み module は次段評価へ進めてよい
- `hold`:
  学習済み module はまだ昇格させない
- `blocked`:
  環境または入力不足で run 自体を信頼しない

## 10. 重要 artifact 一覧

dataset 単位:

- `contract.json`
- `doctor_report.json`
- `dataset/manifest.jsonl`
- `dataset/summary.json`
- `dataset_eval_report.json`
- `collection_decision.json`
- `training_plan.json`
- `next_steps.json`
- `run_summary.json`

学習と昇格:

- `training_run_report.json`
- `model_eval_report.json`
- `promotion_decision.json`

改訂:

- `revision_search_report.json`
- `revised_contract.json`

lineage:

- `artifacts/dataset_registry.jsonl`
- `artifacts/model_registry.jsonl`

external GPU:

- `gpu_handoff_report.json`
- `run_external_gpu.sh`
- `external_finalize_report.json`

## 11. 推奨運用パターン

### 11.1 研究用の通常運用

1. `plan`
2. `campaign --auto-handoff`
3. `promotion_decision.json` を確認
4. `hold` なら `revise`

### 11.2 production dataset の整備

1. `production_runner run --execution-target gpu-handoff`
2. remote で `run_external_gpu.sh`
3. metrics を同期し戻す
4. `production_runner finalize`

### 11.3 別セッションで再開するとき

最初に見る順:

1. `contract.json`
2. `collection_decision.json`
3. `promotion_decision.json`
4. `next_steps.json`
5. `revision_search_report.json`
6. `artifacts/dataset_registry.jsonl`
7. `artifacts/model_registry.jsonl`

## 12. トラブル時の見方

### 12.1 `doctor` が blocked

対処:

```bash
./scripts/bootstrap_env.sh
make doctor
```

### 12.2 dataset が `revise`

見るもの:

- `dataset_eval_report.json`
- `collection_decision.json`
- `revision_search_report.json`

### 12.3 model が `hold`

見るもの:

- `model_eval_report.json`
- `promotion_decision.json`
- `artifacts/model_registry.jsonl`

### 12.4 external finalize が `failed`

まず `external_finalize_report.json` の `missing_outputs` を見る。
通常は remote 側の metrics 未同期が原因。

## 13. 最小コマンド集

smoke:

```bash
make dataset-smoke
```

production handoff:

```bash
make production-campaign
```

production finalize:

```bash
make production-finalize
```

manual finalize:

```bash
./.venv/bin/python -m trm_pipeline.dataset_harness finalize-external \
  --contract artifacts/dataset_agentic_production/contract.json
```

## 14. 運用上の結論

このハーネスは現時点で、TRM 研究の運用基盤として十分に使える。

別セッションで重要なのは次の 3 点だけである。

1. `.venv` を前提に実行する
2. `contract.json` と各 decision artifact を起点に状態を読む
3. external GPU を使った場合は最後に必ず `finalize-external` または `production_runner finalize` を実行する
