# Harness Design Spec 2026-03-31

## 1. 目的

この文書は、2026-03-31 時点の本リポジトリにおけるハーネス設計の仕様と設計判断をまとめたものである。

本ハーネスの目的は次の 2 つである。

1. `Lenia -> dataset collection -> TRM training -> runtime evaluation` を再現可能な実験系として固定すること
2. 収集・学習・評価を単発スクリプトではなく、contract と gate を持つ反復可能な閉ループにすること

この設計は、研究コードを「その場で走ったら終わり」から「次の改善に接続できる系」へ変えることを狙っている。

## 2. 設計原則

ハーネス全体は以下の原則で設計する。

- `doctor first`
  実験や収集の前に実行環境を検査し、壊れた環境では blocked にする。
- `contract-driven`
  収集条件、評価条件、artifact 出力先を事前に JSON contract として固定する。
- `hard gate`
  単なる診断ではなく、昇格可否を `collect / revise / blocked` や promotion decision として明示する。
- `lineage`
  どの contract で、どの dataset / run を作り、どう判定されたかを registry へ残す。
- `closed loop`
  fail したら次回 contract を自動生成し、pass したら下流学習へ handoff する。

## 3. 対象範囲

現時点のハーネスは 2 系統を扱う。

### 3.1 実験ハーネス

対象:

- `TRM-Vm / TRM-As` の runtime mode 比較
- family-aware な acceptance
- promotion decision

主実装:

- `trm_pipeline/experiment_harness.py`

### 3.2 データ収集ハーネス

対象:

- Family A: 受動 Lenia rollout dataset
- Family B: ERIE runtime bootstrap dataset

主実装:

- `trm_pipeline/dataset_harness.py`
- `trm_pipeline/lenia_data.py`
- `trm_pipeline/prepare_trm_va_data.py`

## 4. システム構成

### 4.1 実行環境固定

入口:

- `scripts/bootstrap_env.sh`
- `Makefile`
- `.github/workflows/ci.yml`

役割:

- `.venv` を正準実行環境にする
- `pip`, `pytest`, `numpy`, `torch` の整合性を `doctor` で検査する
- smoke 実行を CI で強制する

### 4.2 Family A: Passive Lenia Harness

対象データ:

- `TRM-A / TRM-B` 用の受動 rollout

収集器:

- `trm_pipeline.lenia_data.generate_rollouts`

主な retained unit:

- warmup 後に成立した 1 episode

主な supervision 単位:

- one-step pair `(S_t, S_(t+1))`

### 4.3 Family B: Agentic Bootstrap Harness

対象データ:

- `TRM-Vm / TRM-As` 用の bootstrap rollout

収集器:

- `trm_pipeline.prepare_trm_va_data.prepare_trm_va_cache`

主な retained unit:

- 1 runtime episode

主な supervision 単位:

- per-step viability sample
- per-step action-score sample

## 5. Contract 仕様

すべての収集 run は contract から開始する。

contract は最低限以下を含む。

- `dataset_name`
- `dataset_kind`
- `purpose`
- `generator`
- `acceptance`
- `artifacts`

### 5.1 `dataset_kind`

値:

- `passive_lenia_pretrain`
- `agentic_bootstrap`

### 5.2 `generator`

Family A:

- `seed_catalog`
- `num_seeds`
- `warmup_steps`
- `record_steps`
- `image_size`
- `target_radius`
- `root_seed`

Family B:

- `seed_catalog`
- `episodes`
- `steps`
- `warmup_steps`
- `seed`
- `image_size`
- `target_radius`
- `target_band_weight`
- `target_g_overshoot_weight`
- `defensive_family_bias`
- `policy_mode_mix`
- quality filter 用閾値

### 5.3 `acceptance`

Family A では主に以下を gate に使う。

- retained successful episodes
- unique seed count
- split 別 episode 数
- seed-disjoint split
- effective one-step sample count
- perturbation coverage
- regime diversity
- stable / chaotic balance

Family B では主に以下を gate に使う。

- retained runtime episodes
- source seed count
- effective step sample count
- family coverage
- policy mode coverage
- runtime mode share
- success / failure mix
- all-actions coverage
- dominant action concentration
- dead trajectory dominant action diversity
- policy entropy
- recovery fraction
- stress-time defensive / exploit fraction

## 6. Preset 設計

dataset harness には 4 つの preset がある。

- `passive_canonical`
- `agentic_canonical`
- `passive_production`
- `agentic_production`

### 6.1 `passive_production`

基準:

- attempted seeds: `240`
- retained successful episodes: `192+`
- split: `144 / 24 / 24`
- `warmup_steps = 32`
- `record_steps = 256`
- effective one-step samples: `49k+`
- `stable` と `chaotic` のどちらも `85%` を超えない

### 6.2 `agentic_production`

基準:

- retained runtime episodes: `192+`
- source seeds: `96+`
- `warmup_steps = 4`
- `steps >= 24`
- effective step samples: `4.5k+`
- runtime mode: `closed_loop / random / no_action`
- 初期 target mix: `50 / 30 / 20`
- non-dead terminal episodes: `55%` から `80%`
- all five actions present
- no single action above `55%`
- dead trajectory dominant action diversity: `2+`

## 7. Artifact 仕様

各 run は最低限以下を出力する。

- `contract.json`
- `doctor_report.json`
- `dataset/manifest.jsonl`
- `dataset/summary.json`
- `dataset_eval_report.json`
- `collection_decision.json`
- `training_plan.json`
- `training_run_report.json`
- `model_eval_report.json`
- `promotion_decision.json`
- `revision_search_report.json`
- `gpu_handoff_report.json`
- `run_external_gpu.sh`
- `external_finalize_report.json`
- `revised_contract.json`
- `next_steps.json`
- `run_summary.json`
- `artifacts/dataset_registry.jsonl`
- `artifacts/model_registry.jsonl`

### 7.1 `collection_decision.json`

状態:

- `collect`
- `revise`
- `blocked`

役割:

- canonical dataset として扱ってよいかを決める

### 7.2 `training_plan.json`

役割:

- gate 通過後にどの学習コマンドを実行するかを固定する

### 7.3 `training_run_report.json`

役割:

- handoff 実行結果を残す
- 各 step の return code と出力末尾を保存する

### 7.4 `revised_contract.json`

役割:

- failed criteria を見て、次回 run 用の contract を自動生成する

### 7.5 `external_finalize_report.json`

役割:

- 外部 GPU 実行後に synced-back artifact を検査する
- `training_run_report.json` を確定する
- `model_eval_report.json` と `promotion_decision.json` を更新する
- `model_registry.jsonl` に remote 実行 lineage を追記する

## 8. 閉ループ運用

### 8.1 基本フロー

1. `bootstrap` で環境を固定する
2. `doctor` で blocked を先に落とす
3. `plan` で contract を生成する
4. `run` または `campaign` で収集する
5. `evaluate` で hard gate を計算する
6. `collect` なら `handoff` に進む
7. `external GPU` を使う場合は `gpu-handoff` を作る
8. remote host で `run_external_gpu.sh` を実行する
9. artifact を同期し戻した後に `finalize-external` で promotion と registry を確定する
10. `revise` なら `revised_contract` を次の入力にする

### 8.2 `campaign`

`campaign` は次の分岐を内包する。

- pass:
  `--auto-handoff` が有効なら学習まで自動実行する
- fail:
  `revised_contract.json` を出して次回条件へ接続する

## 9. Family B の mode mix 設計

Family B の production readiness では、`closed_loop` だけの dataset を禁止する。

そのため収集側では、1 本の dataset を mode 別 sub-run に分解して集約する。

- `dataset/by_mode/closed_loop`
- `dataset/by_mode/random`
- `dataset/by_mode/no_action`

その後、top-level の `manifest.jsonl` と `summary.json` に再統合する。

この集約 summary には少なくとも以下を保持する。

- `policy_mode_counts`
- `family_counts`
- `aggregate_action_counts`
- `aggregate_policy_entropy_mean`
- `aggregate_recovery_fraction_mean`
- `aggregate_stress_defensive_fraction_mean`
- `aggregate_stress_exploit_fraction_mean`
- `rejected_episodes`
- `attempted_episodes`

## 10. 改訂ロジック

`revised_contract` は失敗理由に応じて次回条件を自動更新する。

主な規則:

- retained 数不足:
  `num_seeds` または `episodes` を増やす
- passive regime 偏り:
  `root_seed` を進めて別サンプル構成を取る
- source seed coverage 不足:
  runtime seed を進める
- success/failure mix 偏り:
  `policy_mode_mix` を再配分する
- action collapse:
  `random` と `no_action` の比率を増やす
- stress-response 弱さ:
  `defensive_family_bias` を増やす
- shaping 強すぎ:
  shaping weight を少し下げる

現時点では単発 heuristic ではなく、候補 contract を複数生成して
`revision_search_report.json` に比較結果を残した上で最良候補を採用する。

## 11. Handoff 設計

### 11.1 Family A

下流 handoff:

1. `train_trm_a`
2. `prepare_trm_b_data`
3. `train_trm_b`

### 11.2 Family B

下流 handoff:

1. `train_trm_vm`
2. `train_trm_as`

local handoff では学習を実際に起動し、その結果を `training_run_report.json` に書く。
external GPU handoff では `gpu_handoff_report.json` と `run_external_gpu.sh` を出し、
artifact 同期後に `finalize-external` が `training_run_report.json` と
`promotion_decision.json` を確定する。

## 12. 現時点で満たしていること

現時点で、本ハーネスは次を満たしている。

- 環境自己診断
- contract-driven collection
- production readiness gate
- family-aware / mode-aware evaluation
- registry による lineage 追跡
- fail から revise への自動接続
- pass から training handoff への自動接続
- post-train eval と promotion decision の campaign / finalize 統合
- model registry による checkpoint lineage 追跡
- external GPU handoff と finalize
- smoke と CI による最低限の継続検証

## 13. まだ未完了の点

完成系に近づいたが、まだ次は残る。

- generator parameter search は未実装
- production contract を本番サイズで回した結果の正式承認はまだ未実施
- remote host 側の自動 sync / auto-resume は未実装
- dataset campaign と experiment harness の完全統合は未実装

## 14. 運用コマンド

例:

```bash
./scripts/bootstrap_env.sh
make doctor

./.venv/bin/python -m trm_pipeline.dataset_harness plan \
  --output-root artifacts/dataset_agentic_production \
  --dataset-name agentic_production \
  --dataset-kind agentic_bootstrap \
  --preset agentic_production

./.venv/bin/python -m trm_pipeline.dataset_harness campaign \
  --contract artifacts/dataset_agentic_production/contract.json \
  --auto-handoff
```

passive 側を改訂する場合:

```bash
./.venv/bin/python -m trm_pipeline.dataset_harness revise \
  --contract artifacts/dataset_passive_production/contract.json
```

external GPU を使う場合:

```bash
./.venv/bin/python -m trm_pipeline.dataset_harness gpu-handoff \
  --contract artifacts/dataset_agentic_production/contract.json \
  --provider vastai \
  --remote-root /workspace/criticism_bot

./.venv/bin/python -m trm_pipeline.dataset_harness finalize-external \
  --contract artifacts/dataset_agentic_production/contract.json \
  --status auto
```

## 15. 結論

本ハーネスは、研究用スクリプト群の周囲に「実行前診断」「contract」「hard gate」「lineage」「revise」「handoff」を追加することで、Lenia ベースの TRM 研究を前進可能な閉ループへ変えるための基盤である。

2026-03-31 時点では、土台ではなく、すでに production-ready dataset planning を支える第一版の完成系ハーネスとして扱ってよい段階に入っている。
