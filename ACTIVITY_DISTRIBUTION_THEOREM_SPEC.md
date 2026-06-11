# Activity Distribution Theorem (ADT) v1 / 五定義整合仕様

## 1. 目的

本仕様は、Lenia 群体の運用仕様を `滅・身・界・行・故` の五定義に整合させるための要件定義と実装仕様を定める。
主眼は「群体を不死化すること」ではなく、「滅に開かれたまま因果連鎖が継続すること」である。

## 2. 五定義の操作的定義

- `滅`:
有限存在として、個体・群体ともに消滅可能であること。
- `身`:
状態を持つ実体として、個体が質量・エネルギー・境界完全性を保持すること。
- `界`:
自己/外界の interface が明示され、観測と行為がその interface を経由すること。
- `行`:
環境依存で行為分布が変化し、挑戦/保守の配分が閉ループで更新されること。
- `故`:
痕跡が次世代の生成条件を制約し、世代間の因果連鎖が実装されること。

## 3. 活動分布定理 ADT-v1

群体効用を次で定義する。

`U(p) = E[DeltaGain(p)] - beta * Var[DeltaGain(p)] - gamma * P_ext(p; tau) + eta * I_trace`

- `p`: 挑戦個体比率
- `P_ext`: 消滅確率
- `I_trace`: 痕跡が次世代生成に与える因果影響量（故の指標）
- `beta, gamma, eta >= 0`: 重み

仮定として `E[DeltaGain]` の凹性、`P_ext` の凸増加、`I_trace` の単調増加を置く。
このとき最適 `p*` は通常 `0 < p* < 1` に現れ、全員挑戦/全員保守は非最適となる。

## 4. 要件定義（What）

### 4.1 機能要件

1. `滅` 要件:
`spawn/split/death removal` を実装し、消滅原因を分類記録できること。
2. `身` 要件:
`BodyState` が `energy/mass/age/boundary_integrity/alive` を持つこと。
3. `界` 要件:
観測と行為が `BoundaryInterface` を経由すること。外界直読は標準系で禁止。
4. `行` 要件:
各 step で `p_t` を更新し、挑戦/保守ロールを再割当できること。
5. `故` 要件:
`TraceField` を保持し、`spawn` 条件が痕跡依存で変化すること。
6. 群体消滅は許容するが、`Expected / Degenerate / Policy-forbidden` を区別して判定できること。

### 4.2 非機能要件

1. 同一 seed で再現可能（決定論）
2. 単体モード互換（`single_body_mode=true`）
3. NaN/Inf 非発生
4. 個体数増加時に計算量が破綻しない

### 4.3 受け入れ基準（五定義対応）

- `滅`: `min_survival_steps` を満たしつつ `max_early_extinction_rate` 未満
- `身`: `invalid_body_state_count == 0`（負質量・NaN・ID衝突なし）
- `界`: `boundary_interface_usage_rate == 1.0`（診断モード除く）
- `行`: `action_diversity` と `navigation_rate` が下限以上
- `故`: `trace_ablation_spawn_delta >= epsilon_trace`

`trace_ablation_spawn_delta` は「痕跡あり/なし」での spawn 統計差で測る。

### 4.4 完全性担保要件（Fail-Closed）

本仕様における「完全な群体行動」は、単一 run の見た目ではなく、以下 6 層が全て満たされた時のみ成立とみなす。

1. `仕様契約`:
   完了条件を acceptance 指標として固定する（`mean_p_t`, `mean_challenge_fraction`, `role_switch_events_total`, `mean_aux_nontrivial_action_count` を含む）。
2. `不変条件テスト`:
   NaN/Inf、ID 衝突、`death removal` 不整合、boundary 経路逸脱を単体テストで常時検出する。
3. `反証実験`:
   `trace ablation` と `no_action/random/closed_loop` 比較を必須とし、因果差が出ない場合は不合格。
4. `統計ゲート`:
   seed 群で判定し、平均値だけでなく CI 下限・劣化率・崩壊率で gate 判定する。
5. `再現性`:
   seed・設定・history・summary を保存し、同条件で同等結果を再生成できること。
6. `昇格制御`:
   1 つでも gate 未達なら `promote` せず `revise` に倒す（fail-closed）。

推奨 acceptance 追加キー（ハーネス実装対応）:
- `min_mean_p_t`, `max_mean_p_t`
- `min_mean_challenge_fraction`, `max_mean_challenge_fraction`
- `min_role_switch_events_total`
- `min_mean_aux_nontrivial_action_count`

## 5. 実装仕様（How）

### 5.1 状態モデル

- `WorldState`:
`bodies`, `external_state`, `boundary_state`, `trace_field`, `time`
- `BodyState`:
`id`, `energy`, `mass`, `age`, `alive`, `boundary_integrity`, `role`, `prediction_confidence`, `local_hazard`
- `TraceField`:
`intensity`, `decay_rate`, `provenance_body_id`, `timestamp`

### 5.2 コア更新則

- 挑戦比率:
`p_t = clip(sigmoid(a0 + a1 * buffer_t - a2 * hazard_t - a3 * boundary_damage_t), p_min, p_max)`
- ロール割当スコア:
`score_i = w_e * energy_i + w_c * prediction_confidence_i - w_h * local_hazard_i - w_b * boundary_damage_i`
- 生成確率（故を反映）:
`spawn_logit = s0 + s1 * trace_local + s2 * resource_local - s3 * hazard_local`

### 5.3 step 順序（固定）

1. `BoundaryInterface` 経由で観測
2. 内部状態更新（belief/energy/boundary）
3. `p_t` 推定とロール割当
4. 行為選択・適用（interface 経由）
5. `spawn` 判定（`TraceField` 依存）
6. `split` 判定
7. `death removal` 判定
8. `TraceField` 更新（堆積/減衰）
9. イベント記録（`spawn/split/death/role_switch/trace_update`）

### 5.4 消滅判定ポリシー

- `Expected extinction`: 許容
- `Degenerate extinction`: 不許容（数値不安定、行動固定化、実装不良）
- `Policy-forbidden extinction`: 評価フェーズで不許容

## 6. テスト仕様

### 6.1 単体テスト

- `compute_challenge_ratio()` 単調性
- `assign_roles()` の `K_t` 一致
- `boundary_interface_read/write` の経路監査
- `trace_conditioned_spawn()` の痕跡感度
- `death removal` の ID 一意性

### 6.2 統合テスト

- 1000 step 実行で `p_t`, `N_t`, `extinction_cause`, `trace_metrics` を記録
- seed 固定で再現一致
- `trace ablation` 実験で spawn 統計差が閾値以上

### 6.3 回帰テスト

- 単体モードの既存メトリクス劣化なし
- 既存ハーネス閾値（action diversity/intake/navigation）維持

## 7. 非目標（v1）

- 進化アルゴリズム全体の同時導入
- TRM 学習パイプライン全再設計
- 厳密数学証明

## 8. 実装順序（推奨）

1. `WorldState/BodyState` の五定義対応フィールド追加
2. `BoundaryInterface` 実装と経路監査テスト
3. `TraceField` 実装と `trace_conditioned_spawn()` テスト
4. ロール配分（`p_t`, `assign_roles`）と行為閉ループ接続
5. `spawn/split/death removal` を新 step 順序へ統合
6. ハーネスに `trace_ablation_spawn_delta` と五定義メトリクス追加
