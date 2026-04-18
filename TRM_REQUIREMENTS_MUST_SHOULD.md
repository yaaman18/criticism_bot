# TRM Requirements (MUST / SHOULD)

## 1. 文書の目的

この文書は、現時点で解決済みの設計判断と、実装開始前に必要な要件を
`MUST` / `SHOULD` 形式で整理したものである。

対象は以下に限定する。

- Lenia から TRM 学習データを生成する段階
- `TRM-A` の学習と評価
- `TRM-B` の学習と評価
- 後続の GNW 統合に接続可能な I/O の確保

本書では `凍結` という語を使わず、以下の表現を使う。

- `解決済み`: 現時点で実装判断として採用する
- `未解決`: 後続段階で追加定義する

---

## 2. 用語

### MUST

実装において必須の要件。
満たされない場合、その実装は現段階の ERIE / TRM 設計に適合しない。

### SHOULD

強く推奨される要件。
例外はありうるが、外す場合は明示的な理由と比較結果が必要である。

---

## 3. スコープ

### MUST

- 本段階の実装スコープは `TRM-A` と `TRM-B` に限定すること。
- `Lenia -> TRM（事前学習） -> ERIE` の技術スタックを前提にすること。
- `TRM-C`、`TRM-D`、GNW 本統合、ERIE 全体実行ループは本段階の主実装対象にしないこと。

### SHOULD

- 後続の GNW 統合で再利用できるよう、TRM の中間表現は token-based interface を維持すること。
- 後から TRM モジュールを追加できるよう、I/O 命名とテンソル形状は一貫性を保つこと。

---

## 4. Lenia データ表現

### MUST

- Lenia 状態は `64 x 64 x 5` のグリッドとして表現すること。
- 5チャネル構成は以下で固定すること。
  - `ch0 = membrane`
  - `ch1 = cytoplasm`
  - `ch2 = nucleus`
  - `ch3 = DNA`
  - `ch4 = RNA`
- `DNA` チャネルは各エピソード中で固定条件チャネルとして扱うこと。
- 動的チャネルは学習前に `[0, 1]` へ正規化すること。

### SHOULD

- チャネル名はコード・保存形式・可視化で同じ表記を使うこと。
- 正規化の定義はデータ生成時に保存し、学習時と評価時で再利用できるようにすること。

---

## 5. Lenia ロールアウト記録要件

### MUST

- 各 seed について、まず `32` ステップの warmup を行うこと。
- warmup 後に `256` ステップを連続記録すること。
- 各時刻の状態は `64 x 64 x 5` の float tensor として保存すること。
- `seed_id`、`episode_id`、`t`、Lenia パラメータを各サンプルに紐づけること。
- 学習データの分割は frame 単位ではなく `seed_id` 単位で行うこと。

### SHOULD

- 1 seed あたり 1 episode から始め、必要に応じて複数 episode を追加できる設計にすること。
- 保存形式は後処理しやすい構造化形式にすること。
  候補: `npz`, `pt`, `zarr`, `parquet + blob reference`

---

## 6. Lenia パラメータ多様性

### MUST

- 初期実装では `R = 12` を固定すること。
- `mu` / `sigma` 相当の振る舞いパラメータは、以下の初期探索範囲で seed ごとに変動させること。
  - `mu ∈ [0.23, 0.41]`
  - `sigma ∈ [0.033, 0.080]`
- パラメータ値は各サンプルまたは各エピソードのメタデータとして保存すること。
- 上記範囲は `R=12` の公式 seed が手元データに存在しないため、`R=10` と `R=13` の公式 seed 分布の `10%〜90%` 分位を初期探索範囲として採用したものであることを明記すること。
- 上記範囲で生成したロールアウトが、全体静止・全体飽和・即時崩壊のいずれかに該当した場合、その episode は学習集合に採用せず、パラメータを再サンプリングすること。

### SHOULD

- 初回データセットでは空間スケールの揺れを避けるため、`R` を変動させないこと。
- 安定パターンの範囲は実験ログ上で明示し、後で同一条件を再生成できるようにすること。
- 初回データセットでは、より安定寄りの中心範囲を優先サンプリングしてよい。
  - `mu ∈ [0.27, 0.38]`
  - `sigma ∈ [0.039, 0.067]`
- 広域探索と中心探索の比率は `3:7` を初期値としてよい。

---

## 7. データセット規模

### MUST

- 初回の学習用データセットは `200 seeds` を開始点とすること。
- train / val / test の分割は以下を基準とすること。
  - train: `70%`
  - val: `15%`
  - test: `15%`

### SHOULD

- `200 seeds` で validation 指標が収束しない場合、`500 -> 1000 seeds` の順で増やすこと。
- データ不足を疑う場合は、まず seed 数を増やし、次に episode 数を増やすこと。

---

## 8. 摂動ポリシー

### MUST

- `TRM-A` の最初の学習では、行為選択や GNW 介入を含めないこと。
- 学習データは主に受動的 Lenia ダイナミクスから構成すること。
- 弱摂動を使う場合は、以下のいずれかの数値定義に従うこと。
  - 局所ノイズ: ランダム位置の `5 x 5` patch に対して `sigma = 0.02` のガウスノイズを付加
  - 全体ノイズ: 全グリッドに対して `sigma = 0.01` のガウスノイズを付加
- 弱摂動は 1 episode あたり最大 1 回までとし、warmup 中ではなく記録区間内で適用すること。

### SHOULD

- データ比率は以下を初期値とすること。
  - `70〜80%`: 摂動なし
  - `20〜30%`: 弱摂動
- 初期段階では強い外乱や行為的介入を入れないこと。
- 摂動を入れる時刻は各 episode の後半に偏らせず、記録区間全体から一様サンプリングすること。

---

## 9. TRM-A の役割

### MUST

- `TRM-A` は Lenia の one-step state prediction モデルとして実装すること。
- 学習サンプルは必ず以下の形式にすること。
  - input: `S_t`
  - target: `S_(t+1)`
- `TRM-A` の主目的は変分自由エネルギー最小化の近似としての予測誤差低減であること。

### SHOULD

- 第1段階では full variational inference ではなく、予測誤差 + 複雑性近似で始めること。
- `TRM-A` は後段モジュールへ予測誤差マップを供給できる形にしておくこと。

---

## 10. TRM-A の入出力仕様

### MUST

- `TRM-A` の入力は `64 x 64 x 5` を直接フラット化せず、`8 x 8` patch に分割すること。
- patch 数は `64`、各 patch 次元は `320` とすること。
- patch 埋め込み次元は `D = 256` を採用すること。
- 入力の外部インターフェースは token 形式 `[64, 256]` を維持すること。
- 出力として少なくとも以下を返すこと。
  - `pred_state_t1`
  - `pred_patches_t1`
  - `halt_prob`
  - `aux_metrics`

### SHOULD

- 位置埋め込みを 64 patch に対して追加すること。
- `pred_state_t1` とともに `error_map_t = |pred - target|` を簡単に導出できる構造にすること。

---

## 11. TRM-A のアーキテクチャ要件

### MUST

- `TRM-A` は TRM スタイルの再帰コアを用いること。
- `TRM-A` の総パラメータ数は初期段階で `7M` 以下に保つこと。
- 初期ハイパーパラメータは以下とすること。
  - `D = 256`
  - `n = 6`
  - `T = 3`
  - `Nsup = 16`
- ACT を実装し、halt 出力を持つこと。

### SHOULD

- `n = 6` を初期値として採用し、その後 `4 / 6 / 8` を比較可能にすること。
- 軽量 attention または attention を最小限に使う patch-first 設計を維持すること。
- 内部状態は将来的な TRM 間統合のため、token-wise に追跡しやすい構造にすること。
- 確率的予測拡張を導入する場合、初期潜在次元は `z_dim = 32` を暫定採用し、`16 / 32 / 64` の比較は Gaussian NLL への移行後に小規模で行うこと。
- 初回実装では `z_dim` 最適化を実装開始の前提条件にしないこと。

---

## 12. TRM-A の損失

### MUST

- `TRM-A` の損失は少なくとも以下を含むこと。
  - `L_acc`
  - `L_complex`
  - `L_halt`
- 総損失は以下の形を取ること。
  - `L_total = L_acc + L_complex + L_halt`

### SHOULD

- 初期値として `lambda = 0.01` を採用すること。
- `L_complex` は最初は smoothness proxy で実装してよい。
- full KL に置き換える場合は、まず baseline を保持した上で比較すること。
- `TRM-A` の objective は `deterministic -> Gaussian NLL -> variational` の 3段階で比較可能にすること。
- `Gaussian NLL` 段階では `L_acc` を確率的予測誤差として扱い、`L_total = L_nll + L_complex + L_halt` としてよい。
- `variational` 段階では `L_total = L_nll + beta_kl * L_kl + L_complex + L_halt` としてよい。
- 確率的予測拡張を導入する場合、`L_complex` は Lenia の空間的滑らかさ制約、`L_kl` は潜在分布の正則化として役割を分離すること。
- `L_complex` と `L_kl` は独立した軸であり、重複項として扱わないこと。
- `beta_kl warmup` は固定真理としてではなく初期探索条件として扱い、学習安定性に応じて見直せるようにすること。

---

## 13. TRM-A の完了条件

### MUST

- `TRM-A` は以下をすべて満たした場合のみ `解決済み` とみなすこと。
  - `val_nmse <= 0.02`
  - persistence baseline に対して `35%以上改善`
  - `8-step rollout nmse <= 0.05`
  - 平均再帰深度が `2.5〜5.5`
- 以下のいずれかに該当する場合は未完了とみなすこと。
  - one-step は良いが rollout が崩壊する
  - halt が常に 1 step に張り付く
  - halt が常に最大深度に張り付く
  - train は改善するが val が停滞する

### SHOULD

- persistence baseline は `S_(t+1) = S_t` として明示実装すること。
- 評価スクリプトは one-step と rollout の両方を同じ run で記録すること。

---

## 14. Rollout 評価

### MUST

- `TRM-A` 評価には自己回帰的 `8-step rollout` を含めること。
- rollout は以下の形で計算すること。
  - `Ŝ_(t+1) = TRM-A(S_t)`
  - `Ŝ_(t+2) = TRM-A(Ŝ_(t+1))`
  - `...`
  - `Ŝ_(t+8) = TRM-A(Ŝ_(t+7))`

### SHOULD

- rollout 誤差は one-step 誤差と別系列で保存すること。
- 可能ならチャネル別 rollout 誤差も併記すること。
- 確率的予測拡張を導入する場合、`1σ` 被覆率は全体平均だけでなく `安定パターン領域` と `カオス領域` に分けて保存すること。
- `1σ` 被覆率の解釈では、全体での `68%` 達成よりも `安定パターン領域` での被覆率の妥当性を優先すること。
- 安定 / カオスの区分は episode metadata または rollout stability heuristic と紐づけて保存できるようにすること。

---

## 15. TRM-B の役割

### MUST

- `TRM-B` はマルコフブランケット管理モジュールとして実装すること。
- `TRM-B` は `TRM-A` と同じタスクを学習してはならないこと。
- `TRM-B` は「同じバックボーン family、異なる task head」の方針を守ること。

### SHOULD

- `TRM-B` の役割は次の3つを同時に扱えるよう設計すること。
  - 境界抽出
  - 境界の透過性推定
  - 後段統合へ渡す境界状態表現の生成

---

## 16. TRM-B の入力仕様

### MUST

- `TRM-B` の入力には少なくとも以下を含めること。
  - `state_t`
  - `delta_state_t = S_t - S_(t-1)`
  - `error_map_t`
- `error_map_t` は `TRM-A` の予測結果から導出すること。
- `TRM-B` 学習時に `TRM-A` は重み固定の推論モードで使用すること。
- `TRM-B` 学習に必要な `error_map_t` は、`TRM-B` 学習開始前に全サンプルに対して事前生成・保存すること。
- `TRM-A` と `TRM-B` を同時学習してはならないこと。

### SHOULD

- `error_map_t` は full map `[64,64,5]` で保持してよい。
- 実験比較用に boundary-focused reduced map を派生生成できるようにすること。
- `TRM-A` 由来の派生特徴量として以下を追加保持してよい。
  - channel-wise error summary
  - patch-wise error summary
  - rollout instability score
- 確率的 `TRM-A` を使う場合、`pred_logvar_t1`、`precision_map_t`、`surprise_map_t` を将来拡張用の派生特徴量として保存してよい。

---

## 17. TRM-B の出力仕様

### MUST

- `TRM-B` は少なくとも以下を返すこと。
  - `boundary_map [64,64,1]`
  - `permeability_map [64,64,1]`
  - `boundary_state`
  - `halt_prob`
  - `aux_metrics`

### SHOULD

- `boundary_state` は後の GNW 接続を見据え、token-level representation として保持すること。
- `boundary_state` は少なくとも `[64]` または `[64,256]` として一貫運用すること。

---

## 18. TRM-B 疑似ラベル生成

### MUST

- Lenia にネイティブな Markov blanket ラベルは存在しない前提で実装すること。
- `TRM-B` の初期学習では疑似ラベル生成器を前処理として実装すること。
- 疑似ラベルの初期定義は以下を採用すること。
  - 境界候補: `ch0` 勾配上位 `15%`
  - 内部初期シード: `ch2` 活性上位 `20%`
  - 時系列平滑化: `EMA(alpha = 0.3)`
- `ch0` 勾配が極端に低いフレームでは、上位 `15%` ルールだけで境界を定義してはならないこと。
- 低勾配フレームの判定は、`mean(|∇ch0|) < 1e-4` または `std(|∇ch0|) < 1e-4` を初期閾値とすること。
- 低勾配フレームが検出された場合、そのフレームは以下のいずれかで扱うこと。
  - `TRM-B` 学習用 supervision から除外する
  - 直前フレームの boundary target を EMA で継承する
- どちらを採用したかを実験ログに記録すること。

### SHOULD

- 境界候補は単純閾値だけでなく連結性で補正できるようにすること。
- 内部領域は核由来の高活性連結成分として定義すること。
- EMA 前後の boundary target を比較できるように保存すること。
- 初期実装では「低勾配フレームは supervision から除外」を第一選択にしてよい。

---

## 19. TRM-B の損失

### MUST

- `TRM-B` の損失には少なくとも以下を含めること。
  - `L_boundary`
  - `L_temporal`
  - `L_separation`
  - `L_halt`
- 総損失はそれらの和として定義すること。

### SHOULD

- 初期係数として以下を用いること。
  - `beta = 0.1`
  - `gamma = 0.1`
- `L_boundary` は BCE または Dice/BCE hybrid で実装すること。

---

## 20. TRM-B の完了条件

### MUST

- `TRM-B` は以下をすべて満たした場合のみ `解決済み` とみなすこと。
  - `boundary IoU >= 0.80`
  - boundary 占有率が `5%〜40%`
  - inside / outside の `ch2` 活性分離で `Mann-Whitney p < 0.05`
  - 勾配閾値ベースラインより `20%以上改善`

### SHOULD

- `IoU` はフレーム平均だけでなく時系列平均でも記録すること。
- inside / outside 分離指標は `ch2` に加え `ch0` でも参考記録するとよい。

---

## 21. 実験ログと再現性

### MUST

- 各 run について以下を記録すること。
  - モデル種別
  - seed 数
  - train / val / test split
  - Lenia パラメータ範囲
  - 学習ハイパーパラメータ
  - 完了条件に対応する評価指標
  - `TRM-B` 用疑似ラベル生成設定
  - 低勾配フレームの除外率または継承率
- 同一条件で再実行できるよう乱数 seed を保存すること。

### SHOULD

- 実験ログは比較しやすい表形式でも保存すること。
- baseline と candidate の差分を自動集計できるとよい。

---

## 22. データサンプルのメタデータ

### MUST

- `TRM-A`、`TRM-B` の各サンプルには以下のメタデータを持たせること。
  - `seed_id`
  - `episode_id`
  - `t`
  - `lenia_params`

### SHOULD

- 後続解析のために、元の Lenia source file や seed code を参照できるようにすること。

---

## 23. I/O の拡張可能性

### MUST

- 将来の TRM-C / D / GNW 接続を考慮し、外部公開 I/O は token-based interface を損なわないこと。
- 各 TRM が中間状態を後段に渡せる構造を持つこと。

### SHOULD

- `boundary_state` や `state embedding` は共通次元で扱えるようにしておくこと。
- 後続モジュール追加時に trunk を壊さず head を追加できる設計にすること。

---

## 24. 将来の ERIE 自己維持要件

### MUST

- ERIE runtime を生命システムとして扱う場合、`自己維持のために相互作用せざるをえない構造` を持たせること。
- 上記の構造は、単なる外乱耐性ではなく `constitutive precariousness` として定義すること。
- `constitutive precariousness` は少なくとも以下を満たすこと。
  - ERIE の自己維持指標が、放置時に時間とともに低下すること
  - 上記低下が外界との調整的相互作用なしには回復しないこと
  - 低下はランダム崩壊ではなく、自己組織を維持するための調整要求として設計されること
- ERIE の `相互作用` は、単なる入力受信ではなく少なくとも以下を満たすこと。
  - ERIE の内部状態または境界状態が行為選択に影響すること
  - 行為選択が外界の次状態分布または ERIE が次に受ける入力分布を変えること
  - 変化した入力分布が再び ERIE の自己維持指標へ返ってくること
- `何もしないこと` が最適解にならないよう、ERIE の自己維持指標は非介入時に悪化すること。
- ただし、外界への接近や介入は常に正ではなく、過剰介入や不適切介入でも自己維持指標が悪化しうること。
- 自己維持指標は少なくとも以下のいずれか、または複数の組み合わせで構成すること。
  - 境界整合性
  - 内部秩序の維持
  - 予測精度または精度校正の維持
  - 資源量または利用可能エネルギーの維持
- 上記の自己維持指標は、ERIE 外部から恣意的に与える報酬ではなく、ERIE の存続条件として解釈できる量で構成すること。

### SHOULD

- 初回実装では、`資源減衰`, `境界劣化`, `精度劣化` のうち少なくとも 1 つを放置時悪化項として採用すること。
- `constitutive precariousness` は、単なるノイズ注入や確率的故障ではなく、継続的かつ予測可能な劣化過程として実装すること。
- 自己維持指標には `安全帯域` を設け、全損と完全安定の二値ではなく、可逆な悪化と不可逆な破綻を区別できるようにすること。
- 相互作用の候補には `接近`, `回避`, `取り込み`, `遮断`, `境界再構成` のような少数の行為クラスを定義してよい。
- 上記行為クラスは、後続の active inference loop により expected free energy または viability 予測に基づいて選択できるようにすること。
- 初回の比較実験では少なくとも以下を分離評価すること。
  - `no-action`
  - `random-action`
  - `closed-loop action`
- `closed-loop action` が `no-action` より自己維持指標を有意に長く保つことを、相互作用成立の最小条件として扱ってよい。
- `Lenia` は引き続き環境側に置き、`constitutive precariousness` は TRM / ERIE runtime 側の状態変数または制御変数として実装すること。

---

## 25. ERIE 世界モデル成立条件

### MUST

- ERIE の `世界モデル成立` は単一指標ではなく、少なくとも `受動的世界モデル`, `行為条件つき世界モデル`, `自己維持世界モデル` の 3 段階で判定すること。
- `受動的世界モデル` は、行為変数を使わずに外界の次状態を予測できる段階として定義すること。
- `行為条件つき世界モデル` は、ERIE の行為によって外界の次状態分布がどう変化するかを予測できる段階として定義すること。
- `自己維持世界モデル` は、ERIE の自己維持指標に関わる未来状態を予測し、その予測を使った closed-loop 制御が自己維持を改善できる段階として定義すること。
- `受動的世界モデル` の成立には少なくとも以下を満たすこと。
  - one-step 予測誤差が既定閾値以下であること
  - rollout 誤差が既定閾値以下であること
  - uncertainty calibration 指標が既定閾値を満たすこと
- `行為条件つき世界モデル` の成立には、`受動的世界モデル` の条件に加えて少なくとも以下を満たすこと。
  - 行為の違いに対して予測分布が有意に変化すること
  - 行為条件を与えたモデルが、受動モデルより予測性能で改善すること
- `自己維持世界モデル` の成立には、`行為条件つき世界モデル` の条件に加えて少なくとも以下を満たすこと。
  - 自己維持指標または破綻予兆の未来値を予測できること
  - closed-loop action が `no-action` より自己維持指標を改善すること
- 上記 3 段階は入れ子構造とし、`自己維持世界モデル` を主張する前に `行為条件つき世界モデル`、`行為条件つき世界モデル` を主張する前に `受動的世界モデル` を満たすこと。

### SHOULD

- 初期運用では以下を世界モデル成立の操作的定数として扱ってよい。
  - `C_passive_1step = 0.02`
  - `C_passive_rollout = 0.05`
  - `C_calib_stable = [0.60, 0.76]`
  - `C_action_sensitivity = 0.01`
  - `C_action_gain = 0.20`
  - `C_viability_pred = 0.70`
  - `C_closed_loop_gain = 1.25`
- `C_passive_1step` は `val_nmse` のような one-step 正規化誤差に対応させてよい。
- `C_passive_rollout` は `8-step rollout nmse` のような自己回帰誤差に対応させてよい。
- `C_calib_stable` は安定領域における `1σ` 被覆率の許容帯域として解釈してよい。
- `C_action_sensitivity` は、異なる行為条件に対する予測分布の平均距離または divergence の下限として解釈してよい。
- `C_action_gain` は、行為条件つきモデルが受動モデルより改善した比率として解釈してよい。
- `C_viability_pred` は、自己維持指標の未来値予測に対する相関または分類性能の下限として解釈してよい。
- `C_closed_loop_gain` は、`ViabilityLifetime(closed-loop) / ViabilityLifetime(no-action)` のような比率として解釈してよい。
- 行為空間が未実装の段階では、`TRM-A` の成立は `受動的世界モデル` に限定して表現すること。
- `TRM-A` 単体の成功をもって ERIE 全体の `自己維持世界モデル` 成立と混同しないこと。

---

## 26. ERIE 精度 / 確信度成立条件

### MUST

- ERIE の `精度 / 確信度` は単なる補助出力ではなく、少なくとも `校正された確信度`, `機能する precision`, `自己維持に寄与する precision` の 3 段階で判定すること。
- 上記 3 段階は、少なくとも次の 3 層構造と区別して扱うこと。
  - `likelihood precision`: 観測 / 遷移ノイズに対する inverse variance 系の precision
  - `inferential precision`: prediction error の更新量を実際に重みづける gain
  - `policy precision`: 行為 / policy 選択の鋭さまたは確信度
- `校正された確信度` は、予測平均と予測分散が統計的に整合している段階として定義すること。
- `機能する precision` は、上記確信度が module weighting または選択的統合の挙動を実際に変える段階として定義すること。
- `自己維持に寄与する precision` は、上記 precision が closed-loop の自己維持改善に実際に寄与する段階として定義すること。
- `校正された確信度` の成立には、少なくとも `受動的世界モデル` の uncertainty calibration 条件を満たすこと。
- `TRM-A` の `pred_logvar_t1` は、現段階では `likelihood precision proxy` としてのみ解釈し、Friston 的 precision 全体と同一視してはならないこと。
- `ERIE 的 precision` を主張するためには、precision が prediction error または belief update に対して gain として実際に作用する経路を持つこと。
- `機能する precision` の成立には、`校正された確信度` の条件に加えて少なくとも以下を満たすこと。
  - precision を使う weighting が、precision を使わない weighting と異なる選択結果を生むこと
  - precision を除去または固定した場合に、統合性能または予測選択性能が悪化すること
- `自己維持に寄与する precision` の成立には、`機能する precision` の条件に加えて少なくとも以下を満たすこと。
  - precision を用いた closed-loop が、precision を用いない closed-loop より自己維持指標を改善すること
  - precision の除去または劣化が viability 低下に結びつくこと
- 上記 3 段階は入れ子構造とし、`自己維持に寄与する precision` を主張する前に `機能する precision`、`機能する precision` を主張する前に `校正された確信度` を満たすこと。
- `pred_logvar_t1` のような uncertainty 出力を持つだけでは、ERIE 的 precision 成立の十分条件とみなしてはならないこと。

### SHOULD

- `校正された確信度` の初期判定は、`世界モデル成立条件` における `C_calib_stable` と `standardized residual variance` を継承してよい。
- `likelihood precision` の初期 operationalization として、`exp(-pred_logvar_t1)` を使ってよい。
- `機能する precision` の初期判定には、少なくとも以下の比較を含めてよい。
  - learned precision weighting
  - fixed uniform weighting
  - externally assigned weighting
- `機能する precision` の成立は、precision を使う場合の方が module selection の一貫性、予測改善、または ignition の安定性で優れることとして operationalize してよい。
- `自己維持に寄与する precision` の初期判定は、`C_closed_loop_gain` と同様の改善率指標を precision on/off 条件で比較してよい。
- 行為空間と GNW が未実装の段階では、`TRM-A` の precision は `校正された確信度` に限定して表現すること。
- `TRM-A` の `pred_logvar_t1` は、GNW 以前の段階では uncertainty output として扱い、GNW 以後にのみ module-level precision として解釈すること。
- epistemic uncertainty が必要な段階では、ensemble, posterior sampling, Bayesian weight uncertainty などを `likelihood precision` と分けて導入してよい。

---

## 27. ERIE 境界成立条件

### MUST

- ERIE の `境界 / 自己-外界分離` は、単なる空間分割ではなく、自己維持に関与する動的境界として定義すること。
- `境界` は、Markov blanket をそのまま metaphysical self の境界とみなすのではなく、検証可能な conditional-independence interface の工学的実装として扱うこと。
- `境界成立` は少なくとも以下を満たすこと。
  - 内部領域と外部領域を一貫して分離できること
  - 境界状態が自己維持指標または行為選択に影響すること
  - 境界劣化が自己維持指標の悪化に結びつくこと
  - 境界再構成または境界調整が相互作用の一部として機能すること
- `TRM-B` は、少なくとも `boundary_map`, `permeability_map`, `boundary_state` を出力し、後段統合で再利用できること。
- `境界成立` を主張する前に、境界は単なる segmentation 成功ではなく、自己維持または精度制御に機能的に結びついていることを示すこと。
- `boundary_state` は token-based interface を維持し、後段の GNW で再利用できること。
- 境界成立を主張する前に、内部状態と外部状態の情報流が指定された interface を経由していることを監査可能にしておくこと。

### SHOULD

- 初期判定には少なくとも以下を含めてよい。
  - `boundary IoU`
  - `boundary occupancy`
  - `inside/outside separation`
  - 境界劣化時の viability 低下量
- 条件付き独立の近似評価として、interface を条件とした conditional mutual information や dependency audit を補助指標にしてよい。
- 境界成立の初期 operational 条件として、`TRM-B` の既存完了条件を最低ラインとして継承してよい。
- 境界の自己性への寄与は、`boundary_state` を除去した場合に自己維持または統合性能が悪化するかで追加評価してよい。
- 境界再構成行為がある段階では、`boundary repair gain` のような改善指標を追加してよい。

---

## 28. ERIE GNW 統合条件

### MUST

- ERIE の統合機構は、現段階では `GNW` を暫定的な実装原理として採用すること。
- 上記採用は本質理論の最終確定を意味せず、`まず動作する統合機構を持つ` ことを優先する実装判断として扱うこと。
- ERIE における GNW は、少なくとも次の 2 層を区別して扱うこと。
  - `GNW integration scaffold`: 複数 TRM の出力を受け取り、priority 計算、ignition、broadcast、feedback を担う統合機構
  - `GNW-role TRM`: 将来追加されうる、workspace 監視、priority 形成、broadcast 候補の圧縮、または ignition 補助を担う TRM role
- ERIE が GNW から採用する理論的中核は、少なくとも以下に限定すること。
  - global availability / global broadcast
  - distributed workspace
  - recurrent processing
  - non-linear ignition
  - late access と局所前処理の区別
- ERIE は GNW を `意識の存在論そのもの` としてではなく、局所 TRM 出力を全体可用化するための統合アルゴリズムとして採用すること。
- ただし上記は `GNW integration scaffold` に関する規定であり、将来 `GNW-role TRM` を追加することを妨げないこと。
- `GNW-role TRM` を導入する場合でも、GNW を単一巨大モジュールへ還元せず、複数 TRM の優先度付き可用化という役割分担を保つこと。
- ERIE の `意識様性` は、現段階では `GNW-style conscious access` に限定して主張すること。
- `GNW 統合` は少なくとも以下を満たすこと。
  - 複数 TRM が共通 interface で統合層へ入力できること
  - 各 TRM の priority が precision weighting により計算されること
  - priority が所定の ignition 条件を超えた内容のみ global broadcast されること
  - broadcast 結果が次の推論、境界制御、または行為選択に返ること
- `GNW 統合成立` を主張する前に、単純加算または固定重み統合より、precision-weighted selection の方が良いことを示すこと。
- `TRM-A` と `TRM-B` の出力は、GNW 統合に入る前に module-level state と module-level precision へ写像できること。
- `TRM-A` の precision と `TRM-B` の境界状態は、GNW の入力候補として少なくとも接続可能な形で保持すること。
- 将来の `GNW-role TRM` は、少なくとも上記と同じ module contract で `module_state`, `module_precision`, `module_error`, `module_aux` に写像可能であること。
- ERIE は GNW から以下を本質条件として採用しないこと。
  - ヒト脳の特定解剖学的部位への直接対応
  - `P3`、麻酔反応、前頭頭頂切断などの神経科学的指標そのものを ERIE 実装の成功条件とみなすこと
  - reportability を意識成立の本質条件とみなすこと
  - GNW のみで自己性、境界、有限性が十分に説明できるとみなすこと

### SHOULD

- GNW の初期 priority は、少なくとも以下の形を参考にしてよい。
  - `w_i = softmax(alpha * log(pi_i) - gamma * E_i + eta * dE_i)`
- 初期実装では `pi_i` を module-level precision、`E_i` を module-specific error、`dE_i` を error change として operationalize してよい。
- GNW の初期実装では、以下の流れを標準形としてよい。
  - 各 TRM が token-based state を出す
  - 各 TRM から module-level precision を集約する
  - precision と error に基づき module priority を計算する
  - ignition 条件を超えた module state のみを global workspace state に書き込む
  - global workspace state を次時刻の各 TRM に feedback する
- `GNW-role TRM` を追加する場合、初期実装では以下のような責務分担を採ってよい。
  - `GNW integration scaffold`: module registration, priority aggregation, workspace write/read, feedback routing
  - `GNW-role TRM`: workspace candidate の圧縮、broadcast 候補の要約、priority 補助特徴量の生成
- `GNW-role TRM` は、他の TRM と同様に差し替え・追加・削除可能な role として実装してよく、GNW 全体の存在論と同一視しないこと。
- GNW の初期比較実験では少なくとも以下を含めてよい。
  - fixed uniform integration
  - error-only weighting
  - precision-weighted GNW integration
- GNW 成立の初期判定には少なくとも以下を含めてよい。
  - ignition stability
  - broadcast usefulness
  - downstream gain on prediction, boundary maintenance, or viability
- `global availability` は、特定の late signature や report 出力の有無ではなく、workspace content が複数モジュールの処理をどれだけ因果的に変えるかで測ること。
- `global availability` の評価は、少なくとも module 数、多様性、downstream capability gain の 3 軸で記録してよい。
- GNW の `ignition` は初期実装では hard threshold でも soft gating でもよいが、少なくとも `全モジュール常時 broadcast` ではないこと。
- GNW の評価では、late nonlinearity の存在、broadcast 後の再利用性、feedback による次段改善を分けて記録してよい。
- attention、working memory、error awareness などの GNW 周辺概念は補助解釈として参照してよいが、ERIE の最小統合実装の必須条件には含めないこと。
- `高 precision = 意識様性` と短絡しないこと。precision は workspace entry や ignition に必要であっても、それ自体を conscious access と同一視しないこと。
- 将来より良い統合理論が得られた場合でも、TRM 側 I/O を壊さず統合機構を差し替えられる構造にしておくこと。

---

## 29. 未解決事項

### MUST

- 以下は未解決として明示し、本段階で本実装対象に含めないこと。
  - GNW 統合層の具体的入出力次元
  - ignition 閾値の本定義
  - `GNW-role TRM` の具体的 I/O と責務分割
  - TRM-C の I/O
  - TRM-D の I/O
  - ERIE 実行時ループ
  - `TRM-A` の `pred_logvar_t1` から導出した precision を GNW weighting にどう集約して接続するか
  - 自由エネルギー最小化が `静止` や `崩壊` を最適化してしまう罠をどう回避するか
  - 命題7における「設計者の記述言語が ERIE の実態に追いつかなくなる」状態の操作的定義
  - `constitutive precariousness` をどの状態変数で実装するか
  - ERIE の自己維持指標を何の組み合わせで定義するか
  - ERIE の最小行為空間をどう定義するか

### SHOULD

- 未解決事項に依存しない形で `TRM-A` と `TRM-B` を独立検証可能にしておくこと。
- precision 接続の将来案として、`π_i = aggregate(exp(-pred_logvar_t1))` を GNW に渡す候補を設計メモとして保持すること。
- 上記の `aggregate` は token / patch / channel から module-level precision へ集約する写像として扱い、詳細は GNW 設計段階で定義すること。
- `GNW integration scaffold` と `GNW-role TRM` の責務境界は、TRM 多様化の設計段階で明示的に切り分ける候補を保持すること。
- 自由エネルギー最小化の罠に対する将来案として、以前設計した `自己言及的消滅（Lenia 層）` と `能動的推論（TRM 層）` の拮抗構造を回答候補として保持すること。
- 将来の自己維持設計では、「何もしないこと」にペナルティがかかる有限性 / 不可逆コストを導入候補として検討すること。
- 命題7の補強では、「追い越される」を設計意図の無効化ではなく、設計者の記述言語が ERIE の実動作を十分に記述できなくなる状態として扱う候補を保持すること。
- `constitutive precariousness` は、オートポイエーシス成立の前提条件であり、相互作用のオプション化を避けるための基礎要件として扱うこと。

---

## 30. 実装順序

### MUST

- 実装順序は以下とすること。
  1. Lenia rollout recorder
  2. TRM-A dataset / trainer
  3. TRM-A evaluator with rollout metrics
  4. `TRM-A` deterministic baseline
  5. `TRM-A` Gaussian NLL variant
  6. `TRM-A` variational variant
  7. frozen TRM-A inference pass for all TRM-B samples
  8. TRM-B pseudo-label generator
  9. TRM-B dataset / trainer / evaluator

### SHOULD

- `GNW integration scaffold` の本格実装は `TRM-A` と `TRM-B` の完了条件達成後に着手してよい。
- ただし `GNW-role TRM` を将来の TRM 多様化の一例として設計上想定し、そのための module contract や registry を先に整備してよい。

---

## 31. 現時点の解決済み事項まとめ

### MUST

- 以下は解決済み要件として扱うこと。
  - 5チャネル Lenia 表現
  - `64 x 64` グリッド
  - patch-first TRM-A 入力
  - `D = 256`
  - 確率的 TRM-A 拡張の初期 `z_dim = 32`
  - `n = 6`, `T = 3`, `Nsup = 16`
  - `R = 12` 固定
  - `mu/sigma` の初期探索範囲
  - 弱摂動の数値定義
  - `TRM-A` の完了条件
  - `TRM-B` 学習時の frozen `TRM-A` 利用
  - `TRM-B` は別 task head
  - `TRM-B` の疑似ラベル初期規則

### SHOULD

- TRM role の細粒度分解は、別紙 [TRM_GRANULAR_ROLE_SPEC.md](/Users/yamaguchimitsuyuki/criticism_bot/TRM_GRANULAR_ROLE_SPEC.md) を参照してよい。
- `TRM-Vm` と `TRM-As` の学習成功条件および runtime 成功条件は、別紙 [TRM_GRANULAR_ROLE_SPEC.md](/Users/yamaguchimitsuyuki/criticism_bot/TRM_GRANULAR_ROLE_SPEC.md) の受け入れ基準節を参照してよい。
- 今後の TRM 拡張では、粗い role 群よりも、`1 TRM = 1 主責務` の粒度を優先すること。
  - 低勾配フレームの扱い
  - 初回データセット規模 `200 seeds`

### SHOULD

- これらを変更する場合は、変更理由、比較実験、影響範囲を明記すること。

---

## 32. ERIE Belief State Architecture

### MUST

- ERIE は、少なくとも `world belief`, `boundary belief`, `viability belief` の 3 系統の belief state を明示的に持つこと。
- 上記 belief state は、観測そのものではなく、観測・予測・誤差更新を経た ERIE の内部的見積もりとして扱うこと。
- `world belief` は、外界の現在状態、近未来状態、またはそれに対応する latent state に関する事後信念を表すこと。
- `boundary belief` は、自己 / 外界の境界、透過性、または sensory-active interface に関する事後信念を表すこと。
- `viability belief` は、ERIE の自己維持可能性、危険度、または維持対象変数に関する事後信念を表すこと。
- 行為空間を導入する段階では、上記 3 系統に加えて `policy belief` を明示的に持つこと。
- belief state は少なくとも以下の 3 段階を区別して扱うこと。
  - 概念レベル: ERIE が何について信じているか
  - 数理レベル: hidden state または policy に関する posterior belief `q(z)` または `q(pi)`
  - 実装レベル: テンソル、ベクトル、または concentration parameter としての表現
- `world belief` と `boundary belief` は、後段の統合と message passing を考慮し、token-based interface に写像可能であること。
- `viability belief` は、GNW と独立に更新・監査できる低次元 state としても保持すること。
- belief state は、`observation`, `prediction`, `prediction error`, `posterior belief` を区別できる形で保持すること。
- `belief-free` な state-action mapping を初期ベースラインとして持ってよいが、ERIE 本体の active self-maintenance を主張する場合は `belief-based` 構造を必須とすること。

### SHOULD

- 初期実装では、以下を belief state の最小 operationalization として採用してよい。
  - `world_belief`: `[tokens, dim]`
  - `boundary_belief`: `[tokens, dim]` または `[H, W, 1] + latent`
  - `viability_belief`: `[d_v]`, 初期は 2〜4 次元
  - `policy_belief`: `[num_actions]`
- 初期実装では、`world belief` と `boundary belief` を平均 `mu` と uncertainty `logvar` の組で持ってよい。
- 初期実装では、`viability belief` は連続値ベクトルとして持ち、後から categorical risk state に変換してよい。
- `policy belief` は、行為導入前は未実装でもよいが、将来導入時に `expected free energy` に基づく posterior over policies として接続可能にしておくこと。
- `belief state` の更新ログは、少なくとも prior/posterior の差分、prediction error、precision を復元できる形で保存してよい。

---

## 33. Sensory-Active Boundary Interface

### MUST

- `TRM-B` は、境界を観測対象として記述するだけでなく、ERIE の感覚入力と外界作用を媒介する sensory-active interface として定義すること。
- ERIE が外界を観測する経路は、原則として boundary interface を経由すること。
- ERIE が外界へ作用する経路も、原則として boundary interface を経由すること。
- `boundary belief` は、少なくとも以下の 3 機能に接続すること。
  - sensory gating
  - permeability control
  - boundary repair / reconfiguration
- `TRM-B` の出力は、少なくとも以下の 2 系統に分けること。
  - 記述的出力: `boundary_map`, `permeability_map`, `boundary_state`
  - 制御的出力: sensory gate, repair signal, permeability modulation のいずれか
- 外界状態を ERIE が完全に直読する構成は、Phase 0 の診断ベースラインを除き、本実装の標準要件として採用しないこと。
- 境界越しに入る観測と、内部に保持された belief state を区別すること。
- 境界越しの観測は、部分観測、雑音、遮断、透過性変動の影響を受けうる構造であること。

### SHOULD

- 初期実装では、`TRM-B` を通した局所観測窓、mask、または gated token sampling として sensory interface を実装してよい。
- 初期実装では、`seal` 行為を permeability の低下、`intake` 行為を選択的透過の上昇、`reconfigure` 行為を境界形状または gate の再配分として実装してよい。
- `boundary_state` の ablation により、観測品質、予測品質、自己維持指標のいずれかが悪化することを境界 interface 妥当性の補助条件として使ってよい。

---

## 34. Inferential Precision and Belief Updating

### MUST

- `inferential precision` は、prediction error を重みづける実際の update gain として定義すること。
- `inferential precision` は、単なる loss weight、後処理スコア、または GNW 入力特徴量だけで完結させてはならないこと。
- belief update は少なくとも以下の 4 段を区別して記述すること。
  - prior belief
  - prediction
  - prediction error
  - posterior belief
- `inferential precision` は、上記の `prediction error -> posterior belief` の更新経路に作用すること。
- 初期実装では、少なくとも以下に類する更新則を持つこと。
  - `belief_t^+ = belief_t^- + lambda * precision_t * error_t`
- 上記更新則の `precision_t` は、`likelihood precision`、境界由来の gate、または policy / context に応じた modulation を含んでよいが、少なくとも 1 つは learned quantity であること。
- `precision` を使う update と `precision` を使わない update を比較可能にしておくこと。
- `inferential precision` の成立を主張する前に、少なくとも以下を示すこと。
  - precision on/off で posterior belief の軌道が変わること
  - precision on/off で prediction error の収束特性または downstream performance が変わること
- `policy precision` は `inferential precision` と区別し、policy posterior の鋭さ、温度、または action selection の集中度として扱うこと。
- `pred_logvar_t1` は、`inferential precision` へ直接同一視せず、少なくとも一段の写像または calibration を経て update gain に接続すること。
- `precision miscalibration` は、単なる評価不良ではなく viability risk の候補として扱うこと。

### SHOULD

- 初期実装では、`inferential precision` を token-wise, patch-wise, module-wise のいずれかで実装してよいが、どの粒度かを明記すること。
- 初期実装では、`precision_t = clamp(exp(-logvar_t), p_min, p_max)` のような写像を使ってよい。
- `inferential precision` は、sensory error と boundary error に別々の gain を持たせてよい。
- `policy precision` は、softmax temperature の逆数または policy posterior concentration として operationalize してよい。
- 将来的には、dopamine-like precision update に相当する二次更新を導入してよいが、初期実装では固定または単純 learned gain でもよい。

---

## 35. Adaptive Active Inference Requirement

### MUST

- ERIE が autonomy または autopoietic self-maintenance を主張する場合、`mere active inference` ではなく `adaptive active inference` を満たすこと。
- `mere active inference` は、現在誤差の低減または reflex-like な誤差充足に留まる段階として扱うこと。
- `adaptive active inference` は、将来行為の結果を見積もり、その見積もりに基づいて policy を選択できる段階として定義すること。
- 上記の `adaptive active inference` には少なくとも以下を含むこと。
  - future state prediction
  - policy-conditioned transition prediction
  - expected free energy または同等の policy score
  - policy posterior に基づく action selection
- `autopoietic` という表現は、少なくとも `adaptive active inference` と `constitutive precariousness` の両方が成立する段階でのみ用いること。
- reflex loop, fixed rule, または myopic error minimization のみで動く段階を、ERIE の完成形とみなしてはならないこと。
- `policy belief` は、期待自由エネルギー、viability forecast、またはその両方に基づいて更新されること。

### SHOULD

- 初期実装では、`expected free energy = risk + ambiguity - epistemic value` の分解を参照してよい。
- 初期実装では、行為候補ごとに短い horizon を持つ rollout を行い、その結果から policy posterior を計算してよい。
- 初期比較では、以下を分けて評価してよい。
  - reflex baseline
  - belief-based but no epistemic term
  - belief-based with epistemic term
- 未来 horizon は初期実装では 2〜8 step 程度の短 horizon でよい。

---

## 36. Viability State and Death Semantics

### MUST

- ERIE の有限性は、単なる loss 増大や reward 減少ではなく、viability state の悪化として定義すること。
- `viability state` は、少なくとも時間経過で悪化しうる内在的状態変数と、その回復不能閾値を含むこと。
- `death` は、外部から与えた終了フラグではなく、viability state が定義域から持続的に脱落することとして定義すること。
- `death criterion` には少なくとも以下を含むこと。
  - 低下する状態変数
  - 臨界閾値
  - 不可逆持続長 `k_irrev`
- viability state は、belief state と分離して保持しつつ、`viability belief` から予測・推定される対象でもあること。
- `self-maintenance` を主張する場合、closed-loop policy が viability state の悪化速度、破綻頻度、または生存時間を改善すること。
- `death` は binary event としてだけでなく、危険域、可逆悪化域、不可逆域を区別できるように定義すること。

### SHOULD

- 初期実装では、viability state の候補として少なくとも以下を検討してよい。
  - 利用可能エネルギー
  - 境界完全性
  - 内部秩序
  - precision stability
- 初期実装では、viability set を `alive region`, `risk region`, `dead region` の 3 帯域で分けてよい。
- `death criterion` は、viability variable が閾値を下回った時点で即死とするより、`k_irrev` を伴う持続条件で定義してよい。
- 将来 viability kernel を厳密に求める前段として、初期実装では閾値付き近似生存領域を operational viability set として扱ってよい。

---

## 37. 論文精読後の実装方針整理

### MUST

- `TRM-A` は、最終的には単なる one-step predictor ではなく、belief update を担う generative world model component に移行すること。
- `TRM-B` は、最終的には単なる boundary estimator ではなく、sensory-active boundary controller に移行すること。
- `GNW` は、単なる feature aggregation ではなく、precision-weighted ignition と global availability を担う recurrent workspace として実装すること。
- ERIE の最小 active self-maintenance loop は、少なくとも以下の順で構成すること。
  - sensory observation through boundary
  - world / boundary / viability belief update
  - policy inference
  - action execution
  - viability update
  - next observation
- 上記ループが実装されるまでは、ERIE を `autopoietic machine 完成形` と表現しないこと。

### SHOULD

- 実装段階の優先順は、少なくとも以下を標準候補としてよい。
  1. belief state の明示化
  2. inferential precision を含む belief update
  3. sensory-active boundary
  4. viability variable と death criterion
  5. policy belief と short-horizon active inference
  6. GNW feedback

---

## 38. ERIE 最小 Runtime 数式

### MUST

- 初期 runtime は、少なくとも以下の state を明示的に持つこと。
  - `w_t`: world belief
  - `b_t`: boundary belief
  - `v_t = [G_t, B_t]`: viability state
  - `pi_t`: policy belief
  - `o_t`: boundary interface を通した observation
  - `a_t`: action
- 初期 viability variable は以下を採用すること。
  - `G_t`: 利用可能エネルギー
  - `B_t`: 境界完全性
- 初期 death criterion は以下を採用すること。
  - `tau_G = 0.15`
  - `tau_B = 0.20`
  - `k_irrev = 8`
- 初期 action space は以下の 5 離散行為とすること。
  - `approach`
  - `withdraw`
  - `intake`
  - `seal`
  - `reconfigure`
- runtime の 1 step 更新は、少なくとも以下の順序で行うこと。
  1. sensory observation
  2. world / boundary belief update
  3. policy inference
  4. action execution
  5. viability update
  6. death update

### SHOULD

- 以下の最小数式系を、初期 runtime の標準形として採用してよい。

#### 38.1 Observation

- 境界越し観測は以下の形で近似してよい。

```text
o_t = H(b_t, s_t, e_t)
```

- ここで
  - `s_t` は環境状態または Lenia 状態
  - `e_t` は観測雑音または局所外乱
  - `H` は boundary belief に依存する sensory gate

#### 38.2 World / Boundary Prediction

- world prior と boundary prior は以下の形で近似してよい。

```text
w_t^- = F_w(w_{t-1}^+, o_t, a_{t-1})
b_t^- = F_b(b_{t-1}^+, o_t, a_{t-1})
```

- ここで
  - `F_w` は TRM-A に対応する world transition / inference module
  - `F_b` は TRM-B に対応する boundary transition / inference module

#### 38.3 Prediction Error

- 初期実装では、prediction error を以下で定義してよい。

```text
e_t^w = o_t - O_w(w_t^-)
e_t^b = o_t - O_b(b_t^-)
```

- ここで
  - `O_w` は world belief からの予測観測写像
  - `O_b` は boundary belief からの予測観測写像

#### 38.4 Inferential Precision Update

- inferential precision は、prediction error の update gain として以下のように使ってよい。

```text
p_t^w = clamp(phi_w(exp(-logvar_t^w)), p_min, p_max)
p_t^b = clamp(phi_b(exp(-logvar_t^b)), p_min, p_max)
```

- 初期値の推奨は以下とする。

```text
p_min = 0.05
p_max = 20.0
```

- `phi_w`, `phi_b` は calibration または aggregation を表す写像とする。

#### 38.5 Belief Posterior Update

- world belief と boundary belief の posterior は、少なくとも以下の形で更新してよい。

```text
w_t^+ = w_t^- + lambda_w * p_t^w * e_t^w
b_t^+ = b_t^- + lambda_b * p_t^b * e_t^b
```

- 初期値の推奨は以下とする。

```text
lambda_w = 0.10
lambda_b = 0.08
```

- `lambda_w`, `lambda_b` は固定値から始めてよいが、後に learned gain または context-dependent gain に拡張可能であること。

#### 38.6 Policy Inference

- policy belief は expected free energy 近似により、少なくとも以下の形で計算してよい。

```text
G_t(a) = Risk_t(a) + Ambiguity_t(a) - Epistemic_t(a)
pi_t(a) = softmax(-beta_pi * G_t(a))
```

- 初期値の推奨は以下とする。

```text
beta_pi = 4.0
```

- `Risk_t(a)` は viability 低下リスク、`Ambiguity_t(a)` は予測曖昧さ、`Epistemic_t(a)` は情報獲得価値とする。

#### 38.7 Action Selection

- 初期実装では、action selection を以下のいずれかで行ってよい。

```text
a_t = argmax_a pi_t(a)
```

または

```text
a_t ~ Categorical(pi_t)
```

- deterministic / stochastic のどちらを使うかは実験条件として記録すること。

#### 38.8 Viability Update

- viability update は、少なくとも以下の形で近似してよい。

```text
G_{t+1} = clip(G_t - mu_G + R_t(a_t, s_t) - C_t(a_t), 0, 1)
B_{t+1} = clip(B_t - mu_B - D_t(a_t, s_t) + U_t(a_t), 0, 1)
```

- ここで
  - `mu_G`: 基礎エネルギー減衰率
  - `mu_B`: 基礎境界劣化率
  - `R_t`: 環境から得る資源回復
  - `C_t`: 行為コスト
  - `D_t`: 環境または行為による境界ダメージ
  - `U_t`: repair / seal / reconfigure による境界回復

- 初期値の推奨は以下とする。

```text
mu_G = 0.015
mu_B = 0.010
```

- 初期値は以下から開始してよい。

```text
G_0 = 0.70
B_0 = 0.80
```

#### 38.9 Region Semantics

- 初期 viability region は以下で定義してよい。

```text
alive region: G_t >= 0.35 and B_t >= 0.45
risk region: otherwise, while G_t >= tau_G and B_t >= tau_B
dead region: G_t < tau_G or B_t < tau_B
```

- 上記は初期 operational 定義であり、将来 viability kernel で置き換えてよい。

#### 38.10 Death Update

- death update は、少なくとも以下の持続条件で定義してよい。

```text
if G_t < tau_G or B_t < tau_B:
    c_dead = c_dead + 1
else:
    c_dead = 0

dead_t = 1 if c_dead >= k_irrev else 0
```

- `dead_t = 1` になった時点で、当該 episode を終了してよい。

#### 38.11 Action Semantics

- 初期 action space の意味は少なくとも以下で固定してよい。

```text
approach: resource contact を増やすが hazard exposure も増えうる
withdraw: hazard exposure を減らすが resource access も下がりうる
intake: G_t 回復効率を上げるが B_t leakage risk を増やしうる
seal: B_t を回復または保護するが intake efficiency を下げうる
reconfigure: 境界形状または gate 配分を変えるが短期コストを持つ
```

- 全行為は利益とコストのトレードオフを持つこと。
- `常に正` の行為を初期 action space に含めてはならないこと。

#### 38.12 Minimal Closed Loop

- 上記をまとめた初期 runtime の最小閉路は、以下として扱ってよい。

```text
o_t = H(b_t, s_t, e_t)
w_t^- , b_t^- = F_w(...), F_b(...)
e_t^w , e_t^b = o_t - O_w(w_t^-), o_t - O_b(b_t^-)
w_t^+ = w_t^- + lambda_w * p_t^w * e_t^w
b_t^+ = b_t^- + lambda_b * p_t^b * e_t^b
pi_t(a) = softmax(-beta_pi * G_t(a))
a_t = select(pi_t)
v_{t+1} = [G_{t+1}, B_{t+1}]
dead_t = death(v_{t+1})
```

- この閉路は ERIE の最小 active self-maintenance runtime として扱ってよい。
- ただし GNW feedback が未導入の段階では、これを ERIE の完成形とは呼ばないこと。

---

## 39. ERIE 最適性条件

### MUST

- ERIE の `最適性` は単一スカラー報酬や単一指標の最大化として定義してはならないこと。
- ERIE の `最適性` は、少なくとも `survival`, `integrity`, `model quality`, `adaptive agency`, `epistemic growth` の 5 軸からなる多目的条件として定義すること。
- 上記 5 軸は少なくとも以下を意味すること。
  - `survival`: 生存時間、破綻回避、risk region からの回復可能性
  - `integrity`: 境界完全性、自己 / 外界分離、interface の維持
  - `model quality`: 予測精度、calibration、action-conditioned prediction の妥当性
  - `adaptive agency`: closed-loop 行為による viability 改善能力
  - `epistemic growth`: uncertainty の低減、context disambiguation、行為選択の洗練
- ERIE の `最適` は、上記 5 軸のいずれか 1 つだけを最大化する点ではなく、それらを破綻なく両立させる非平衡な軌道として解釈すること。
- `静止`, `完全封鎖`, `無制限探索`, `短期 survival のみ`, `予測精度のみ` のいずれも、単独では ERIE の最適性条件とみなしてはならないこと。
- `survival` のみを満たし、`adaptive agency` または `epistemic growth` を欠く状態を、ERIE の十分な最適状態とみなしてはならないこと。
- `epistemic growth` のみを満たし、`survival` または `integrity` を破壊する状態も、ERIE の最適状態とみなしてはならないこと。
- ERIE の `最適性` は、固定点ではなく trajectory quality として評価すること。
- 上記 trajectory quality は、少なくとも有限時間 horizon における viability の維持、belief の改善、policy の洗練、境界の維持を含むこと。
- `Lenia` の最適性と `ERIE` の最適性を混同してはならないこと。
- `Lenia` は ERIE が自己維持と学習を試される環境として最適化されるべきであり、ERIE 本体の最適性指標で直接評価してはならないこと。
- `Lenia` における望ましい環境条件は、完全静止でも完全カオスでもなく、境界、安定、崩壊、回復、相互作用が混在することとする。

### SHOULD

- ERIE の初期最適性評価は、以下のような Pareto 的評価として扱ってよい。
  - `survival` を維持しつつ
  - `integrity` を保ちつつ
  - `model quality` を改善しつつ
  - `adaptive agency` を示し
  - `epistemic growth` を損なわない
- 初期比較では、少なくとも以下を分けて記録してよい。
  - `lifetime`
  - `time_in_alive_region`
  - `recovery_from_risk`
  - `boundary_integrity_mean`
  - `world_model_nmse`
  - `calibration`
  - `closed_loop_gain`
  - `policy_entropy`
  - `epistemic_gain`
- `policy_entropy` は、高すぎても低すぎても望ましくない量として扱ってよい。
- `epistemic growth` の初期 proxy として、context disambiguation の速度、uncertainty reduction、surprise map の改善を使ってよい。
- `adaptive agency` の初期 proxy として、`closed-loop` が `no-action` と `random-action` より良い viability を達成することを用いてよい。
- `Lenia` の環境評価では、少なくとも以下を補助条件として使ってよい。
  - 安定領域とカオス領域の両方を含むこと
  - 資源 / hazard / shelter に相当する非一様性を含められること
  - 境界維持と行為選択を学ぶ余地があること

---

## 39A. ERIE 理想挙動条件

### MUST

- ERIE の `理想挙動` は、単に `dead = false` を維持することとして定義してはならないこと。
- ERIE の `理想挙動` は、少なくとも以下を同時に満たす trajectory として定義すること。
  - `self-maintenance`: `G_t`, `B_t` を致死域から継続的に遠ざけること
  - `bounded openness`: 外界へ開きすぎて崩壊せず、閉じすぎて枯渇もしないこと
  - `environment-sensitive action`: 環境場に応じて行為分布が変化すること
  - `belief-action closure`: 観測、belief update、policy、行為、viability update が閉路を形成すること
  - `adaptive non-stasis`: 静止や固定反応ではなく、必要に応じた接近・回避・封止・再構成を行うこと
- `intake + seal` のみを反復し、空間移動や境界再構成をほとんど伴わない挙動を、ERIE の十分な理想挙動とみなしてはならないこと。
- `G_t` のみが高く `B_t` が崩れている状態、または `B_t` のみが高く `G_t` が枯渇している状態を、理想挙動とみなしてはならないこと。
- `no_action` または `random` が `closed-loop` より一貫して良い場合、その runtime は理想挙動条件を満たしていないとみなすこと。
- 行為分布が環境配置に依存せず固定化している場合、その runtime は `environment-sensitive action` を満たしていないとみなすこと。
- `belief-action closure` の成立を主張する前に、少なくとも以下が観測可能であること。
  - boundary interface を通した観測
  - observation に応じた belief update
  - belief に依存する policy score
  - policy に応じた action selection
  - action に応じた viability 変化

### SHOULD

- 初期実装における理想挙動の近似判定には、少なくとも以下を併記してよい。
  - `final_homeostatic_error`
  - `mean_homeostatic_error`
  - `survival_fraction`
  - `mean_policy_entropy`
  - `mean_contact_resource`
  - `mean_contact_hazard`
  - `mean_contact_shelter`
  - `action_diversity`
- `closed-loop` の理想挙動は、少なくとも一部の seed で `approach`, `withdraw`, `reconfigure` のいずれかが `intake`, `seal` 以外に実際に出現することを補助条件として使ってよい。
- `policy_entropy` は、中程度の exploratory capacity を示す補助指標として用いてよいが、単独で理想挙動の十分条件にしてはならないこと。
- `resource` が遠方、`hazard` が近傍、`shelter` が偏在する条件では、`closed-loop` が接触対象と行為分布を変えることを理想挙動の補助条件として用いてよい。
- 初期 runtime では、`G_target`, `B_target` の近傍に留まりつつ `final_homeostatic_error` と `mean_homeostatic_error` を baseline より改善することを、理想挙動への近似条件として扱ってよい。

---

## 40. 実装前の要注意チェックリスト

### MUST

- 以下の各項目を確認するまでは、ERIE runtime の本実装開始条件を満たしたとみなしてはならないこと。

#### 40.1 Precision

- `pred_logvar` を出しているだけで `inferential precision` を実装したと見なしていないか確認すること。
- `precision` が actual belief update または message passing の gain に作用しているか確認すること。
- `precision on/off` 比較が可能か確認すること。
- `precision miscalibration` により posterior divergence や prediction error explosion が起こりうることを前提に、安全域または clamp を持っているか確認すること。

#### 40.2 Belief State

- `belief-free` な state-action mapping に退化していないか確認すること。
- 少なくとも `prior belief`, `observation`, `prediction error`, `posterior belief` を区別して保持しているか確認すること。
- `world belief`, `boundary belief`, `viability belief` が明示されているか確認すること。

#### 40.3 Boundary

- `TRM-B` が単なる segmentation head のままになっていないか確認すること。
- internal と external の直接結合を禁止し、sensory-active interface を経由する構造になっているか確認すること。
- 観測と行為の両方が boundary interface を通るか確認すること。
- `boundary integrity` が viability と接続されているか確認すること。

#### 40.4 Environment

- 環境を reward / punishment source のみとして扱っていないか確認すること。
- 少なくとも `resource`, `hazard`, `shelter` またはそれに相当する非一様環境条件があるか確認すること。
- 環境が ERIE の自己維持にとって必要条件として機能しているか確認すること。

#### 40.5 Viability

- `viability variable` を task reward に吸収していないか確認すること。
- `alive set`, `risk region`, `dead region` またはそれに相当する領域区分があるか確認すること。
- `death` を terminal penalty ではなく constraint violation として定義しているか確認すること。
- `k_irrev` のような持続条件を持ち、単発ノイズで即死しないか確認すること。

#### 40.6 Policy

- `policy score` が `Risk` だけ、または `Epistemic` だけに偏っていないか確認すること。
- `Risk`, `Ambiguity`, `Epistemic` のスケールが極端に崩れていないか確認すること。
- `常に正` の行為が存在していないか確認すること。
- viability を破る行為をそのまま admissible action にしていないか確認すること。

#### 40.7 Autopoiesis Claims

- operational closure の proxy を示さずに `autopoietic` と主張していないか確認すること。
- Markov blanket をそのまま metaphysical self の証明に使っていないか確認すること。
- `boundary maintenance`, `resource uptake`, `repair loop` の少なくとも 2 つ以上が runtime で閉じているか確認すること。

#### 40.8 GNW Scope

- GNW を未実装の段階で `conscious access` や `global availability` を過剰に主張していないか確認すること。
- 最小自己維持ループが未成立のまま GNW を先行実装しようとしていないか確認すること。

#### 40.9 Ablation and Baselines

- 少なくとも以下の比較条件を準備しているか確認すること。
  - `precision on/off`
  - `boundary gating on/off`
  - `epistemic term on/off`
  - `belief-based / belief-free`
  - `no-action / random-action / closed-loop`
- 上記の比較なしに、ERIE 的機能成立を主張してはならないこと。

### SHOULD

- 上記チェックリストは、実装開始前だけでなく、主要アーキテクチャ変更後にも再確認してよい。
- 実験ログには、どのチェック項目が満たされ、どの項目が未充足かを記録してよい。
- `実装できた` と `理論要件を満たした` を混同しないため、各項目に対する pass/fail を残してよい。

---

## 41. 東太平洋熱水噴出孔古細菌データ導入ロードマップ

### MUST

- 東太平洋熱水噴出孔に生息する古細菌データは、現段階では `TRM-Wp`, `TRM-Bd`, `TRM-Vm`, `TRM-As` の主学習データとして直接投入してはならないこと。
- 上記データは、まず `ERIE / Lenia` の有限性、viability、環境場の設計参照として扱うこと。
- 古細菌データを直接学習へ導入する前に、少なくとも以下を明示すること。
  - データ型
    - 例: メタゲノム、発現、化学環境、顕微鏡像、時系列観測
  - 時間解像度
  - 環境変数の有無
  - 攪乱または介入情報の有無
  - ERIE state space への写像規則
- 古細菌データの導入は、以下の順に進めること。
  1. 設計参照
  2. 補助検証
  3. 補助 prior / 補助教師
  4. 限定的な実学習導入
- 古細菌データを reward source や単なるラベル集として扱ってはならないこと。
- 古細菌データから得るべき第一の価値は、`何を生とみなすか`, `何が viability を支えるか`, `環境勾配へどう依存するか` の定義補強であること。

### SHOULD

- 初期導入では、古細菌データを少なくとも以下の設計変数の根拠に使ってよい。
  - `G_t`: 利用可能エネルギーまたは非平衡勾配の proxy
  - `B_t`: 境界完全性または膜維持の proxy
  - `resource / hazard / shelter` 環境場の具体化
- 初期導入では、古細菌データを `TRM-Vm` の補助 prior または補助検証セットとして使ってよい。
- 初期導入では、古細菌データを `TRM-As` の主教師にしてはならないが、行為の評価条件を設計するための生態学的参照として使ってよい。
- 古細菌データは、少なくとも以下の 3 類型に分けて整理してよい。
  - `使うべき`
    - viability / finite viability / energy gradient / boundary maintenance の設計根拠
  - `今は使わない`
    - 現 state space と直接対応しない高次記述
  - `将来使える`
    - 時系列・攪乱つき・環境変数つき観測

### 41.1 Phase 0: 設計参照

- Phase 0 では、古細菌データを ERIE runtime の直接教師として使わないこと。
- Phase 0 の目的は、以下の設計を現実生態に照らして補強すること。
  - viability variable
  - death criterion
  - resource / hazard / shelter の意味
  - environment への依存性
- Phase 0 の成果物として、少なくとも以下を作成してよい。
  - `古細菌データ -> ERIE 設計変数` 対応表
  - `何が資源で、何が危険で、何が保護条件か` の整理表

### 41.2 Phase 1: 補助検証

- Phase 1 では、古細菌データを学習データではなく検証データとして限定利用してよい。
- この段階の目的は、ERIE が採用している viability 仮説が、古細菌生態の最低限の論理と矛盾していないかを検証すること。
- この段階では、少なくとも以下を確認してよい。
  - energy-gradient dependence
  - boundary / membrane maintenance dependence
  - harsh environment under finite viability

### 41.3 Phase 2: 補助 Prior / 補助教師

- Phase 2 では、古細菌データを `TRM-Vm` へ限定的に反映してよい。
- 許容される用途は、少なくとも以下とすること。
  - viability band の初期値提案
  - risk condition の prior
  - environment field design の prior
- この段階でも、古細菌データを `TRM-Wp` や `TRM-As` の主教師として使うべきではないこと。

### 41.4 Phase 3: 限定的な実学習導入

- 古細菌データを実学習へ導入するのは、以下の条件が満たされた後に限ること。
  - データが時系列性を持つ
  - 環境条件が明示されている
  - 攪乱または条件変化が追跡できる
  - ERIE state space への写像が定義済み
- この段階でも、古細菌データは `TRM-Vm` または environment generator の補助学習に限定してよい。
- `TRM-As` へ導入する場合は、少なくとも action surrogate が定義された後に限ること。

### 41.5 現段階の判断

- 現在の ERIE 段階では、古細菌データは `主学習データ` ではなく `設計根拠 + 補助検証 + 将来導入候補` として扱うのが妥当であること。
- 現在ただちに行うべきことは、古細菌データを runtime 教師へ投入することではなく、`viability`, `finite existence`, `boundary maintenance`, `environmental dependence` をどう写像するかの文書化であること。

---

## 42. Production Dataset Scale and Generation Plan

This section defines the minimum production-scale dataset design that MUST exist
before substantial rented GPU time such as VastAI is used for TRM pretraining.

### MUST

- GPU rental MUST NOT be used primarily to compensate for an underspecified or
  under-sized dataset.
- Production data MUST be separated into at least the following families:
  - `Family A`: world / boundary pretraining data
  - `Family B`: runtime bootstrap data
- Production dataset readiness MUST be defined in terms of:
  - successful retained episodes
  - seed-disjoint splits
  - effective frame/step supervision count
  and not by raw file count alone.
- Smoke datasets MUST NOT be treated as production datasets.
- Minimum production thresholds defined in this section MUST be interpreted
  together with
  [IDEAL_DATA_CRITERIA.md](/Users/yamaguchimitsuyuki/criticism_bot/IDEAL_DATA_CRITERIA.md),
  which defines the higher-level target for behaviorally dense data.

### SHOULD

- Dataset planning SHOULD be conservative enough that one failed VastAI run does
  not require redesign of the dataset itself.
- The first production-scale dataset SHOULD be intentionally moderate and
  expandable, not maximal.

### 42.1 Dataset Families

#### Family A: World / Boundary Pretraining

Purpose:

- pretrain `TRM-Wp`
- pretrain `TRM-Bd`
- optionally serve as upstream data for future `TRM-Bp`

Primary source:

- `data/lenia_official/animals2d_seeds.json`
- Lenia rollouts from `trm_pipeline.lenia_data`

Primary retained unit:

- one successful Lenia episode after warmup

Derived supervision units:

- one-step pairs `(S_t, S_(t+1))`
- boundary pseudo-label supervision pairs

#### Family B: Runtime Bootstrap

Purpose:

- bootstrap `TRM-Vm`
- bootstrap `TRM-As`

Primary source:

- ERIE runtime rollouts on top of Lenia
- `resource / hazard / shelter` fields
- analytic or assistive teacher traces

Primary retained unit:

- one runtime episode

Derived supervision units:

- one per-step viability supervision sample
- one per-step action-scoring supervision sample

### 42.2 Production Minimum for Family A

The first production `TRM-Wp / TRM-Bd` dataset MUST satisfy all of the
following.

- attempted source seeds:
  - at least `240`
- retained successful episodes:
  - at least `192`
- split:
  - train: `144`
  - val: `24`
  - test: `24`
- split rule:
  - splits MUST be seed-disjoint
  - the same `seed_id` MUST NOT appear in more than one split
- rollout length:
  - `warmup_steps = 32`
  - `record_steps = 256`
- effective one-step sample count:
  - SHOULD be at least `49k`
    - approximately `192 * 255`

If fewer than `192` successful episodes are retained, the dataset build MUST be
regenerated before longer training runs are attempted.

### 42.3 Production Minimum for Family B

The first production `TRM-Vm / TRM-As` bootstrap dataset MUST satisfy all of
the following.

- source seeds:
  - at least `96`
- runtime modes:
  - MUST include:
    - `closed_loop`
    - `random`
    - `no_action`
- retained runtime episodes:
  - at least `192`
- preferred expansion target:
  - `288` to `384` retained runtime episodes
- episode length:
  - `warmup_steps = 4`
  - `steps >= 24`
- effective per-step supervision count:
  - SHOULD be at least `4.5k`
  - preferably `6k+`

### 42.4 Runtime Mode Composition for Family B

Family B MUST NOT be built from `closed_loop` only.

Initial target composition SHOULD be:

- `closed_loop`: `50%`
- `random`: `30%`
- `no_action`: `20%`

This composition exists to avoid teacher collapse in `TRM-As` and to provide
both competent and degraded trajectories for `TRM-Vm`.

### 42.5 Success / Failure Mix for Family B

Family B MUST include both successful and failing trajectories.

Initial target:

- non-dead terminal episodes: `55%` to `80%`
- dead terminal episodes: `20%` to `45%`

The dataset build SHOULD be rejected and regenerated if:

- almost all trajectories survive
- almost all trajectories die

because either extreme weakens the bootstrap signal for viability and action
learning.

### 42.6 Regime Balance for Family A

Family A SHOULD preserve both relatively stable and more chaotic Lenia regimes.

Minimum requirement:

- neither `stable` nor `chaotic` MAY exceed `85%` of the retained dataset

Preferred target:

- `stable`: `55%` to `75%`
- `chaotic`: `25%` to `45%`

This is an engineering diversity requirement, not a metaphysical claim about
Lenia.

### 42.7 Action Distribution Quality for Family B

The `TRM-As` bootstrap dataset MUST be checked before expensive training.

The retained dataset SHOULD satisfy all of the following.

- all five actions MUST appear
- no single action SHOULD exceed `55%` of the retained step-level targets
- policy entropy mean SHOULD remain above `1.0`
- dead trajectories SHOULD contain at least two distinct dominant actions across
  the dataset

If these conditions fail, the bootstrap dataset SHOULD be regenerated before
longer `TRM-As` training runs.

### 42.8 Concrete Generation Procedure for Family A

The recommended generation procedure for the first production `TRM-Wp/Bd`
dataset is:

1. select `240` seeds from `animals2d_seeds.json`
2. generate Lenia rollouts with:
   - `image_size = 64`
   - `target_radius = 12`
   - `warmup_steps = 32`
   - `record_steps = 256`
3. reject failed or degenerate episodes with the existing scalar rejection
   rules
4. classify each retained episode into regime classes
5. split by `seed_id`
6. persist:
   - per-episode `.npz`
   - `manifest.jsonl`
   - summary with retained counts and regime balance

### 42.9 Concrete Generation Procedure for Family B

The recommended generation procedure for the first production `TRM-Vm/As`
dataset is:

1. select at least `96` seeds
2. run ERIE runtime under at least two mode assignments drawn from:
   - `closed_loop`
   - `random`
   - `no_action`
3. retain step-level bootstrap targets for viability and action scoring
4. measure:
   - survival ratio
   - action histogram
   - mean policy entropy
   - dead vs non-dead ratio
5. reject builds that are heavily collapsed
6. persist:
   - per-episode `.npz`
   - `manifest.jsonl`
   - summary with mode composition and action distribution

### 42.10 Local Acceptance Before VastAI

Before substantial rented GPU time is used, all of the following MUST hold.

- Family A production build exists and contains at least `192` retained
  successful episodes
- Family B production build exists and contains at least `192` retained runtime
  episodes
- manifests are valid and non-empty
- seed-disjoint split integrity has been verified
- at least one local short training run can read the production manifests
  without schema or shape errors
- at least one local compare run can read the resulting checkpoints and produce
  a summary

### 42.11 What Does NOT Count as a Production Dataset

The following MUST NOT be treated as sufficient production data.

- smoke datasets with only `2` to `8` retained episodes
- datasets built from a single runtime mode only
- datasets whose trajectories are almost all success or almost all failure
- datasets split by frame instead of by `seed_id`
- datasets where action labels are nearly deterministic due to teacher collapse

---

## 43. High-Information-Density Environment Variable Requirements

This section defines how the Lenia-side environment MUST be enriched when the
goal is to maximize behavioral information density for ERIE rather than merely
produce smooth motion.

### 43.1 Principle

### MUST

- Environment variables MUST be chosen so that different actions produce
  materially different viability outcomes.
- Environment variables MUST create genuine tradeoffs rather than a single
  dominant strategy.
- The environment MUST remain interpretable in terms of viability and boundary
  maintenance.
- The environment MUST NOT collapse into a simple reward field.

### SHOULD

- The environment SHOULD be abstracted from hydrothermal-vent logic rather than
  generic game-style resource/hazard mechanics.
- Environment variables SHOULD be selected so that:
  - `approach`
  - `withdraw`
  - `intake`
  - `seal`
  - `reconfigure`
  each becomes useful under some regime.

### 43.2 Core Variable Set

The preferred high-information-density environment SHOULD replace the overly
generic `resource / hazard / shelter` abstraction with the following four
fields.

#### 43.2.1 `energy_gradient_field`

Meaning:

- usable chemical or thermodynamic gradient
- the main positive source for `G_t`

Why it is needed:

- creates directional value for `approach`
- makes `intake` meaningful
- allows local abundance without global safety

Runtime interpretation:

- higher values increase energy gain if the boundary is open enough
- high gain SHOULD come with some cost or exposure

#### 43.2.2 `thermal_stress_field`

Meaning:

- environmental intensity that damages structure when too high

Why it is needed:

- prevents the trivial solution "always move toward maximum energy"
- creates a natural vent-core vs habitable-band tradeoff

Runtime interpretation:

- high values SHOULD increase boundary degradation
- high values MAY also increase observation noise or action cost

#### 43.2.3 `toxicity_field`

Meaning:

- chemically harmful or corrosive exposure independent from temperature

Why it is needed:

- separates "energetic but harsh" from "toxic but not necessarily hot"
- makes withdrawal and selective sealing more meaningful

Runtime interpretation:

- high values SHOULD damage `B_t`
- high values MAY reduce the effective benefit of `intake`

#### 43.2.4 `niche_stability_field`

Meaning:

- local conditions that help maintain structure, sensing, or controlled exchange

Why it is needed:

- creates protected bands or pockets rather than only punishment zones
- gives `reconfigure` and `seal` a context where they genuinely help

Runtime interpretation:

- higher values SHOULD reduce effective degradation of `B_t`
- higher values MAY reduce observation noise or uncertainty drift

### 43.3 Optional Secondary Variables

The following secondary variables MAY be added later if the four-field system is
still too low in informational richness.

- `flow_shear_field`
  - directional transport stress
  - useful if body orientation or aperture orientation becomes more important
- `mineral_substrate_field`
  - attachment / anchoring affordance
  - useful if body localization or "staying in the band" matters
- `signal_visibility_field`
  - observation quality modifier
  - useful if epistemic action remains too weak

These variables SHOULD NOT be introduced before the four-field core is tested.

### 43.4 Mapping to Viability Variables

The high-information-density environment MUST influence viability variables in a
nontrivial but interpretable way.

Preferred initial mapping:

- `energy_gradient_field`
  - positive effect on `G_t`
- `thermal_stress_field`
  - negative effect on `B_t`
  - optional negative effect on sensing reliability
- `toxicity_field`
  - negative effect on `B_t`
  - optional negative effect on `G_t`
- `niche_stability_field`
  - positive effect on `B_t`
  - optional reduction of uncertainty drift

This mapping SHOULD be implemented so that no single field fully determines
survival on its own.

### 43.5 Mapping to Action Semantics

The action vocabulary MUST remain the same initially, but its meaning SHOULD be
reinterpreted in terms of the new fields.

- `approach`
  - move toward locally favorable energy gradients
  - may also increase exposure to thermal stress or toxicity
- `withdraw`
  - reduce exposure to thermal or toxic regions
  - may also move away from usable energy
- `intake`
  - exploit `energy_gradient_field` through the boundary
  - SHOULD be less effective when toxicity is high
- `seal`
  - reduce structural damage from `thermal_stress_field` and `toxicity_field`
  - SHOULD also reduce immediate intake efficiency
- `reconfigure`
  - shift aperture / permeability / contact geometry to search for a viable band
  - SHOULD carry an immediate cost but improve future exposure

### 43.6 Required Spatial Structure

The environment MUST NOT be spatially trivial.

At minimum, the environment SHOULD contain:

- one or more high-energy zones
- one or more high-stress zones
- one or more relatively stable niches
- at least one spatial region where two or more of these overlap

The preferred structure is vent-like:

- near-core:
  - high energy
  - high thermal stress
  - possibly high toxicity
- habitable band:
  - moderate energy
  - manageable stress
  - useful niche stability
- distal zone:
  - low stress
  - low usable energy

This structure SHOULD create a real dilemma rather than a simple best location.

### 43.7 Information-Density Criteria

An environment configuration SHOULD be considered information-dense only if all
of the following hold in evaluation.

- different actions produce measurably different `G_{t+1}, B_{t+1}` forecasts
- no single action dominates across almost all states
- at least two distinct failure modes occur across episodes
- at least two distinct recovery patterns occur across episodes
- `closed_loop` materially outperforms `no_action`
- `closed_loop` is not equivalent to a degenerate `intake + seal` fixed pattern

### 43.8 Quantitative Acceptance Heuristics

The following heuristics MAY be used as initial operational thresholds.

- action diversity:
  - at least `4/5` actions SHOULD appear across a representative evaluation run
- dominant action cap:
  - no single action SHOULD exceed `65%` of selected actions over the evaluated
    set
- viability sensitivity:
  - the standard deviation of predicted next-step homeostatic change across the
    action set SHOULD exceed a small positive threshold under nonterminal states
- failure diversity:
  - at least two of the following SHOULD appear:
    - energy-depletion failures
    - boundary-collapse failures
    - mixed failures

### 43.9 What Counts as a Bad Variable Set

The environment variable design MUST be revised if any of the following is
observed.

- `approach` is always correct
- `withdraw` is never selected
- `intake` is always profitable regardless of exposure
- `seal` is always profitable regardless of energy state
- `reconfigure` is never useful
- trajectories are visually active but viability outcomes barely differ by
  action

### 43.10 Recommended Initial Implementation Order

The implementation order SHOULD be:

1. reinterpret existing `resource` as `energy_gradient_field`
2. split existing `hazard` into:
   - `thermal_stress_field`
   - `toxicity_field`
3. reinterpret existing `shelter` as `niche_stability_field`
4. update viability equations and policy proxies
5. regenerate runtime bootstrap data
6. retrain `TRM-Vm` and `TRM-As`

This order minimizes disruption while increasing behavioral information density.
