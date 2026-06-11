# ERIE Life Viewer

openFrameworks + GLSL で ERIE/Lenia runtime を生命体っぽく描画するための最小 viewer です。

## 最短セットアップ

- openFrameworks 0.12 以降
- `openframeworks/erie_life_viewer/config.make` の `OF_ROOT` がローカル環境に合っていること

```bash
./scripts/prepare_erie_life_viewer.sh \
  --npz artifacts/multispecies_runtime_check/erie_20260404_seed_000215.npz
```

これで以下を一括実行します。

- `.npz` から viewer 用 `manifest.json` / `frames/*.png` を export
- `bin/data/session/` へ同期
- `make` で viewer を build

起動:

```bash
make -C openframeworks/erie_life_viewer RunRelease
```

## Project Generator UI の補足

Project Generator を使う場合、`Project path` と `Template` は上部の `create / update` タブにあります。  
右上ギアの `settings` タブには `openFrameworks path` / `Platform` / `Version` しか表示されません。

期待する構成:

```text
bin/data/session/manifest.json
bin/data/session/frames/life_0000.png
bin/data/session/frames/field_0000.png
bin/data/session/frames/body_0000.png
bin/data/session/frames/aura_0000.png
```

## 操作

- `space`: 再生 / 停止
- `left/right`: フレーム移動
- `[` `]`: 再生速度変更
- `1` `2`: overlay 強度変更
- `3` `4`: pulse 強度変更
- `f`: フルスクリーン
- `r`: manifest 再読込

## 描画方針

- `life`: membrane / cytoplasm / nucleus の疑似生体色
- `field`: resource / hazard / shelter の環境色
- `body`: occupancy / boundary / permeability の身体色
- `aura`: uncertainty の光輪

Shader は学術図ではなく、
- 生物膜の縁の発光
- 体表の脈動
- 環境場による屈折
- uncertainty の霧

を前に出す構成にしています。
