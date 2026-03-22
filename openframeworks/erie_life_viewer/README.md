# ERIE Life Viewer

openFrameworks + GLSL で ERIE/Lenia runtime を生命体っぽく描画するための最小 viewer です。

## 前提

- openFrameworks 0.12 以降
- 先に runtime `.npz` から frame export を作ること

```bash
python3 -m trm_pipeline.export_erie_openframeworks_frames \
  --npz artifacts/lenia_runtime_check/erie_20260318_seed_000510.npz \
  --output-root artifacts/of_viewer_export_smoke
```

## openFrameworks project の作り方

1. Project Generator で新しい empty app をこのディレクトリに作る  
   `openframeworks/erie_life_viewer`
2. 生成後、`src/` と `bin/data/shaders/` をこのディレクトリの内容で上書きする
3. export 済みデータを `bin/data/session/` にコピーする  
   `manifest.json` と `frames/*.png`

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
