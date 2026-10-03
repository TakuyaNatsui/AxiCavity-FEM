# AxiCavity-FEM アーキテクチャ（v3）

パッケージ設計と各層の責務を記述する（計算コアは v2 系のまま。GUI は v3 で PySide6 に置換）。

## レイヤ構成

```
cli / gui              ← ユーザ入口
  ↓
reports                ← 可視化・レポート（純表示層）
  ↓
fem_tm0 / fem_hom      ← ソルバ専用層
  ↓
physics_core           ← 規約のセーフティ層（PBC 符号・P_flow 公式の一元管理）
  ↓
shared                 ← 数学・I/O・規約の共通基盤
```

import は上から下への一方向のみ（循環なし）。

## 各層の責務

### shared
- 形状関数・面積座標・ガウス求積（DOF に依存しない数学）
- Gmsh メッシュ I/O（境界条件分類含む）
- HDF5 統一スキーマ Read/Write
- CLI 共通引数定義

### physics_core
- 時間規約 e^{+jωt}
- PBC 位相符号 `x_max = e^{-jθ} * x_min`（一元化、回帰テストあり）
- P_flow ラインインテグラル（一元化、回帰テストあり）
- 規格化規約

### fem_tm0 / fem_hom
- それぞれの DOF（H_φ スカラー節点 / Nédélec エッジ + 節点）に固有の処理
- 要素行列・全体組立・境界条件適用・固有値解析
- 場の再構成・パラメータ計算

### reports
- Matplotlib による可視化
- HTML レポート生成
- 電磁場マップ出力

### cli / gui
- ユーザ入口。reports/fem_tm0/fem_hom の公開 API を組み合わせて使う。
- v3 の GUI（PySide6）は内部でも層を分ける: `gui/core`（純 Python: 文書・スケッチ幾何・変換）←
  `gui/sketch`（planegcs）← `gui/app/store.py`（状態と Undo）← `gui/ui`（ウィジェット）。`gui/jobs` は
  gmsh・solve・post・report・export を**子プロセス**で実行する（`cli.cmd_*` を argv ごと呼ぶ）。
  `gui/io` / `gui/project` はプロジェクト `.axiproj` と結果履歴、`gui/batch.py` は GUI を使わない解析
  （`axicavity-fem-run` と Python の `Project`）。
- 入口は `gui/launcher.py` 1 つにまとまっている（GUI・コア CLI・バッチ・GUI の子プロセス `job`・`version`・`selftest`）。
  Windows 版（EXE）はこれをそのまま main にする。EXE は gmsh（GPL）を本体に入れないので、`.msh` の読込は
  `gui/mshlite`（gmsh の読込 API の純 Python 版。コアの `import gmsh` に差し込む）、生成は `gui/jobs/meshing.py` から
  別プロセスのメッシャ（`mesher/axicavity_mesh.py`、埋め込み Python + gmsh）。詳細は DEVELOPER_GUIDE の「Windows 版（EXE）」。

詳細は別ドキュメント:
- [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md): ファイル別コードマップ・規約と落とし穴・拡張レシピ（開発はまずこれ）
- [HDF5_SCHEMA.md](HDF5_SCHEMA.md): HDF5 スキーマ仕様
- [BC_NAMING.md](BC_NAMING.md): 境界条件命名規約
- [README.md](README.md): docs 索引
