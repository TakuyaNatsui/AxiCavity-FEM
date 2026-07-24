# AxiCavity-FEM v2 アーキテクチャ

ver2 のパッケージ設計と各層の責務を記述する。

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
- GUI はサブプロセスで CLI を呼び出す（または直接 import 呼出）

詳細は別ドキュメント:
- [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md): ファイル別コードマップ・規約と落とし穴・拡張レシピ（開発はまずこれ）
- [HDF5_SCHEMA.md](HDF5_SCHEMA.md): HDF5 スキーマ仕様
- [BC_NAMING.md](BC_NAMING.md): 境界条件命名規約
- [README.md](README.md): docs 索引
