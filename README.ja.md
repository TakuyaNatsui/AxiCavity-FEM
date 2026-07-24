# AxiCavity-FEM

**軸対称 RF 空洞の共振モードを解く 2 次元有限要素法ソルバ。
形状作成・メッシュ生成・後処理・レポート作成まで一貫して行えます。**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-373%20passing-brightgreen.svg)](tests/)

English README: [README.md](README.md)

![5 セル加速管の TM0 モード](docs/images/tm0_accelerating_structure.png)

AxiCavity-FEM は回転体形状の (z, r) 半平面上でヘルムホルツ方程式を解き、共振周波数・
電磁場分布に加えて、設計で実際に使う工学パラメータ（Q 値、R/Q、実効加速電圧、蓄積
エネルギー、壁損失、電力流、群速度、減衰定数）を出力します。

加速に使う**軸対称 TM0 モード**と、ビーム不安定性の原因になる**高次方位角モード
（HOM, n ≥ 1）**の両方に対応し、定在波でも、セル間位相差をもつ進行波でも計算できます。

---

## 特徴

**物理**

- 軸対称 **TM0** モード（H<sub>φ</sub> 定式化・節点要素）— 定在波／進行波
- 任意の方位角次数 n ≥ 1 の**高次モード**（E 場定式化・辺要素）
- 任意の位相差を与える**周期境界条件**と、位相スキャン（`0:180:20`）による分散曲線
- **誘電体**: 領域ごとの比誘電率と、**誘電正接 tanδ** による誘電体損失
  （`1/Q = 1/Q_wall + 1/Q_diel`）
- 1 次・2 次三角形要素

**得られる量**

| 量 | |
|---|---|
| f, Q, Q_wall, Q_diel | 共振周波数と各種 Q 値 |
| R/Q, V_eff | シャントインピーダンスと実効加速電圧（指定した β の走行時間係数込み） |
| U, P_loss, P_diel | 蓄積エネルギー・壁損失・誘電体損失 |
| P_flow, v_g, α | ポート通過電力・群速度・減衰定数（進行波） |

**ワークフロー**

- **形状エディタ GUI** — 点・直線・円弧で輪郭を作成、線分ごとに境界条件を設定、
  多領域・穴あき形状、領域ごとの材料、ワンクリックでメッシュ生成
- **パラメトリック形状** — 変数（`a = 100`, `c = a + b`）を定義し座標欄に式を入力。
  変数を変えると形状が追従して再計算されます
- **Superfish `.af` の入出力** — 既存形状の取り込みと書き出し
- **出力**: HDF5（スキーマ文書化済み）、テキストサマリ、場マップ PNG、
  直線上・矩形領域・軸上の場データ（HDF5／テキスト）、自己完結 HTML レポート、
  進行波モードの GIF アニメーション
- **Result Viewer** — モードを切り替えて場を確認し、グラフ上をダブルクリックして
  その点の場の値を読み取れます

<!--
GUI スクリーンショットを撮影したらここに追加する:

| 形状エディタ | 結果ビューア |
|---|---|
| ![Multi-Region Editor](docs/images/gui_editor.png) | ![Result Viewer](docs/images/gui_result_viewer.png) |
-->

| 高次ダイポールモード (n = 1) | メッシュ・境界条件・誘電体界面 |
|---|---|
| ![HOM n=1](docs/images/hom_dipole_n1.png) | ![Mesh overview](docs/images/mesh_overview.png) |

---

## インストール

Python 3.10 以降が必要です。

```bash
git clone https://github.com/TakuyaNatsui/AxiCavity-FEM.git
cd AxiCavity-FEM
pip install -e .          # ソルバ + コマンドライン
pip install -e ".[gui]"   # GUI も使う場合（wxPython）
```

clone せずに直接入れることもできます。

```bash
pip install "git+https://github.com/TakuyaNatsui/AxiCavity-FEM.git"
```

Gmsh は pip の依存として入るので、別途インストールする必要はありません
（Linux では OpenGL ランタイム `libglu1-mesa` が必要になる場合があります）。

任意: `pip install pypardiso` を入れると疎行列分解が Intel MKL PARDISO に切り替わります。
**計算結果は変わりません**。自由度 2 万程度を超える大きなメッシュで数倍速くなります。

---

## クイックスタート

### コマンドライン

サンプル形状を [`samples/`](samples/) に同梱しています。次の例は半径 50 mm・
長さ 100 mm の円筒空洞（ピルボックス）です。

```bash
axicavity-fem solve  --type tm0 -m samples/cylinder100mm.msh --elem-order 2 --num-modes 6 -o pillbox.h5
axicavity-fem post   --type tm0 -i pillbox.h5 --cond 5.8e7
axicavity-fem report --type tm0 -i pillbox.h5 -o pillbox_report
```

`solve` でモードを求め、`post` で工学パラメータを追加し（この例は銅壁 σ = 5.8×10⁷ S/m）、
`report` で場マップ入りの HTML を書き出します。各ステップは HDF5 の隣にテキストサマリも
出力します。最低次モードは 2.294851 GHz となり、解析解 `j₀₁c/2πa` と小数 6 桁まで一致します。

最低次からではなく特定の帯域を探索したいときは次のようにします。

```bash
axicavity-fem solve --type tm0 -m samples/s-band_1cell.msh --num-modes 4 --target-freq 2.856 -o sband.h5
```

高次モードは方位角次数を並べて指定し、進行波はセル間位相差を与えます。

```bash
axicavity-fem solve --type hom -m samples/s-band_1cell.msh --az-order 0 1 2 -o hom.h5
axicavity-fem solve --type tm0 -m samples/s-band_1cell.msh -p 120 -o traveling.h5
```

全オプションは `axicavity-fem <command> --help`、または
[ユーザーマニュアル](USER_MANUAL.md) を参照してください。

### GUI

```bash
axicavity-fem-gui
```

1. **Multi-Region Editor** — `samples/` のサンプルを開く（または新しく輪郭を描く）、
   線分ごとに境界条件を設定して `.msh` を書き出す。
2. **FEM Analysis** — メッシュのパスは自動で入ります。**Run Solver** →
   **Run Post-Process** → **Create Report** または **View Results** の順に押します。

---

## 精度

- **球形空洞と球ベッセル解析解の比較**: 最も細かいメッシュで相対誤差 ~1e-8、
  収束次数 ≈ 4（TM0・HOM とも）
  （[`examples/accuracy_verification/`](examples/accuracy_verification/)）
- **ピルボックス TM010**: `j₀₁c/2πa` と小数 6 桁まで一致
- **誘電体損失**: 一様充填で `Q_diel = 1/tanδ` が機械精度で成立し、周波数は
  解析解 TM010 と相対誤差 1.4e-7
  （[`examples/dielectric_loss/`](examples/dielectric_loss/)）
- **数値積分**: TM0 は既定で 7 点（5 次精度）求積を使います。収束検証
  （[`examples/quadrature_convergence/`](examples/quadrature_convergence/)）により、
  従来の 4 点則は 2 次要素の質量項を過小積分して偽モードを生じるのに対し、
  7 点則ではどのメッシュでも偽モードが出ないことを確認しています。
- 返される固有対はすべて**残差チェック**済みで、収束していないモードが偽の共振周波数
  として結果に混入することはありません。

設計用途では 2 次要素（`--elem-order 2`）を使ってください。同じメッシュサイズで
1 次要素より 1〜2 桁高精度です。

---

## ドキュメント

| | |
|---|---|
| [USER_MANUAL.md](USER_MANUAL.md) | GUI の操作手順と CLI の全オプション |
| [PHYSICS_AND_CONVENTIONS.md](PHYSICS_AND_CONVENTIONS.md) | 時間規約・規格化・電力流・周期境界の符号 |
| [docs/BC_NAMING.md](docs/BC_NAMING.md) | PEC / E-short / M-short / None の物理的意味 |
| [docs/HDF5_SCHEMA.md](docs/HDF5_SCHEMA.md) | 出力ファイルの構造 |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md) | 開発者向け |
| [examples/](examples/) | 精度検証・パラメータスイープの実行可能なスクリプト |
| [CHANGELOG.md](CHANGELOG.md) | 変更履歴 |

---

## version 1 との関係

version 2 は version 1 の完全な書き直しです。pip でインストールできるパッケージ化、
統一コマンドライン、新しい形状エディタ、多領域・誘電体対応、自動テスト、
文書化されたファイル形式を備えています。物理は同じで、数値も version 1 と一致します
（開発時にモードごとに照合済み）。

version 1 は同じリポジトリの tag
[`v1.0`](https://github.com/TakuyaNatsui/AxiCavity-FEM/releases/tag/v1.0) と branch
[`v1`](https://github.com/TakuyaNatsui/AxiCavity-FEM/tree/v1) から引き続き利用できます。

---

## 引用

本コードが論文等に寄与した場合は、次の情報で引用してください。

```bibtex
@software{AxiCavityFEM,
  author  = {Natsui, Takuya},
  title   = {{AxiCavity-FEM}: a 2-D axisymmetric finite-element solver for RF cavity resonant modes},
  version = {2.3.0},
  year    = {2026},
  url     = {https://github.com/TakuyaNatsui/AxiCavity-FEM}
}
```

## ライセンス

MIT — [LICENSE](LICENSE) を参照してください。

実行時依存ライブラリはそれぞれのライセンスに従います（**Gmsh**: GPL（リンク例外付き）、
**wxPython**: wxWindows Library Licence（LGPL 相当））。pip で導入する分には本コードの
ライセンスは変わりませんが、これらを同梱して再配布する場合はそれぞれの条件が適用されます。

## 開発への参加

バグ報告・プルリクエストは
[GitHub Issues](https://github.com/TakuyaNatsui/AxiCavity-FEM/issues) へお願いします。
プルリクエストの前にテストを実行してください。

```bash
pip install -e ".[dev]"
pytest
```

GUI のレイアウトは [wxGlade](https://wxglade.sourceforge.net/) が
`main_frame_ui.wxg` / `result_viewer_ui.wxg` から生成しています。`*_ui.py` を直接
編集せず、`.wxg` を編集して再生成してください。動作は手書きのサブクラス
（`main_frame.py`, `result_viewer.py`）側にあります。
