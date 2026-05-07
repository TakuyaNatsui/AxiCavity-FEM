# AxiCavity-FEM

**加速器空洞共振器 TM0／HOM 解析のための 2 次元軸対称電磁場 FEM**

[English](README.md) | [日本語](README.ja.md)

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey)
![GUI](https://img.shields.io/badge/GUI-wxPython-orange)

AxiCavity-FEM は、軸対称 RF 空洞の共振モードを有限要素法で解くツールです。基本 TM0 モードと任意の方位角次数の高次モード（HOM）を、定在波・進行波の両方について計算できます。Webb 階層基底による 2 次要素を採用し、高い周波数精度を達成しています。wxPython による GUI で Gmsh ベースのメッシュ生成・固有モード解析・ポストプロセス・HTML / GIF レポートを一貫して実行でき、バッチ用途に等価な CLI も用意しています。

---

## スクリーンショット

### GUI ワークフロー

<table>
  <tr>
    <td width="33%"><img src="docs/images/mesh_edit_UI_screenshot.png" alt="メッシュ編集タブ" /></td>
    <td width="33%"><img src="docs/images/FEM_Analysis_screenshot.png" alt="FEM Analysis タブ" /></td>
    <td width="33%"><img src="docs/images/Result_Viewer_screenshot.png" alt="Result Viewer" /></td>
  </tr>
  <tr>
    <td align="center"><sub>1. 形状作成・メッシュ生成タブ</sub></td>
    <td align="center"><sub>2. FEM Analysis（ソルバー + ポストプロセス）</sub></td>
    <td align="center"><sub>3. 対話的な Result Viewer</sub></td>
  </tr>
</table>

### 解析結果の例

<table>
  <tr>
    <td width="50%"><img src="docs/images/mesh_overview.png" alt="メッシュ概要" /></td>
    <td width="50%"><img src="docs/images/tm0_traveling_field.png" alt="TM0 進行波の電磁場" /></td>
  </tr>
  <tr>
    <td align="center"><sub>メッシュ概要（S バンド進行波構造）</sub></td>
    <td align="center"><sub>TM0 進行波の E／H 場（θ=120°）</sub></td>
  </tr>
  <tr>
    <td width="50%"><img src="docs/images/hom_n1_field.png" alt="HOM n=1 モード" /></td>
    <td width="50%"><img src="docs/images/axial_ez.png" alt="軸上 Ez 分布" /></td>
  </tr>
  <tr>
    <td align="center"><sub>HOM n=1 双極モード（E／H 場）</sub></td>
    <td align="center"><sub>軸上 E<sub>z</sub> 分布（実部／虚部）</sub></td>
  </tr>
</table>

---

## 主な機能

### ソルバー
- **TM0 モード** — スカラー $H_\phi$ 定式化、定在波・進行波両対応
- **HOM（n ≥ 1）** — Nédélec エッジ DOF + 節点 $E_\phi$ DOF、任意の方位角次数、複数 `n` を一括計算
- **Webb 階層基底 2 次要素** — O(h⁴) 収束、球形空洞ベンチマークで周波数誤差 < 0.01 %
- **周期境界の位相スキャン** — 進行波構造の分散曲線を `0:180:20` のような構文で自動掃引

### ワークフロー
- **wxPython GUI** に Gmsh メッシュ生成を統合（形状エディタ → メッシュ → 解析 → ビューア が 1 ウィンドウ）
- **等価な CLI** によりスクリプト・バッチ実行が可能。GUI から実行された全コマンドは `command.log` に記録される
- **インタラクティブ Result Viewer** — カラーマップ、電気力線、ベクトル図、ダブルクリックでその点の電磁場値を表示、進行波の時間発展 GIF を保存
- **HTML レポート** — Q 値、R/Q、シャントインピーダンス、群速度、減衰定数などの工学パラメータと、各モードのフィールド図を自動生成
- **電磁場マップ出力** — Area / Line / Axis のサンプリング、HDF5 + TXT 出力、複素振幅または瞬時値、ビーム計算用の電力スケーリング対応

---

## インストール

```bash
git clone https://github.com/TakuyaNatsui/AxiCavity-FEM.git
cd AxiCavity-FEM
pip install -r requirements.txt
```

**動作環境:** Python 3.10 以上（wxPython 4.2+ で 3.10–3.13 をサポート）。

**注意点:**
- `wxPython` と `gmsh` を含むすべての依存関係が PyPI 経由で Windows / Linux にインストール可能です。
- Linux 環境では wxPython のビルド回避のため Conda 環境を推奨します。
- Gmsh は Python バインディングが PyPI から直接配布されているため、別途 Gmsh を入れる必要はありません。

---

## クイックスタート

### GUI

```bash
python app.py
```

GUI 内部のワークフローは 3 ステップ:

1. **Shape & Mesh タブ** — 空洞断面を点・円弧で作図し、メッシュサイズを設定して `.msh` を出力。
2. **FEM Analysis タブ** — TM0 / HOM を選び、要素次数（**2 次推奨**）、モード数、必要なら位相スキャンを設定 → **Run Solver** → **Run Post-Process** → **Create HTML Report**。
3. **Result Viewer** — モードを対話的に切り替えて確認。フィールドマップや GIF アニメーションを書き出し可能。

### CLI（TM0 定在波の最小例）

```bash
# 解析
python FEM_code/run_analysis.py \
    -m mesh_and_result/cylinder50mm_2nd.msh \
    --elem-order 2 --num-modes 5 -o result.h5

# ポストプロセス + レポート生成
python FEM_code/post_process_unified.py -i result.h5 -c 5.8e7 -b 1.0
```

GUI 操作の詳細・全 CLI オプション（HOM ソルバー、進行波位相スキャン、電磁場マップ出力など）は [USER_MANUAL.md](USER_MANUAL.md) を参照してください。

---

## プロジェクト構成

```
AxiCavity-FEM/
├── app.py                      # GUI エントリポイント
├── MyFrame.py                  # メイン GUI ロジック
├── ResultViewer.py             # 対話的な結果ビューア
├── plot_common.py              # 共通可視化ユーティリティ
├── FEM_code/                   # TM0 ソルバー・ポストプロセス・フィールド出力
├── FEM_HOM_code/               # HOM（n ≥ 1）ソルバー・ポストプロセス・フィールド出力
├── mesh_and_result/            # サンプルメッシュとレポート出力
├── docs/images/                # README 用スクリーンショット
└── USER_MANUAL.md              # ユーザー向けマニュアル（日本語）
```

---

## ドキュメント

| ファイル | 内容 |
|------|------|
| [USER_MANUAL.md](USER_MANUAL.md) | ユーザー向け操作マニュアル（日本語）。GUI チュートリアル、CLI リファレンス、実例 |
| [PHYSICS_AND_CONVENTIONS.md](PHYSICS_AND_CONVENTIONS.md) | FEM 定式化、時間規約 $e^{+j\omega t}$、周期境界の符号、工学パラメータ定義 |
| [Helmholtz_2D_FEM.pdf](Helmholtz_2D_FEM.pdf) | 軸対称ヘルムホルツ FEM の数学的導出 |
| [CLAUDE.md](CLAUDE.md) | 開発に用いた AI エージェント向けプロジェクト指示 |

---

## 精度検証

球形共振器（R = 100 mm）の解析的固有周波数と比較して検証しています。2 次要素では 1 次要素に対して誤差が 1〜2 桁減少し、理論収束次数 O(h⁴) を達成します。

| メッシュサイズ | 要素次数 | TM₀₁₁ 誤差 | TM₁₁₁ 誤差 | TE₁₁₁ 誤差 | TM₂₁₁ 誤差 |
|---|---|---|---|---|---|
| 10 mm（粗） | 1 次 | 〜 1 % | 〜 0.1 % | 〜 0.1 % | 〜 0.1 % |
| 10 mm（粗） | **2 次** | **0.12 %** | **0.0004 %** | **0.0014 %** | **0.0005 %** |
| 2.5 mm（細） | 2 次 | < 0.01 % | < 0.001 % | < 0.001 % | < 0.001 % |

この精度を達成する定式化の詳細は `PHYSICS_AND_CONVENTIONS.md` を参照してください。

---

## 引用

学術用途で AxiCavity-FEM をご利用の場合は、以下をご引用ください:

```bibtex
@software{axicavity_fem,
  title  = {AxiCavity-FEM: 2D Axisymmetric Electromagnetic FEM for Accelerator Cavity TM0 / HOM Analysis},
  author = {Takuya Natsui},
  year   = {2026},
  url    = {https://github.com/TakuyaNatsui/AxiCavity-FEM}
}
```

---

## ライセンス

MIT License で配布しています。詳細は [LICENSE](LICENSE) を参照してください。

なお、本プロジェクトは実行時に wxPython（LGPL）および Gmsh（GPL with linking exception）の Python バインディングに依存します。AxiCavity-FEM 自体のソースコードは MIT ライセンスです。

---

## 謝辞

本プロジェクトは **Claude（Anthropic）** との反復的な協働開発によって構築されました。
