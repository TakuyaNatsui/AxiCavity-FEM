# AxiCavity-FEM 開発者ガイド

> **目的**: 新しい開発者が、コードの構成・役割・規約・落とし穴を短時間で把握し、
> 安全に改良を進められるようにするための総合ドキュメント。
> まずこのファイルを読み、必要に応じて各専門ドキュメント（[ARCHITECTURE](ARCHITECTURE.md) /
> [HDF5_SCHEMA](HDF5_SCHEMA.md) / [BC_NAMING](BC_NAMING.md) / 上位の
> [PHYSICS_AND_CONVENTIONS.md](../PHYSICS_AND_CONVENTIONS.md)）へ進むこと。

このプロジェクトは、軸対称空洞共振モードの 2 次元有限要素法（FEM）解析ツールである。
version 1（別リポジトリ tag `v1.0`）の機能を保ちつつ、共通基盤の抽出・パッケージ化・
統一 CLI/GUI 化を行ったのが version 2（本パッケージ `axicavity_fem`）である。

- **TM0 モード**: 軸対称（方位角次数 m=0）。未知数は H_φ スカラー（節点 DOF）。
- **HOM（High Order Mode）**: 方位角次数 n≥0。未知数は電界 E（Nédélec エッジ DOF + rE_θ 節点 DOF）。
- 定在波（standing）と進行波（traveling, 周期境界）の両方を扱う。

---

## 0. クイックスタート

```bash
pip install -e ".[dev]"          # コア + テスト依存（GUI は ".[dev,gui]"）
pytest -q                        # 全テスト（現状 373 passed）

# 一気通貫の CLI ワークフロー
axicavity-fem solve  --type tm0 -m mesh.msh --elem-order 2 --num-modes 10 -o r.h5
axicavity-fem post   --type tm0 -i r.h5 --cond 5.8e7 --beta 1.0
axicavity-fem report --type tm0 -i r.h5 -o r_report      # index.html
axicavity-fem plot   --type tm0 -i r.h5 -m 0 -o map.png
axicavity-fem export --type tm0 -i r.h5 -m 0 --shape axis --npts 500 -o axis
axicavity-fem info   -i r.h5
axicavity-fem-gui                # GUI
```

検証用メッシュとサンプル形状: [`../samples/`](../samples/)
- `cylinder100mm.msh`（円筒空洞・2 次要素。解析解 TM010 と照合できる）,
  `s-band_1cell.msh`（S バンド 1 セル・周期境界あり）, `sphere50mm.msh`（球形空洞）。
- `.gmshproj`（Multi-Region Editor のプロジェクト）と `.materials.json` も同梱してあるので、
  他の形状は GUI か `export_msh_multi_region()` でメッシュを再生成できる。

---

## 1. レイヤ構成と依存方向

import は **上から下への一方向のみ**（循環なし）。これを壊さないこと。

```
cli / gui                 ← ユーザ入口
  ↓
reports                   ← 可視化・出力（純表示層）
  ↓
fem_tm0 / fem_hom         ← ソルバ専用層（DOF 固有処理）
  ↓
physics_core              ← 規約セーフティ層（PBC 符号・P_flow 公式の一元管理）
  ↓
shared                    ← 数学・I/O・規約の共通基盤
```

- `cli` と `gui` は互いに依存しない（GUI はサブプロセスで CLI を起動する）。
- `reports` は `fem_tm0`/`fem_hom` の場再構成（field_recon）を使う。
- `physics_core` は「過去のバグを二度と繰り返さない」ための薄い規約層（後述 §4）。

---

## 2. ディレクトリ・ファイル別コードマップ

各行: `ファイル` — 役割（主な公開シンボル）。`(stub)` は未実装プレースホルダ。

### shared/ — 数学・I/O・規約の共通基盤（DOF 非依存）

| ファイル | 役割（公開シンボル） |
|---|---|
| `constants.py` | 物理定数 `C0, MU0, EPS0`。 |
| `element_functions.py` | 面積座標・節点形状関数・Whitney/Webb エッジ基底。`calculate_triangle_area_double`, `calculate_area_coordinates`, `grad_area_coordinates`, `calculate_quadratic_nodal_shape_functions`, `grad_quadratic_nodal_shape_functions`, `calculate_edge_shape_functions`, `calculate_edge_shape_functions_2nd`, `calculate_curl_edge_shape_functions_2nd`。 |
| `quadrature.py` | 三角形ガウス求積。`integration_points_triangle`（dict, n=1/3/4/7/"3_tm0_midpoint"）, `gaussian_quadrature_triangle`（クロージャ方式）, `calculate_triangle_area`。 |
| `edge_index.py` | `create_edge_index_map(simplices)` → `{(n_i,n_j): edge_idx}, num_edges`。 |
| `mesh_loader.py` | Gmsh 読込。`MeshData`（TM0 用: nodes, elements, element_order, physical_groups）, `load_mesh`; `HOMMeshData` + `load_mesh_hom`（エッジ + HOM 境界分類）; `compute_hom_boundary`（境界分類のみ、H5 からの再計算に使う）。 |
| `boundary_groups.py` | BC 名分類。`BoundaryClassification`, `classify_boundaries`（PEC/E-short/M-short、`Dirichlet`→PEC エイリアス）, `find_axis_nodes`, `pec_boundary_geometry`（PEC 境界線/中点の描画用幾何。GUI/レポート共用、matplotlib 非依存）, `bc_boundary_segments`（BC 別＋BC 未指定 `"None"` の境界辺。レポートの色分け用）, `mean_edge_length`（平均メッシュサイズ）。 |
| `eigensolver.py` | 一般化固有値ソルバ。`solve_eigenmodes`(dense), `_lobpcg`, `solve_eigenmodes_eigsh`(実対称 shift-invert), `solve_eigenmodes_eigs`(複素・進行波)。 |
| `hdf5_io.py` | 統一 HDF5 スキーマ v2.0。`write_results`, `read_results`, `append_post_process`, `dump_info`, `is_v2`, `load_v1_legacy`。 |
| `cli_common.py` | `parse_phase_list`（"0:180:20" 等）, `add_common_solver_args`。 |
| `superfish_io.py` | Superfish `.af` 入出力（GUI 非依存, `MultiRegionGeometry` ベース）。`import_superfish`, `export_superfish`, `validate_superfish_exportable`。単位換算・劣弧選択は旧実装と数値一致。 |
| `conventions.py` | (stub) 時間規約メモのみ。 |
| `logging.py`, `sparse_io.py` | (stub) 未実装。 |

### physics_core/ — 規約セーフティ層

| ファイル | 役割 |
|---|---|
| `pbc_phase.py` | 進行波 PBC 位相 **`e^{-jθ}`** を一元管理。`pbc_phase_factor(_deg)`, `apply_pbc_complex`, `real_imag_pbc_matrix`。回帰テスト `test_pbc_phase_sign.py`。 |
| `power_flow.py` | P_flow ラインインテグラルの数値プリミティブ。`gauss_3point_unit_interval`, `edge_line_integral_3point`, `tm0_p_flow_coefficient`（=−π/ωε₀）。回帰テスト `test_power_flow_lineintegral.py`。 |
| `normalization.py` | (stub) 未実装。 |

### fem_tm0/ — TM0 ソルバ（DOF: H_φ スカラー節点）

| ファイル | 役割 |
|---|---|
| `element_matrices.py` | ベクトル化要素行列。`element_matrices_1st`, `element_matrices_2nd`（曲線アイソパラメトリック対応, (E,n,n) を返す）。 |
| `assembly.py` | `assemble_global_matrices(mesh)` → (K,M) CSR。 |
| `boundary.py` | `find_nodes_on_r0_boundary`, `build_dirichlet_nodes`（M-short ∪ 軸 r=0）, `identify_periodic_boundaries`, `create_transformation_matrix`（位相は pbc_phase 経由）, `apply_bc_transformation`。 |
| `solver.py` | `TM0Result`, `solve_tm0_standing`, `solve_tm0_traveling`, `solve_tm0`（ファイル→解）。 |
| `field_recon.py` | `TM0FieldReconstructor`（H_φ→E_z/E_r 再構成、点/節点/軸スキャン）。 |
| `post_process.py` | `compute_tm0_parameters`（U・P_loss・Q・V_eff・R/Q・群速度）, `tm0_p_flow_zmin`, `calculate_boundary_integral_quadratic`, `TM0Parameters`/`TM0ModeParameters`。 |

### fem_hom/ — HOM ソルバ（DOF: Nédélec エッジ + rE_θ 節点）

| ファイル | 役割 |
|---|---|
| `element_matrices.py` | `element_matrices_1st_batch`（Whitney）, `element_matrices_2nd_batch`（Webb 階層, 曲線対応）。 |
| `assembly.py` | `assemble_global_matrices_1st/2nd`, `assemble_global_matrices(mesh)`, `matrix_size_hom`, `get_edge_orientation_sign`。 |
| `boundary.py` | 変換行列各種（実/複素, 1次/2次）、`find_periodic_boundary_pairs`, `get_pec_dof_indices_1st/2nd`, `apply_bc_transformation(_hermitian)`, `reconstruct_eigenvector_transformation`。位相は pbc_phase 経由。 |
| `solver.py` | `HOMModeSet`, `HOMResult`, `solve_hom_standing/traveling`, `solve_hom`。 |
| `field_recon.py` | `split_hom_dofs`（フラット固有ベクトル→CT/LN・LT/LN・face・E_θ）, `calculate_E_theta_from_rE_theta`, `HOMFieldReconstructor`。 |
| `post_process.py` | `compute_hom_parameters`（U・P_loss・Q・群速度等）, `calc_p_flow_hom`, `get_pec_edges`, `HOMParameters`/`HOMModeParameters`。 |

### reports/ — 可視化・出力（純表示層, matplotlib）

| ファイル | 役割 |
|---|---|
| `export_fields.py` | `export_fields`（area/line/axis を H5/TXT 出力）, `sample_area`, `sample_line`。 |
| `plot_common.py` | `STYLE`, `make_triangulation`, `make_triangulation_vis`（2 次細分割）, `plot_scalar_field`, `new_panel_grid`, `save_figure`, `render_gif`（固定キャンバスで GIF 生成）。**import 時に Agg バックエンドを設定**。 |
| `plot_tm0.py` / `plot_hom.py` | `plot_tm0_mode` / `plot_hom_mode`（Result Viewer 準拠の場マップ PNG。TM0=H_phi カラー＋電気力線＋ベクトル＋PEC線＋軸上 Ez、HOM=E/H 2 パネル）、`animate_tm0_mode` / `animate_hom_mode`（進行波 GIF）。描画は「setup＋1 フレーム描画」に分離し静止画/GIF で共用。 |
| `report_builder.py` | `build_report`（場マップ + 工学パラメータ表の index.html）。 |

### cli/ — 統一 CLI

| ファイル | 役割 |
|---|---|
| `main.py` | `build_parser`（argparse, サブコマンド定義）, `cli_main`（ディスパッチ）。entry point: `axicavity-fem`。 |
| `cmd_solve.py` / `cmd_post.py` / `cmd_info.py` / `cmd_export.py` / `cmd_plot.py` / `cmd_report.py` | 各サブコマンドの `run(args)`。 |
| `summary_txt.py` | 結果の簡易テキストサマリ（`.h5` → `.txt`）。`render_solve_summary` / `render_post_summary` / `write_solve_summary` / `write_post_summary` / `txt_path_for`。 |

### gui/ — wxPython GUI

| ファイル | 役割 |
|---|---|
| `app.py` | `main()`（entry point `axicavity-fem-gui`）。 |
| `main_frame_ui.py` | wxGlade 生成 UI（`MyFrameUI`）。**人間が wxGlade で編集** → 再生成可能。手で複雑な変更をしない。v2.3 で 2 タブ構成（Multi-Region Editor / FEM Analysis）。 |
| `main_frame.py` | `MyFrame`（ロジック）。Solver/Post/Report を **統一 CLI 呼び出し**で実行（`_build_base_cmd` + subprocess）。Superfish `.af` 入出力・座標欄の数値プレビュー・旧形式 `.gmshproj` の MR 変換読込もここ。 |
| `multi_region_editor.py` | `MultiRegionEditorPanel`（多領域メッシュ形状エディタ、matplotlib キャンバス。点/線/円弧/ループ/領域編集）。 |
| `result_viewer_ui.py` | wxGlade 生成 UI（`ResultViewerUI`）。 |
| `result_viewer.py` | `ResultViewer`（結果ビューア。read_results + field_recon + export_fields に再配線）。描画は `_draw_on(fig, time_phase)` に集約（画面更新と GIF で共用）。TM0=H_phi カラー＋電気力線＋ベクトル、HOM=E/H 2 パネル。PEC 境界線は常時表示、`Show E-wall` で PEC エッジ中点に電場ベクトル（TM0）。`Save GIF...`（進行波）。**plot_common は import しない**（三角形分割・PEC 幾何は shared 経由）。 |

---

## 3. データモデル

### 3.1 統一 HDF5 スキーマ（[HDF5_SCHEMA.md](HDF5_SCHEMA.md) 詳細）

- 固有ベクトルは **フラットな 1 配列**で格納（standing: `eigenvector`、traveling: `eigenvector_re/_im`）。
  TM0（節点）と HOM（エッジ+節点）を統一。HOM の分割は読み戻し後 `split_hom_dofs` で行う。
- `results/n{N}/standing/mode_{K}/` と `results/n{N}/traveling/phase_{xxxxx}/mode_{K}/`。
  phase キーは θ_deg×10 のゼロ埋め 5 桁（120.0° → `phase_01200`）。
- post は `/post_process/...` に attrs（Q, R_over_Q, U_stored, P_loss, P_flow_zmin, group_velocity, ...）を追記。
- `read_results(path)` → dict（`mesh`, `parameters`, `results_by_n`, `post_process`, `solver_type`）。
  `mesh["edge_index_map"]` は読み込み時に復元される（HOM）。

### 3.2 HOM の DOF 番号体系（重要）

- 1 次: `[0, num_edges)` エッジ（Whitney）、`[num_edges, ...)` 節点（n>0）。
- 2 次: `2e`=エッジ e の CT/LN、`2e+1`=LT/LN、`2·num_edges + 2k(+1)`=要素 k の face DOF(N7,N8)、
  `2·num_edges + 2·num_elem + i`=節点 i（n>0）。
- 符号: CT/LN のみエッジ向き（昇順 +1）に応じて ±1。LT/LN・face・node は +1。
- 節点 DOF は **rE_θ**（r 倍した量）。物理 E_θ へは `calculate_E_theta_from_rE_theta`（n=1 は軸上を近傍平均で補間）。

### 3.3 主な dataclass

`MeshData`/`HOMMeshData`（shared.mesh_loader）, `TM0Result`/`HOMResult`,
`TM0Parameters`/`TM0ModeParameters`, `HOMParameters`/`HOMModeParameters`。

---

## 4. 規約と落とし穴（必読）

### 4.1 境界条件（[BC_NAMING.md](BC_NAMING.md)）
- 正規名は **PEC / E-short / M-short** の 3 つのみ。`Dirichlet` は PEC エイリアス（警告付き）。
- FEM 行列上の扱いは **DOF により反転**:
  - TM0: PEC/E-short → Neumann（自然）、**M-short → Dirichlet (H_φ=0)**、**軸 r=0 → Dirichlet**。
  - HOM: PEC/E-short → Dirichlet (E_tan=0)、M-short → Neumann、軸 r=0 は n=0 で Neumann / n≥1 で Dirichlet。
- 軸 r=0 を TM0 で Dirichlet にするのは物理的に正しい（H_φ は軸上で消える）。

### 4.2 進行波 PBC 位相
- 必ず `physics_core.pbc_phase` 経由。規約は **`x_max = e^{-jθ}·x_min`**（時間規約 e^{+jωt}）。
  直接 `exp(±1j*θ)` を書かないこと（過去に符号バグを出した箇所）。

### 4.3 P_flow
- z=z_min 断面のラインインテグラル。TM0 係数 **−π/(ωε₀)**、HOM 係数 **+π/(ωμ₀)**。
  数値プリミティブ（3 点ガウス線積分）は `physics_core.power_flow.edge_line_integral_3point`。

### 4.4 eigsh の不安定性（TM0 定在波）
- `eigsh(sigma, which='LA')` は σ 近傍で **数値的に不安定**で、同一行列でも ARPACK の
  乱数始ベクトル依存で返すモード集合がばらつく（K を増やすと ARPACK エラーも）。
- → このため周波数の比較テストは「モード集合の一致」ではなく **基本モード TM010 の値対応**
  で行う。また `shared/eigensolver.py` が全モードの残差をチェックし、収束していない
  モードを出力から除外する（§4.9）。

### 4.5 蓄積エネルギー U（HOM）
- eigsh の M 正規化（vᴴMv=1）により、HOM の U は全モードで **2π·ε₀**（≈5.563e-11）になる
  （HOM 行列は係数 2 を省略した構成のため）。これは正常。Q の差は P_loss が担う。

### 4.6 gmsh のグローバル状態
- `load_mesh` は呼び出しごとに `finalize()→initialize()` で**フレッシュな状態**にしてから読む。
  これを怠ると 2 次曲線メッシュの再パラメータ化が前段の状態に依存し、結果がぶれる（実害あり、修正済み）。

### 4.7 matplotlib バックエンド
- `reports/plot_common.py` は import 時に **Agg** を設定（CLI/オフスクリーン用）。
- GUI の `result_viewer.py` は埋め込み **WXAgg** キャンバスを使うため、**plot_common を import しない**
  （三角形分割はインラインで持つ）。両者を混ぜないこと。

### 4.8 求積次数（TM0）
- TM0 の要素行列は既定で **7 点（Dunavant 5 次）求積**を使う（`n_quad` 引数で 4 点に変更可）。
  2 次要素の質量項 `r·G_i·G_j` は 5 次多項式で、4 点則（3 次精度）では過小積分となり
  M が数値的に不定化して偽モードを生む。7 点則はこれを厳密に積分する。
- HOM はエッジ要素の別定式のため、7 点でも偽モードが残ることがある（残差フィルタで除去）。

### 4.9 残差チェックと偽モード除去
- `shared/eigensolver.py` は返された固有対の相対残差 ‖Kx−λMx‖/‖λMx‖ を全モードで評価し、
  閾値（1e-2）超のモードを出力から除外する。不足分は要求モード数を増やして最大 3 回まで
  解き直して補充し、それでも足りなければ警告して収束分のみを返す。
- **収束した固有対の数値は一切変えない**。この安全網の検証は `tests/test_residual_retry.py`。

---

## 5. 検証戦略

数値に関わる変更を入れたら、次のいずれかで裏を取るテストを必ず追加する。

| 方法 | 例 | 判定 |
|---|---|---|
| **解析解との比較** | 円筒空洞 TM010 = `j01·c/(2πa)`、球形空洞と球ベッセル解、一様充填の `Q_diel = 1/tanδ` | 相対誤差（例 1e-6）または収束次数 |
| **回帰（既知の期待値）** | 周期境界の位相符号、P_flow の線積分 | rtol 1e-9 程度 |
| **等価性** | 同じ物理を別経路で解いた結果の一致（例: 一様 ε_r の 1 領域と 2 領域、CLI と直接 API 呼び出し、PARDISO と SuperLU） | rtol 1e-6 |
| **往復 I/O** | HDF5・`.gmshproj`・Superfish `.af` の書き出し → 読み込み | 完全一致 |
| **スモーク** | plot / report / GUI 構築 / 生成スクリプトの `compile()` | 例外なし・生成物が非空 |

固有ベクトルには符号・位相の任意性があるため、2 つの実装を比べるときは
**同じ固有ベクトルを両者に与える**設計にすると判定が安定する。

> version 1 との数値一致は開発時に検証済み（全体行列 K/M のビット一致、TM0/HOM の周波数、
> ポストプロセスの end-to-end 比較）。その比較テストは version 1 のソースを必要とするため
> 公開リポジトリには含めていない。

---

## 6. 拡張レシピ

### 6.1 CLI サブコマンドを追加する
1. `cli/main.py` の `build_parser` にサブパーサと引数を追加。
2. `cli/cmd_<name>.py` に `run(args) -> int` を実装。
3. `cli_main` のディスパッチに `if args.command == "<name>": from . import cmd_<name>; return cmd_<name>.run(args)`。

### 6.2 ポストプロセスにパラメータを足す
1. `fem_tm0/post_process.py`（または hom）の `compute_*_parameters` で計算し dataclass に追加。
2. `cli/cmd_post.py` の `_*_params_to_attrs` に H5 attrs キーを追加。
3. レポート表に出すなら `reports/report_builder.py:_param_keys` に追記。

### 6.3 新しい可視化を足す
- `reports/plot_common.py` のヘルパを使い `reports/plot_*.py` に関数追加 → `cli/cmd_plot.py` から呼ぶ。
- GUI に出すなら `gui/result_viewer.py` の描画（インライン）に追加（plot_common は import しない）。

### 6.4 GUI を変更する
- レイアウトは wxGlade。ソースはリポジトリ直下の **`main_frame_ui.wxg`** と
  **`result_viewer_ui.wxg`**。いずれも `option="0"`（単一ファイル）＋`path` を対象
  `src/.../gui/*_ui.py` に直接指定してあるので、wxGlade で対象 `.wxg` を開き
  **Generate Source** すればその `*_ui.py` が直接更新される（CamelCase 改名不要）。
- `*_ui.py` は生成物なので手で複雑な変更をしない。ロジックは `main_frame.py` / `result_viewer.py` に書く。
- ハンドラ未実装の追加が必要なときは、まず `.wxg` にイベントを足して再生成 → ロジック側でオーバーライド。

---

## 7. 既知の制約・今後の課題

- **誘電体の工学パラメータ**: V_eff / R/Q は**軸 r=0 とポート断面が真空**であることを前提と
  している。共振周波数・U・P_flow は誘電体があっても正確で、前提が破れる場合はソルバが警告する。
- **μ_r と磁性損失は未対応**（`mu_r` は読み込むが解には使っていない）。
- **tanδ は摂動法**なので目安として tanδ ≲ 0.05 まで。また tanδ は solve 時に HDF5 へ保存
  されるため、値を変えたら solve から実行し直す必要がある。
- **HOM の偽モード**: エッジ要素の定式では 7 点求積でも偽モードが残ることがあり、
  残差フィルタ（§4.9）で除去している。根本的な解消は未対応。
- **TM0 定在波 `eigsh(which='LA')` のモード選択不安定性**（§4.4）。
- **シフト値が ε_r に追従しない**: 誘電体を多く含む空洞では最低次モードを取り逃す場合がある。
  その場合は `--target-freq` で探索したい周波数を明示する。
- **レポートの GIF** はモード数×フレーム数を生成するため遅く、1 モードで数 MB になり得る
  （`optimize=False`）。フレーム数・最適化のチューニングは要検討。
- **表示系（GUI 操作・PNG/HTML/GIF の見た目）は自動テストで検証できない** → 人の目で確認する。
- 未実装のスタブ: `shared/logging.py`, `shared/sparse_io.py`, `physics_core/normalization.py`
  （必要になったら実装）。

---

## 8. 関連ドキュメント

- [ARCHITECTURE.md](ARCHITECTURE.md) — レイヤ設計の概要。
- [HDF5_SCHEMA.md](HDF5_SCHEMA.md) — 出力ファイル仕様。
- [BC_NAMING.md](BC_NAMING.md) — 境界条件の物理-数学対応表。
- [../PHYSICS_AND_CONVENTIONS.md](../PHYSICS_AND_CONVENTIONS.md) — 物理規約（時間規約・規格化・P_flow 等）。
- [../USER_MANUAL.md](../USER_MANUAL.md) — エンドユーザ向け操作手順。
- [../CHANGELOG.md](../CHANGELOG.md) — 変更履歴。
