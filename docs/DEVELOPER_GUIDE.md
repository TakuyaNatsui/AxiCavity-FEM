# AxiCavity-FEM 開発者ガイド（バージョン 3: 計算コアは 2.3.0 と同じ、GUI は PySide6）

> **バージョン 3 について**: 本ガイドの計算コア（§1〜§5、§7）の記述はバージョン 2 系からそのまま有効（コアは
> 2.3.0 と同じコードで、違いは `__version__` だけ。`tests/test_core_identical.py` が SHA-256 の一覧で監視）。
> GUI は wxPython から PySide6 のリボン UI に置き換わった（§2 gui/、§4.7、§6.3〜6.5）。Windows 版（EXE）は §10。

> **目的**: 新しいセッション（または新しい開発者）が、コードの構成・役割・規約・落とし穴を
> 短時間で把握し、安全に改良を進められるようにするための総合ドキュメント。
> まずこのファイルを読み、必要に応じて各専門ドキュメント（[ARCHITECTURE](ARCHITECTURE.md) /
> [HDF5_SCHEMA](HDF5_SCHEMA.md) / [BC_NAMING](BC_NAMING.md) / 上位の
> [PHYSICS_AND_CONVENTIONS.md](../PHYSICS_AND_CONVENTIONS.md)）へ進むこと。

このプロジェクトは、軸対称空洞共振モードの 2 次元有限要素法（FEM）解析ツールである。
ver1（GitHub のタグ `v1.0`）の機能を保ちつつ、共通基盤の抽出・パッケージ化・統一 CLI/GUI 化を行ったのが ver2
（本パッケージ `axicavity_fem`）、GUI を作り直したのが ver3。ver1 と数値が一致することは ver2 の開発中に確認した
（その比較テストは ver1 のコードが要るので公開リポジトリには含めない）。

- **TM0 モード**: 軸対称（方位角次数 m=0）。未知数は H_φ スカラー（節点 DOF）。
- **HOM（High Order Mode）**: 方位角次数 n≥0。未知数は電界 E（Nédélec エッジ DOF + rE_θ 節点 DOF）。
- 定在波（standing）と進行波（traveling, 周期境界）の両方を扱う。

---

## 0. クイックスタート

```bash
git clone https://github.com/TakuyaNatsui/AxiCavity-FEM.git && cd AxiCavity-FEM
pip install -e ".[gui,viz3d,dev]"                        # コア + GUI（PySide6, planegcs）+ 3D 表示 + テスト依存
PYTHONUTF8=1 python -m pytest -q -p no:cacheprovider     # 全テスト（実ウィンドウを 1 秒開く）
AXICAVITY_SKIP_REAL_WINDOW=1 python -m pytest -q         # 実ウィンドウの確認を飛ばす

# 一気通貫の CLI ワークフロー（v2.3 と同じ）
axicavity-fem-run samples/cylinder100mm.gmshproj --out pillbox      # メッシュ → solve → post（pillbox/model.msh）
axicavity-fem solve  --type tm0 -m pillbox/model.msh --elem-order 2 --num-modes 10 -o r.h5
axicavity-fem post   --type tm0 -i r.h5 --cond 5.8e7 --beta 1.0      # r.h5 に追記
axicavity-fem report --type tm0 -i r.h5 -o r_report                  # index.html
axicavity-fem plot   --type tm0 -i r.h5 -m 0 -o map.png
axicavity-fem export --type tm0 -i r.h5 -m 0 --shape axis --npts 500 -o axis
axicavity-fem info   -i r.h5
axicavity-fem-gui [project.axiproj | old.gmshproj | result.h5]      # GUI
python -m axicavity_fem.gui.launcher selftest   # GUI と同じ子プロセス経路で円筒空洞を解く（EXE の selftest と同じ）
python tools/verify_samples.py     # samples の全サンプルを v3 の経路で解析して参照結果（あれば）と比較
python tools/manual_screenshots.py   # USER_MANUAL 用のスクリーンショット（実ウィンドウ）
```

テストのメッシュは `tests/conftest.py` の `sample_mesh` fixture が `samples/*.gmshproj` からその場で作る。
参照結果（`samples/` の `.h5` / `.txt`）はリポジトリに入れていないので、それが要るテストは無ければ skip になる。

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

- `cli` と `gui` は互いに依存しない。v3 の GUI は **子プロセス**（`python -m axicavity_fem.gui.jobs.runner`、
  EXE では `AxiCavity-FEM.exe job`）で `cli.cmd_*` の `run(args)` を argv ごと呼ぶ（CLI と同じ経路・引数検証。
  gmsh も子プロセスでだけ初期化する）。
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
| `multi_region_model.py` | 多領域の形状 `MultiRegionGeometry`（点・Segment（線 / 円弧、BC 名）・Loop・Region（材料タグ・ε_r・tanδ・式））、`.gmshproj` の JSON（`load_json` / `save_json`）、変数と式。標準ライブラリだけ（メッシャでも使う）。 |
| `gmsh_export_occ.py` | `MultiRegionGeometry` → gmsh（OpenCASCADE + `fragment`）→ `.msh`（`export_msh_multi_region`）、`.geo` と自己完結の Python スクリプトの書き出し、`inspect_msh_physical_groups`。gmsh・標準ライブラリ・`multi_region_model` だけ（メッシャでも使う）。 |
| `material_resolver.py` | 領域の材料表（`.materials.json` のサイドカー）と要素ごとの ε_r / tanδ。 |
| `expression_eval.py` | 変数と式の安全な評価器（`ast`。`eval` は使わない）。GUI の式もこれ。 |
| `mesh_loader.py` | Gmsh 読込。`MeshData`（TM0 用: nodes, elements, element_order, physical_groups）, `load_mesh`; `HOMMeshData` + `load_mesh_hom`（エッジ + HOM 境界分類）; `compute_hom_boundary`（境界分類のみ、H5 からの再計算に使う）。 |
| `boundary_groups.py` | BC 名分類。`BoundaryClassification`, `classify_boundaries`（PEC/E-short/M-short、`Dirichlet`→PEC エイリアス）, `find_axis_nodes`, `pec_boundary_geometry`（PEC 境界線/中点の描画用幾何。GUI/レポート共用、matplotlib 非依存）, `bc_boundary_segments`（BC 別＋BC 未指定 `"None"` の境界辺。レポートの色分け用）, `mean_edge_length`（平均メッシュサイズ）。 |
| `eigensolver.py` | 一般化固有値ソルバ。`solve_eigenmodes`(dense), `_lobpcg`, `solve_eigenmodes_eigsh`(実対称 shift-invert), `solve_eigenmodes_eigs`(複素・進行波)。 |
| `hdf5_io.py` | 統一 HDF5 スキーマ（schema_version 2.2）。`write_results`, `read_results`, `append_post_process`, `dump_info`, `is_v2`, `load_v1_legacy`。 |
| `cli_common.py` | `parse_phase_list`（"0:180:20" 等）, `add_common_solver_args`。 |
| `superfish_io.py` | Superfish `.af` 入出力（GUI 非依存, `MultiRegionGeometry` ベース）。`import_superfish`, `export_superfish`, `validate_superfish_exportable`。単位換算・劣弧選択は旧実装と数値一致。 |
| `logging.py`, `sparse_io.py` | (stub) 未実装。 |

### physics_core/ — 規約セーフティ層

| ファイル | 役割 |
|---|---|
| `pbc_phase.py` | 進行波 PBC 位相 **`e^{-jθ}`** を一元管理。`pbc_phase_factor(_deg)`, `apply_pbc_complex`, `real_imag_pbc_matrix`。回帰テスト `test_pbc_phase_sign.py`。 |
| `power_flow.py` | P_flow ラインインテグラルの数値プリミティブ。`gauss_3point_unit_interval`, `edge_line_integral_3point`, `tm0_p_flow_coefficient`（=−π/ωε₀）。回帰テスト `test_power_flow_lineintegral.py`。 |
| `conventions.py` | (stub) 時間規約メモのみ。 |
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

### gui/ — PySide6 リボン GUI（v3）

下位層は上位層を import しない: `core`（純 Python）← `sketch`（planegcs）← `app/store.py` ← `ui`。`jobs` / `io` /
`project` は Qt の QObject を持つが UI を import しない。

| ファイル | 役割 |
|---|---|
| `app/__init__.py`, `__main__.py` | `main()`（entry point `axicavity-fem-gui`、`python -m axicavity_fem.gui`）。引数に `.axiproj` / `.gmshproj` / `.af` / `.h5`。アプリ名とアイコン（`ui/app_icon.svg`）。 |
| `launcher.py` | 入口を 1 つにまとめたもの（EXE の main）: GUI / コア CLI（`solve` …）/ `run`（バッチ）/ `job`（GUI の子プロセス）/ `version` / `selftest`。§10。 |
| `frozen.py`, `app_info.py` | EXE（Nuitka）かどうかと EXE のパス（`GetModuleFileNameW`）、版と同梱物（PARDISO・メッシュ生成の方式・3D 表示）の情報。 |
| `mshlite/{reader,gmsh_api}.py` | `.msh`（4.1 / 2.2、ASCII / binary）の純 Python リーダと、gmsh の読込 API の互換モジュール（EXE で `sys.modules["gmsh"]` に差し込む）。§10。 |
| `core/document.py`, `serialize.py`, `expressions.py`, `hashes.py` | `AxiDocument`（meta / params / sketch / regions / boundaries / mesh / analysis / post / report / view）と JSON、式評価（`shared/expression_eval` のアダプタ）、形状 / メッシュ / モデルのハッシュ（結果が「形状変更前」かの判定）。 |
| `core/sketch/{geometry,model,profiles,snap,edit_ops,planarize}.py` | スケッチ幾何（EM-CAD-py からコピー）、閉領域の検出（純 Python）、点挿入 / 再接続 / 線⇄円弧、交差・接触での分割。 |
| `core/sketch/reference.py` | 参照ジオメトリ（原点 `ref-origin`・z 軸 `ref-z-axis`・r 軸 `ref-r-axis`）。`projection = "reference"` の構築要素としてスケッチに常にあり、ソルバーでは固定、閉領域・変換・ハッシュ・数には入らない。「r=0 / z=0 に固定」と作図の自動拘束は「点を曲線上に（軸）」（`axis_constraint` / `axis_constraints_for` / `implied_redundant`）。 |
| `core/convert.py` | スケッチ + 設定 ⇄ `MultiRegionGeometry`（`to_multi_region` / `from_multi_region`）、有効 BC の自動規則（`effective_bc`: 明示 → 内部界面 → 軸 → PEC）、領域設定の再対応付け（`resolve_regions`）、`check_geometry`（Issue の一覧）。 |
| `core/legacy.py`, `io/{axiproj,gmshproj,superfish}.py` | 旧形式 `.gmshproj` / Superfish の取り込み・書き出し、プロジェクト `.axiproj` + `.axiproj.data/`（結果履歴の走査 `scan_results`、`result.json`、`command.log`）。 |
| `sketch/{solver,tools,annotations,controller}.py` | planegcs ソルバー（式付きの点は固定 / 片側固定）、作図・拘束ツール、注釈、`SketchController`（ツール状態と自動平面化）。 |
| `app/store.py` | `DocumentStore`: 文書・Undo（deepcopy スナップショット）・ツール・選択・拘束・パラメータ・物理設定（`set_boundary` / `set_region`）・設定（`set_mesh` / `set_analysis` / `set_post` / `set_report` / `set_units`）。UI はシグナルで追従。 |
| `project/controller.py` | `ProjectController`: new / open / save / save_as、ハッシュ、`prepare_mesh` / `prepare_run`（自動保存）、結果の登録・完了・選択・名前変更・削除、`log_command`。 |
| `jobs/runner.py`, `pipeline.py`, `analysis_pipeline.py` | 子プロセスの入口と本体: mesh / export_msh（gmsh → `.msh` + `.materials.json` + `mesh_preview.npz`）、analysis（メッシュ → `cmd_solve` → `cmd_post` → 要約）、post / report / export。イベントは標準出力の JSON 行、fd 1 は `log.txt` へ付け替え、`cancel.flag`。 |
| `jobs/meshing.py`, `mesher/axicavity_mesh.py`（リポジトリ直下） | メッシュ生成の入口 `generate_msh`（開発・pip はプロセス内の gmsh、EXE は別プロセスのメッシャ）と、メッシャ本体。§10。 |
| `jobs/session.py`, `mesh_folder.py`, `mesh_controller.py`, `analysis_controller.py`, `commands.py` | `JobSession`（QProcess、同時 1 本）、メッシュの世代フォルダと再利用判定、解析の投入と結果への記録、CLI 相当の argv（wx 版と同じ文字列）。 |
| `ui/main_window.py` | リボン 7 タブ（`_action` / `_menu_button`）、ブラウザツリー、`QStackedWidget{SketchView, ResultView}`、右パネルの `QStackedWidget`、ログ、ステータスバー、配置の保存。 |
| `ui/sketch_view.py`, `sketch_panels.py`, `params_panel.py`, `constraint_actions.py` | スケッチ画面（overlay = sketch / physics / mesh）、スケッチパネル（Z/R 式・拘束一覧）、パラメータ表。 |
| `ui/physics_panel.py`, `mesh_panel.py`, `analysis_panel.py`, `results_panel.py`, `panel_helpers.py` | タブに連動する右パネル。 |
| `ui/result_renderer.py`, `result_model.py`, `result_view.py` | 場の描画（Qt 非依存。ver2.3 `result_viewer._draw_on` の移植。`Figure` だけを受け取る）、表示中の結果と選択・オプション、`FigureCanvasQTAgg` のビュー。 |
| `ui/dialogs/{run_confirm,export_field,gif,about}.py` | 実行確認、場の書き出し、GIF、バージョン情報。 |
| `batch.py` | `.axiproj` を GUI なしで: `Project.open` → `set_params`（`DocumentStore.update_param` で式付きの点と拘束を解き直す。QApplication 不要）→ `run`（`analysis_pipeline.run_job` を同じプロセスで。既定の出力は `batch/<日時>-<種類>/`、`register=True` で結果履歴）。CLI `axicavity-fem-run`（`main`）。 |
| `core/revolve.py` | 2D 三角形分割を z 軸まわりに回転した wedge グリッド、節点振幅から 3D の E / H（e^{jnφ}: cos 型 E_z・E_r・H_φ、sin 型 E_φ・H_z・H_r）、Ψ の等高線の折れ線。numpy と matplotlib.tri だけ。 |
| `ui/view3d_scene.py`, `ui/view3d_window.py` | 3D 表示（M10）: `Scene3D(plotter)`（pyvista。壁面・子午面・横断面・矢印・電気力線・スカラーバー。`pv.Plotter(off_screen=True)` でテスト）、`View3DController`（結果モデルに追従、アニメーション、PNG / GIF）、`View3DPanel`、`View3DWindow`（ToolWindow + QtInteractor。offscreen では作らない）。 |
| `ui/icon_set.py`, `ui/icons/*.svg`, `tools/make_icons.py` | アイコン（`ACTION_ICONS`: i18n キー → SVG 名。SVG は生成スクリプトの出力そのもの）。 |
| `i18n/tr.py`, `ja.json`, `en.json` | 表示文字列（`tr("section.key", **placeholders)`）。 |

---

## 3. データモデル

### 3.1 統一 HDF5 スキーマ（schema_version 2.2。[HDF5_SCHEMA.md](HDF5_SCHEMA.md) 詳細）

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

## 4. 規約と落とし穴（新セッション必読）

### 4.1 境界条件（[BC_NAMING.md](BC_NAMING.md)）
- 正規名は **PEC / E-short / M-short** の 3 つのみ。`Dirichlet` は PEC エイリアス（警告付き）。
- FEM 行列上の扱いは **DOF により反転**:
  - TM0: PEC/E-short → Neumann（自然）、**M-short → Dirichlet (H_φ=0)**、**軸 r=0 → Dirichlet**。
  - HOM: PEC/E-short → Dirichlet (E_tan=0)、M-short → Neumann、軸 r=0 は n=0 で Neumann / n≥1 で Dirichlet。
- 軸 r=0 を TM0 で Dirichlet にするのは ver1 準拠かつ物理的に正しい（H_φ は軸上で消える）。

### 4.2 進行波 PBC 位相
- 必ず `physics_core.pbc_phase` 経由。規約は **`x_max = e^{-jθ}·x_min`**（時間規約 e^{+jωt}）。
  直接 `exp(±1j*θ)` を書かないこと（ver1 で符号バグの実績あり）。

### 4.3 P_flow
- z=z_min 断面のラインインテグラル。TM0 係数 **−π/(ωε₀)**、HOM 係数 **+π/(ωμ₀)**。
  数値プリミティブ（3 点ガウス線積分）は `physics_core.power_flow.edge_line_integral_3point`。

### 4.4 eigsh の不安定性（TM0 定在波）
- ver1 由来の `eigsh(sigma, which='LA')` は σ 近傍で **数値的に不安定**で、同一行列でも
  ARPACK の乱数始ベクトル依存で返すモード集合がばらつく（K を増やすと ARPACK エラーも）。
- → ver2 の開発中、ver1 との一致は「全体行列のビット一致」と基本モード TM010 の値対応で確かめた。
  実用堅牢化（`which='LM'`+ソート等）は未対応の検討課題。

### 4.5 蓄積エネルギー U（HOM）
- eigsh の M 正規化（vᴴMv=1）により、HOM の U は全モードで **2π·ε₀**（≈5.563e-11）になる
  （HOM 行列は係数 2 を省略した構成のため）。これは正常。Q の差は P_loss が担う。

### 4.6 gmsh のグローバル状態
- `load_mesh` は呼び出しごとに `finalize()→initialize()` で**フレッシュな状態**にしてから読む。
  これを怠ると 2 次曲線メッシュの再パラメータ化が前段の状態に依存し、結果がぶれる（実害あり、修正済み）。
- EXE では gmsh を使わず `gui/mshlite` が同じ API で読む（gmsh の読込と要素・Physical・節点の並びが一致することを
  `tests/test_mshlite.py` で確認。座標は gmsh の Windows 版の数値の読み取りの丸めで最下位ビットが違うことがある）。

### 4.7 matplotlib バックエンド
- `reports/plot_common.py` は import 時に **Agg** を設定（CLI/オフスクリーン用）。
- GUI（`gui/ui/result_renderer.py` / `result_view.py`）は `Figure` + `FigureCanvasQTAgg` だけを使い、
  **`pyplot` / `plot_common` / `matplotlib.use` を import しない**（`tests/test_no_pyplot_in_gui.py` が監視）。
  三角形分割・PEC 幾何・誘電体界面は `shared/boundary_groups.py` 経由。GIF はオフスクリーンの Agg キャンバスに同じ
  `ResultRenderer.draw` で描く。
- gmsh は非メインスレッドで initialize できず、C レベルで fd 1 に書く。GUI プロセスでは gmsh を初期化せず、
  メッシュ生成・solve・post・report・export は **子プロセス**（`gui/jobs/runner.py`。fd 1 を `log.txt` に付け替え）。
  GEO / Python スクリプトの書き出しはテキスト生成だけなのでメインスレッドで行う。EXE ではメッシュ生成だけがさらに
  別プロセスのメッシャで動く（§10）。

---

## 5. 検証戦略

| テスト | 何を確かめるか |
|---|---|
| `test_core_identical.py` | 計算コアのファイルが変わっていない（SHA-256 の一覧 `core_files.sha256`） |
| 解析解との比較（`test_target_frequency.py`、`examples/accuracy_verification/` など） | 球形空洞・ピルボックスの周波数、誘電体損失 `Q_diel = 1/tanδ` |
| 回帰（`test_pbc_phase_sign.py`、`test_power_flow_lineintegral.py`） | 過去のバグ（PBC 位相の符号、P_flow の係数）を繰り返さない |
| `test_convert_roundtrip.py` | 全サンプルの取り込み → 書き出しの往復で、メッシュ（Physical 名・節点数）と周波数が一致 |
| `test_mshlite.py`、`test_meshing.py`、`test_launcher.py` | EXE の構成（gmsh 無しの読込・メッシャ・入口）が開発環境と同じ結果になる |
| `test_known_bugs_v23.py` | v2.3 の GUI の既知バグ 14 件が再現しない |
| GUI（`test_ui_*.py`、`test_sketch_*.py`） | offscreen での配線・ツール・パネル。見た目は手動で確認 |

新しい数値機能を足したら、解析解か、既存の結果との一致テストを足す（固有ベクトルの任意性を避けるため、
可能なら同じ固有ベクトルを両方に与える設計にする）。v1 との比較（ver2 開発時）は GitHub のタグ `v2.3.0` 以前の履歴を参照。

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
- GUI に出すなら `gui/ui/result_renderer.py` の `ResultRenderer.draw`（`ViewOptions` にオプションを足し、
  `results_panel.py` のチェックボックスと `main_window.py` の `RESULT_OPTIONS` に結線）。plot_common は import しない。

### 6.4 GUI を変更する（v3）
- **状態は `app/store.py` の `DocumentStore` に置き、UI はシグナルで追従する**（UI から直接 document を書き換えない）。
  設定を足すときは `core/document.py` の dataclass → `core/serialize.py` → `store.set_*` → パネル → i18n の順。
- リボンのボタンは `main_window._build_actions` の `_action(tab, "section.key", slot, shortcut, checkable)`。
  表示名は `i18n/ja.json` / `en.json` の `section.key`、説明は `tip.section.key`（**全ボタンに書く**。
  `tests/test_docs_manual.py` がマニュアルとの対応を、`tests/test_ui_main_window.py` がツールチップの有無を見る）。
  アイコンは `ui/icon_set.py` の `ACTION_ICONS` に名前を足し、`tools/make_icons.py` の `ICONS` に線画を書いて
  `python tools/make_icons.py` で SVG を書き出す（`tests/test_ui_icon_set.py` が生成物との一致を見る）。
- 重い処理は子プロセスへ: `jobs/runner.py` の `JOB_KINDS` と `run_child` に種類を足し、`pipeline.py` /
  `analysis_pipeline.py` に本体を書く。GUI 側は `JobSession.submit(kind, job_dir)` → `job_finished` で受ける。
  CLI 相当のコマンドは `jobs/commands.py` で組み立て `project.log_command` で `command.log` に残す。
- ダイアログは `ui/dialogs/`。テストでは `exec` を monkeypatch して受け付ける。
- テスト: UI は offscreen（`QT_QPA_PLATFORM=offscreen`、pytest-qt）。`MainWindow` は `qtbot.addWidget` しない
  （fixture の後始末より先に close されて未保存確認で止まる）。実ウィンドウの配置は `tests/_main_window_check.py`
  を別プロセスで（`test_ui_main_window_real.py`）。
- 変更したら `python tools/manual_screenshots.py` で `docs/images/v3_*.png` を撮り直し、USER_MANUAL.md を更新する。

### 6.5 v2.3 の wx GUI との対応
| v2.3 | v3 |
|---|---|
| `gui/multi_region_editor.py`（点/線/円弧/ループ/領域の手動編集） | `gui/core/sketch/*` + `sketch/*` + `ui/sketch_view.py`（閉領域は自動検出） |
| `gui/main_frame.py`（FEM タブ、subprocess で CLI） | `ui/analysis_panel.py` + `jobs/analysis_controller.py`（子プロセスで `cmd_*` を直接） |
| `gui/result_viewer.py` `_draw_on` | `ui/result_renderer.py` `ResultRenderer.draw` |
| `.gmshproj`（単一ファイル） | `.axiproj` + `.axiproj.data/`（`io/axiproj.py`。`.gmshproj` は取り込み / 書き出し） |
| `command.log`（プロジェクトの隣） | `<名前>.axiproj.data/command.log` |

---

## 7. ver1 → ver2 対応表（移植元）

| ver2 | ver1 |
|---|---|
| `shared/element_functions.py` | `FEM_HOM_code/FEM_element_function.py` |
| `shared/quadrature.py` | `*/gaussian_quadrature_triangle.py` |
| `shared/mesh_loader.py`(+boundary_groups) | `FEM_HOM_code/mesh_reader.py`, `FEM_code/...load_gmsh_mesh` |
| `shared/eigensolver.py` | `FEM_HOM_code/eigensolver.py` |
| `physics_core/pbc_phase.py` | `FEM_HOM_code/boundary_conditions.py`(位相部) |
| `fem_tm0/*` | `FEM_code/FEM_helmholtz_TM0_calclation.py`, `field_calculator.py`, `post_process_unified.py` |
| `fem_hom/*` | `FEM_HOM_code/element_assembly.py`, `boundary_conditions.py`, `field_calculator_hom.py`, `post_process_hom.py` |
| `reports/export_fields.py` | `*/export_field_data.py` |
| `reports/plot_*`, `report_builder.py` | `plot_common.py`, `FEM_*/plot_utils*.py`, `*_html_report` |
| `gui/*`（v2.3） | `MyFrame.py`, `MyFrameUI.py`, `PointLineEditorPanel.py`, `ResultViewer*.py`, `app.py` |
| `gui/*`（v3） | 同じ作者の 3 次元 FEM の GUI（EM-CAD-py。非公開）のスケッチ・リボン・子プロセス実行の仕組み、v2.3 `gui/result_viewer.py` |

---

## 8. 既知の未対応・今後の課題

- v3 GUI の 3D 表示（pyvista）は `QT_QPA_PLATFORM=offscreen` では `QtInteractor` が落ちるので、GUI テストでは
  ウィンドウを作らない（コントローラは偽の scene で、描画は `pv.Plotter(off_screen=True)` で、ウィンドウは
  実ウィンドウの `_main_window_check.py` で）。メッシュを作り直すと gmsh の版差で曲線境界上の節点が動き、
  周波数が同梱の参照結果と 1e-4 程度ずれる（`tools/verify_samples.py` がコアの差とメッシュの差を分けて報告）。
  手元の参照 h5（`axicavity_fem_version` が 2.3.0 より前）は 4 点求積で作られていて、最終コアとは 1e-3 程度ずれる
  （同スクリプトは版を見て許容を切り替える）。
- GIF アニメーション（進行波）: 結果タブの「GIF 保存…」とレポートの `--animate`（固定キャンバス Agg + `render_gif`）。
- TM0 定在波 `eigsh(which='LA')` のモード選択不安定性（§4.4）。`which='LM'`+ソートでの堅牢化は要再検証。
- 表示系（GUI ウィンドウ操作・PNG/HTML/GIF の見た目）は自動テスト不可 → 手動確認。
- レポート GIF はモード数×フレーム数を生成するため遅く、1 モード数 MB になり得る（`optimize=False`）。
  フレーム数・最適化のチューニングは要検討。
- stub: `shared/logging.py`, `shared/sparse_io.py`, `physics_core/normalization.py`,
  `physics_core/conventions.py`（必要になったら実装）。

---

## 9. 関連ドキュメント

- [ARCHITECTURE.md](ARCHITECTURE.md) — レイヤ設計の概要。
- [HDF5_SCHEMA.md](HDF5_SCHEMA.md) — 出力ファイル仕様。
- [BC_NAMING.md](BC_NAMING.md) — 境界条件の物理-数学対応表。
- [../PHYSICS_AND_CONVENTIONS.md](../PHYSICS_AND_CONVENTIONS.md) — 物理規約（時間規約・規格化・P_flow 等）。
- [../USER_MANUAL.md](../USER_MANUAL.md) — エンドユーザ向け操作手順（v3 GUI のチュートリアルと v2.3 → v3 対応表）。
- [../packaging/README.md](../packaging/README.md) — Windows 版（EXE）の作り方。
- [../CHANGELOG.md](../CHANGELOG.md) — リリースノート。

---

## 10. Windows 版（EXE）

`AxiCavity-FEM.exe` は Nuitka の standalone（onefile ではない）で作る 1 つの EXE で、GUI とコマンドラインを兼ねる。
作り方は [../packaging/README.md](../packaging/README.md)。

**入口**: `gui/launcher.py` の `main`（Nuitka の main スクリプト `packaging/AxiCavity-FEM.py` から呼ぶ）。引数なし・
`gui`・`.axiproj` などのファイル → GUI、`solve|post|info|export|plot|report` → `cli.main.cli_main`、`run` →
`gui.batch.main`、`job <kind> <dir> --events` → `gui.jobs.runner.main`（GUI の子プロセス）、`version`、`selftest`。
`--windows-console-mode=attach` なので、端末から呼べば出力が端末に出て、ダブルクリックならコンソールは出ない。

**子プロセス**: Nuitka standalone の `sys.executable` は存在しない `<dist>/python.exe` を指すので、EXE のパスは
`frozen.executable_path()`（`GetModuleFileNameW`）で得る。`runner.program_command()` は EXE なら `[EXE, "job"]`
（runner の種類名 post / report / export はコア CLI と同じ名前なので `job` の下に置く）。EXE かどうかは
`"__compiled__" in globals()`（Nuitka は `sys.frozen` を設定しない）。

**gmsh を本体に入れない理由と仕組み**: pypardiso が使う Intel MKL のライセンス（Intel Simplified Software License）は
GPL のソフトウェアと組み合わせて配布することを認めていない。gmsh（GPL-2.0-or-later）を同じプロセスに入れると
本体が GPL の結合物になるので、gmsh は別プログラムにする:

- `.msh` の読込（コアの `shared/mesh_loader.py` の `import gmsh`）: launcher / runner が起動時に
  `mshlite.install_as_gmsh()` で `sys.modules["gmsh"]` に純 Python の互換モジュールを置く（コアは無変更）。
  ビルドでは `--nofollow-import-to=gmsh`。
- メッシュ生成（`export_msh_multi_region`）: `jobs/meshing.generate_msh` が `mesher/python.exe mesher/axicavity_mesh.py
  mesh job.json` を子プロセスで動かす（`CREATE_NO_WINDOW`、Job Object で親が終われば一緒に終わる）。メッシャは
  埋め込み Python（PSF）+ pip の gmsh（GPL）+ `axicavity_mesh.py` + コアの `gmsh_export_occ.py` / `multi_region_model.py`
  （ソースのまま。numpy 不要）で、gmsh のソース tarball と GPL 全文を同梱する（`packaging/build_mesher.py`）。
- GEO / Python スクリプトの書き出しはテキスト生成だけなので、本体の互換モジュールのままで動く。

開発環境で同じ構成を試すには `AXICAVITY_GMSH=lite`（gmsh の代わりに mshlite）と `AXICAVITY_MESHER=process`
（メッシュ生成をメッシャのプロセスで）。`python -m axicavity_fem.gui.launcher selftest` をこの 2 つを付けて走らせると、
本体・子プロセスとも gmsh を読み込まずに解析まで通る（`tests/test_launcher.py`）。

**Intel MKL**: pip の `mkl` の DLL（`mkl_rt`・`mkl_core`・`mkl_tbb_thread`・CPU 別の `mkl_avx*` など）と `tbb12.dll` を
`Library/bin/` にデータファイルとして入れ、`MKL_THREADING_LAYER=TBB` と `PYPARDISO_MKL_RT` を main スクリプトで設定する。
intel-openmp（別の EULA）は入れない。

**落とし穴**: ビルド先は OneDrive の外（同期・ロック）。Claude デスクトップアプリから実行するときは `%LOCALAPPDATA%` も
避ける（MSIX の仮想化で別の場所に書かれる）。`--python-flag=isolated` なので PYTHONPATH などは効かない。
アイコン・翻訳ファイルの同梱漏れは黙って空になるので `selftest` で確かめる。Nuitka は `pyvista` の import で
`pooch` などを連れてくる（`sympy`・`requests` などは使わないので `--nofollow-import-to`）。Nuitka の EXE は既定で
引数のどこかに `-m <x>` / `-c <x>` があると自己呼び出しとみなして止まるので、コア CLI の `-m <mesh>` のために
`--no-deployment-flag=self-execution` を付ける。zig のキャッシュはビルド先ごとに分ける（共有すると別のプロジェクトの
定数が入った EXE ができ、"Frozen object named 'encodings' is invalid" で起動しない。packaging/README.md）。
qtpy は import 時に GPL-3.0 のみの Qt Data Visualization を読みにいくので、ビルドから外している（MKL と同じプロセスに
GPL のものを載せない。packaging/README.md の「ライセンス上の注意」）。
