# 2次元有限要素法 電磁場解析ツール ユーザーマニュアル

本ツールは、Gmshを利用したメッシュ作製から、有限要素法による電磁場解析、ポストプロセス（工学パラメータ計算）、可視化・レポート生成までを一貫して行うGUIアプリケーションです。円筒座標系の軸対称 TM0 モード（定在波・進行波）および高次方位角モード（HOM）を対象とした空洞共振器・加速管の設計・解析に対応しています。

---

## 全体ワークフロー

```
[形状作成] → [メッシュ生成] → [FEM計算 (Solver)] → [パラメータ計算 (Post-Process)]→ [結果確認] → [レポート生成] 
   タブ1         タブ1            タブ2              タブ2                        タブ2           タブ2
```

---

## 1. 起動方法

```bash
python app.py
```

ウィンドウが表示されたら、上部のタブで操作を切り替えます。

- **タブ1: Mesh Geometry Edit** — 形状の作成とメッシュ生成
- **タブ2: FEM Analysis TM0** — TM0 モードの定在波・進行波の計算とポストプロセス（RadioBox で切替）
- **タブ3: FEM Analysis HOM** — 高次方位角モード（HOM）の定在波・進行波の計算と電場可視化

---

## 2. タブ1：形状作成とメッシュ生成

### 2.1 初期設定

1. **単位の選択**: 「Unit」プルダウンから設計単位（m / cm / mm / inch）を選択します。
2. **表示範囲の設定**: 「X min/max」「Y min/max」を入力し「Update Limits」を押してグラフ範囲を調整します。

### 2.2 形状の作成（打点とループ閉鎖）

1. グラフエリアをクリックして頂点を順に打ちます（自動的に直線で結ばれます）。
2. **3点以上** 打ったら「Close Loop」ボタンを押して形状を閉じます。

> **注意**: 円筒座標のr軸（Y軸）が **r ≥ 0** の範囲で形状を作成してください。
> ループの向きは **反時計回り（CCW）** を推奨します。時計回りの場合は自動的に警告が出るので「Reverse Loop」で反転してください。

### 2.3 形状の微調整

「Edit Mode」ラジオボタンで **Edit Points** / **Edit Lines** を切り替えます。

**Edit Points（点編集）**
- **移動**: グラフ上の点をドラッグ。
- **座標指定**: 点を選択し X・Y 欄に数値入力 → 「Update Coords」。
- **挿入**: 線分の上をダブルクリック。
- **削除**: 点を選択 → 「Delete Point」。

**Edit Lines（線編集）**
- **円弧化**: 直線を選択 → 「Convert to 90deg Arc」。
- **円弧調整**: 円弧を選択し「Center X/Y」を編集 → 「Update Center」。
- **境界条件設定**: 「Physical Name」欄に境界条件名を入力します（例: `PEC`, `Dirichlet`）。ここで設定した名前が FEM 解析で境界条件として認識されます。

### 2.4 メッシュ生成とエクスポート

1. **Mesh Size (lc)**: 節点の間隔を入力します（単位は選択した Unit と同じ）。
2. **Mesh order**: 「1st」（1次要素）または「2nd」（2次要素）を選択します。高精度解析には2次要素を推奨します。
3. **Export MSH (Direct)**: Gmsh SDK を内部で呼び出してメッシュファイル（.msh）を直接生成します。生成されたパスは「FEM Analysis」タブの「Mesh File」に自動セットされます。

> その他のエクスポート:
> - **Export to .geo**: Gmsh の形状ファイルを出力。Gmsh GUI で確認可能。
> - **Export Python Script**: Gmsh Python API スクリプトを出力。
> - **Export Superfish**: Superfish 形式（.af）で出力。

### 2.5 プロジェクトの保存と管理

| 操作 | 方法 |
|---|---|
| 保存 | `File > Save` または `Ctrl+S` → `.gmshproj` 形式 |
| 読み込み | `File > Load Project` |
| Superfish インポート | `File > Import Superfish File`（.af 形式）|

---

## 3. タブ2：TM0 解析（定在波・進行波統合）

共振空洞の TM0 モードを解析します。**Wave Type** RadioBox で定在波（Standing Wave）と進行波（Traveling Wave）を切り替えます。

### 3.1 ソルバーの実行

1. 「Mesh File (.msh)」にメッシュファイルのパスを入力するか、「Browse...」で選択します（タブ1からの自動セットも可）。
2. **Number of Modes**: 計算する固有モード数（デフォルト: 10）を設定します。
3. **Mesh Order**: FEM 要素次数（1: 1次要素 / 2: 2次要素）を確認します。高精度解析には2次要素を推奨。
4. **Wave Type**: 「Standing Wave」または「Traveling Wave」を選択します。
5. **Phase Shift (deg)**: 進行波選択時のみ有効。周期境界のセル間位相差を入力します（後述の形式を参照）。
6. 「**Run Solver**」を押します。

**出力ファイル**:
- 定在波: `*_analysis.h5`、`*_frequencies.txt`
- 進行波: `*_traveling_analysis.h5`、`*_frequencies.txt`
- `.h5` にはメッシュ・固有値・固有ベクトルが HDF5 形式で格納されます。

FEM ログにリアルタイムで進捗が表示されます。

### 3.2 ポストプロセス（パラメータ計算）

1. 「Raw Result File」にソルバー出力の `.h5` ファイルが自動セットされていることを確認します（または Browse で選択）。
2. **Conductivity [S/m]**: 空洞壁面の導電率（デフォルト: 無酸素銅 `5.8e7` S/m）を入力します。
3. **Beta (v/c)**: 加速粒子の相対速度 β を入力します（光速粒子なら `1.0`）。
4. 「**Run Post-Process**」を押します。

**計算される物理量（定在波）**:

| パラメータ | 記号 | 説明 |
|---|---|---|
| 蓄積エネルギー | $U$ [J] | 空洞内の電磁エネルギー |
| 壁面損失 | $P_{loss}$ [W] | 表面抵抗による損失電力 |
| Q 値 | $Q = \omega U / P_{loss}$ | エネルギー損失の逆数（共振の鋭さ） |
| 実効電圧 | $V_{eff}$ [V] | 軸上の加速電圧 |
| R/Q | $R/Q$ [Ω] | 加速効率の指標（β 依存） |

**出力ファイル** (`*_processed.h5` および `*_parameters.txt`):
- `_processed.h5`: 規格化された固有ベクトルと計算パラメータを格納
- `_parameters.txt`: 全パラメータのテキストサマリー

### 3.3 レポート生成

1. 「Post-Processed Result File」に処理済み `.h5` ファイルが自動セットされていることを確認します。
2. 「**Create animation**」チェックボックスは進行波モードでのみ意味を持ちます（オフ：アニメ無し / オン：時間発展 GIF を作成）。アニメーション生成は時間がかかるため、デフォルトはオフです。
3. 「**Create HTML Report**」を押します（チェック OFF でも分布図の生成があるため、それなりに時間がかかります）。
4. 「**Open HTML Report**」を押すとブラウザでレポートが開きます。

#### TM0 モードのレポート内容

- メッシュ概要図
- 各モードの磁場コンター + 電場ベクトル図（定在波）または Real/Imag/Magnitude（進行波）
- 軸上電場 $E_z$ 分布図
- 全パラメータのサマリー表（Q, R/Q, シャントインピーダンス, 群速度, 減衰定数 など）
- 進行波モードで「Create animation」がオンの場合、各モードの時間発展 GIF アニメーション

#### HOM モードのレポート内容

- メッシュ概要図
- 各 (n, 位相シフト, モード) について E-field と H-field の左右並置プロット
- 軸上 $E_z$ 分布図
- 蓄積エネルギー $U$、壁損 $P_{loss}$、Q 値のサマリー表
- 進行波モードで「Create animation」がオンの場合、時間発展 GIF アニメーション

HOM レポートは方位角オーダー $n$ ごとに章立てされ、進行波の場合は位相シフト値ごとに分かれます。

### 3.4 結果の対話的な確認（Result Viewer）

「**Result Viewer**」ボタンを押すと、インタラクティブなビューアが起動します。

| 操作 | 機能 |
|---|---|
| Mode プルダウン | 表示するモードの選択 |
| H_theta (Color) | 磁場 $H_\theta$ のカラーマップ表示 |
| E Lines / Levels | 電気力線（等高線）の表示・本数調整 |
| E Vectors / Z・R | 電場ベクトルの表示・格子点数調整 |
| Show Mesh | 元のメッシュ形状を重畳表示 |
| グラフをダブルクリック | その座標の電磁場値（$H_\theta, E_z, E_r, \|E\|$）をポップアップ表示 |

---

## 4. タブ3：HOM 解析（Higher-Order Mode）

加速管・空洞の高次方位角モード（azimuthal order n ≥ 1）および TM0（n=0）の定在波・進行波を計算します。

### 4.1 ソルバーの実行

1. 「Mesh File (.msh)」でメッシュファイルを選択します。
2. **Number of Modes**: 計算するモード数（デフォルト: 10）。
3. **Element Order**: **「2nd」（2次要素）を強く推奨します**。2次要素は1次要素と比べて精度が1〜2桁向上します（後述「解析精度について」参照）。
4. **Mode Orders (e.g. 0 1 2)**: 計算する方位角次数 n をスペース区切りで入力します。
   - `0` → TM0 モード
   - `1` → TE/TM の n=1（双極子）モード
   - `0 1 2` → 複数次数を一括計算
5. **Wave Type**: 「Standing Wave」または「Traveling Wave」を選択します。
6. **Phase Shift (deg)**: 進行波選択時のみ有効。位相形式は TM0 タブと同じです（後述）。
7. **Output File**: 出力 `.h5` ファイルのパスを指定します（空欄時は `*_hom.h5` に自動設定）。
8. 「**Run Solver**」を押します。

**出力ファイル**:
- `*_hom.h5`: 主計算結果（HDF5）
  - `/mesh/`: メッシュ頂点・要素・エッジ情報
  - `/results/n{n}/Normal/mode_{i}/`: 定在波モード（固有ベクトル・周波数・固有値を格納）
  - `/results/n{n}/Periodic/PB_Phase_XXX/`: 進行波モードの固有ベクトル（複数位相対応）
- `*_hom_frequencies.txt`: 計算直後に自動生成される周波数一覧（約15桁精度）。計算完了後すぐに周波数を確認できます。

### 4.2 電場の可視化

1. 「Result File (.h5)」で計算結果ファイルを選択します。
2. **n**: 表示する方位角次数。
3. **Mode**: モードインデックス（0 始まり）。
4. **Phase (deg)**: 表示する位相（進行波の場合に有効）。
5. **Snapshot (deg)**: 進行波の瞬時場を表示する時間位相（空欄時は実部を表示）。
6. **Density**: 表示するベクトルの密度（0.1〜1.0 程度）。
7. 「**Plot Field**」を押します。

**表示内容**:
- n=0: $E_z$、$E_r$ のベクトル場 + メッシュ・PEC境界の重畳表示
- n≥1: 左パネルに $E_z$/$E_r$ ベクトル、右パネルに $E_\theta$ コンター

---

## 5. コマンドログとテキスト出力

GUIから実行したすべてのコマンドは、プロジェクトディレクトリの **`command.log`** に自動記録されます。これにより:
- 後から CLI でバッチ処理を行うときのコマンドがすぐに確認できます。
- 計算条件（メッシュファイル、モード数、位相、導電率など）の履歴が残ります。

**生成されるテキストファイルの一覧**:

| ファイル名 | 生成タイミング | 内容 |
|---|---|---|
| `*_frequencies.txt` | Run Solver 完了後 | 計算周波数の一覧 |
| `*_parameters.txt` | Run Post-Process 完了後 | Q値・R/Q等の全パラメータ表 |
| `command.log` | コマンド実行のたびに追記 | 実行コマンド履歴（タイムスタンプ付き） |

---

## 6. コマンドライン（CLI）での実行

GUI を使わずにコマンドラインから直接実行できます。TM0 と HOM の引数体系は統一されています。

### 6.1 位相指定の共通形式（`-p/--phase`）

TM0・HOM どちらも `-p` で同じ記法を使用します（単位: 度）。

| 指定方法 | 例 | 意味 |
|---|---|---|
| 単一値 | `-p 120` | 120° のみ |
| カンマ区切り | `-p "60,90,120"` | 60°、90°、120° の3点 |
| レンジ（start:end:step） | `-p "0:180:20"` | 0° から 180° を 20° 刻み（10点） |

> **定在波と進行波の切り替え**: `-p 0`（または `-p` 省略）で定在波、それ以外の値を指定すると自動的に進行波として計算されます。

---

### 6.2 TM0 ソルバー（`FEM_code/run_analysis.py`）

| 引数 | 短縮形 | デフォルト | 説明 |
|---|---|---|---|
| `--mesh` | `-m` | （必須） | メッシュファイル (.msh) |
| `--elem-order` | — | `2` | FEM 要素次数（1 または 2） |
| `--num-modes` | — | `10` | 計算する固有モード数 |
| `--output` | `-o` | `analysis_result.h5` | 出力 HDF5 ファイル |
| `--phase` | `-p` | `0.0` | 位相シフト（定在波: 0、進行波: 非0） |

```bash
# 定在波（2次要素、10モード）
python FEM_code/run_analysis.py -m cavity.msh --elem-order 2 --num-modes 10 -o result.h5

# 進行波（単一位相 120°）
python FEM_code/run_analysis.py -m cavity.msh --elem-order 2 --num-modes 10 -o result.h5 -p 120

# 進行波（位相スキャン: 0°〜180° を 20° 刻み）
python FEM_code/run_analysis.py -m cavity.msh --elem-order 2 --num-modes 10 -o result.h5 -p "0:180:20"
```

---

### 6.3 TM0 ポストプロセス（`FEM_code/post_process_unified.py`）

```bash
# パラメータ計算のみ（高速）
python FEM_code/post_process_unified.py -i result.h5 -o result_processed.h5 -c 5.8e7 -b 1.0 --mode calc

# レポート生成のみ（アニメ無し → 高速）
python FEM_code/post_process_unified.py -i result_processed.h5 --mode report --no-anim

# レポート生成のみ（進行波の場合は時間発展アニメも作成）
python FEM_code/post_process_unified.py -i result_processed.h5 --mode report

# 一括実行（計算 + レポート）
python FEM_code/post_process_unified.py -i result.h5 -o result_processed.h5 -c 5.8e7 -b 1.0
```

`--no-anim` を付けると進行波モードでも GIF アニメーション生成をスキップします。

---

### 6.4 HOM ソルバー（`FEM_HOM_code/run_analysis.py`）

| 引数 | 短縮形 | デフォルト | 説明 |
|---|---|---|---|
| `--mesh` | `-m` | （必須） | メッシュファイル (.msh) |
| `--az-order` | — | `[0]` | 方位角モード次数 n（複数指定可） |
| `--elem-order` | — | `1` | FEM 要素次数（**2 を強く推奨**。精度が1〜2桁向上） |
| `--num-modes` | — | `10` | 出力するモード数 |
| `--output` | `-o` | 自動生成 | 出力 HDF5 ファイル |
| `--phase` | `-p` | `0.0` | 位相シフト（定在波: 0、進行波: 非0） |

```bash
# n=0（TM0）定在波 — 2次要素を推奨
python FEM_HOM_code/run_analysis.py -m cavity.msh --az-order 0 --elem-order 2 --num-modes 10 -o hom_result.h5

# n=0, 1, 2 を一括計算（定在波、2次要素）
python FEM_HOM_code/run_analysis.py -m cavity.msh --az-order 0 1 2 --elem-order 2 --num-modes 10 -o hom_result.h5

# n=1 進行波（単一位相 120°）
python FEM_HOM_code/run_analysis.py -m cavity.msh --az-order 1 --elem-order 2 --num-modes 10 -o hom_result.h5 -p 120

# n=1 進行波（位相スキャン: 0°〜180° を 20° 刻み）
python FEM_HOM_code/run_analysis.py -m cavity.msh --az-order 1 --elem-order 2 --num-modes 10 -o hom_result.h5 -p "0:180:20"
```

計算完了後、`hom_result_frequencies.txt` が自動生成され、全モードの周波数を約15桁精度で確認できます。

---

### 6.4.1 HOM ポストプロセス + レポート生成（`FEM_HOM_code/post_process_hom.py`）

```bash
# パラメータ計算のみ（_processed.h5 が出力される）
python FEM_HOM_code/post_process_hom.py hom_result.h5 --cond 5.8e7 --mode calc

# レポート生成のみ（アニメ無し → 高速、デフォルト）
python FEM_HOM_code/post_process_hom.py hom_result_processed.h5 --mode report --no-anim

# レポート生成のみ（進行波モードの GIF アニメーションを含む）
python FEM_HOM_code/post_process_hom.py hom_result_processed.h5 --mode report

# 一括実行（パラメータ計算 + レポート、--no-anim はデフォルトで適用したい場合のみ付与）
python FEM_HOM_code/post_process_hom.py hom_result.h5 --cond 5.8e7 --no-anim
```

レポートは `<入力 h5 ベース名>_report/index.html` に生成されます。各 $n$ ごと、進行波の場合は位相シフトごとに章立てされ、モードごとに E-field と H-field の左右並置プロット、軸上 $E_z$ 分布、工学パラメータ（$U$, $P_{loss}$, Q）が含まれます。

---

### 6.5 HOM 電場可視化（`FEM_HOM_code/plot_hom_field.py`）

```bash
# n=0, モード0 の電場（定在波）
python FEM_HOM_code/plot_hom_field.py hom_result.h5 --n 0 --mode 0

# n=1, モード0 の電場（進行波、位相 120°）
python FEM_HOM_code/plot_hom_field.py hom_result.h5 --n 1 --mode 0 --periodic --phase 120

# 瞬時場（時間位相 45°）
python FEM_HOM_code/plot_hom_field.py hom_result.h5 --n 1 --mode 0 --periodic --phase 120 --snapshot 45

# ベクトル密度を変更して保存
python FEM_HOM_code/plot_hom_field.py hom_result.h5 --n 0 --mode 0 --density 0.5 --save
```

---

### 6.6 電磁場マップ出力（`*/export_field_data.py`）

ビーム計算等のために、解析結果から指定領域の電磁場マップを HDF5 と TXT で出力します。TM0 と HOM で同一の引数体系を採用しています。

**出力形状**:

| `--shape` | 内容 | 必要パラメータ |
|---|---|---|
| `area` | 矩形領域 (`z_min..z_max` × `r_min..r_max`) のグリッドマップ | `--z-range`, `--r-range`, `--nz`, `--nr` |
| `line` | 任意の 2 点 P1, P2 を結ぶ直線上 | `--p1 z,r`, `--p2 z,r`, `--npts` |
| `axis` | 軸上 (`r=0`) の直線（簡易指定） | `--z-range`, `--npts` |

**入力電力スケーリング**: `--scale-to-power P [W]` を指定すると、現在の規格化（軸上ピーク E_z = 1 V/m）を解除し、空洞内の壁面損失が P [W] となるように電磁場全体を再スケーリングして出力します。本機能は `processed.h5` (Run Post-Process 完了後のファイル) でのみ利用可能で、メタデータに `p_loss_W`, `stored_energy_J`, `q_factor` が併記されます。

**出力モード（進行波）**:

| 指定 | 内容 |
|---|---|
| デフォルト | **複素振幅** (Re/Im 列) を出力。後段ツールで任意の時間発展ができる |
| `--instant` | 指定 `--time-phase` での実瞬時値を出力 |

定在波は本質的に実数値のため、常に実瞬時値（peak）を出力します。

**出力フォーマット**: `--format {h5,txt,both}` (default `both`)。HDF5 と TXT の両方、またはいずれか単独を選べます。

**TM0 の例**:

```bash
# 矩形領域（1 kW 入射相当にスケーリング）
python FEM_code/export_field_data.py -i result_processed.h5 -m 0 \
    --shape area --z-range 0,0.1 --r-range 0,0.05 \
    --nz 200 --nr 100 --scale-to-power 1000.0 \
    -o field_area

# 軸上電場（V_eff の検算用、500 点サンプル、HDF5 のみ）
python FEM_code/export_field_data.py -i result_processed.h5 -m 0 \
    --shape axis --npts 500 --scale-to-power 1000.0 \
    --format h5 -o field_axis

# 進行波の実瞬時値（位相 120°、time-phase 30°）
python FEM_code/export_field_data.py -i result_processed.h5 -m 0 \
    --shape area --phase 120 --instant --time-phase 30 \
    -o field_TW_instant
```

**HOM の例**:

```bash
# n=1 ダイポール、m=0 の矩形マップ
python FEM_HOM_code/export_field_data.py -i hom_result_processed.h5 \
    --n 1 -m 0 --shape area --nz 200 --nr 100 --scale-to-power 1.0 \
    -o hom_field_area

# 任意の直線上（z=0,r=0.01 から z=0.1,r=0.04 へ）
python FEM_HOM_code/export_field_data.py -i hom_result_processed.h5 \
    --n 0 -m 1 --shape line --p1 0.0,0.01 --p2 0.1,0.04 --npts 500 \
    -o hom_field_line

# 進行波（n=0, mode 0, 位相 120°）の複素振幅出力 (default)
python FEM_HOM_code/export_field_data.py -i hom_result_processed.h5 \
    --n 0 -m 0 --shape axis --npts 500 --phase 120 -o hom_field_axis_TW

# 進行波の実瞬時値（time-phase 45° スナップショット）
python FEM_HOM_code/export_field_data.py -i hom_result_processed.h5 \
    --n 0 -m 0 --shape axis --npts 500 --phase 120 \
    --instant --time-phase 45 -o hom_field_axis_t45
```

> HOM では複素振幅出力時、内部で `time-phase = 0°` と `90°` の 2 スナップショット
> $F(t)$ から複素振幅 $\tilde{F} = F(0) - j\,F(\pi/2)$ を再構築しています
> （物理規約 $e^{+j\omega t}$）。

**共通の出力構造**（HDF5）:

- `attrs`: `mode_index`, `frequency_GHz`, `phase_shift_deg`, `time_phase_deg`, `scale_factor`, `target_power_W`, `p_loss_W`, `stored_energy_J`, `q_factor`, `is_complex`, `shape`, など
- `area`: `z_vec`, `r_vec`, `H_theta`, `Ez`, `Er`, `E_abs`, `mask` (HOM は加えて `E_theta`, `Hz`, `Hr`, `H_abs`)
- `line`/`axis`: `s`, `distance`, `z`, `r`, `mask` + 各場の成分（同上）
- 場の dtype は `is_complex` に応じて実数 / 複素数のどちらか

**TXT** は `#` で始まるヘッダにメタデータを記述し、空白区切りで列値を出力します。複素振幅出力時は各場成分が `Re(*) Im(*)` の 2 列に展開されます。最初のデータ行直前に `Columns: ...` が記載されているので、後段ツールで読み込む際の参考にしてください。

#### Result Viewer からの実行

Result Viewer 下部に追加された **「Export Area...」「Export Line...」「Export Axis...」** ボタンから GUI 経由で同じ出力ができます。

ダイアログで以下を選択できます:
- **Output mode**: `Complex (Re/Im)` / `Instant real` （定在波では自動的に Instant real 固定）
- **Output format**: `Both (HDF5 + Text)` / `HDF5 only` / `Text only`
- **Scale to power**: チェックを入れると指定 W にスケーリング
- 領域・グリッド数・直線端点など

現在表示中のモード・位相・時間位相が自動で CLI 引数に反映され、実行コマンドは `command.log` に追記されます。後から CLI で再実行・バッチ化したい場合に便利です。

---

### 6.7 GIF アニメーション保存

**進行波モード**のみ利用可能な機能です。現在 Result Viewer で表示中のモードを時間位相 θ=0°〜360° でアニメーション化し、GIF ファイルとして保存します。

#### 使い方

Result Viewer 下部の **「Save GIF...」** ボタンをクリックするとダイアログが開きます。

| 設定項目 | 説明 | デフォルト |
|---|---|---|
| 1ループのフレーム数 | θ=0°〜360° を何コマに分割するか（4〜360） | 36 |
| FPS | アニメーションの再生速度（フレーム/秒） | 12 |
| 出力ファイル名 | 保存先 `.gif` ファイルパス | `<h5ファイルと同ディレクトリ>_mode<N>_anim.gif` |

「OK」をクリックすると進行状況ダイアログが表示され、全フレームのレンダリング完了後に GIF ファイルが保存されます。

#### アニメーション内容

現在の表示設定がそのまま反映されます：

- **H_theta 表示**（カラーマップ）のオン/オフ
- **E Lines**（電気力線）の本数
- **Vectors**（電場ベクトル）のオン/オフとグリッド数
- **Show Mesh** のオン/オフ

グラフタイトルには周波数、位相シフト (phase)、時刻位相 (θ) が記載されます。

#### 注意事項

- GIF 生成には **Pillow** ライブラリが必要です（`pip install Pillow`）。
- フレーム数が多いほど生成時間とファイルサイズが増加します。目安：36フレーム 12fps ≈ 3秒ループ。
- 定在波モードでは「Save GIF...」ボタンは押せますが、クリック時にメッセージを表示して処理を中断します。

---

## 7. 解析精度について

### 要素次数の選択

2次要素（6節点三角形, `--elem-order 2`）は1次要素と比べて大幅に精度が向上するため、**特に HOM 解析では2次要素を推奨**します。

球形共振器（半径 R=100mm）での精度検証（`accuracy_verification_HOM/run_verification_hom.py`）:

| メッシュサイズ | 要素次数 | TM_011 誤差 | TM_111 誤差 | TE_111 誤差 | TM_211 誤差 |
|---|---|---|---|---|---|
| 10 mm（粗） | 1次 | 〜1% | 〜0.1% | 〜0.1% | 〜0.1% |
| 10 mm（粗） | **2次** | **0.12%** | **0.0004%** | **0.0014%** | **0.0005%** |
| 2.5 mm（細） | 2次 | < 0.01% | < 0.001% | < 0.001% | < 0.001% |

2次要素では理論収束次数 O(h⁴)（メッシュサイズを半分にすると誤差が 1/16 に減少）を達成しています。

### メッシュサイズの目安

- **TM0 解析**: セル長（空洞長）の **1/5〜1/10** を目安にします。
- **HOM 解析**: 最高周波数モードの波長の **1/10 以下** を推奨します。
- 円弧境界（球形・楕円形など）を含む形状は、2次メッシュ（Mesh order: 2nd）を使用すると境界形状が正確に表現されます。
- 軸上（r=0）の特異点は L'Hôpital 則を適用して適切に処理されます。

### 2次要素の計算速度について

2次要素のアセンブリは Webb 階層基底 + NumPy ベクトル化実装により、要素数に対してほぼ線形（O(N¹)）のスケーリングを示します。1次要素のスカラー実装と比較して 20〜40 倍高速です（モード次数 n に依存）。

---

## 8. 注意事項

- **PEC 境界**: 物理グループ名が `PEC`、`Dirichlet`、`E-short` のいずれかである境界が電気壁（Dirichlet 条件）として認識されます。`M-short`（磁気壁）は自然境界条件として何も設定しなければ自動的に適用されます。HOM（n≥1）では z 軸上（r=0）も自動的に Dirichlet 条件が付加されます。
- **ループの向き**: 反時計回り（CCW）を推奨します。電磁場解析コードはこれを前提とします。
- **Gmsh のインストール**: Python パッケージの `gmsh` が必要です（`pip install gmsh`）。
- **処理の順序**: Run Solver → Run Post-Process → Create HTML Report の順で実行してください。Result Viewer は Raw / Processed どちらの H5 ファイルでも開けます。
