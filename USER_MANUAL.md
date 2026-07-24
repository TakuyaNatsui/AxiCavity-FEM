# AxiCavity-FEM ユーザーマニュアル

本ツールは、Gmsh を利用したメッシュ作製から、有限要素法による電磁場解析、ポストプロセス（工学パラメータ計算）、可視化までを一貫して行うツールです。円筒座標系の軸対称 TM0 モード（定在波・進行波）および高次方位角モード（HOM）を対象とした空洞共振器・加速管の設計・解析に対応しています。**統一 CLI**（`axicavity-fem`）と **GUI**（`axicavity-fem-gui`）の 2 つの入口があり、どちらからでも同じ計算を実行できます。

主な機能:

- **ソルバ**: TM0 / HOM、定在波・進行波（周期境界の位相差指定・位相スキャン）、1 次／2 次要素。
- **形状とメッシュ**: **Multi-Region Editor**（タブ1）で点・直線・円弧から輪郭を作り、線分ごとに境界条件を設定して Gmsh でメッシュ化します。複数領域・穴あき（ドーナツ型）形状・領域ごとの材料（比誘電率 ε_r、誘電正接 tanδ）に対応。単一の真空空洞は **Simple モード**でほぼワンクリック、多領域・誘電体は **Advanced モード**で扱います（§3）。
- **変数と数式**: 座標・寸法・材料の入力欄に `a+b` のような式を書け、**Variables** 表で変数を定義できます。変数を変えると形状が追従して再計算されます（§3.5）。
- **後処理**: Q・Q_wall・Q_diel、R/Q、V_eff、蓄積エネルギー、壁損失・誘電体損失、ポート通過電力、群速度、減衰定数。
- **出力**: HDF5、テキストサマリ、場マップ PNG、場データ（HDF5／テキスト）、自己完結 HTML レポート、進行波の GIF アニメーション、対話的な Result Viewer。

計算上の注意:

- **偽モードの自動除去**: 返される固有対はすべて残差チェックされ、収束していないモードは出力から除外されます（不足分は要求数を増やして解き直します）。粗いメッシュでは「収束モードが要求数に届かず N/M 本のみ返します」という警告とともに要求より少ないモード数が返ることがありますが、**返るものはすべて実在のモード**です。モード数が足りない場合はメッシュを細かくしてください。
- **高速化（任意）**: `pip install pypardiso` を入れると、大規模メッシュ（目安 2 万自由 DOF 以上）の定在波 TM0 / HOM 解析で疎行列分解が Intel MKL PARDISO（マルチコア）に切り替わり数倍速くなります。**計算結果は変わりません**。小規模メッシュや進行波（複素）解析では従来どおりの動作です。

---

## 全体ワークフロー

```
[形状作成] → [メッシュ生成] → [FEM計算 (solve)] → [パラメータ計算 (post)] → [結果確認・レポート]
```

---

## 0. インストール

```bash
pip install -e .          # コア（CLI）
pip install -e ".[gui]"     # GUI も使う場合（wxPython）
```

## 1. 起動方法

### 1.1 コマンドライン (CLI)

```bash
# 定在波 TM0（2次要素・10モード）
axicavity-fem solve --type tm0 -m mesh.msh --elem-order 2 --num-modes 10 -o result.h5
# HOM（方位角次数 0/1/2）
axicavity-fem solve --type hom -m mesh.msh --az-order 0 1 2 --elem-order 2 -o hom.h5
# 進行波（位相 120°、スキャンは "0:180:20"）
axicavity-fem solve --type tm0 -m mesh.msh -p 120 -o tw.h5
# 探索したい周波数を指定（2.856 GHz 近傍のモードを 4 本）
axicavity-fem solve --type tm0 -m mesh.msh --num-modes 4 --target-freq 2.856 -o band.h5
# ポストプロセス（Q値・R/Q・群速度等を計算して追記）
axicavity-fem post  --type tm0 -i result.h5 --cond 5.8e7 --beta 1.0
# HDF5 構造ダンプ
axicavity-fem info  -i result.h5
```

主なオプション: `-m/--mesh`, `--type {tm0,hom}`, `--elem-order {1,2}`, `--num-modes`,
`--az-order`（HOM）, `-p/--phase`（位相）, `--target-freq`（探索周波数 [GHz]）,
`-o/--output`, `--cond`, `--beta`。
位相は単値 `120`、リスト `60,90,120`、レンジ `0:180:20`（start:end:step）。

`--target-freq`（TM0/HOM 共通）は shift-invert のシフト値を
sigma = (2πf/c)² に設定し、**その周波数に近いモードから順に** `--num-modes` 本を
返す。省略時は従来どおりメッシュの r_max から sigma を自動推定し、**最低次から**
返す（数値は従来と完全に同一）。高い帯域のモードだけを見たいときに、モード数を
増やして下から全部計算する必要がなくなる。なお「近さ」は固有値 λ=k²（∝ f²）で
測るため、指定値より下側のモードがやや選ばれやすい。

### 1.2 GUI

```bash
axicavity-fem-gui
```

ウィンドウが表示されたら、上部のタブで操作を切り替えます。

- **タブ1: Multi-Region Editor** — 多領域・誘電体・穴あき対応のメッシュエディタ。**Simple モード**（単一真空空洞を簡単に）と **Advanced モード**（多領域・誘電体）を切替。形状作成・メッシュ生成・Superfish 入出力（§2）・変数と数式（§3.5）もここで行います。
- **タブ2: FEM Analysis** — Solver（solve）と Post-Process（post）の実行、HTML レポート生成（report）、Result Viewer の起動。Analysis Mode（TM0/HOM）・Wave Type（Standing/Traveling）を RadioBox で切替。内部で統一 CLI `axicavity-fem` を呼び出します。

> **旧形式ファイルの互換**: v2 の単一領域 `.gmshproj`（schema_version なし）を
> `File > Load Project` で開くと、自動的に Multi-Region 形式（1 つの Vacuum
> 領域）へ変換して読み込みます。保存すると現行形式で書き出されます。

---

## 2. Superfish 形式 (.af) の入出力

Superfish（LANL の 2D 空洞計算コード）の `.af` 形式を Multi-Region Editor と
やり取りできます。座標の単位は Draw Area の **Unit** 選択に従って cm と換算します。

| 操作 | 方法 |
|---|---|
| インポート | `File > Import Superfish File...`（.af 形式）|
| エクスポート | `File > Export Superfish .af...`（Ctrl+A）|

- **インポート**: `.af` の点列を読み込み、閉じた輪郭なら Vacuum 領域を自動生成します。
  境界条件（BC）は `.af` に情報が無いため、すべて `PEC` として読み込みます。
  輪郭が閉じていない場合は点・線のみ読み込むので、Simple モードの **Close Loop**
  で閉じてください。
- **エクスポート**: `.af` は単一領域の境界しか表現できないため、**領域が 1 つ・
  穴（hole）なし・閉ループ**のときのみ出力できます。条件を満たさない場合はエラー
  ダイアログで理由を表示します。BC 情報は `.af` には保存されません。円弧は劣弧
  （≤180°）として書き出されます。

---

## 3. タブ1：Multi-Region Editor（多領域・誘電体メッシュ）

形状の作成からメッシュ生成までを行うエディタです。**Simple モード**（既定）で単一の真空空洞を簡単に作成でき、**Advanced モード**で複数領域・誘電体（ε_r）・穴あき（ドーナツ）形状を扱えます。内部の幾何モデルは両モードで共通なので、いつでも切り替えられます。

> **メッシュ生成の方式（CCW/CW を気にしなくてよい理由）**: 本エディタは Gmsh の OpenCASCADE カーネルでメッシュを生成します。ループの向き（時計回り／反時計回り）は自動補正されるため、向きを意識する必要はありません。

### 3.1 モードの切替

右ペイン最上部の **View Mode** ラジオボックスで切り替えます。

| モード | 用途 |
|---|---|
| **Simple**（既定） | 単一の真空空洞。点を打って閉じるだけで真空領域が自動生成される。 |
| **Advanced** | 複数領域・誘電体・穴あき。Loop と Region を明示的に定義する。 |

Advanced で複数領域などを作った状態から Simple に戻すと警告が出ますが、内部データは保持されます（表示が簡易になるだけ）。

### 3.2 Simple モードの操作

#### 3.2.1 形状を描く（Edit Points）

1. 上部の **Draw Area** で Unit（m/cm/mm/inch）と Z・R の表示範囲を設定し「Update」。
2. キャンバスをクリックして点を順に打ちます。**直前の点との直線が自動で生成**されます（円筒座標なので R ≥ 0 の範囲で描いてください）。
3. **3 点以上**打ったら「**Close Loop**」を押すと、最後の点と最初の点が結ばれて形状が閉じ、**Vacuum 領域（ε_r = 1）が自動生成**されます。
4. 以降は **Edit Points / Edit Lines** のラジオボックスで編集します（点が 1 つ以上あると有効化）。

#### 3.2.2 点と線の自動切替（Simple モードの便利機能）

Simple モードでは、クリックした対象に応じて編集モードが**自動的に切り替わります**。

| 現在のモード | 操作 | 動作 |
|---|---|---|
| Edit Points | 既存の点をクリック | 点を選択（そのままドラッグで移動） |
| Edit Points | 線の上をクリック | **Edit Lines へ自動切替**し、その線を選択 |
| Edit Points | 線の上をダブルクリック | その位置に点を挿入（形状はそのまま） |
| Edit Lines | 点の近くをクリック | **Edit Points へ自動切替**し、その点を選択 |
| Edit Lines | 線の上をダブルクリック | 点を挿入して Edit Points へ切替 |

ラジオボックスの表示も自動で同期します。

#### 3.2.3 点の編集

- **移動**: 点を選択してドラッグ。または Z・R 欄に数値入力 →「Update」。
- **挿入**: 線の上をダブルクリック（前後の線が分割されてループは保たれます）。
- **削除**: 点を選択 →「Delete」。**前後の点を結ぶ線が自動生成され、ループが保たれます**（Mesh Geometry Edit と同じ挙動）。

#### 3.2.4 線（segment）の編集

Edit Lines で線を選択すると、**Selected Segment** ボックスで操作できます。

- **境界条件 (BC) の変更**: ドロップダウンで選択（次節参照）。
- **円弧化**: 「Convert to Arc」で直線を劣弧（短い方の円弧）に変換。
- **円弧の調整**: 「Center Z / R」を編集 →「Update Center」で膨らみを変更。
- **直線化**: 「Convert to Line」で円弧を直線に戻す。
- 円弧の端点をドラッグすると自動的に直線に戻ります。
- Simple モードでは **Delete Segment は無効**（ループが壊れるため）。線を消したい場合は点の削除で行います。

> **境界条件と線の色**
>
> | BC 名 | 色 | 意味 |
> |---|---|---|
> | `PEC` | 濃いオレンジ | 完全電気導体（金属壁）。壁損失計算に含まれる。 |
> | `E-short` | 青 | 電気壁（対称境界）。 |
> | `M-short` | 緑 | 磁気壁（対称境界）。 |
> | `None` | 灰色 | 境界条件を付けない線。**領域間の共有（内部）境界**や **軸 r=0 の線**に使う。 |
>
> 線を選択すると**色はそのままで太く**表示されます。`None` は Gmsh の境界グループ（Physical Curve）を作りません。軸 r=0 の境界はソルバが幾何的に自動判定するため、軸の線は `None` にしておくと見た目が整います（解析結果には影響しません）。詳細は [docs/BC_NAMING.md](docs/BC_NAMING.md)。

#### 3.2.5 メッシュ出力

下部の **Mesh Output** で:

1. **lc**: 節点間隔（Unit と同じ単位）。
2. **Order**: 1st（1次要素）/ 2nd（2次要素）。高精度には 2nd 推奨。
3. **Export MSH**: `.msh` を生成。同時に **`<名前>.materials.json`**（領域ごとの ε_r 等）も書き出され、FEM Analysis タブで自動的に読み込まれます。
4. **Export to .geo** / **Export Python Script**: それぞれ Gmsh の `.geo` スクリプト、Gmsh Python API スクリプトを出力（学習・再現用）。

出力した `.msh` のパスは自動的に FEM Analysis タブ（タブ2）の「Mesh File」にセットされます。

### 3.3 Advanced モードの操作（多領域・誘電体）

Advanced モードでは、形状を **点 → 線（segment）→ ループ（Loop）→ 領域（Region）** の順で明示的に組み立てます。

1. **Edit Mode** ラジオボックスで操作を選択:
   - **Add Points**: クリックで点を追加。
   - **Edit Points**: 点の移動・選択。
   - **Add Segments**: 2 点を順にクリックして線を作成。
   - **Build Loop**: 線を順にクリックして 1 つの閉ループにまとめ「Close Loop」で確定。
   - **Edit BC**: 線を選択して境界条件を設定。
2. **Loops**: 作成済みループの一覧。穴（ドーナツの内側）にするループもここで作ります。
3. **Regions**: 領域を定義します。
   - **Name**: 領域名（例 `Vacuum`, `Dielectric_1`）。
   - **Outer Loop**: 外周ループを選択。
   - **Holes**: 穴にするループにチェック（複数可）。
   - **material_tag**: 材料タグ（英数字。例 `vacuum`, `dielectric_1`）。**領域間で重複不可**。
   - **eps_r**: 比誘電率（実数、> 0）。真空は 1.0。
   - **tan_d**: 誘電正接 tanδ（実数、>= 0）。無損失は 0.0。指定すると
     post で誘電体損失 Q_diel が計算され、Q 値に反映されます。
   - 「Add」で領域を追加、「Update」で選択中の領域を更新、「Delete」で削除。

> **穴あき（ドーナツ型）領域の作り方**
> 1. 外周用ループと穴用ループの 2 つを作る。
> 2. Regions で Outer Loop に外周を選び、Holes で穴ループにチェック → Add。
>
> 共有境界の節点は OpenCASCADE の `fragment` により自動的に一致します。

> **誘電体ロード空洞の作り方**
> 1. 真空領域と誘電体領域をそれぞれ作る（隣接する内部境界の線は BC を `None` に）。
> 2. 誘電体領域の Region で `eps_r` に比誘電率（例 4.0）を設定。損失を扱う場合は
>    `tan_d` に誘電正接（例 1e-4）も設定。
> 3. Export MSH で `.msh` + `.materials.json` が出力され、FEM Analysis タブで ε_r / tanδ が自動反映されます。
>
> **誘電体損失 Q_diel（摂動法）**: `tan_d > 0` の領域があると、post 実行時に
> 誘電体損失 `P_diel` と `Q_diel` が計算され、出力の `Q` は**壁損失+誘電体損失込みの
> 合計 Q**（`1/Q = 1/Q_wall + 1/Q_diel`）になります。壁損失のみの値は `Q_wall` として
> 別途出力されます。一様充填では `Q_diel = 1/tanδ`。固有値問題は実数のまま解く摂動法
> なので計算時間は変わらず、tanδ ≪ 1（目安 0.05 以下）で有効です。
> tanδ は solve 時に H5 へ保存されるため、**tanδ を変えたら solve から再実行**して
> ください（既存 H5 に post だけ再実行しても反映されません）。
>
> **注意（既知の制約）**: 誘電体ロード時、共振周波数（固有値）は正確ですが、軸・ポート上に誘電体があると工学パラメータ（U, P_flow, V_eff, Q, R/Q）が不正確になる場合があり、ソルバが警告を出します。詳細は [docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md) の「既知の制約」を参照。

### 3.4 プロジェクトの保存と読み込み

`File > Save` / `Ctrl+S` で `.gmshproj`（JSON）形式に保存します。保存時のスキーマは `schema_version="2.3"`（tan_delta を含む）です。旧スキーマ（2.1 / 2.2）も後方互換で読み込めます（tan_delta が無ければ 0.0 扱い）。**単一領域だけを持つ旧形式（schema_version なし）は、読み込み時に自動で Multi-Region 形式（1 つの Vacuum 領域）へ変換**されます。

### 3.5 変数と数式

座標・寸法・材料を**変数と数式**で定義できます。1 か所（Variables 表）の値を変えると、
その値を使った箇所がすべて追従して再計算されるため、寸法違いの形状づくりや調整が容易です。

#### 3.5.1 変数定義テーブル（Variables）

Multi-Region Editor 右ペインの **Variables** 表で変数を定義します。3 列構成です。

| 列 | 内容 |
|---|---|
| `Name` | 変数名（英字・数字・アンダースコア。先頭は英字/`_`） |
| `Expression` | 値または式（例 `100`, `a+b`, `2*pi*r0`） |
| `Value` | 評価結果（自動表示・読み取り専用） |

- 例: `a`=`100`、`b`=`10`、`c`=`a + b` と入力すると `Value` 列に `100` / `10` / `110` が出ます。
- 変数は**上の行から順に評価**され、上の行で定義した変数を下の行で参照できます（前方参照は不可）。
- 末尾の空行に入力すると自動で行が増えます。名前か式が空の行は無視されます。

#### 3.5.2 入力欄で数式を使う

定義した変数は、次の入力欄で数式として使えます（**Update** で確定）。

- Selected Point の **Z / R**
- 円弧の中心 **Z / R**（Update Center）
- Draw Area の **Z / R 範囲**、**mesh size（lc）**
- Region の **eps_r**（比誘電率）

例: Selected Point の Z 欄に `a + b`、R 欄に `sqrt(a)` と入力して **Update** すると、その点が
`(110, 10)` に移動します。入力が不正（未定義変数・構文エラー等）ならエラーダイアログが出て、
その欄は更新されません。

> **数値プレビュー**: Point の Z/R 欄と円弧中心の Z/R 欄では、式を入力すると
> 隣に評価済みの数値が `= 110.000000` のように灰色で表示されます（素の数値のときは
> 何も表示しません）。選択の切替・Update・Variables 変更・入力中のいずれでも即座に更新
> され、式が実際にいくつになるかを確認できます。評価できない式は `= ?` と表示します。

**使える演算・関数**（`math` 準拠の安全な評価器。`eval` は使いません）:

- 演算子: `+  -  *  /  //  %  **`、単項マイナス、括弧 `( )`
- 関数: `sin` `cos` `tan` `asin` `acos` `atan` `atan2` `sqrt` `exp` `log` `log10` `log2`
  `hypot` `degrees` `radians` `floor` `ceil` `fabs` `abs` `min` `max` など
- 定数: `pi` `e` `tau`

#### 3.5.3 式の保持と変数による再計算

- **入力した式そのものが保持されます**（点座標・円弧中心・mesh_size・eps_r）。別の点を選んでから
  戻ると、欄には数値ではなく式（例 `a+b`）が表示されます。`.gmshproj`（schema 2.2）にも式で保存されます。
  素の数値（例 `12.5`）は定数として扱い、式は保存しません。
- **Variables 表を変更すると、式を持つ座標・寸法・材料が自動で再計算**され図形が動きます。
  円弧は端点に追従して中心・半径が再算出されます（直線化はしません）。
- 式を持つ点を**マウスでドラッグすると、その式は破棄され定数**になります（意図的な上書き）。

#### 3.5.4 Export Python Script への反映

**Export Python** で出力したスクリプトには、冒頭に `from math import *` が出力され、
座標や `_LC`（mesh size）は**式のまま**（例 `gmsh.model.occ.addPoint((a+b) * _SCALE, ...)`）
埋め込まれます。**Export GEO / MSH は評価済みの数値**で出力されます（式は Python のみ）。

スクリプト全体が **`build_model()` 関数**にまとめられ、
`if __name__ == '__main__':` から呼ばれる形になりました。変数表の変数は
`user_vars` で上書きできます。

```python
def build_model(out_msh=_OUT_MSH, mesh_order=_MESH_ORDER, **user_vars):
    a = user_vars.get('a', 40)
    L = user_vars.get('L', 50)
    c = user_vars.get('c', a + L)   # 依存変数は上書き後の a, L から再計算
    ...
    return out_msh


if __name__ == '__main__':
    build_model()
```

`python cavity_mesh.py` と実行すれば従来どおり `.msh` を生成し、外部スクリプトから
`build_model(a=42, out_msh='trial.msh')` と呼べばパラメータを変えたメッシュを
繰り返し生成できます。**周波数の自動調整**などに使えます
（サンプル: [`examples/frequency_tuning/`](examples/frequency_tuning/)）。

#### 3.5.5 使い方の流れ（例）

1. Variables に `r0`=`50`、`gap`=`5` を定義。
2. 点や円弧を作り、座標欄に `r0`、`r0-gap` などの式を入れて **Update**。
3. Variables の `r0` を `60` に変更 → 形状が追従して動くのを確認。
4. **Export Python** で出力し、生成された `.py` の変数定義や式を必要に応じて編集。
   （メッシュ生成自体は `.msh` 出力、または生成 `.py` を実行して行います。）

---

## 4. タブ2：TM0 解析（定在波・進行波統合）

共振空洞の TM0 モードを解析します。**Wave Type** RadioBox で定在波（Standing Wave）と進行波（Traveling Wave）を切り替えます。

### 4.1 ソルバーの実行

1. 「Mesh File (.msh)」にメッシュファイルのパスを入力するか、「Browse...」で選択します（Multi-Region Editor からの自動セットも可）。
2. **Number of Modes**: 計算する固有モード数（デフォルト: 10）を設定します。
3. **Mesh Order**: FEM 要素次数（1: 1次要素 / 2: 2次要素）を確認します。高精度解析には2次要素を推奨。
4. **Wave Type**: 「Standing Wave」または「Traveling Wave」を選択します。
5. **Phase Shift (deg)**: 進行波選択時のみ有効。周期境界のセル間位相差を入力します（後述の形式を参照）。
6. **Predicted frequency (GHz)**（任意）: 計算したい周波数を入れると、その周波数に
   近いモードから順に **Number of Modes** 本を計算します（内部で固有値ソルバのシフト値
   sigma =(2πf/c)² を設定。CLI の `--target-freq` と同じ）。**空欄なら従来どおり**
   最低次のモードから計算します。高い帯域のモードだけを見たいときに使います。
7. 「**Run Solver**」を押します。

**出力ファイル**:
- 定在波: `*_analysis.h5`、`*_frequencies.txt`
- 進行波: `*_traveling_analysis.h5`、`*_frequencies.txt`
- `.h5` にはメッシュ・固有値・固有ベクトルが HDF5 形式で格納されます。

FEM ログにリアルタイムで進捗が表示されます。

### 4.2 ポストプロセス（パラメータ計算）

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

**出力ファイル** (`*_processed.h5` および同名の `*_processed.txt`):
- `_processed.h5`: 規格化された固有ベクトルと計算パラメータを格納
- `_processed.txt`: 周波数・Q・R/Q などのテキストサマリー（§6 参照）

### 4.3 レポート生成

1. 「Post-Processed Result File」に処理済み `.h5` ファイルが自動セットされていることを確認します。
2. 「**Create animation**」チェックボックスは進行波モードでのみ意味を持ちます（オフ：アニメ無し / オン：時間発展 GIF を作成）。アニメーション生成は時間がかかるため、デフォルトはオフです。
3. 「**Create HTML Report**」を押します（チェック OFF でも分布図の生成があるため、それなりに時間がかかります）。
4. 「**Open HTML Report**」を押すとブラウザでレポートが開きます。

#### TM0 モードのレポート内容

- メッシュ概要図（境界条件エッジを色分け表示: PEC=オレンジ / E-short=青 / M-short=緑 / **None=黒**）と、節点数・要素数・**平均メッシュサイズ（平均辺長）**
- 各モードの磁場コンター + 電場ベクトル図（定在波）または Real/Imag/Magnitude（進行波）
- 軸上電場 $E_z$ 分布図
- 全パラメータのサマリー表（Q, R/Q, シャントインピーダンス, 群速度, 減衰定数 など）
- 進行波モードで「Create animation」がオンの場合、各モードの時間発展 GIF アニメーション

#### HOM モードのレポート内容

- メッシュ概要図（境界条件エッジを色分け表示: PEC=オレンジ / E-short=青 / M-short=緑 / **None=黒**）と、節点数・要素数・**平均メッシュサイズ（平均辺長）**
- 各 (n, 位相シフト, モード) について E-field と H-field の左右並置プロット
- 軸上 $E_z$ 分布図
- 蓄積エネルギー $U$、壁損 $P_{loss}$、Q 値のサマリー表
- 進行波モードで「Create animation」がオンの場合、時間発展 GIF アニメーション

HOM レポートは方位角オーダー $n$ ごとに章立てされ、進行波の場合は位相シフト値ごとに分かれます。

### 4.4 結果の対話的な確認（Result Viewer）

「**Result Viewer**」ボタンを押すと、インタラクティブなビューアが起動します。

| 操作 | 機能 |
|---|---|
| Mode プルダウン | 表示するモードの選択 |
| H_theta (Color) | 磁場 $H_\theta$ のカラーマップ表示 |
| E Lines / Levels | 電気力線（等高線）の表示・本数調整 |
| Vectors / z・r | 電場ベクトルの表示・格子点数調整 |
| Show Mesh | 元のメッシュ形状を重畳表示 |
| Show E-wall | PEC 境界での電場ベクトル表示（各 PEC エッジ中点に $E_z, E_r$ を赤で。TM0 のみ） |
| Save GIF... | 進行波モードを時間位相 θ=0°〜360° でアニメーション化し GIF 保存（§7.7） |
| グラフをダブルクリック | その座標の電磁場値（$H_\theta, E_z, E_r, \|E\|$）をポップアップ表示 |

> PEC 境界線（黒太線）は常時表示されます。TM0 は H_phi カラー＋電気力線＋電場ベクトル、
> HOM は左右に E 場 / H 場（方位角成分カラー＋面内ベクトル）の 2 パネルで表示されます。

---

## 5. タブ2：HOM 解析（Higher-Order Mode）

> TM0 と HOM は同じ **FEM Analysis タブ（タブ2）** の Analysis Mode RadioBox で切り替えます。本節は HOM を選んだ場合の操作です。

加速管・空洞の高次方位角モード（azimuthal order n ≥ 1）および TM0（n=0）の定在波・進行波を計算します。

### 5.1 ソルバーの実行

1. 「Mesh File (.msh)」でメッシュファイルを選択します。
2. **Number of Modes**: 計算するモード数（デフォルト: 10）。
3. **Element Order**: **「2nd」（2次要素）を強く推奨します**。2次要素は1次要素と比べて精度が1〜2桁向上します（後述「解析精度について」参照）。
4. **Mode Orders (e.g. 0 1 2)**: 計算する方位角次数 n をスペース区切りで入力します。
   - `0` → TM0 モード
   - `1` → TE/TM の n=1（双極子）モード
   - `0 1 2` → 複数次数を一括計算
5. **Wave Type**: 「Standing Wave」または「Traveling Wave」を選択します。
6. **Phase Shift (deg)**: 進行波選択時のみ有効。位相形式は TM0 タブと同じです（後述）。
7. **Predicted frequency (GHz)**（任意）: TM0 と同じく、入力するとその周波数に近い
   モードから順に計算します（方位角次数 n ごとに適用）。空欄なら従来どおり最低次から。
8. **Output File**: 出力 `.h5` ファイルのパスを指定します（空欄時は `*_hom.h5` に自動設定）。
9. 「**Run Solver**」を押します。

**出力ファイル**:
- `*_hom.h5`: 主計算結果（HDF5）
  - `/mesh/`: メッシュ頂点・要素・エッジ情報
  - `/results/n{n}/Normal/mode_{i}/`: 定在波モード（固有ベクトル・周波数・固有値を格納）
  - `/results/n{n}/Periodic/PB_Phase_XXX/`: 進行波モードの固有ベクトル（複数位相対応）
- `*_hom_frequencies.txt`: 計算直後に自動生成される周波数一覧（約15桁精度）。計算完了後すぐに周波数を確認できます。

### 5.2 電場の可視化

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

## 6. コマンドログとテキスト出力

GUIから実行したすべてのコマンドは、プロジェクトディレクトリの **`command.log`** に自動記録されます。これにより:
- 後から CLI でバッチ処理を行うときのコマンドがすぐに確認できます。
- 計算条件（メッシュファイル、モード数、位相、導電率など）の履歴が残ります。

**生成されるテキストファイルの一覧**:

| ファイル名 | 生成タイミング | 内容 |
|---|---|---|
| `<結果>.txt` | Run Solver（`solve`）完了後 | 解析タイプ・メッシュ名・要素次数・モード数・位相と、各モードの共振周波数 |
| `<結果>.txt` | Run Post-Process（`post`）完了後 | 上記に加え Q・R/Q・V_eff・U・P_loss・P_flow・群速度・減衰定数 |
| `command.log` | コマンド実行のたびに追記 | 実行コマンド履歴（タイムスタンプ付き） |

`.txt` は **HDF5 の出力先と同じ場所・同じ名前で拡張子だけ `.txt`** にしたものです
（例 `result.h5` → `result.txt`、`result_processed.h5` → `result_processed.txt`）。
CLI から `solve` / `post` を実行した場合も同じファイルが生成されます。
**ファイルの中身は英語**で、周波数は約 15 桁（10 桁以上の精度）で出力されます。

```
AxiCavity-FEM result summary (post)
generated: 2026-07-15 15:16:57
==================================================
analysis type : TM0
mesh file     : cav.msh
element order : 2
num modes     : 4
phase         : 0.0
--------------------------------------------------
[n=0] standing
  mode 0:
      f       = 2.86856339647306 GHz
      Q       = 1.800990e+04
      R/Q     = 2.070193e+02 ohm
      ...
```

---

## 7. コマンドライン（CLI）での実行

GUI で行える操作はすべてコマンドラインからも実行できます（GUI 自身も内部で同じ
コマンドを呼んでいます）。サブコマンドは 6 つです。

| サブコマンド | 役割 | 節 |
|---|---|---|
| `solve` | 固有値解析（周波数と固有ベクトル） | §7.2 |
| `post` | 工学パラメータ（Q, R/Q, V_eff, 損失, 電力流…）の計算 | §7.3 |
| `info` | 出力 HDF5 の構造ダンプ | §7.4 |
| `plot` / `report` | 場マップ PNG と HTML レポート | §7.5 |
| `export` | 電磁場マップのデータ出力（HDF5 / テキスト） | §7.6 |

各サブコマンドの全オプションは `axicavity-fem <サブコマンド> --help` でも確認できます。

### 7.1 位相指定の共通形式（`-p/--phase`）

TM0・HOM どちらも `-p` で同じ記法を使用します（単位: 度）。

| 指定方法 | 例 | 意味 |
|---|---|---|
| 単一値 | `-p 120` | 120° のみ |
| カンマ区切り | `-p "60,90,120"` | 60°、90°、120° の3点 |
| レンジ（start:end:step） | `-p "0:180:20"` | 0° から 180° を 20° 刻み（10点） |

> **定在波と進行波の切り替え**: `-p 0`（または `-p` 省略）で定在波、それ以外の値を指定すると自動的に進行波として計算されます。

---

### 7.2 `solve` — 固有値解析

```bash
axicavity-fem solve --type {tm0,hom} -m MESH.msh [options] -o OUT.h5
```

| 引数 | 短縮形 | デフォルト | 説明 |
|---|---|---|---|
| `--type` | — | （必須） | `tm0`（軸対称）または `hom`（高次方位角モード） |
| `--mesh` | `-m` | （必須） | 入力メッシュファイル (.msh) |
| `--elem-order` | — | `2` | FEM 要素次数（1 または 2）。設計用途では 2 を推奨 |
| `--num-modes` | — | `10` | 計算する固有モード数 |
| `--phase` | `-p` | `0.0` | 位相シフト [度]。`0` で定在波、それ以外で進行波（記法は §7.1） |
| `--target-freq` | — | （自動） | 探索したい周波数 [GHz]。指定するとこの周波数に近いモードから順に返す。省略時はメッシュ半径から自動推定し最低次から返す |
| `--az-order` | — | `0` | **HOM のみ**: 方位角次数 n をスペース区切りで複数指定（例 `0 1 2`） |
| `--materials` | — | （自動） | 材料定義 JSON。省略時は `<mesh>.materials.json` を自動探索 |
| `--output` | `-o` | `<mesh>_<type>.h5` | 出力 HDF5 |

```bash
# 定在波（2次要素、10モード）
axicavity-fem solve --type tm0 -m cavity.msh --elem-order 2 --num-modes 10 -o result.h5

# 進行波（単一位相 120°）
axicavity-fem solve --type tm0 -m cavity.msh -p 120 -o result_TW.h5

# 進行波（位相スキャン: 0°〜180° を 20° 刻み → 分散曲線用）
axicavity-fem solve --type tm0 -m cavity.msh -p "0:180:20" -o result_scan.h5

# HOM（方位角次数 0/1/2 を一括計算）
axicavity-fem solve --type hom -m cavity.msh --az-order 0 1 2 --num-modes 5 -o hom.h5

# 2.856 GHz 近傍のモードを 4 本だけ計算
axicavity-fem solve --type tm0 -m cavity.msh --num-modes 4 --target-freq 2.856 -o band.h5
```

> 誘電体を含む解析では、Multi-Region Editor が `.msh` と一緒に書き出す
> `<mesh>.materials.json`（領域ごとの `eps_r` / `mu_r` / `tan_delta`）が自動で読まれます。
> tanδ は solve 時に HDF5 へ保存されるため、**tanδ を変えたら post だけでなく solve から
> 実行し直してください**。

---

### 7.3 `post` — 工学パラメータの計算

`solve` の出力を読み、Q・R/Q・V_eff・蓄積エネルギー・損失・電力流などを計算して追記します。

```bash
axicavity-fem post --type {tm0,hom} -i RAW.h5 [-o OUT.h5] [--cond S/m] [--beta β]
```

| 引数 | 短縮形 | デフォルト | 説明 |
|---|---|---|---|
| `--type` | — | （自動） | 省略時は HDF5 に記録された `solver_type` を使う |
| `--input` | `-i` | （必須） | `solve` が出力した HDF5 |
| `--output` | `-o` | （入力に追記） | 別ファイルに書きたい場合に指定 |
| `--cond` | — | `5.8e7` | 壁の導電率 [S/m]（既定は銅） |
| `--beta` | — | `1.0` | 粒子の β = v/c。走行時間係数に使う（**TM0 のみ**） |

```bash
# 銅壁、光速粒子
axicavity-fem post --type tm0 -i result.h5 --cond 5.8e7 --beta 1.0

# 別ファイルに出力（元の生データを残す）
axicavity-fem post --type hom -i hom.h5 -o hom_processed.h5
```

---

### 7.4 `info` — HDF5 の構造を確認する

```bash
axicavity-fem info -i result.h5
```

グループ・データセット・属性を一覧表示します。どのモードが入っているか、`post` が
実行済みかを確認したいときに使います。

---

### 7.5 `plot` / `report` — 図と HTML レポート

```bash
axicavity-fem plot   --type {tm0,hom} -i FILE.h5 [-o OUT.png] [options]
axicavity-fem report --type {tm0,hom} -i FILE.h5 [-o OUT_DIR]  [options]
```

`plot` は 1 モードの場マップを PNG で出力し、`report` は全モードの図と工学パラメータの
表をまとめた自己完結 HTML（`index.html`）を出力ディレクトリに書き出します。

| 引数 | 短縮形 | デフォルト | 対象 | 説明 |
|---|---|---|---|---|
| `--input` | `-i` | （必須） | 両方 | `solve` / `post` の出力 HDF5 |
| `--output` | `-o` | （自動） | 両方 | `plot` は PNG パス、`report` は出力ディレクトリ |
| `--mode` | `-m` | `0` | plot | モード番号 |
| `--n` | — | （自動） | plot | 方位角次数（HOM） |
| `--phase` | — | （先頭） | plot | 進行波の位相 [度] |
| `--time-phase` | — | `0` | 両方 | 瞬時値の時間位相 [度] |
| `--no-mesh` | — | — | 両方 | 場マップにメッシュを重ねない |
| `--animate` | — | — | report | 進行波モードに GIF アニメーションを追加する |
| `--dpi` | — | `120` | 両方 | 画像解像度 |

```bash
# モード 0 の場マップを PNG に
axicavity-fem plot --type tm0 -i result.h5 -m 0 -o mode0.png

# HTML レポート（進行波はアニメーション付き）
axicavity-fem report --type tm0 -i result_TW.h5 -o report_dir --animate
```

---

### 7.6 `export` — 電磁場マップのデータ出力

ビーム計算等のために、解析結果から指定領域の電磁場マップを HDF5 と TXT で出力します。TM0 と HOM で同一の引数体系を採用しています。

```bash
axicavity-fem export --type {tm0,hom} -i FILE.h5 -o OUT_BASE --shape {area,line,axis} [options]
```

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
axicavity-fem export --type tm0 -i result_processed.h5 -m 0 \
    --shape area --z-range 0,0.1 --r-range 0,0.05 \
    --nz 200 --nr 100 --scale-to-power 1000.0 \
    -o field_area

# 軸上電場（V_eff の検算用、500 点サンプル、HDF5 のみ）
axicavity-fem export --type tm0 -i result_processed.h5 -m 0 \
    --shape axis --npts 500 --scale-to-power 1000.0 \
    --format h5 -o field_axis

# 進行波の実瞬時値（位相 120°、time-phase 30°）
axicavity-fem export --type tm0 -i result_processed.h5 -m 0 \
    --shape area --phase 120 --instant --time-phase 30 \
    -o field_TW_instant
```

**HOM の例**:

```bash
# n=1 ダイポール、mode 0 の矩形マップ
axicavity-fem export --type hom -i hom_processed.h5 \
    --n 1 -m 0 --shape area --nz 200 --nr 100 --scale-to-power 1.0 \
    -o hom_field_area

# 任意の直線上（z=0,r=0.01 から z=0.1,r=0.04 へ）
axicavity-fem export --type hom -i hom_processed.h5 \
    --n 0 -m 1 --shape line --p1 0.0,0.01 --p2 0.1,0.04 --npts 500 \
    -o hom_field_line

# 進行波（n=0, mode 0, 位相 120°）の複素振幅出力 (default)
axicavity-fem export --type hom -i hom_processed.h5 \
    --n 0 -m 0 --shape axis --npts 500 --phase 120 -o hom_field_axis_TW

# 進行波の実瞬時値（time-phase 45° スナップショット）
axicavity-fem export --type hom -i hom_processed.h5 \
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

### 7.7 GIF アニメーション保存

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
- **Show E-wall**（PEC 境界の電場ベクトル, TM0）のオン/オフ
- PEC 境界線（黒太線）は常時表示

グラフタイトルには周波数、位相シフト (phase)、時刻位相 (θ) が記載されます。HOM は E/H の 2 パネルが
そのまま動画化されます。なお同じアニメーションは HTML レポート生成時に「Create animation」を
オンにすることでも各進行波モードに追加できます（CLI: `axicavity-fem report ... --animate`）。

#### 注意事項

- GIF 生成には **Pillow** ライブラリが必要です（`pip install Pillow`）。
- フレーム数が多いほど生成時間とファイルサイズが増加します。目安：36フレーム 12fps ≈ 3秒ループ。
- 定在波モードでは「Save GIF...」ボタンは押せますが、クリック時にメッセージを表示して処理を中断します。

---

## 8. 解析精度について

### 要素次数の選択

2次要素（6節点三角形, `--elem-order 2`）は1次要素と比べて大幅に精度が向上するため、**特に HOM 解析では2次要素を推奨**します。

球形共振器（半径 R=100mm）での精度検証（[`examples/accuracy_verification/`](examples/accuracy_verification/)）:

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

## 9. 注意事項

- **PEC 境界**: 物理グループ名が `PEC`、`Dirichlet`、`E-short` のいずれかである境界が電気壁（Dirichlet 条件）として認識されます。`M-short`（磁気壁）は自然境界条件として何も設定しなければ自動的に適用されます。HOM（n≥1）では z 軸上（r=0）も自動的に Dirichlet 条件が付加されます。
- **ループの向き**: 反時計回り（CCW）を推奨します。電磁場解析コードはこれを前提とします。
- **Gmsh のインストール**: Python パッケージの `gmsh` が必要です（`pip install gmsh`）。
- **処理の順序**: Run Solver → Run Post-Process → Create HTML Report の順で実行してください。Result Viewer は Raw / Processed どちらの H5 ファイルでも開けます。
