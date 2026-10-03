# 境界条件命名規約

ver2 では物理境界を表す Physical Group 名を **3 種類のみ**に統一する。数学用語（`Dirichlet` 等）は廃止。

## 3 種類の物理境界

| Physical Group 名 | 物理的意味 | Wall Loss (P_loss) 積分 |
|---|---|---|
| `PEC` | 完全電気導体（物理的金属壁） | **含む** |
| `E-short` | 対称境界としての電気壁（仮想壁）。`E_tan = 0` を課すが Wall Loss には寄与しない | 除外 |
| `M-short` | 完全磁気導体／対称境界としての磁気壁 | 除外 |

軸 r=0 は Physical Group 指定不要（座標 `r < tol` で自動検出）。

## v2.1 追加: `None`（境界グループを付けない線）

segment の BC 名には `None` を選べる（v2.1 から。v3 の GUI では物理タブの「None」ボタン。領域の共有境界と軸の上の線には
自動で `None` が付き、ほかの外周は既定で `PEC`）。これは
**Physical Curve グループを生成しない線**を表す特別な値で、次の 2 用途に使う。

| 用途 | 説明 |
|---|---|
| 領域間の共有（内部）境界 | 場が連続する内部界面（誘電体界面・領域分割線）。壁条件を付けない。 |
| 軸 r=0 の線 | 軸の扱いはソルバが座標 `r < tol` で**自動判定**するため、明示 BC は不要。`None` にしておくと描画が灰色になり外壁と区別しやすい。 |

- データモデル: `shared/multi_region_model.py` の `BC_NONE = "None"`、
  `ALLOWED_BC_NAMES = ("PEC", "E-short", "M-short", "None")`。
- Gmsh エクスポート: `shared/gmsh_export_occ.py` は `bc_name == "None"` の segment を
  **1D Physical Curve グループから除外**する（.msh / .geo / 生成 Python の 3 経路すべて）。
- 解析への影響: `None` は境界グループを作らないだけで、軸 r=0 の Dirichlet 等は
  座標ベースで自動適用されるため、**FEM 結果は変わらない**（見た目の整理用）。

## FEM 行列上の数学的扱い（DOF により反転）

同じ物理境界でも、DOF の種類により Dirichlet と Neumann の扱いが入れ替わる。

| BC 名 | TM0 (DOF: H_φ スカラー節点) | HOM (DOF: Nédélec エッジ + 節点) |
|---|---|---|
| `PEC` | **Neumann**（自然境界、H_φ の法線微分関連） | **Dirichlet** (`E_tan = 0`) |
| `E-short` | **Neumann** | **Dirichlet** (`E_tan = 0`) |
| `M-short` | **Dirichlet** (`H_φ = 0`) | **Neumann**（自然境界） |
| 軸 r=0（TM0） | **Dirichlet** (`H_φ = 0`) | — |
| 軸 r=0, n=0（HOM） | — | Neumann（自然成立） |
| 軸 r=0, n≥1（HOM） | — | **Dirichlet** 強制 |

### 直感的理解

- **PEC / E-short** = 電気壁（電界の接線成分ゼロ）
  - HOM では E を DOF にするため，直接 Dirichlet として行列に強制される。
  - TM0 では H_φ を DOF にするため，電気壁条件は H_φ の境界微分項として自動で満たされる（Neumann）。
- **M-short** = 磁気壁（磁界の接線成分ゼロ）
  - TM0 では H_φ = 0 を強制（Dirichlet）。
  - HOM では E に対する自然境界条件として表現される。
- **軸 r=0**（対称軸）
  - TM0 では H_φ が軸上で恒等的に消えるため，常に **Dirichlet** (`H_φ = 0`) を強制する。
  - HOM では n=0 のとき自然境界（Neumann），n≥1 のとき `E_θ = 0` 等のため **Dirichlet** を強制する。

## エイリアス（後方互換）

ver1 のメッシュファイルをそのまま使えるように、以下のエイリアスを受け入れる。検出時には WARN ログを出す。

| 標準名 | 受け入れる旧名 |
|---|---|
| `PEC` | `Dirichlet`, `dirichlet`, `pec` |
| `E-short` | `E_short`, `Eshort`, `e-short` |
| `M-short` | `M_short`, `Mshort`, `m-short` |

`--strict-bc-names` フラグを CLI に付けると未知名でエラー終了する（CI 用）。
