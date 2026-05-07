# 物理規約・計算方法リファレンス

本ドキュメントは、TM0 コード (`FEM_code/`) および HOM コード (`FEM_HOM_code/`) に共通する
物理規約・座標系・計算方法を一元的にまとめたものです。
各コードの修正・拡張の際は本ドキュメントを参照し、規約の一貫性を保ってください。

---

## 目次

1. [座標系](#1-座標系)
2. [時間規約（最重要）](#2-時間規約最重要)
3. [周期境界条件と進行波](#3-周期境界条件と進行波)
4. [有限要素定式化](#4-有限要素定式化)
5. [電磁場の再構成](#5-電磁場の再構成)
6. [工学パラメータ計算](#6-工学パラメータ計算)
7. [境界条件の種類と P_loss への影響](#7-境界条件の種類と-p_loss-への影響)

---

## 1. 座標系

円筒座標 $(z, r, \phi)$ を使用する。

- $z$: 軸方向（加速管・キャビティの軸）
- $r$: 半径方向 ($r \ge 0$)
- $\phi$: 方位角

**方位角モード** $n$ について、電磁場は $e^{jn\phi}$ に比例すると仮定する。

| $n$ | モード種別 |
|---|---|
| 0 | 軸対称モード (TM0, TM01n 等) |
| 1 | 双極子モード (TM11, TE11 等) |
| $n \ge 2$ | 高次方位角モード |

`FEM_code/` は $n=0$ 専用、`FEM_HOM_code/` は任意の $n$ に対応する。

---

## 2. 時間規約（最重要）

### 採用規約：物理規約 $e^{+j\omega t}$

本コードは **物理規約 $e^{+j\omega t}$** を採用する。
複素振幅 $\tilde{A}$ と瞬時値の関係は：

$$A(t) = \mathrm{Re}\bigl[\tilde{A}\, e^{+j\omega t}\bigr]$$

### Maxwell 方程式（複素振幅形式）

$$\nabla \times \boldsymbol{E} = -j\omega\mu_0 \boldsymbol{H}$$
$$\nabla \times \boldsymbol{H} = +j\omega\varepsilon_0 \boldsymbol{E}$$

$E_z$, $E_r$ と $H_\phi$ の関係（TM0 モード、軸対称）：

$$E_z = \frac{1}{j\omega\varepsilon_0}\left(\frac{H_\phi}{r} + \frac{\partial H_\phi}{\partial r}\right)
      = \frac{-j}{\omega\varepsilon_0}\left(\frac{H_\phi}{r} + \frac{\partial H_\phi}{\partial r}\right)$$

$$E_r = -\frac{1}{j\omega\varepsilon_0}\frac{\partial H_\phi}{\partial z}
      = \frac{j}{\omega\varepsilon_0}\frac{\partial H_\phi}{\partial z}$$

コード内の対応箇所：

```python
# field_calculator.py (TM0)
coeff = -1j / (omega * self.eps0)   # E = (-j/ωε) × curl_H
```

```python
# field_calculator_hom.py (HOM)
coef = 1j / (self.omega * self.mu_0)  # H = (+j/ωμ) × curl_E
```

### アニメーションの時間発展

時刻 $t$ に相当する位相角 $\theta_\text{time}$ (degree 単位) のとき：

$$E(z, r, t) = \mathrm{Re}\bigl[\tilde{E}(z,r)\, e^{j\theta_\text{time}}\bigr]$$

```python
comp_phase = np.exp(1j * np.deg2rad(theta_time))
Ez_plot = (Ez_comp * comp_phase).real
```

$\theta_\text{time}$ を 0° → 360° に連続変化させることで時間発展アニメーションを生成する。

---

## 3. 周期境界条件と進行波

### 原理

周期長 $L$ の構造で +z 方向に伝搬する進行波（位相速度 $v_\phi > 0$）は：

$$\boldsymbol{E}(z, r, t) = \mathrm{Re}\bigl[\tilde{\boldsymbol{E}}(z, r)\, e^{+j\omega t}\bigr]$$

Bloch 条件より、1 セル分の伝搬で空間依存部の位相が $-kL$（$k > 0$）変化するため：

$$\tilde{\boldsymbol{E}}(z + L, r) = e^{-jkL}\, \tilde{\boldsymbol{E}}(z, r)$$

位相アドバンス $\theta = kL$（$\theta > 0$ で +z 方向伝搬）とおくと：

$$\boxed{x_\text{max} = e^{-j\theta} \cdot x_\text{min}}$$

> **注意（過去の誤りについて）**: 2025年4月以前のコードは `e^{+jθ}` を使用していたため、  
> 進行波のアニメーションが −z 方向を向いていた。上記 `e^{-jθ}` が正しい物理規約に基づく定義。

### コード実装

```python
# FEM_HOM_code/boundary_conditions.py  （L544, L712）
# FEM_code/FEM_helmholtz_TM0_calclation.py  （L692 付近）
phase_factor = np.exp(-1j * theta)   # ← 物理規約: e^{-jθ}
```

2N×2N 実数展開の場合（余弦・正弦成分）：

$$x_j = (\cos\theta - j\sin\theta)\, x_i
\quad\Rightarrow\quad
\begin{cases}
\mathrm{Re}(x_j) = \cos\theta \cdot \mathrm{Re}(x_i) \\
\mathrm{Im}(x_j) = -\sin\theta \cdot \mathrm{Re}(x_i)
\end{cases}$$

### 分散曲線の解釈

位相アドバンス $\theta$ を 0° → 180° でスキャンすると、各 $\theta$ で共振周波数 $f(\theta)$ が得られる。
これが $\omega$-$\beta$（分散）曲線の第 1 ブリルアン・ゾーン内の情報を与える。

| $\theta$ | 対応モード |
|---|---|
| 0° | 定在波（$k=0$、周期境界と同位相） |
| 90° | $\lambda/4$ モード |
| 180° | $\pi$ モード（最短セル長でシンクロナス条件） |

---

## 4. 有限要素定式化

### TM0 コード（$n=0$）

**DOF**: 節点磁場 $H_\phi$ のみ（スカラー節点要素）

固有値問題：

$$\mathbf{K}\, \boldsymbol{u} = k^2 \mathbf{M}\, \boldsymbol{u}, \quad k = \omega/c$$

行列要素（要素 $e$、節点 $i$, $j$）：

$$K_{ij}^{(e)} = 2\pi\int_e \frac{1}{r}\left(\frac{\partial G_i}{\partial z}\frac{\partial G_j}{\partial z}
                  + \frac{\partial G_i}{\partial r}\frac{\partial G_j}{\partial r}\right) r\, dz\, dr
                  + 2\pi\int_e \frac{G_i G_j}{r}\, dz\, dr$$

$$M_{ij}^{(e)} = 2\pi\int_e G_i G_j r\, dz\, dr$$

> $2\pi$ 因子は方位角積分 $\int_0^{2\pi} d\phi = 2\pi$ に由来するが、後段の物理量計算で $\pi$ を乗じる
> 慣習があるため、行列自体には $\pi$ のみが係数として残る実装もある。コード内コメントを確認すること。

**固有ベクトル規格化**: 軸上ピーク電場 $\max_z |E_z(z, r=0)| = 1\ \mathrm{V/m}$ になるよう規格化。

### HOM コード（任意 $n$）

**DOF**: Nédélec（Whitney）エッジ要素 + 節点スカラー要素（$n>0$）

- エッジ DOF: $E_z$, $E_r$ の接線成分を表す
- 節点 DOF: $rE_\phi$ のスカラー自由度（$n>0$ の場合のみ）

1次 Nédélec では curl（$\nabla \times \boldsymbol{E}$）が**要素内定数**となるため、
$H$ は各要素内で一定値をとる（節点補間ではない）。

2次要素（Webb 階層基底）では CT/LN + LT/LN + Face DOF の組み合わせを使用する。

---

## 5. 電磁場の再構成

### TM0 の場の計算

固有ベクトル $\boldsymbol{u}$ は節点 $H_\phi$ 値（実数、定在波 / 複素数、進行波）。

$$E_z = \frac{-j}{\omega\varepsilon_0}\left(\frac{H_\phi}{r} + \frac{\partial H_\phi}{\partial r}\right), \quad
E_r = \frac{j}{\omega\varepsilon_0}\frac{\partial H_\phi}{\partial z}$$

軸上 ($r \to 0$) はロピタルの定理 $\lim_{r\to0} H_\phi/r = \partial H_\phi/\partial r$ を適用：

$$E_z\big|_{r=0} = \frac{-2j}{\omega\varepsilon_0}\frac{\partial H_\phi}{\partial r}, \quad E_r = 0$$

### HOM の場の計算

ファラデー則 $\nabla \times \boldsymbol{E} = -j\omega\mu_0 \boldsymbol{H}$ より：

$$H_z = \frac{j}{\omega\mu_0}\left(-\frac{1}{r}\frac{\partial(r E_\phi)}{\partial r} + \frac{jn}{r}E_r\right)$$

$$H_r = \frac{j}{\omega\mu_0}\left(-jn\frac{E_z}{r} - \frac{\partial E_\phi}{\partial z}\right)$$

$$H_\phi = \frac{j}{\omega\mu_0}(\nabla \times \boldsymbol{E})_{zr\text{-part}}
         = \frac{j}{\omega\mu_0}\left(\frac{\partial E_r}{\partial z} - \frac{\partial E_z}{\partial r}\right)$$

1次 Nédélec では $(\nabla \times \boldsymbol{E})_{zr} = (e_{z1} + e_{z2} + e_{z3}) \times 2/A_\text{elem}$ （定数）。

### 軸上 ($r = 0$) の磁場の正しい扱い

$H_r$ と $H_z$ の式には $1/r$ 項が含まれるため、$r \to 0$ で不定形 $0/0$ が生じる場合がある。

#### $H_z$ ($r \to 0$)

$\nabla \times \boldsymbol{E}$ の $z$ 成分は：

$$H_z = \frac{j}{\omega\mu_0}\frac{1}{r}\left[\frac{\partial(r E_\phi)}{\partial r} - jn E_r\right]$$

分子 $= \partial(rE_\phi)/\partial r - jn E_r$ について：軸上では $rE_\phi = 0$（PEC 的正則条件）かつ $E_r = 0$（$n \ne 0$ の場合は対称性）なので $0/0$。  
Bessel 関数の解析解では $J_n(k_c r)/r$ の極限から $H_z(r=0) = 0$ が成り立つ（$n \ge 1$）。

#### $H_r$ ($r \to 0$)

$$H_r = \frac{j}{\omega\mu_0}\left(-jn\frac{E_z}{r} - \frac{\partial E_\phi}{\partial z}\right)$$

- $n = 0$：$H_r = (j/\omega\mu_0)(-\partial E_\phi/\partial z) = 0$（$n=0$ では $E_\phi$ DOF なし）
- $n \ge 1, r \to 0$：分子 $-jn E_z = 0$（軸上では $E_z(r=0) = 0$）かつ $\partial E_\phi/\partial z$ も → 0 となるため、
  ロピタルの定理を適用すると：

$$H_r\big|_{r=0} = \frac{j}{\omega\mu_0}\left(-jn\frac{\partial E_z}{\partial r}\bigg|_{r=0} - \frac{\partial^2 (s_\theta/r)}{\partial z\,\partial r}\bigg|_{r=0}\right)$$

ここで $s_\theta = rE_\phi$（節点 DOF）とした。1 次要素では 2 階微分がゼロになるため：

$$H_r\big|_{r=0} \approx \frac{j}{\omega\mu_0}\left(-jn\frac{\partial E_z}{\partial r}\bigg|_{r=0}\right) = \frac{n}{\omega\mu_0}\frac{\partial E_z}{\partial r}\bigg|_{r=0}$$

#### モード別の $r=0$ における磁場値

| モード | $n$ | $H_z(r=0)$ | $H_r(r=0)$ |
|---|---|---|---|
| TM01 | 0 | 0（TM モード） | 0（$n=0$） |
| TM11 | 1 | 0（TM モード） | **非ゼロ**（$\partial E_z/\partial r$ 寄与） |
| TM21 | 2 | 0（TM モード） | 0（$J_2'(0) = 0$） |
| TE11 | 1 | 0（$J_1(0)=0$） | **非ゼロ**（$\partial E_z/\partial r$ 寄与） |
| TE21 | 2 | 0（$J_2(0)=0$） | 0（$J_2'(0) = 0$） |

> **コード実装**: `field_calculator_hom.py` の `calculate_fields()` では、$r < 10^{-9}$ m の点に対して
> 上記ロピタル極限（1次微分版）を適用する。$n = 0$ は $H_r = H_z = 0$、$n \ge 1$ は
> shape function の $\partial/\partial r$ 微分から $H_r$ を計算し、$H_z = 0$ とする。

---

## 6. 工学パラメータ計算

### 蓄積エネルギー $U$

共振時（時間平均全エネルギー = ピーク磁場エネルギー = ピーク電場エネルギー）：

**TM0**（$H_\phi$ 節点ベクトル $\boldsymbol{u}$、FEM 質量行列 $\mathbf{M}$）:

$$U = \pi\mu_0\,\mathrm{Re}\bigl(\boldsymbol{u}^\dagger \mathbf{M}\, \boldsymbol{u}\bigr)$$

**HOM**（エッジ/節点 DOF から再構成した $\boldsymbol{E}$、ガウス求積）:

$$U = \frac{1}{2}\varepsilon_0 \cdot 2\pi \int\!\!\int |\tilde{E}|^2\, r\, dz\, dr
    = \pi\varepsilon_0 \int\!\!\int |\tilde{E}|^2\, r\, dz\, dr$$

定在波の係数は $1/2$（ピーク $\to$ 時間平均）。

> **規約の一貫性**: TM0 は $\mu_0|H|^2$、HOM は $\varepsilon_0|E|^2$ を使うが、
> 共振時のエネルギー等分配 ($U_E = U_H$) から両者は一致する。

### 壁面損失 $P_\text{loss}$

表面抵抗 $R_s = \sqrt{\omega\mu_0 / (2\sigma)}$ を用いて：

$$P_\text{loss} = \frac{1}{2}R_s \oint_S |H_\text{tan}|^2\, dS
= \pi R_s \int_\text{PEC wall} |H_\phi|^2\, r\, dl$$

**重要**: $P_\text{loss}$ の積分は**物理的 PEC 壁のみ**を対象とする。
E-short（対称境界）・M-short（PMC）・軸（$r=0$）は含めない。

### Q 値

$$Q = \frac{\omega\, U}{P_\text{loss}}$$

### 実効電圧 $V_\text{eff}$

粒子が速度 $v = \beta c$ で +z 方向に通過するとき（物理規約 $e^{+j\omega t}$ での導出）：

$$V_\text{eff} = \left|\int_0^L E_z(z, r=0)\, e^{+j\omega z/(\beta c)}\, dz\right|$$

- $e^{+j}$ の符号は物理規約に基づく（工学規約 $e^{-j\omega t}$ では $e^{-j\omega z/\beta c}$）
- シンクロナス条件では $E_z \propto e^{-jkz}$（前進波）なので積分が最大になる

### シャントインピーダンス

$$R_\text{eff} = \frac{V_\text{eff}^2}{P_\text{loss}}, \quad
\frac{R}{Q} = \frac{V_\text{eff}^2}{\omega\, U}$$

進行波構造では単位長あたりの量 $r = R_\text{eff} / L_\text{cell}$ [Ω/m] を用いる。

### 群速度と減衰定数（進行波のみ）

$$v_g = \frac{P_\text{flow} \cdot L_\text{cell}}{U}, \quad
\alpha = \frac{P_\text{loss}}{2\, P_\text{flow}\, L_\text{cell}}$$

### $P_\text{flow}$ の計算方法

$P_\text{flow}$ はポインティングベクトルの $z$ 成分を $z = z_\text{min}$ 断面で面積分して求める。

TM0 モードでは $\psi = H_\phi$（DOF）、$E_r = \frac{j}{\omega\varepsilon_0}\frac{\partial\psi}{\partial z}$ なので、

$$S_z = \frac{1}{2}\mathrm{Re}[E_r H_\phi^*] = -\frac{1}{2\omega\varepsilon_0}\,\mathrm{Im}\!\left[\frac{\partial\psi}{\partial z}\psi^*\right]$$

断面積分（方位角 $\int_0^{2\pi}d\phi = 2\pi$ を含む）：

$$\boxed{P_\text{flow} = -\frac{\pi}{\omega\varepsilon_0}\int_0^{r_\text{max}} \mathrm{Im}\!\left[\frac{\partial\psi}{\partial z}\psi^*\right] r\, dr}$$

> **旧実装の誤り（2026年4月以前）**: 全要素の2D域を積分し、かつ $1/r$ を被積分関数に含めていたため、
> 次元的にも物理的にも正しくない量を計算していた。群速度の符号は合っていたが値が誤っていた。

#### FEM による数値実装（`FEM_code/post_process_unified.py: calc_p_flow`）

1. $z = z_\text{min}$ に属するノードを座標で特定（許容誤差 $= L_\text{cell} \times 10^{-6}$）
2. 全要素を走査し、$z_\text{min}$ 境界に乗るエッジを収集（重複排除）
3. 各エッジについて 3 点ガウス求積でラインインテグラルを計算
   - エッジパラメータ $t \in [0,1]$ と要素の重心座標 $(L_1,L_2,L_3)$ のマッピング：
     - 辺 0–1: $(1-t,\ t,\ 0)$
     - 辺 1–2: $(0,\ 1-t,\ t)$
     - 辺 2–0: $(t,\ 0,\ 1-t)$
   - 各ガウス点で $\partial\psi/\partial z = \sum_i (\partial G_i/\partial z)\,\psi_i$、$\psi = \sum_i G_i\psi_i$、$r = \sum_i L_i r_i$ を計算
   - Jacobian $= L_\text{edge}/2$（エッジ端点間の $r$ 方向距離）
4. $P_\text{flow} = -(\pi/\omega\varepsilon_0)\sum_\text{edges}(\text{edge integral})$

1次・2次メッシュ両対応。$r=0$（軸上）は $\psi=0$ のため自然にゼロ寄与。

#### 有限差分法による検証

位相アドバンス $\theta$ を 1° だけ変化させた 2 点の FEM 計算から：

$$v_g^\text{FD} = \frac{L_\text{cell}\cdot 2\pi\,\Delta f}{\Delta\theta}$$

Sバンド加速空洞 1 セル（$L = 35\ \text{mm}$）で $\theta = 120°$, $121°$ の結果を比較したところ、
全 10 モードにわたって $P_\text{flow}$ 法との差は最大 0.005 $c$ 以内（相対誤差数%）で一致を確認。

---

## 7. 境界条件の種類と P_loss への影響

| PhysicalGroup 名 | FEM 行列の扱い | P_loss 積分 | 備考 |
|---|---|---|---|
| `PEC` | Dirichlet ($E_\text{tan}=0$) | **含む** | 物理的金属壁 |
| `Dirichlet` | Dirichlet ($E_\text{tan}=0$) | **含む** | PEC と同一扱い |
| `E-short` | Dirichlet ($E_\text{tan}=0$) | **含まない** | 対称境界（仮想壁）|
| `M-short` | 自然境界条件 (何もしない) | 含まない | PMC、表面電流なし |
| 軸 $r=0$, $n=0$ | 自然境界条件 | 含まない | $H_\phi=0$ は成立 |
| 軸 $r=0$, $n\ge1$ | Dirichlet 強制 | 含まない | $E_z=E_r=0$, $r=0$ |

> **実装上の注意**: `mesh_reader.py` は `boundary_edges_pec`（FEM 行列用、E-short を含む）と
> `boundary_edges_pec_loss`（P_loss 用、E-short を除く）を別々に保存する。
> `post_process_hom.py` は `boundary_edges_pec_loss` を優先使用し、
> 旧ファイルには座標フィルタ（z = z_min のエッジを除外）で対応する。

---

## 参考：符号規約の比較

| 量 | 本コードの式 | 工学規約 ($e^{-j\omega t}$) |
|---|---|---|
| 時間依存 | $e^{+j\omega t}$ | $e^{-j\omega t}$ |
| 前進波の空間依存 | $e^{-jkz}$ | $e^{+jkz}$ |
| PBC（+z 前進波） | $x_\text{max} = e^{-j\theta} x_\text{min}$ | $x_\text{max} = e^{+j\theta} x_\text{min}$ |
| Faraday の法則 | $\nabla\times E = -j\omega\mu H$ | $\nabla\times E = +j\omega\mu H$ |
| $V_\text{eff}$ 位相因子 | $e^{+j\omega z/\beta c}$ | $e^{-j\omega z/\beta c}$ |

---

*最終更新: 2026-04-29*  
*対象コード: `FEM_code/` (TM0), `FEM_HOM_code/` (HOM)*
