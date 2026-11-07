# 物理規約・計算方法リファレンス

本ドキュメントは、TM0 コード (`fem_tm0/`) および HOM コード (`fem_hom/`) に共通する
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

`fem_tm0/` は $n=0$ 専用、`fem_hom/` は任意の $n$ に対応する。

> **HOM のコアの変数**: HOM のコアは $E_\phi$ の代わりに $E_\theta = E_\phi / j$（$H_z$, $H_r$ も同じく物理量の $1/j$ 倍）を持つ。要素行列 $r|c|^2 + |n\boldsymbol{E}_t - \nabla(rE_\theta)|^2/r$ を実数にする取り方で、$c = \partial_z E_r - \partial_r E_z$。

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
# fem_tm0/field_recon.py (TM0)
coeff = -1j / (omega * self.eps0)   # E = (-j/ωε) × curl_H
```

```python
# fem_hom/field_recon.py (HOM)
coef = 1j / (self.omega * self.mu0)  # H = (+j/ωμ) × curl_E
```

### アニメーションの時間発展

時刻 $t$ に相当する位相角 $\theta_\text{time}$ (degree 単位) のとき：

$$E(z, r, t) = \mathrm{Re}\bigl[\tilde{E}(z,r)\, e^{j\theta_\text{time}}\bigr]$$

```python
comp_phase = np.exp(1j * np.deg2rad(theta_time))
Ez_plot = (Ez_comp * comp_phase).real
```

$\theta_\text{time}$ を 0° → 360° に連続変化させることで時間発展アニメーションを生成する。

> 進行波では固有ベクトルの全体位相が不定（§4）なので、$\theta_\text{time} = 0°$ の図がどの瞬間かは解き直すと変わる。工学量には影響しない。

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
# physics_core/pbc_phase.py の pbc_phase_factor（TM0 も HOM もこれを使う）
return cmath.exp(-1j * theta_rad)   # ← 物理規約: e^{-jθ}
```

実部と虚部に分けて書くと次のとおり（ソルバは複素の変換行列を使うので、この形は使っていない。`physics_core/pbc_phase.py` の `real_imag_pbc_matrix` がこの式）：

$$x_j = (\cos\theta - j\sin\theta)\, x_i
\quad\Rightarrow\quad
\begin{cases}
\mathrm{Re}(x_j) = \cos\theta \cdot \mathrm{Re}(x_i) + \sin\theta \cdot \mathrm{Im}(x_i) \\
\mathrm{Im}(x_j) = -\sin\theta \cdot \mathrm{Re}(x_i) + \cos\theta \cdot \mathrm{Im}(x_i)
\end{cases}$$

### 周期面にある壁の点（HOM, $n \ge 1$）

周期面のエッジの対の符号は、エッジの基底の向き（番号の小さい節点 → 大きい節点）が $r$ 方向で揃うなら $+1$、逆なら $-1$（三角形の向きによらない。3.0.0 までは三角形の向きの一致を仮定していて、$z$ 方向に領域を分けたセルで周波数が全く違っていた）。

周期条件は両面の値を結びつけるだけで、値を 0 にはしない（壁が片方の周期面だけに触れるとき、もう片方の面の対の節点も壁の像なので固定する）。周期面の節点のうち周期面以外の PEC 壁に接するもの（外壁の角、アイリスの先端、同軸の内導体、$r = 0$ の端で軸の辺が PEC 扱いになる点）は $E_\theta = 0$ なので、周期条件でつながず PEC として固定する（`fem_hom/boundary.py` の `find_periodic_boundary_pairs`）。磁気壁（M-short）の角は固定せず周期条件でつなぐ。

3.0.0 までは $r = r_\text{max}$ の点だけを周期条件から外していたので、$r < r_\text{max}$ の壁の点に $E_\theta = 0$ が課されず、$n \ge 1$ の周波数が低く出ていた（CHANGELOG）。

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

$$K_{ij}^{(e)} = 2\int_e \left[r\,\nabla G_i\cdot\nabla G_j + \frac{G_i G_j}{r}
                  + \frac{\partial (G_i G_j)}{\partial r}\right] dz\, dr$$

$$M_{ij}^{(e)} = 2\int_e G_i G_j\, r\, dz\, dr$$

> 係数 2 は両辺に共通なので固有値には効かない。方位角積分の $2\pi$ は行列に含まないので、物理量の計算で補う（§6 の $U$）。

**固有ベクトル規格化**: post の工学量は、軸上ピーク電場 $\max_z |\tilde{E}_z(z, r=0)| = 1\ \mathrm{V/m}$ になるよう固有ベクトルを規格化して計算する。$\tilde{E}_z$ は複素振幅（定在波では実数）。規格化は post の中だけで行い、HDF5 の固有ベクトルは solve の値のまま保存する。軸がメッシュに無いときは規格化しない（係数 1）。場の書き出しの目標壁損失（`--scale-to-power`）は、規格化によらない比 $P_\text{loss}/U$ に保存された固有ベクトルの $U$ を掛けて、保存された固有ベクトルの壁損失に換算してから倍率を求める。

> **進行波の全体位相**: 進行波の固有ベクトルは複素数で、全体に $e^{j\varphi}$ を掛けたものも固有ベクトルである。$\varphi$ は ARPACK の初期ベクトルで決まり、解き直すと変わる（SciPy 1.16 では新しいプロセスの最初の解は毎回同じ位相、SciPy 1.18 では実行のたびに変わる）。工学量が $\varphi$ に依存しないよう、規格化と電圧積分（§6）は実部（ある瞬間の値）ではなく複素振幅で行う。3.0.0 までの実装は実部を使っていたので、進行波の U・P_loss・P_diel・P_flow・V_eff・V_acc・R/Q がこの位相に依存していた（CHANGELOG）。

### HOM コード（任意 $n$）

**DOF**: Nédélec（Whitney）エッジ要素 + 節点スカラー要素（$n>0$）

- エッジ DOF: $E_z$, $E_r$ の接線成分を表す
- 節点 DOF: $rE_\theta$（$= rE_\phi/j$。§1）のスカラー自由度（$n>0$ の場合のみ）

1次 Nédélec では curl（$\nabla \times \boldsymbol{E}$）が**要素内定数**となるため、
$H$ は各要素内で一定値をとる（節点補間ではない）。

2次要素（Webb 階層基底）では CT/LN + LT/LN + Face DOF の組み合わせを使用する。

---

## 5. 電磁場の再構成

### TM0 の場の計算

固有ベクトル $\boldsymbol{u}$ は節点 $H_\phi$ 値（実数、定在波 / 複素数、進行波）。進行波の $\boldsymbol{u}$ は全体位相 $e^{j\varphi}$ が不定（§4）。

$$E_z = \frac{-j}{\omega\varepsilon_0}\left(\frac{H_\phi}{r} + \frac{\partial H_\phi}{\partial r}\right), \quad
E_r = \frac{j}{\omega\varepsilon_0}\frac{\partial H_\phi}{\partial z}$$

軸上 ($r \to 0$) はロピタルの定理 $\lim_{r\to0} H_\phi/r = \partial H_\phi/\partial r$ を適用：

$$E_z\big|_{r=0} = \frac{-2j}{\omega\varepsilon_0}\frac{\partial H_\phi}{\partial r}, \quad E_r = 0$$

### HOM の場の計算

ファラデー則 $\nabla \times \boldsymbol{E} = -j\omega\mu_0 \boldsymbol{H}$ より：

$$H_z = \frac{j}{\omega\mu_0}\cdot\frac{1}{r}\left(\frac{\partial(r E_\phi)}{\partial r} - jnE_r\right)$$

$$H_r = \frac{j}{\omega\mu_0}\left(\frac{jn}{r}E_z - \frac{\partial E_\phi}{\partial z}\right)$$

$$H_\phi = \frac{j}{\omega\mu_0}(\nabla \times \boldsymbol{E})_{zr\text{-part}}
         = \frac{j}{\omega\mu_0}\left(\frac{\partial E_r}{\partial z} - \frac{\partial E_z}{\partial r}\right)$$

コアの変数（§1。$H_z$, $H_r$, $E_\theta$ はいずれも $E_\phi$ などの物理量の $1/j$ 倍）では 2 項が同相になる。`field_recon` と壁損失の計算はこの形を使う：

$$H_z = \frac{j}{\omega\mu_0}\left(\frac{1}{r}\frac{\partial(r E_\theta)}{\partial r} - \frac{n}{r}E_r\right), \quad
H_r = \frac{j}{\omega\mu_0}\left(\frac{n}{r}E_z - \frac{\partial E_\theta}{\partial z}\right)$$

1次 Nédélec では $(\nabla \times \boldsymbol{E})_{zr} = (e_{z1} + e_{z2} + e_{z3}) / A_\text{elem}$ （定数。$A_\text{elem}$ は要素の面積）。

### 軸上 ($r = 0$) の磁場の正しい扱い

$H_r$ と $H_z$ の式には $1/r$ 項が含まれるため、$r \to 0$ で不定形 $0/0$ が生じる場合がある。

#### $H_z$ ($r \to 0$)

$\nabla \times \boldsymbol{E}$ の $z$ 成分は：

$$H_z = \frac{j}{\omega\mu_0}\frac{1}{r}\left[\frac{\partial(r E_\phi)}{\partial r} - jn E_r\right]$$

分子 $= \partial(rE_\phi)/\partial r - jn E_r$ について：軸上で $E_\phi$ と $E_r$ は $n \ge 2$ では 0、$n = 1$ では非ゼロで $E_\phi = jE_r$（軸上の $\boldsymbol{E}$ は一様な横向きの場）である。どちらも分子は 0 になるので $0/0$。  
TE モードの解析解は $H_z \propto J_n(k_c r)$ なので、$J_n(0) = 0$ から $H_z(r=0) = 0$ が成り立つ（$n \ge 1$）。

#### $H_r$ ($r \to 0$)

$$H_r = \frac{j}{\omega\mu_0}\left(\frac{jn}{r}E_z - \frac{\partial E_\phi}{\partial z}\right)$$

- $n = 0$：$H_r = (j/\omega\mu_0)(-\partial E_\phi/\partial z) = 0$（$n=0$ では $E_\phi$ DOF なし）
- $n \ge 1, r \to 0$：分子 $jn E_z = 0$（軸上では $E_z(r=0) = 0$）なので、ロピタルの定理を適用すると：

$$H_r\big|_{r=0} = \frac{j}{\omega\mu_0}\left(jn\frac{\partial E_z}{\partial r}\bigg|_{r=0} - \frac{\partial^2 s}{\partial z\,\partial r}\bigg|_{r=0}\right)$$

ここで $s = rE_\phi$（節点 DOF は $s/j$）とした。

#### モード別の $r=0$ における磁場値

| モード | $n$ | $H_z(r=0)$ | $H_r(r=0)$ |
|---|---|---|---|
| TM01 | 0 | 0（TM モード） | 0（$n=0$） |
| TM11 | 1 | 0（TM モード） | **非ゼロ**（$\partial E_z/\partial r$ 寄与） |
| TM21 | 2 | 0（TM モード） | 0（$J_2'(0) = 0$） |
| TE11 | 1 | 0（$J_1(0)=0$） | **非ゼロ**（$\partial E_\phi/\partial z$ 寄与） |
| TE21 | 2 | 0（$J_2(0)=0$） | 0（$J_2'(0) = 0$） |

> **コード実装**: `fem_hom/field_recon.py` の `calculate_fields()` は $|r| \le 10^{-9}$ m の点で $H_z = 0$ とし、$H_r$ は上の極限を数値で評価せず正則性で決める。
> $n = 1$ の軸上の $\boldsymbol{H}$ は一様な横向きの場なので（$\boldsymbol{E}$ と同じく）$H_\phi = jH_r$、コアの変数では $H_r = -H_\theta$。$n \ne 1$ は $H_r = 0$。
> $H_\theta$ は要素の $\partial_z E_r - \partial_r E_z$ から直接求まる。2 次要素で、ピルボックスの TM110、導波管セルの TM11・TE11 の解析解と振幅で $h$ = 2 mm なら 0.35% 以内、$h$ = 4 mm なら 0.7% 程度、複素数で 2 mm なら 2% 程度。1 次要素の $H_\theta$ は要素で一定なので、軸上の値は収束しない（1 mm で典型 16%）。
> 3.0.0 までは $n \ge 1$ で $H_r = (j/\omega\mu_0)(-\partial E_\theta/\partial z)$ として $n\,\partial E_z/\partial r$ の項を落としていた（TM110 でほぼ 0、TM11 で真値の 0.3 倍程度）。
> $P_\text{loss}$ と $P_\text{flow}$ は軸上を評価しない。
> $n = 1$ の軸上の節点の $E_\theta$ は $E_r$（節点を囲む要素の平均）とする（$E_\phi = jE_r$、コアの変数で $E_\theta = E_r$）。節点 DOF は $rE_\theta = 0$ なので $E_\theta$ はここからは決まらない。3.0.0 までは隣接節点の平均だった。この置き換えは場の再構成だけで、工学量（$U$、$P_\text{loss}$、$P_\text{flow}$）は通らない。
> 軸から要素の最長の辺の 5% より近い点は、$H_z$ と $H_r$ を軸上と同じく正則性で決める（$(n/r)$ の項は 0/0 に近く、3.0.0 までは $h$ = 2 mm の $r = 0.1$ mm で $H_z$ が最大の $H$ の 400% に達した）。対象は軸に頂点が接する要素だけで、軸の無いメッシュ（細い内導体）には効かない。修正後は 2 次要素の $h$ = 2 mm、$n = 1$ で $H_r$ の振幅 0.7%、複素数で帯のすぐ外は最大 11%、$H_z$ 6% 以内。$n \ge 2$ の帯のすぐ外は 2 mm で最大の $H$ の 30〜40%（$n/r$ が雑音を増幅する）。1 次要素の軸の近くは信頼できない。

---

## 6. 工学パラメータ計算

### 蓄積エネルギー $U$

共振時（時間平均全エネルギー = ピーク磁場エネルギー = ピーク電場エネルギー）：

**TM0**（$H_\phi$ 節点ベクトル $\boldsymbol{u}$、FEM 質量行列 $\mathbf{M}$）:

$$U = \frac{1}{2}\mu_0 \cdot 2\pi \int\!\!\int |H_\phi|^2\, r\, dz\, dr
    = \frac{\pi}{2}\mu_0\,\mathrm{Re}\bigl(\boldsymbol{u}^\dagger \mathbf{M}\, \boldsymbol{u}\bigr)$$

**HOM**（エッジ/節点 DOF から再構成した $\boldsymbol{E}$、ガウス求積）:

$$U = \frac{1}{2}\varepsilon_0 \cdot 2\pi \int\!\!\int \varepsilon_r |\tilde{E}|^2\, r\, dz\, dr
    = \pi\varepsilon_0 \int\!\!\int \varepsilon_r |\tilde{E}|^2\, r\, dz\, dr$$

定在波でも進行波でも係数は $1/2$（ピーク振幅に対する全エネルギー）。3.0.0 までの HOM は定在波だけ係数が 1 で、$U$・$Q_\text{wall}$・$P_\text{diel}$ が 2 倍だった（$n = 0$ で $\tan\delta = 0$ なら $Q$ も。CHANGELOG）。

> **規約の一貫性**: TM0 は $\mu_0|H|^2$、HOM は $\varepsilon_0|E|^2$ を使うが、
> 共振時のエネルギー等分配 ($U_E = U_H$) から両者は一致する。

> **HOM の $n \ge 1$ の絶対値**: $U$ と $P_\text{loss}$ は $e^{jn\phi}$ の場（全周で $|E|$, $|H|$ が一様）の値で、方位角積分は $2\pi$。同じピーク振幅の $\cos n\phi$ のモードでは $U$ も $P_\text{loss}$ も半分になり、$Q$ は同じ。

### 壁面損失 $P_\text{loss}$

表面抵抗 $R_s = \sqrt{\omega\mu_0 / (2\sigma)}$ を用いて：

$$P_\text{loss} = \frac{1}{2}R_s \oint_S |H_\text{tan}|^2\, dS
= \pi R_s \int_\text{PEC wall} \left(|H_\phi|^2 + |H_z t_z + H_r t_r|^2\right) r\, dl$$

$(t_z, t_r)$ は $(z, r)$ 面内の壁の接線。$n = 0$ では第 2 項は 0。$H_z$, $H_r$ は §5 のコアの変数の式で計算する。3.0.0 までの HOM は curl の 2 項の間に余分な $j$ があり、$n \ge 1$ の $P_\text{loss}$・$Q$・$\alpha$ が誤っていた（CHANGELOG）。

**重要**: $P_\text{loss}$ の積分は**物理的 PEC 壁のみ**を対象とする。
E-short（対称境界）・M-short（PMC）・軸（$r=0$）は含めない。

> **既知の制約**: HOM の $P_\text{loss}$ は、曲線の壁では 2 次要素でもメッシュの 1 次でしか収束せず、誤差の向きは形状による（半径 50 mm の球の TM101, $n = 0$ で $Q$ / 解析解は $h$ = 8 / 4 / 2 / 1 mm で 0.904 / 0.953 / 0.975 / 0.988。S バンドのセルの $n = 0$ は多くのモードで TM0 より高く、最低次の 3 モードで、$h$ = 2.5 mm の端面 PEC で最大 3%、端面 E-short で最大 4.5%）。周波数の誤差は球で $3\times10^{-5}$ 以下、TM0 の解析は球のメッシュで解析解と 0.1% 以内。

### Q 値

$$Q = \frac{\omega\, U}{P_\text{loss} + P_\text{diel}}$$

ver2.3 以降、`Q` は**全損失込みの合計 Q** である（無損失誘電体
$\tan\delta = 0$ のときは従来どおり $Q = \omega U / P_\text{loss}$ とビット同一）。
壁損失のみの Q は `Q_wall`、誘電体損失のみの Q は `Q_diel` として別途出力される。

### 誘電体損失 $Q_\text{diel}$（ver2.3、摂動法）

領域ごとの誘電正接 $\tan\delta$（複素誘電率
$\varepsilon = \varepsilon_0\varepsilon_r(1 - j\tan\delta)$ の損失）を、
**実固有値解を変えない摂動法**で扱う。時間平均の誘電体損失は

$$P_\text{diel} = \frac{\omega\varepsilon_0}{2}
\sum_e \varepsilon_r^{(e)} \tan\delta^{(e)} \int_e |\tilde{E}|^2\, 2\pi r\, dA$$

実装では規約係数の取り違えを避けるため、**同一の求積機構で評価した
2 つの積分の比**として計算する：

$$Q_\text{diel} = \frac{\sum_e \varepsilon_r^{(e)} \int_e |\tilde{E}|^2\, r\, dA}
{\sum_e \varepsilon_r^{(e)} \tan\delta^{(e)} \int_e |\tilde{E}|^2\, r\, dA},
\qquad P_\text{diel} = \frac{\omega U}{Q_\text{diel}}, \qquad
\frac{1}{Q} = \frac{1}{Q_\text{wall}} + \frac{1}{Q_\text{diel}}$$

- **一様充填では $Q_\text{diel} = 1/\tan\delta$ が恒等的に成立**する
  （分子 = 分母 / $\tan\delta$。物理的には共振時のエネルギー等分配
  $U_E = U_H = U/2$ と上式 $P_\text{diel}$ の帰結 $P_\text{diel} = \omega \tan\delta\, U$）。
- TM0 は $H_\phi$ から再構成した $E$（`field_recon` と同一の式）を
  7 点求積で直接積分（`fem_tm0/post_process._dielectric_weight_integrals`）。
  $E$ は $H$ の微分で 1 次精度が落ちるため、**部分充填**の損失配分は
  周波数より収束が 1 次遅い（2 次要素を推奨）。HOM は E 場ネイティブで
  `_stored_energy` と同一機構（重み $\varepsilon_r \tan\delta$）を流用する。
- 摂動法はモード形状の変化を無視する近似であり、$\tan\delta \ll 1$
  （目安 $\lesssim 0.05$）で有効。固有値問題は実数のままで、
  eigsh / PARDISO 経路・計算時間に影響しない。
- $R_\text{shunt}$・$\alpha$（減衰定数）は従来どおり**壁損失ベース**のまま。
- $\mu_r$ と磁性損失 $\tan\delta_m$ は未対応（スコープ外）。

### 実効電圧 $V_\text{eff}$

粒子が速度 $v = \beta c$ で +z 方向に通過するとき（物理規約 $e^{+j\omega t}$ での導出）：

$$V_\text{eff} = \left|\int_0^L \tilde{E}_z(z, r=0)\, e^{+j\omega z/(\beta c)}\, dz\right|, \quad
V_\text{acc} = \left|\int_0^L \tilde{E}_z(z, r=0)\, dz\right|$$

- $\tilde{E}_z$ は複素振幅（定在波では実数、進行波は複素数のまま積分する。§4）
- $V_\text{acc}$ は走行時間因子を含まない電圧（$V_\text{eff}$ の $\beta\to\infty$ の極限。進行波ではセルごとに足せない量で、加速電圧の意味は持たない）
- $e^{+j}$ の符号は物理規約に基づく（工学規約 $e^{-j\omega t}$ では $e^{-j\omega z/\beta c}$）
- シンクロナス条件では $E_z \propto e^{-jkz}$（前進波）なので積分が最大になる
- 軸に誘電体があるときの $\tilde{E}_z$ は $\frac{1}{j\omega\varepsilon_0\varepsilon_r}\cdot 2\,\partial_r H_\phi$ で、要素の $\varepsilon_r$ で割る。軸を横切る界面では $D_z$ が連続なので $E_z$ が跳ぶ。規格化（§4）もこの値で行う。3.0.0 までは $\varepsilon_r$ で割っていなかった（CHANGELOG）

### シャントインピーダンス

$$R_\text{eff} = \frac{V_\text{eff}^2}{P_\text{loss}}, \quad
\frac{R}{Q} = \frac{V_\text{eff}^2}{\omega\, U}$$

進行波構造では単位長あたりの量 $r = R_\text{eff} / L_\text{cell}$ [Ω/m] を用いる。

> **規格化によらない量**: Q・R/Q・$R_\text{eff}$・$r$・$v_g$・$\alpha$ は $|\boldsymbol{u}|^2$ の比なので規格化係数によらない。U・P_loss・P_diel・P_flow は $|\boldsymbol{u}|^2$、V_eff・V_acc は $|\boldsymbol{u}|$ に比例するので、絶対値は、TM0 では §4 の規格化（軸上ピーク 1 V/m）での値、HOM では保存された固有ベクトルの値（$\boldsymbol{v}^\dagger\mathbf{M}\boldsymbol{v} = 1$ で $U \approx \pi\varepsilon_0$）である。

### 群速度と減衰定数（進行波のみ）

$$v_g = \frac{P_\text{flow} \cdot L_\text{cell}}{U}, \quad
\alpha = \frac{P_\text{loss}}{2\, P_\text{flow}\, L_\text{cell}}$$

後退波（$v_g < 0$）では $\alpha$ も負になる。$\theta = 0°, 180°$ では $P_\text{flow}$ が丸め誤差の大きさなので、$v_g$ はほぼ 0 で、$\alpha$ は意味を持たない。

### $P_\text{flow}$ の計算方法

$P_\text{flow}$ はポインティングベクトルの $z$ 成分を $z = z_\text{min}$ 断面で面積分して求める。

TM0 モードでは $\psi = H_\phi$（DOF）、$E_r = \frac{j}{\omega\varepsilon_0}\frac{\partial\psi}{\partial z}$ なので、

$$S_z = \frac{1}{2}\mathrm{Re}[E_r H_\phi^*] = -\frac{1}{2\omega\varepsilon_0}\,\mathrm{Im}\!\left[\frac{\partial\psi}{\partial z}\psi^*\right]$$

断面積分（方位角 $\int_0^{2\pi}d\phi = 2\pi$ を含む）：

$$\boxed{P_\text{flow} = -\frac{\pi}{\omega\varepsilon_0}\int_0^{r_\text{max}} \mathrm{Im}\!\left[\frac{\partial\psi}{\partial z}\psi^*\right] r\, dr}$$

> **旧実装の誤り（2026年4月以前）**: 全要素の2D域を積分し、かつ $1/r$ を被積分関数に含めていたため、
> 次元的にも物理的にも正しくない量を計算していた。群速度の符号は合っていたが値が誤っていた。

HOM では $S_z = \frac{1}{2}\mathrm{Re}[E_r H_\phi^* - E_\phi H_r^*]$ に §5 の式を代入して、コアの変数 $E_\theta$ で：

$$\boxed{P_\text{flow} = \frac{\pi}{\omega\mu_0}\int_0^{r_\text{max}} \left\{\mathrm{Im}[E_r c^*]\, r - n\,\mathrm{Im}[E_\theta E_z^*] + \mathrm{Im}\!\left[E_\theta \left(\frac{\partial E_\theta}{\partial z}\right)^{\!*}\right] r\right\} dr}, \quad c = \frac{\partial E_r}{\partial z} - \frac{\partial E_z}{\partial r}$$

> 3.0.0 までの実装は第 2 項が実部で、$E_z$ を持つ $n \ge 1$ の進行波の $P_\text{flow}$・$v_g$・$\alpha$ が誤っていた（CHANGELOG）。

#### FEM による数値実装（`fem_tm0/post_process.py: tm0_p_flow_zmin`）

1. $z = z_\text{min}$ に属するノードを座標で特定（許容誤差 $= L_\text{cell} \times 10^{-6}$）
2. 全要素を走査し、$z_\text{min}$ 境界に乗るエッジを収集（重複排除）
3. 各エッジについて 3 点ガウス求積でラインインテグラルを計算
   - エッジパラメータ $t \in [0,1]$ と要素の重心座標 $(L_1,L_2,L_3)$ のマッピング：
     - 辺 0–1: $(1-t,\ t,\ 0)$
     - 辺 1–2: $(0,\ 1-t,\ t)$
     - 辺 2–0: $(t,\ 0,\ 1-t)$
   - 各ガウス点で $\partial\psi/\partial z = \sum_i (\partial G_i/\partial z)\,\psi_i$、$\psi = \sum_i G_i\psi_i$、$r = \sum_i L_i r_i$ を計算
   - Jacobian $= L_\text{edge}$（エッジ端点間の $r$ 方向距離。重みは $t \in [0,1]$ で和が 1）
4. $P_\text{flow} = -(\pi/\omega\varepsilon_0)\sum_\text{edges}(\text{edge integral})/\varepsilon_r$（$\varepsilon_r$ はそのエッジが属する要素の値。$E_r \propto 1/\varepsilon_r$ のため）

1次・2次メッシュ両対応。$r=0$（軸上）は $\psi=0$ のため自然にゼロ寄与。

#### 有限差分法による検証

位相アドバンス $\theta$ を変えた 2 点の FEM 計算から：

$$v_g^\text{FD} = \frac{L_\text{cell}\cdot 2\pi\,\Delta f}{\Delta\theta}$$

Sバンド加速空洞 1 セル（$L = 35\ \text{mm}$）で $\theta = 119°$, $121°$ の中心差分を $\theta = 120°$ の $P_\text{flow}$ 法と比較したところ、
全 10 モードにわたる $P_\text{flow}$ 法との差は、$h$ = 1.25 mm で最大 0.004 $c$、サンプルの 5 mm メッシュで最大 0.015 $c$ で、メッシュとともに小さくなる。
HOM の $n$ = 1, 2 では、同じセルのサンプルのメッシュ、モード 0〜5 で差は 0.005 $c$ 以内（$\theta$ = 119°, 121° の差分。`tests/test_hom_post_analytic.py` は 0.003 $c$ か相対 3.5% の大きい方を許容として監視する）。

---

## 7. 境界条件の種類と P_loss への影響

| PhysicalGroup 名 | FEM 行列の扱い | P_loss 積分 | 備考 |
|---|---|---|---|
| `PEC` | Dirichlet ($E_\text{tan}=0$) | **含む** | 物理的金属壁 |
| `Dirichlet` | Dirichlet ($E_\text{tan}=0$) | **含む** | PEC と同一扱い |
| `E-short` | Dirichlet ($E_\text{tan}=0$) | **含まない** | 対称境界（仮想壁）|
| `M-short` | 自然境界条件 (何もしない) | 含まない | PMC、表面電流なし |
| 軸 $r=0$, $n=0$ | 自然境界条件 | 含まない | $H_\phi=0$ は成立 |
| 軸 $r=0$, $n\ge1$ | Dirichlet 強制（軸の両端の節点と、それに接する軸の辺は端面のラベルに従う） | 含まない | $E_z=0$, $rE_\theta=0$（$E_r$ は拘束しない。$n=1$ では軸上で非ゼロ。端面が None / M-short のとき軸の両端は自由だが、周波数への影響はメッシュの誤差より小さい） |

> **壁の辺の判定**: 両端が壁の節点でも内部の辺は壁ではない（鋭角の角の対角、要素 1 層の隙間を渡る辺）。2 次要素は中点節点も壁の節点である辺、1 次要素はメッシュの線要素（`/mesh/physical_edges`）の辺を壁とする。線要素を持たない古い 1 次要素の h5 は領域の外周の辺だけを数え、`post` が警告する。Dirichlet と壁損失の両方に同じ判定を使う。片方だけだと誤差が打ち消し合わなくなり $Q$ が悪化する。
>
> **実装上の注意**: `shared/mesh_loader.py` の `compute_hom_boundary` は `boundary_edge_indices`（FEM 行列用、E-short を含む）と
> `pec_loss_edge_indices`（P_loss 用、E-short を除く）を別々に作る。`fem_hom/post_process.py` の壁損失は `pec_loss_edge_indices` を使う。

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

*最終更新: 2026-11-07*  
*対象コード: `fem_tm0/` (TM0), `fem_hom/` (HOM)*
