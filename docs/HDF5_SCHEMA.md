# HDF5 スキーマ仕様 (schema_version = 2.2)

ver2 で TM0 と HOM の出力ファイル構造を統一する。ver2.1 で誘電体（領域別 ε_r）の
保存を追加、ver2.3（schema_version=2.2）で誘電正接 tanδ の保存を追加した
（後方互換: v2.0/2.1 ファイルも読める）。

## トップレベル

| パス | 型 | 内容 |
|---|---|---|
| `/` (attrs) | — | `schema_version="2.2"`, `solver_type` (`"tm0"` or `"hom"`), `axicavity_fem_version`（solve を実行した版。post では更新されない） |
| `/mesh/` | group | メッシュデータ |
| `/materials/` | group | 領域別材料（ver2.1、誘電体メッシュのみ） |
| `/parameters/` | group | CLI 引数 + 物理規約定数（ver2.3: `--target-freq` 指定時のみ attr `target_freq_GHz` が増える） |
| `/matrices/` | group | M_global, K_global（オプション） |
| `/results/` | group | 固有値・固有ベクトル |
| `/post_process/` | group | post コマンド実行で追加 |

## /mesh/

| パス | 形状 | 説明 |
|---|---|---|
| `vertices` | (Nn, 2) float64 | `[z, r]` |
| `simplices` | (Ne, k) int32 | k=3（1次）または 6（2次） |
| `edge_map_keys` | (Ned, 2) int32 | エッジマップキー |
| `edge_map_values` | (Ned,) int32 | エッジマップ値 |
| `physical_groups/<name>` | (n,) int32 | 各物理グループのノードインデックス |
| `physical_edges/<name>` | (m, 2) int | **3.0.1**: 1D の物理グループの線要素（角の節点の対）。1 次要素の壁の判定に使う。無ければ古いファイルで、1 次要素の `post` は領域の外周の辺だけを壁に数えて警告する |
| `eps_r_per_element` | (Ne,) float64 | **ver2.1**: 要素ごと比誘電率。材料の表があるときに存在（真空でも書かれる）。H→E 換算で E を ÷ε_r するために使う |
| `tan_delta_per_element` | (Ne,) float64 | **ver2.3**: 要素ごと誘電正接 tanδ。材料の表があるときに存在（無ければ無損失扱い）。post の Q_diel 計算に使う |
| (attrs) | — | `num_nodes`, `num_edges`, `elem_order` |

## /materials/ （ver2.1、任意）

多領域/誘電体メッシュを solve したときのみ作られる。各領域の材料タグごとに属性を持つ。

| パス | 型 | 説明 |
|---|---|---|
| `<material_tag>/` (attrs) | — | `eps_r`（比誘電率）, `mu_r`（比透磁率）, `tan_delta`（誘電正接、ver2.3） |

> 後方互換: `/mesh/eps_r_per_element` と `/materials/` は**無ければ真空扱い**（`None`）。
> v2.0 ファイル（これらを持たない）もそのまま読め、TM0/HOM の場再構成は真空式になる。
> `tan_delta_per_element` / `tan_delta` 属性も同様に無ければ無損失（`None`/0.0）扱い。
> 注意: 領域 ID は H5 に保存されないため、**tanδ を効かせるには solve から再実行**が
> 必要（既存 H5 に post だけ再実行しても tanδ は付与できない）。

## /results/n{N}/standing/mode_{K}/

固有ベクトルは **フラットな 1 配列** ``eigenvector`` で格納する。これにより TM0
（節点 H_φ スカラー）と HOM（エッジ + 節点ハイブリッド）を統一的に扱える。
HOM の場再構成・ポストプロセスは、読み戻したフラット配列を
``fem_hom.field_recon.split_hom_dofs`` で CT/LN・LT/LN・face・節点（rE_theta）へ
分割する（DOF レイアウトはメッシュ情報から一意に決まる）。

| パス | 形状 | 説明 |
|---|---|---|
| `eigenvector` | (N,) float64 | フラット固有ベクトル（DOF 全体） |
| (attrs) | — | `eigenvalue_k2`, `frequency_GHz`, `frequency_Hz`, `mode_type="standing"` |

`N` は TM0: 節点数、HOM 1次: `Ned + (Nn if n>0)`、HOM 2次:
`2·Ned + 2·Ne + (Nn if n>0)`。

## /results/n{N}/traveling/phase_{xxx}/mode_{K}/

`phase_{xxx}` は θ\_deg × 10 のゼロ埋め 5 桁文字列（例: 120.0° → `"phase_01200"`）。

固有ベクトルは複素数のため `*_re`, `*_im` で分離保存：
- `eigenvector_re`, `eigenvector_im`  （形状 (N,)）

（attrs）`theta_rad`, `theta_deg`, `eigenvalue_k2`, `frequency_GHz`, `mode_type="traveling"`

固有ベクトルは solve の値のまま保存する（post で規格化しない）。全体位相 e^{jφ} は不定で、同じメッシュでも解き直すと変わる。

## /post_process/n{N}/{standing|traveling}/.../mode_{K}/

post コマンド実行後に追加（再実行すると `/post_process` を作り直す。固有ベクトルは変更しない）。TM0 の `U_stored`・`P_loss`・`P_diel`・`P_flow_zmin`・`V_eff`・`V_acc` は max_z |E_z(r=0)| = 1 V/m（E_z は複素振幅）に規格化した値。

> 3.0.0 以前の post が書いた次の値は誤り（CHANGELOG）。スキーマは同じなので、solve し直さなくても post を再実行すれば正しい値に置き換わる。
> ただし周波数と固有ベクトルも誤りの結果は solve し直す: HOM の進行波の n≥1 で周期面に達する壁が外半径より内側にあるセル（アイリス付き、同軸）や外壁が M-short のセル、HOM の進行波で z 方向に領域を分けたセルと約 40 µm より細かいメッシュ、HOM で 90° より小さい角や要素 1 層の隙間がある形状と軸の途中に接する壁（USER_MANUAL §13）。
> - 進行波 TM0 の `U_stored`・`P_loss`・`P_diel`・`P_flow_zmin`・`V_eff`・`V_acc`・`R_over_Q`（E_z の実部で計算していた）
> - 軸に誘電体がある TM0 の同じ値、定在波を含む（軸上 E_z を ε_r で割っていなかった）
> - HOM の定在波の `U_stored`・`Q_wall`・`P_diel`（n=0 を含めて 2 倍）と `Q`（n=0 で tanδ=0 なら 2 倍、tanδ>0 ではそれより小さい差）
> - HOM の n≥1 の `P_loss`・`Q`・`Q_wall`・`attenuation`（壁損失の curl の余分な j）
> - HOM の進行波の n≥1 の `P_flow_zmin`・`group_velocity`・`attenuation`（P_flow の第 2 項が実部だった）
> - TM0 の 1 次要素で 90° より小さい角や要素 1 層の隙間がある形状の `P_loss`・`Q`・`Q_wall`・`attenuation`（内部の辺を壁損失に数えていた）

| attrs | 説明 |
|---|---|
| `Q` | 品質係数（**ver2.3 以降は全損失込みの合計 Q**。tanδ=0 なら従来値と同一） |
| `Q_wall` | 壁損失のみの Q = ωU/P_loss（ver2.3、従来の `Q` に相当） |
| `Q_diel` | 誘電体損失のみの Q（ver2.3。0 = 無損失） |
| `frequency_GHz` | 共振周波数 [GHz] |
| `R_over_Q` | R/Q [Ω]（TM0 のみ） |
| `V_eff` | 実効電圧 [V]（TM0 のみ） |
| `V_acc` | 走行時間因子を含まない電圧 [V]（TM0 のみ） |
| `U_stored` | 蓄積エネルギー [J] |
| `P_loss` | 壁面損失 [W] |
| `P_diel` | 誘電体損失 [W]（ver2.3） |
| `P_flow_zmin` | z=z_min 断面ポインティングフラックス [W] |
| `group_velocity` | 群速度 [m/s]（定在波では 0） |
| `attenuation` | 減衰定数 [1/m]（定在波では 0） |
| `v_phase_c` | 位相速度 / c（HOM のみ。定在波では 0） |

## v1 (ver1) ファイルからの読み込み

`shared/hdf5_io.py` の `load_v1_legacy(path)` で ver1 (TM0 フラット型 / HOM ネスト型) を ver2 in-memory 構造に変換する。`schema_version` 属性の有無で自動判別。

書き出しは常に v2 のみ。
