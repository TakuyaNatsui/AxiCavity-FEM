# サンプル形状 / Sample geometries

Multi-Region Editor のプロジェクトファイル（`.gmshproj`）と、それに対応する材料定義
（`.materials.json`）を置いています。GUI の `File > Load Project` で開けます。

一部はメッシュ（`.msh`）も同梱してあり、インストール直後にそのままソルバを実行できます。
同梱していない形状は、GUI でプロジェクトを開いて **Export MSH** を押すか、
`export_msh_multi_region()` でメッシュを生成してください。

The `.gmshproj` files are Multi-Region Editor projects (open them with `File > Load Project`);
`.materials.json` holds the per-region material. Meshes (`.msh`) are included for a few of
them so the solver can be run immediately; regenerate the others with **Export MSH**.

| ファイル | 形状 | 用途 |
|---|---|---|
| `cylinder100mm` **(.msh 同梱)** | 円筒空洞 r = 50 mm, L = 100 mm | 最も基本的な検証用。TM010 = 2.294851 GHz が解析解 `j₀₁c/2πa` と小数 6 桁まで一致する |
| `cylinder_L200mm` | 円筒空洞 r = 50 mm, L = 200 mm | 上と同じで長さ違い |
| `sphere50mm` **(.msh 同梱)** | 球形空洞 r = 50 mm | 球ベッセル関数の解析解と比較できる。`examples/accuracy_verification/` が使用 |
| `s-band_1cell` **(.msh 同梱)** | S バンド加速空洞 1 セル（周期境界あり） | 定在波・進行波（位相スキャン）両方の例。TM0 定在波 2.615 GHz / 進行波 2π/3 モード 2.856 GHz |
| `s-band_2cell` / `3cell` / `4cell` / `Ncell` | 同 2/3/4 セル、および N セル生成用のテンプレート | 多セル構造・分散曲線 |
| `acc2_pipe_flat` | ビームパイプ付き 5 セル加速管（全長 348 mm） | 実用的な形状の例。README の図はこの形状 |
| `PF_cavity` | 蓄積リング用大型空洞（r 最大 235 mm） | 大きなメッシュの例 |
| `diel_simple1` | 誘電体を一部に置いた矩形空洞（ε_r = 2） | 誘電体の最小例 |
| `diel_simple2` **(.msh 同梱)** | 2 領域（ε_r = 1 / 10） | 誘電体境界での場の不連続を見る |
| `diel_test1` | 5 領域・誘電体リング（ε_r = 9.64, tanδ = 5.7e-6） | 誘電体損失 Q_diel の例。多領域メッシュの確認用 |

`acc2_pipe_flat.af` は Superfish 形式の入出力サンプル、`*.py` は **Export Python Script**
で書き出した自己完結のメッシュ生成スクリプトです。

## 実行例

```bash
axicavity-fem solve --type tm0 -m samples/cylinder100mm.msh --elem-order 2 --num-modes 6 -o pillbox.h5
axicavity-fem post  --type tm0 -i pillbox.h5 --cond 5.8e7
```
