# サンプル形状 / Sample geometries

バージョン 2 の形状ファイル（`.gmshproj`）と Superfish の入出力サンプル（`.af`）です。メッシュや結果は同梱していません
（どれも数秒で作れます）。

- **GUI**: 「ファイル」→「取り込み」→「形状の取り込み」で開く（パラメータ・式付きの点・境界条件・材料がそのまま入る）。
  「名前を付けて保存」で `.axiproj` プロジェクトになります。
- **コマンドライン**: `axicavity-fem-run samples/cylinder100mm.gmshproj --out pillbox` でメッシュ生成 → 解析 → 後処理を
  まとめて行い、`pillbox/` に `model.msh` と結果が出ます（Windows 版は `AxiCavity-FEM.exe run …`）。

These are version 2 geometry files (`.gmshproj`) and a Superfish example (`.af`); meshes and results are not
included. Open them in the GUI with *File → Import → Import shape*, or run
`axicavity-fem-run samples/<name>.gmshproj --out <folder>` to mesh, solve and post-process in one go.

| ファイル | 形状 | 用途 |
|---|---|---|
| `cylinder100mm` | 円筒空洞 r = 50 mm, L = 100 mm | 最も基本的な検証用。TM010 = 2.294851 GHz が解析解 `j₀₁c/2πa` と小数 6 桁まで一致する |
| `cylinder_L200mm` | 円筒空洞 r = 50 mm, L = 200 mm | 上と同じで長さ違い |
| `sphere50mm` | 球形空洞 r = 50 mm | 球ベッセル関数の解析解と比較できる。`examples/accuracy_verification/` が使用 |
| `s-band_1cell` | S バンド加速空洞 1 セル（パラメータ付き） | 定在波・進行波（位相スキャン）両方の例。TM0 定在波 2.615 GHz / 進行波 2π/3 モード 2.856 GHz |
| `s-band_2cell` / `3cell` / `4cell` / `Ncell` | 同 2/3/4 セル、および N セル生成用のテンプレート | 多セル構造・分散曲線 |
| `acc2_pipe_flat` | ビームパイプ付き 5 セル加速管（全長 348 mm） | 実用的な形状の例。README の図はこの形状 |
| `PF_cavity` | 蓄積リング用大型空洞（r 最大 235 mm） | 大きなメッシュの例 |
| `diel_simple1` | 誘電体を一部に置いた矩形空洞（ε_r = 2） | 誘電体の最小例 |
| `diel_simple2` | 2 領域（ε_r = 1 / 10） | 誘電体境界での場の不連続を見る |
| `diel_test1` | 5 領域・誘電体リング（ε_r = 9.64, tanδ = 5.7e-6、パラメータ付き） | 誘電体損失 Q_diel の例。多領域メッシュの確認用 |

`acc2_pipe_flat.af` は Superfish 形式の入出力サンプル（「ファイル」→「取り込み」→「Superfish」）、
`s-band_1cell.py` は「Python スクリプト」書き出しの例（gmsh で `.msh` を作る自己完結のスクリプト）です。
