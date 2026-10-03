# Windows 版（EXE）の作り方

`AxiCavity-FEM.exe`（GUI とコマンドラインを兼ねる 1 つの EXE）を [Nuitka](https://nuitka.net/) の standalone で作り、
zip にして GitHub の Releases で配布する。設計の説明は [docs/DEVELOPER_GUIDE.md](../docs/DEVELOPER_GUIDE.md) の
「Windows 版（EXE）」。

## 必要なもの

- Windows、Python 3.13（ビルドに使う Python と同じ版の埋め込み用 Python をメッシャに入れる）
- このリポジトリを editable install し、EXE に入れるものを全部入れる:

  ```bash
  pip install -e ".[gui,viz3d,accel,dev]"
  pip install nuitka ordered-set zstandard
  ```

- C コンパイラ: MSVC（Visual Studio Build Tools）が無ければ Nuitka が zig を自動で取得する（初回のみダウンロード）
- ネットワーク（初回のみ）: 埋め込み Python と gmsh のソース（`build_mesher.py --download` が取得してキャッシュ）

## 作る

```bash
python packaging/build_exe.py --build-dir C:\build\axicavity
```

ビルド先は **OneDrive の外**に置く（数万ファイルの同期とロックで遅く・失敗しやすい）。Claude デスクトップアプリから
実行するときは `%LOCALAPPDATA%` も避ける（MSIX の仮想化で別の場所に書かれる）。初回は 30〜60 分、2 回目以降は
zig のキャッシュ（`<build-dir>/zig-cache/`）で短くなる。`--dry-run` で Nuitka のコマンドだけ表示、`--skip-nuitka` で
後処理と zip だけやり直す。

zig のキャッシュをビルド先ごとに分けているのは、Nuitka の既定の共有キャッシュだと別のプロジェクト（EM-CAD-py など）の
ビルドで作った定数のオブジェクトが返ることがあるため（定数のソース `__constants_data___constant.c` は名前もフラグも
同じで、`#embed` するファイルは前回の絶対パスで照合される）。その EXE は "Frozen object named 'encodings' is invalid"
で起動しない。`build_exe.py` はビルド後に EXE の定数を照合し、食い違えば止まる（そのときは `zig-cache/` を消して作り直す）。

出来上がるもの:

```
<build-dir>/
  AxiCavity-FEM.dist/              配布するフォルダ（zip の中身）
    AxiCavity-FEM.exe              本体（GUI + コマンドライン）
    Library/bin/                   Intel MKL / TBB の DLL（PARDISO）
    mesher/                        メッシュ生成（埋め込み Python + gmsh、GPL。build_mesher.py が組み立てる）
    samples/  docs/images/  USER_MANUAL.md  PHYSICS_AND_CONVENTIONS.md
    LICENSE  README.txt  THIRD_PARTY_NOTICES.txt
  AxiCavity-FEM-<版>-win64.zip     Releases に添付するもの
  nuitka-report.xml                同梱モジュールの一覧（THIRD_PARTY_NOTICES の元）
  mesher/  downloads/  app.ico     中間生成物
  AxiCavity-FEM.build/  zig-cache/ Nuitka の C ソースと zig のキャッシュ（再ビルドを速くする）
```

## 確かめる

```bash
python packaging/smoke_test_exe.py C:\build\axicavity\AxiCavity-FEM.dist --gui-seconds 20
```

`version`（PARDISO・メッシャ・3D 表示の検出）、`selftest`（GUI と同じ子プロセス経路）、`run`（バッチ）、
コア CLI（`solve` / `post` / `report` / `info`）、`job mesh`、2 万自由度超での PARDISO、GUI の起動を確かめる。
最後に実ウィンドウで作図 → メッシュ → 解析 → 結果 → 3D 表示を一通り操作する。

## ライセンス上の注意（同梱物）

| 同梱物 | ライセンス | 扱い |
|---|---|---|
| AxiCavity-FEM | MIT | `LICENSE` |
| Qt / PySide6 / shiboken6 | LGPL-3.0 | 動的リンク（DLL をそのまま同梱。差し替え可能）、全文を THIRD_PARTY_NOTICES に |
| planegcs | LGPL-2.1 | 同上 |
| Intel MKL / TBB（pypardiso 用） | Intel Simplified Software License | 改変せずに同梱、ライセンス全文を THIRD_PARTY_NOTICES に。intel-openmp（別の EULA）は入れない |
| gmsh | GPL-2.0-or-later | **本体には入れない**。`mesher/`（別プログラム）に入れ、gmsh のソース（`mesher/src/`）と GPL 全文を同梱 |
| numpy / scipy / matplotlib / h5py / VTK / pyvista など | BSD / MIT / PSF 系 | THIRD_PARTY_NOTICES に表記 |

MKL のライセンスは GPL のソフトウェアと組み合わせて配布することを認めていないため、gmsh を本体と同じプロセスに
入れない（本体の `.msh` 読込は `axicavity_fem.gui.mshlite`、生成は `mesher/`）。同じ理由で、オープンソース版が
GPL-3.0 のみの Qt のアドオン（Qt Data Visualization・Charts・Graphs など）も入れない。qtpy は import 時に
`QtDataVisualization` を読みにいく（無ければ黙って飛ばす）ので `--nofollow-import-to` で外し、`build_exe.py` は
ビルド後に dist にそれらの DLL が無いことを確かめる。

THIRD_PARTY_NOTICES は Nuitka のレポートのモジュールから配布物を引くが、`importlib.metadata.packages_distributions()`
は `__main__`（本体の main スクリプト）を PyMuPDF（AGPL。同梱していない）に対応させるので、`__main__` は除いている。

## ファイル

| ファイル | 役割 |
|---|---|
| `AxiCavity-FEM.py` | Nuitka の main スクリプト（同梱物の場所を環境変数で教えて `axicavity_fem.gui.launcher.main` を呼ぶ） |
| `build_exe.py` | アイコン → メッシャ → Nuitka → 同梱物 → zip |
| `build_mesher.py` | `mesher/` の組み立て（埋め込み Python・gmsh・`../mesher/axicavity_mesh.py`・コアの 2 モジュール・ソース） |
| `make_ico.py` | `src/axicavity_fem/gui/ui/app_icon.svg` → `app.ico` |
| `third_party_notices.py` | Nuitka のレポートから THIRD_PARTY_NOTICES.txt を作る |
| `smoke_test_exe.py` | 出来上がった dist の動作確認 |
| `README_dist.txt` | 配布物の `README.txt`（日本語・英語） |
| `licenses/` | dist-info に全文が無いライセンス（LGPL / GPL） |
