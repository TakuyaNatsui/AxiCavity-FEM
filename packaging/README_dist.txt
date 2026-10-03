AxiCavity-FEM {version} — Windows 版
======================================

軸対称空洞（加速空洞など）の共振モード（TM0 と高次モード、定在波・進行波）を 2 次元有限要素法で解析します。
GUI でパラメトリックに形状を描き、メッシュ生成・解析・結果表示（3D を含む）まで行えます。
ソースコード・マニュアル・質問: https://github.com/TakuyaNatsui/AxiCavity-FEM

起動
----
  AxiCavity-FEM.exe をダブルクリックすると GUI が起動します。

  初回は Windows の「PC が保護されました」（SmartScreen）が出ることがあります（コード署名をしていないため）。
  「詳細情報」→「実行」で起動できます。

  このフォルダの中身はまとめて使います。AxiCavity-FEM.exe だけを別の場所へ移さないでください
  （ショートカットを作るのは構いません）。

  使い方: USER_MANUAL.md（日本語。テキストエディタやブラウザで読めます）
  サンプル: samples\ の .gmshproj（GUI の「ファイル」→「取り込み」→「形状の取り込み」で開く）

コマンドライン（コマンドプロンプト / PowerShell で、このフォルダから）
--------------------------------------------------------------------
  AxiCavity-FEM.exe run samples\cylinder100mm.gmshproj          メッシュ → 解析 → 後処理をまとめて
  AxiCavity-FEM.exe run cavity.axiproj --set a=45 --json out.json   パラメータを変えて解析
  AxiCavity-FEM.exe solve --type tm0 -m model.msh --elem-order 2 --num-modes 6 -o result.h5
  AxiCavity-FEM.exe post  --type tm0 -i result.h5 --cond 5.8e7
  AxiCavity-FEM.exe report --type tm0 -i result.h5 -o report
  AxiCavity-FEM.exe version        版と同梱物（PARDISO・メッシャ・3D 表示）の確認
  AxiCavity-FEM.exe selftest       動作確認（円筒空洞を解き、TM010 を解析解と比べる）

  pip 版の axicavity-fem / axicavity-fem-run と同じサブコマンド・オプションです（USER_MANUAL.md §10・§11）。

動かないとき
------------
  - 「VCRUNTIME140.dll が見つかりません」などと出る:
    Microsoft Visual C++ 2015-2022 再頒布可能パッケージ（x64）を入れてください。
    https://aka.ms/vs/17/release/vc_redist.x64.exe
  - ウイルス対策ソフトが誤って検出することがあります（Nuitka で作った署名なしの EXE のため）。
  - 報告は GitHub の Issues へ。「AxiCavity-FEM.exe selftest」の出力を添えてください。

アンインストール
----------------
  このフォルダを削除します。設定（表示言語・最近使ったファイル・ウィンドウの配置）はレジストリの
  HKEY_CURRENT_USER\Software\AxiCavity-FEM にあります（不要なら削除してください）。

フォルダの構成
--------------
  AxiCavity-FEM.exe           本体（GUI とコマンドライン）
  Library\bin\                Intel MKL（PARDISO。大きなメッシュの固有値計算を速くする）
  mesher\                     メッシュ生成の別プログラム（gmsh を含み GPL で配布。ソース同梱）
  samples\                    サンプル形状
  docs\images\                USER_MANUAL.md の図
  USER_MANUAL.md              ユーザーマニュアル（日本語）
  PHYSICS_AND_CONVENTIONS.md  物理量と規約の説明
  LICENSE                     AxiCavity-FEM のライセンス（MIT）
  THIRD_PARTY_NOTICES.txt     同梱している第三者ソフトウェアのライセンス
  （そのほかの .dll / .pyd / フォルダは Python と依存ライブラリ）

ライセンス
----------
  AxiCavity-FEM 本体は MIT License です。同梱の第三者ソフトウェアはそれぞれのライセンスに従います
  （Qt / PySide6・planegcs: LGPL、Intel MKL / TBB: Intel Simplified Software License など。THIRD_PARTY_NOTICES.txt）。
  mesher\ は gmsh を含む GPL の別プログラムで、本体は子プロセスとして呼ぶだけです（mesher\README.txt）。


------------------------------------------------------------------------------
English
------------------------------------------------------------------------------

AxiCavity-FEM computes the resonant modes of axisymmetric RF cavities (TM0 and higher-order modes, standing and
traveling waves) with the 2-D finite-element method: parametric sketching, meshing, solving and result display (incl. 3D).
Source code, manual and issues: https://github.com/TakuyaNatsui/AxiCavity-FEM

Start: double-click AxiCavity-FEM.exe. If Windows SmartScreen says "Windows protected your PC" (the executable is
not code-signed), choose "More info" -> "Run anyway". Keep the whole folder together.

Command line (from this folder):
  AxiCavity-FEM.exe run samples\cylinder100mm.gmshproj
  AxiCavity-FEM.exe solve --type tm0 -m model.msh --elem-order 2 --num-modes 6 -o result.h5
  AxiCavity-FEM.exe version  /  AxiCavity-FEM.exe selftest
The subcommands and options are those of the pip version (axicavity-fem / axicavity-fem-run).

If it does not start ("VCRUNTIME140.dll was not found"), install the Microsoft Visual C++ 2015-2022 Redistributable
(x64): https://aka.ms/vs/17/release/vc_redist.x64.exe

Uninstall: delete this folder. Settings are stored under HKEY_CURRENT_USER\Software\AxiCavity-FEM.

Licenses: AxiCavity-FEM is MIT licensed. Bundled third-party software keeps its own license (Qt / PySide6 and
planegcs: LGPL; Intel MKL / TBB: Intel Simplified Software License; see THIRD_PARTY_NOTICES.txt). The mesher folder is
a separate GPL program containing Gmsh, shipped with its source; the main program only runs it as a child process.
