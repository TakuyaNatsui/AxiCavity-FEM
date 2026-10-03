# パラメータスキャンの例（`.axiproj` を Python から解析する）

GUI で作ったプロジェクト（`.axiproj`）のパラメータを Python から変えて解析し、周波数を読みます。
パラメータを変えるたびに、式付きの点と寸法拘束がスケッチの拘束ソルバで解き直されます（GUI と同じ）。

```bash
python scan_pillbox.py                          # ピルボックスを作り、半径 a を振って TM010 を解析解と比べる
python scan_pillbox.py cav.axiproj a 40 42 44   # 手持ちのプロジェクトのパラメータ a を振る
```

最小の形:

```python
from axicavity_fem.gui.batch import Project

p = Project.open("cav.axiproj")
for a in (40, 42, 44):
    p.set_params(a=a)
    r = p.run(post=False)          # メッシュ生成 → 固有値解析（結果は <名前>.axiproj.data/batch/ に）
    print(a, r.frequencies()[0])   # GHz
```

コマンドラインなら `axicavity-fem-run cav.axiproj --set a=42 --json out.json`（Windows 版は `AxiCavity-FEM.exe run …`）。
API と CLI の詳細は [USER_MANUAL.md](../../USER_MANUAL.md) §11。必要なもの: `pip install -e ".[gui]"`（PySide6・planegcs）。
