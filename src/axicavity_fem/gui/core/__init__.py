"""ver3 GUI の純 Python コア（Qt / gmsh / matplotlib 非依存）.

- ``document``: ドキュメントモデル（スケッチ・領域・境界条件・設定）
- ``serialize``: ドキュメント ⇄ JSON
- ``expressions``: 式評価（ver2.3 ``shared/expression_eval`` のアダプタ）
- ``sketch``: 2D スケッチの幾何・編集・閉領域検出・スナップ（EM-CAD-py から転用）
- ``convert``: スケッチ ⇄ ``MultiRegionGeometry``（M2）
- ``hashes``: 形状・メッシュ・モデルのハッシュ
"""
