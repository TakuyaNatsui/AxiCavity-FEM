"""2D スケッチのコア（EM-CAD-py ``emcad/core/sketch`` からの転用。純 Python）.

- ``geometry``: ベクトル・円弧・多角形のユーティリティ（無変更）
- ``model``: エンティティの追加・削除・統合・当たり判定・2D フィレット（``add_point`` に座標式を追加）
- ``profiles``: 共有点でつながる曲線から閉領域（外周＋穴）を検出（無変更）
- ``snap``: スナップとグリッド間隔（無変更）
- ``edit_ops``: ver2.3 の Multi-Region Editor にあった編集（点挿入・点削除再接続・線⇄円弧・端点結合）
"""
