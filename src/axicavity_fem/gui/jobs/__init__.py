"""子プロセスでのジョブ実行（メッシュ・解析・post・レポート・書き出し）.

- :mod:`.runner`: 子プロセスの入口（``python -m axicavity_fem.gui.jobs.runner <kind> <job_dir> --events``）とイベント形式
- :mod:`.pipeline`: 各ジョブの本体（純 Python + ver2.3 のコア。Qt 非依存。子プロセスで動く）
- :mod:`.session`: 親側の QProcess 管理（同時に 1 本、キャンセル・強制停止）
- :mod:`.mesh_folder` / :mod:`.mesh_controller`: メッシュタブの生成物の置き場所と状態

gmsh は非メインスレッドで initialize できず、C レベルで fd 1 に書くので、メッシュ生成・解析はすべて子プロセスで行う
（GUI プロセスでは gmsh を initialize しない）。
"""
