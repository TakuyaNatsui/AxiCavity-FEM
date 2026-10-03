"""MainWindow を実ウィンドウで開く確認（_main_window_check.py を別プロセス・既定のプラットフォームで実行）.

数秒ウィンドウが開く。表示の無い環境（CI など）では AXICAVITY_SKIP_REAL_WINDOW=1 で飛ばす。
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("planegcs")
CHECK = Path(__file__).with_name("_main_window_check.py")


@pytest.mark.skipif(os.environ.get("AXICAVITY_SKIP_REAL_WINDOW") == "1", reason="実ウィンドウを開かない設定")
@pytest.mark.skipif(sys.platform.startswith("linux") and not os.environ.get("DISPLAY"), reason="表示が無い")
def test_main_window_in_a_real_window(tmp_path):
    env = dict(os.environ)
    env.pop("QT_QPA_PLATFORM", None)                 # 実ウィンドウ
    env["PYTHONUTF8"] = "1"
    result = subprocess.run([sys.executable, str(CHECK), str(tmp_path)], capture_output=True,
                            text=True, encoding="utf-8", errors="replace", env=env, timeout=120)
    assert result.returncode == 0 and "OK" in result.stdout, \
        f"stdout:\n{result.stdout[-3000:]}\nstderr:\n{result.stderr[-5000:]}"
