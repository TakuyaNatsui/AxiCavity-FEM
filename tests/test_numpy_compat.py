"""NumPy の新しい版で削除された API に依存しないことの確認（np.trapz と np.in1d は NumPy 2.4 で削除）."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

SCRIPT = """
import sys
import numpy as np
for name in ("trapz", "in1d"):
    if hasattr(np, name):
        delattr(np, name)
        np.__all__.remove(name)      # SciPy が from numpy import * をするので、ここからも除く
from axicavity_fem.cli.main import cli_main
msh, raw, processed = sys.argv[1:]
assert cli_main(["solve", "--type", "tm0", "-m", msh, "--elem-order", "2", "--num-modes", "1", "-o", raw]) == 0
assert cli_main(["post", "--type", "tm0", "-i", raw, "-o", processed]) == 0
"""


@pytest.mark.skipif(not hasattr(np, "trapezoid"), reason="np.trapezoid は NumPy 2.0 から")
def test_tm0_solve_and_post_without_removed_numpy_functions(sample_mesh, tmp_path):
    msh = sample_mesh("cylinder100mm", 2)
    done = subprocess.run([sys.executable, "-c", SCRIPT, str(msh), str(tmp_path / "raw.h5"), str(tmp_path / "post.h5")],
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert done.returncode == 0, done.stdout + done.stderr
    assert (tmp_path / "post.h5").exists()
