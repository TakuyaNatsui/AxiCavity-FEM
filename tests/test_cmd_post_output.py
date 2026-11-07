"""``post`` は計算と追記が済むまで出力に触らない: 失敗しても前の post の値が残る（3.0.0 までは最初に入力を出力へ写していた）."""

from __future__ import annotations

import os
import stat

import pytest

pytest.importorskip("gmsh")
h5py = pytest.importorskip("h5py")

from axicavity_fem.cli import cmd_post  # noqa: E402
from axicavity_fem.cli.main import cli_main  # noqa: E402
from axicavity_fem.shared.hdf5_io import read_results  # noqa: E402


@pytest.fixture
def files(tmp_path, sample_mesh):
    raw, processed = tmp_path / "raw.h5", tmp_path / "processed.h5"
    assert cli_main(["solve", "--type", "tm0", "-m", str(sample_mesh("cylinder100mm", 2)), "--elem-order", "2",
                     "--num-modes", "2", "-o", str(raw)]) == 0
    assert cli_main(["post", "--type", "tm0", "-i", str(raw), "-o", str(processed)]) == 0
    return raw, processed


def _p_loss(path):
    return read_results(path)["post_process"][0]["standing"][0]["P_loss"]


def test_failed_computation_keeps_the_output(files, monkeypatch):
    raw, processed = files
    before = processed.read_bytes()

    def fail(*args):
        raise RuntimeError("post failed")

    monkeypatch.setattr(cmd_post, "_post_tm0", fail)
    with pytest.raises(RuntimeError):
        cli_main(["post", "--type", "tm0", "-i", str(raw), "-o", str(processed)])
    assert processed.read_bytes() == before


def test_input_that_is_not_hdf5_is_refused_and_keeps_the_output(files, capsys):
    raw, processed = files
    before = processed.read_bytes()
    raw.write_bytes(b"not an hdf5 file")
    capsys.readouterr()
    assert cli_main(["post", "--type", "tm0", "-i", str(raw), "-o", str(processed)]) == 1
    assert "HDF5" in capsys.readouterr().out
    assert processed.read_bytes() == before


@pytest.mark.parametrize("in_place", [False, True], ids=["output", "in-place"])
def test_failed_append_keeps_the_file_and_leaves_no_temporary(files, tmp_path, monkeypatch, in_place):
    raw, processed = files
    target = raw if in_place else processed
    before = target.read_bytes()

    def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr(cmd_post, "append_post_process", fail)
    assert cli_main(["post", "--type", "tm0", "-i", str(raw), "-o", str(target)]) == 1
    assert target.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))


def test_output_held_by_another_program_is_not_destroyed(files, tmp_path, monkeypatch, capsys):
    """Windows では開かれたファイルを置き換えられない（ここでは os.replace の失敗で表す）. 値は失われない."""
    raw, processed = files
    before = processed.read_bytes()

    def busy(src, dst):
        raise PermissionError(13, "in use", dst)

    monkeypatch.setattr(cmd_post.os, "replace", busy)
    capsys.readouterr()
    assert cli_main(["post", "--type", "tm0", "-i", str(raw), "-o", str(processed)]) == 1
    assert "前の内容は残っています" in capsys.readouterr().out
    assert processed.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))


def test_read_only_input_gives_a_writable_output(files, tmp_path):
    """入力の属性を写さない（3.0.0 は読み取り専用が出力へ写り、post をやり直せなくなった）."""
    raw, _ = files
    output = tmp_path / "from_read_only.h5"
    os.chmod(raw, stat.S_IREAD)
    try:
        assert cli_main(["post", "--type", "tm0", "-i", str(raw), "-o", str(output)]) == 0
        assert _p_loss(output) > 0.0
        assert os.access(output, os.W_OK)
        assert cli_main(["post", "--type", "tm0", "-i", str(raw), "-o", str(output)]) == 0
    finally:
        os.chmod(raw, stat.S_IWRITE | stat.S_IREAD)


def test_read_only_output_is_refused_before_computing(files, monkeypatch, capsys):
    raw, processed = files

    def must_not_run(*args):
        raise AssertionError("計算まで進んだ")

    monkeypatch.setattr(cmd_post, "_post_tm0", must_not_run)
    os.chmod(processed, stat.S_IREAD)
    try:
        capsys.readouterr()
        assert cli_main(["post", "--type", "tm0", "-i", str(raw), "-o", str(processed)]) == 1
        assert "読み取り専用" in capsys.readouterr().out
    finally:
        os.chmod(processed, stat.S_IWRITE | stat.S_IREAD)


@pytest.mark.parametrize("name, message", [("no/such/x.h5", "フォルダがありません"), ("o.txt", ".txt"), (".", "フォルダです")],
                         ids=["missing-folder", "txt", "folder"])
def test_bad_output_path_is_refused_before_computing(files, tmp_path, monkeypatch, capsys, name, message):
    raw, _ = files

    def must_not_run(*args):
        raise AssertionError("計算まで進んだ")

    monkeypatch.setattr(cmd_post, "_post_tm0", must_not_run)
    capsys.readouterr()
    assert cli_main(["post", "--type", "tm0", "-i", str(raw), "-o", str(tmp_path / name)]) == 1
    assert message in capsys.readouterr().out
    assert not (tmp_path / "o.txt").exists()


def test_output_keeps_new_values_when_only_the_text_summary_fails(files, monkeypatch):
    """.txt は表計算ソフトが開いていることがある. h5 はもう新しい値なので失敗にしない."""
    raw, processed = files
    p_loss = _p_loss(processed)

    def fail(*args):
        raise PermissionError(13, "in use", "processed.txt")

    monkeypatch.setattr("axicavity_fem.cli.summary_txt.write_post_summary", fail)
    with pytest.warns(UserWarning, match="テキストサマリ"):
        assert cli_main(["post", "--type", "tm0", "-i", str(raw), "-o", str(processed), "--cond", str(5.8e7 / 4.0)]) == 0
    assert _p_loss(processed) == pytest.approx(2.0 * p_loss, rel=1e-9)


def test_post_again_after_a_new_solve_replaces_the_stale_solution(files, sample_mesh):
    """既存の出力へも必ず入力を写す: 解き直した固有解が出力に入る."""
    raw, processed = files
    assert cli_main(["solve", "--type", "tm0", "-m", str(sample_mesh("cylinder100mm", 2)), "--elem-order", "2",
                     "--num-modes", "4", "-o", str(raw)]) == 0
    assert cli_main(["post", "--type", "tm0", "-i", str(raw), "-o", str(processed)]) == 0
    results = read_results(processed)
    assert len(results["results_by_n"][0]["standing"]["frequencies"]) == 4
    assert len(results["post_process"][0]["standing"]) == 4


def test_same_file_spelled_differently_runs_in_place(files):
    raw, _ = files
    spelled = os.path.join(os.path.dirname(raw), ".", os.path.basename(raw))
    assert cli_main(["post", "--type", "tm0", "-i", str(raw), "-o", spelled]) == 0
    assert _p_loss(raw) > 0.0


def test_version1_file_is_refused_with_a_clear_message(files, tmp_path, capsys):
    _, processed = files
    before = processed.read_bytes()
    v1 = tmp_path / "v1.h5"
    with h5py.File(v1, "w") as f:                    # ver1 形式は schema_version を持たない
        f["mesh/nodes"] = [[0.0, 0.0]]
    capsys.readouterr()
    assert cli_main(["post", "--type", "tm0", "-i", str(v1), "-o", str(processed)]) == 1
    assert "ver1" in capsys.readouterr().out
    assert processed.read_bytes() == before


def test_success_replaces_the_values_and_leaves_the_input(files):
    raw, processed = files
    before = raw.read_bytes()
    p_loss = _p_loss(processed)
    assert cli_main(["post", "--type", "tm0", "-i", str(raw), "-o", str(processed), "--cond", str(5.8e7 / 4.0)]) == 0
    assert _p_loss(processed) == pytest.approx(2.0 * p_loss, rel=1e-9)      # R_s ∝ 1/√σ
    assert raw.read_bytes() == before
