"""``.axiproj`` をコマンドラインと Python から扱う（GUI を開かずに: パラメータ変更 → メッシュ → solve → post）.

Python::

    from axicavity_fem.gui.batch import Project
    p = Project.open("cav.axiproj")
    p.set_params(a=45, L="(c/f/3)*1e3")      # 式も可。式付きの点と拘束を解き直す（矛盾なら ValueError）
    p.analysis.numModes = 4                   # 設定は文書の dataclass をそのまま
    r = p.run()                               # メッシュ → solve → post（同じプロセス）
    r.frequencies()                           # np.ndarray [GHz]（n = 先頭 / 位相 = 先頭）
    r.modes                                   # [{"n", "phase", "index", "f_GHz", "Q", "R_over_Q", ...}, ...]
    p.save_as("cav_a45.axiproj")              # 変えた状態を別名で保存（元は書き換えない）

コマンドライン（``axicavity-fem-run`` = ``python -m axicavity_fem.gui.batch``）::

    axicavity-fem-run cav.axiproj --set a=45 --set t=6 --modes 4 --json out.json
    axicavity-fem-run cav.axiproj --info
    axicavity-fem-run cav.axiproj --type hom --az-order 0 1 --wave traveling --phases 120

結果は既定で ``<名前>.axiproj.data/batch/<日時>-<種類>/``（GUI の結果履歴には出ない。``register=True`` / ``--register`` で
``results/NNNN-<種類>/`` に登録して GUI のツリーに出す）。実行したコマンドは ``command.log`` に残る。
計算は :mod:`~axicavity_fem.gui.jobs.analysis_pipeline`（GUI の子プロセスと同じ経路。argv は wx 版と同じ文字列）。
入力に ``.gmshproj`` / Superfish ``.af`` も取れる（プロジェクトの無い形状。結果は ``<stem>_batch/`` に）。
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .app.store import DocumentStore
from .core.convert import ConvertError, check_geometry, to_multi_region
from .core.document import AxiDocument, AnalysisSettings, MeshSettings, PostSettings, ReportSettings
from .core.expressions import resolve_params
from .core.hashes import geometry_hash, mesh_hash, model_hash
from .core.legacy import GMSHPROJ_SUFFIX, SUPERFISH_SUFFIX, load_gmshproj
from .io.axiproj import (
    PROJECT_SUFFIX,
    ProjectFile,
    append_command_log,
    data_dir_of,
    next_result_id,
    read_project,
    result_kind,
    write_project,
    write_result_meta,
)
from .io.superfish import load_superfish
from .jobs.analysis_pipeline import run_job
from .jobs.commands import build_commands, command_line
from .jobs.pipeline import GEOMETRY_FILE, JOB_FILE, MESH_FILE

LogFn = Callable[[str], None]
GENERATOR = "axicavity-fem batch"


@dataclass
class RunResult:
    """1 回の解析の結果（``result.json`` の summary と同じ内容）."""

    dir: Path
    kind: str
    summary: dict
    files: dict
    elapsed: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    registered: bool = False

    @property
    def modes(self) -> list[dict]:
        return list(self.summary.get("modes") or [])

    @property
    def n_orders(self) -> list[int]:
        return [int(n) for n in self.summary.get("nOrders") or []]

    @property
    def phases(self) -> list[float]:
        return [float(p) for p in self.summary.get("phases") or []]

    def mode_table(self, n: Optional[int] = None, phase: Optional[float] = None) -> list[dict]:
        """指定の n / 位相のモード（省略は先頭）. 定在波は phase=None."""
        modes = self.modes
        if not modes:
            return []
        if n is None:
            n = self.n_orders[0] if self.n_orders else modes[0]["n"]
        if phase is None and self.phases:
            phase = self.phases[0]
        return [m for m in modes if int(m["n"]) == int(n)
                and ((m.get("phase") is None and phase is None)
                     or (m.get("phase") is not None and phase is not None and abs(float(m["phase"]) - float(phase)) < 1e-9))]

    def frequencies(self, n: Optional[int] = None, phase: Optional[float] = None) -> np.ndarray:
        """共振周波数 [GHz]（モード番号順）."""
        return np.asarray([m["f_GHz"] for m in self.mode_table(n, phase)], dtype=float)

    def values(self, key: str, n: Optional[int] = None, phase: Optional[float] = None) -> np.ndarray:
        """post の量（"Q", "R_over_Q", "V_eff", ...）をモード番号順に（無ければ nan）."""
        return np.asarray([m.get(key, np.nan) if m.get(key) is not None else np.nan
                           for m in self.mode_table(n, phase)], dtype=float)

    def file(self, key: str) -> Optional[Path]:
        name = self.files.get(key)
        return self.dir / name if name else None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["dir"] = str(self.dir)
        return d


class Project:
    """``.axiproj``（または形状ファイル）を読み、パラメータ・設定を変え、解析を実行する."""

    def __init__(self, document: AxiDocument, path: Optional[Path] = None, project_file: Optional[ProjectFile] = None,
                 source: Optional[Path] = None):
        self.path = Path(path) if path is not None else None       # .axiproj のパス（形状ファイルだけなら None）
        self.source = Path(source) if source is not None else self.path
        self.project_file = project_file or ProjectFile(document=document, generator=GENERATOR)
        self.store = DocumentStore()
        self.store.load_document(document)
        self.warnings: list[str] = []

    # ---- 読み込み ---------------------------------------------------------

    @classmethod
    def open(cls, path: str | Path) -> "Project":
        """``.axiproj`` / ``.gmshproj`` / ``.af`` を開く."""
        path = Path(path)
        name = path.name.lower()
        if name.endswith(PROJECT_SUFFIX):
            pf = read_project(path)
            return cls(pf.document, path=path, project_file=pf)
        if name.endswith(GMSHPROJ_SUFFIX):
            doc, warnings = load_gmshproj(path)
        elif name.endswith(SUPERFISH_SUFFIX):
            doc, warnings = load_superfish(path)
        else:
            raise ValueError(f"対応していないファイルです: {path}（.axiproj / .gmshproj / .af）")
        project = cls(doc, path=None, source=path)
        project.warnings = list(warnings)
        return project

    @classmethod
    def from_document(cls, document: AxiDocument) -> "Project":
        """文書（``core.document.AxiDocument``）から（保存先は ``save_as`` で決める）."""
        return cls(document)

    # ---- 文書・設定 -------------------------------------------------------

    @property
    def document(self) -> AxiDocument:
        return self.store.document

    @property
    def analysis(self) -> AnalysisSettings:
        return self.document.analysis

    @property
    def mesh(self) -> MeshSettings:
        return self.document.mesh

    @property
    def post(self) -> PostSettings:
        return self.document.post

    @property
    def report(self) -> ReportSettings:
        return self.document.report

    @property
    def params(self) -> dict[str, str]:
        """パラメータ名 → 式."""
        return {p.name: p.expression for p in self.document.params}

    @property
    def param_values(self) -> dict[str, float]:
        """パラメータ名 → 評価した値."""
        return dict(self.store.params())

    def set_params(self, **values) -> None:
        """パラメータの式を変える（数値でも式でも）. 式付きの点と拘束を解き直す.

        名前が無ければ KeyError、式が評価できない・拘束が解けなければ ValueError。どちらのときもこの呼び出しで
        変えたパラメータはすべて元に戻る（文書は呼ぶ前の状態のまま）。
        """
        by_name = {p.name: p.id for p in self.document.params}
        missing = [name for name in values if name not in by_name]
        if missing:
            raise KeyError(f"パラメータ {', '.join(map(repr, missing))} がありません（{', '.join(by_name) or 'なし'}）")
        applied = 0
        try:
            for name, value in values.items():
                expression = value if isinstance(value, str) else repr(float(value))
                self.store.update_param(by_name[name], expression=expression)     # Undo 履歴に 1 つ積む
                applied += 1
                self._raise_if_invalid(name)
        except Exception:
            self._rollback(applied)
            raise

    def add_param(self, name: str, value) -> None:
        """パラメータを表の末尾に足す（名前が既にあれば ValueError。失敗したら足さない）."""
        if name in self.params:
            raise ValueError(f"パラメータ {name!r} は既にあります（set_params で変えてください）")
        pid = self.store.add_param()
        try:
            self.store.update_param(pid, name=name, expression=value if isinstance(value, str) else repr(float(value)))
            self._raise_if_invalid(name)
        except Exception:
            self._rollback(2)
            raise

    def _rollback(self, steps: int) -> None:
        for _ in range(steps):
            if not self.store.undo():
                break

    def _raise_if_invalid(self, name: str) -> None:
        try:
            resolve_params(self.document.params)            # パラメータ表そのもの（上の行から順に）
        except Exception as exc:  # noqa: BLE001 — 評価器の例外（未定義の名前・構文）
            raise ValueError(f"パラメータ {name} の変更で表が評価できません: {exc}") from exc
        errors = list(self.store.expression_errors)
        info = self.store.solve_info
        if errors:
            raise ValueError(f"パラメータ {name} の変更で式が評価できません: {'; '.join(errors)}")
        if info is not None and info.status in ("conflict", "failed", "invalid"):
            raise ValueError(f"パラメータ {name} の変更で拘束が解けません: {info.status} {info.error or ''}")

    def set_analysis(self, **patch) -> None:
        self.store.set_analysis(**patch)

    def set_mesh(self, **patch) -> None:
        self.store.set_mesh(**patch)

    def set_post(self, **patch) -> None:
        self.store.set_post(**patch)

    def check(self) -> tuple[list[str], list[str]]:
        issues = check_geometry(self.document)
        return ([i.message for i in issues if i.level == "error"], [i.message for i in issues if i.level != "error"])

    # ---- 保存 -------------------------------------------------------------

    def save(self) -> Path:
        if self.path is None:
            raise ValueError("保存先がありません（save_as を使ってください）")
        return self.save_as(self.path)

    def save_as(self, path: str | Path) -> Path:
        pf = ProjectFile(document=self.document, active_result=self.project_file.active_result,
                         ui=dict(self.project_file.ui), generator=GENERATOR)
        written = write_project(path, pf)
        self.path = written
        self.project_file = pf
        return written

    # ---- 出力先 ------------------------------------------------------------

    @property
    def data_dir(self) -> Optional[Path]:
        return data_dir_of(self.path) if self.path is not None else None

    def default_out_dir(self, kind: str) -> Path:
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        if self.data_dir is not None:
            base = self.data_dir / "batch"
        elif self.source is not None:
            base = self.source.with_name(self.source.stem + "_batch")
        else:
            base = Path.cwd() / "axicavity_batch"
        folder = base / f"{stamp}-{kind}"
        n = 1
        while folder.exists():
            n += 1
            folder = base / f"{stamp}-{kind}-{n}"
        return folder

    # ---- 実行 -------------------------------------------------------------

    def run(self, out_dir: Optional[str | Path] = None, post: Optional[bool] = None, register: bool = False,
            log: Optional[LogFn] = None, mesh_file: Optional[str | Path] = None) -> RunResult:
        """メッシュ → solve →（post）を実行する.

        ``out_dir``: 出力フォルダ（省略で ``batch/<日時>-<種類>/``）。``post``: None なら文書の設定。
        ``register``: プロジェクトの結果履歴（``results/NNNN-<種類>/``）に登録する（保存済みのプロジェクトだけ）。
        ``mesh_file``: 既存の ``.msh`` を使う（省略で geometry.json から gmsh で作る）。
        """
        log = log or (lambda _m: None)
        doc = self.document
        errors, _warnings = self.check()
        if errors:
            raise ValueError("実行できません: " + "; ".join(errors))
        try:
            converted = to_multi_region(doc, strict=True)
        except ConvertError as exc:
            raise ValueError(str(exc)) from exc
        run_post = doc.analysis.runPost
        if post is not None:                              # この回だけ（文書の設定は変えない）
            doc.analysis.runPost = bool(post)
        try:
            cmds = build_commands(doc)
        finally:
            doc.analysis.runPost = run_post
        kind = cmds["kind"]
        if register:
            if self.data_dir is None:
                raise ValueError("register=True は保存済みのプロジェクトだけです（save_as を先に）")
            rid = next_result_id(self.data_dir, kind)
            folder = self.data_dir / "results" / rid
        elif out_dir is not None:
            folder = Path(out_dir)
        else:
            folder = self.default_out_dir(kind)
        folder.mkdir(parents=True, exist_ok=True)
        converted.geom.save_json(folder / GEOMETRY_FILE)
        job = {"kind": kind, "solverType": doc.analysis.type, "wave": doc.analysis.wave,
               "meshOrder": int(doc.mesh.order), "units": doc.meta.units,
               "argv": {"solve": cmds["solve"], "post": cmds["post"]},
               "files": {"mesh": MESH_FILE, "raw": cmds["raw"], "processed": cmds["processed"]},
               "createdAt": _dt.datetime.now().isoformat(timespec="seconds"), "generator": GENERATOR,
               "params": self.params}
        if mesh_file is not None:
            import shutil
            shutil.copy2(mesh_file, folder / MESH_FILE)
            materials = Path(mesh_file).with_suffix(".materials.json")
            if materials.exists():
                shutil.copy2(materials, folder / "model.materials.json")
            job["meshReused"] = str(mesh_file)
        (folder / JOB_FILE).write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
        label = folder.name if register else f"batch/{folder.name}"
        if self.data_dir is not None:
            for argv in [cmds["solve"]] + ([cmds["post"]] if cmds["post"] else []):
                append_command_log(self.data_dir, f"[{label}] {command_line(argv)}")
        if register:
            write_result_meta(folder, number=int(folder.name.split("-", 1)[0]), kind=kind,
                              createdAt=_dt.datetime.now().isoformat(timespec="seconds"), status="running",
                              geometryHash=geometry_hash(doc), meshHash=mesh_hash(doc), modelHash=model_hash(doc),
                              generator=GENERATOR, label="")
        try:
            result = run_job("analysis", folder, progress=lambda stage, msg: log(f"[{stage}] {msg}"), log=log)
        except Exception as exc:
            if register:
                write_result_meta(folder, status="error", error=f"{type(exc).__name__}: {exc}",
                                  finishedAt=_dt.datetime.now().isoformat(timespec="seconds"))
            raise
        files = dict(result.get("files") or {})
        summary = result.get("summary") or {}
        if register:
            write_result_meta(folder, status="done", finishedAt=_dt.datetime.now().isoformat(timespec="seconds"),
                              summary=summary, files=files)
        return RunResult(dir=folder, kind=kind, summary=summary, files=files, elapsed=result.get("elapsedS") or {},
                         warnings=list(result.get("warnings") or []), registered=register)


def run_project(path: str | Path, params: Optional[dict] = None, out_dir: Optional[str | Path] = None,
                register: bool = False, log: Optional[LogFn] = None, **settings) -> RunResult:
    """1 行版: ``run_project("cav.axiproj", {"a": 45}, numModes=4)``（settings は analysis の項目）."""
    project = Project.open(path)
    if params:
        project.set_params(**params)
    if settings:
        project.set_analysis(**settings)
    return project.run(out_dir=out_dir, register=register, log=log)


# ---------------------------------------------------------------------------
# コマンドライン
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="axicavity-fem-run",
                                description=".axiproj を読み、パラメータを変えて メッシュ → solve → post を実行する")
    p.add_argument("project", help=".axiproj（または .gmshproj / .af）")
    p.add_argument("--set", dest="sets", action="append", default=[], metavar="NAME=EXPR",
                   help="パラメータの式（数値または式。複数可）")
    p.add_argument("--type", choices=["tm0", "hom"], help="解析の種類")
    p.add_argument("--wave", choices=["standing", "traveling"], help="定在波 / 進行波")
    p.add_argument("--phases", help='進行波の位相 [deg]（"120" / "60,90" / "0:180:20"）')
    p.add_argument("--az-order", dest="az_order", nargs="+", type=int, help="HOM の方位角次数")
    p.add_argument("--modes", type=int, help="モード数")
    p.add_argument("--target-freq", dest="target_freq", type=float, help="予測周波数 [GHz]")
    p.add_argument("--no-post", dest="no_post", action="store_true", help="post を実行しない")
    p.add_argument("--cond", type=float, help="壁の導電率 [S/m]")
    p.add_argument("--beta", type=float, help="β = v/c（TM0）")
    p.add_argument("--lc", type=float, help="メッシュサイズ（座標の単位）")
    p.add_argument("--order", type=int, choices=[1, 2], help="メッシュ次数 = 要素次数")
    p.add_argument("--out", help="出力フォルダ（省略で <名前>.axiproj.data/batch/<日時>-<種類>/）")
    p.add_argument("--register", action="store_true", help="GUI の結果履歴（results/NNNN-<種類>/）に登録する")
    p.add_argument("--json", dest="json_path", help="結果の要約（summary）を JSON に書く")
    p.add_argument("--save-as", dest="save_as", help="変更した状態を別名の .axiproj に保存する")
    p.add_argument("--info", action="store_true", help="パラメータと設定を表示して終わる")
    p.add_argument("--no-run", dest="no_run", action="store_true", help="解析は実行しない（--save-as と）")
    p.add_argument("--quiet", "-q", action="store_true", help="進捗を出さない")
    return p


def _parse_set(text: str) -> tuple[str, str]:
    if "=" not in text:
        raise ValueError(f"--set は NAME=EXPR の形で: {text!r}")
    name, expr = text.split("=", 1)
    return name.strip(), expr.strip()


def _print_info(project: Project, out=None) -> None:
    out = out or sys.stdout
    doc = project.document
    print(f"project : {project.path or project.source}", file=out)
    print(f"units   : {doc.meta.units}", file=out)
    values = project.param_values
    for name, expr in project.params.items():
        value = values.get(name)
        shown = f"{value:g}" if isinstance(value, (int, float)) else "?"
        print(f"  {name:12s} = {expr:20s} → {shown}", file=out)
    a = doc.analysis
    print(f"mesh    : lc = {doc.mesh.sizeExpr or doc.mesh.size} {doc.meta.units}, order {doc.mesh.order}", file=out)
    print(f"analysis: {a.type} {a.wave}" + (f" phases {a.phases}" if a.wave == "traveling" else "")
          + (f" n = {a.azOrders}" if a.type == "hom" else "") + f", {a.numModes} modes"
          + (f", target {a.targetFreqGHz} GHz" if a.targetFreqGHz else "") + f", post {'on' if a.runPost else 'off'}",
          file=out)


def _print_result(result: RunResult, out=None) -> None:
    out = out or sys.stdout
    print(f"output  : {result.dir}", file=out)
    for n in result.n_orders or [0]:
        phases = result.phases or [None]
        for phase in phases:
            table = result.mode_table(n, phase)
            if not table:
                continue
            head = f"[n={n}] " + ("standing" if phase is None else f"traveling phase={phase:g}deg")
            print(head, file=out)
            for m in table:
                line = f"  mode {m['index']:2d}: f = {m['f_GHz']:.6f} GHz"
                if m.get("Q") is not None:
                    line += f"  Q = {m['Q']:.4e}"
                if m.get("R_over_Q") is not None:
                    line += f"  R/Q = {m['R_over_Q']:.4e}"
                print(line, file=out)


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        project = Project.open(args.project)
        for line in project.warnings:
            print(f"warning: {line}", file=sys.stderr)
        if args.sets:
            project.set_params(**dict(_parse_set(s) for s in args.sets))
        patch = {}
        if args.type:
            patch["type"] = args.type
        if args.wave:
            patch["wave"] = args.wave
        if args.phases:
            patch["phases"] = args.phases
        if args.az_order:
            patch["azOrders"] = " ".join(str(n) for n in args.az_order)
        if args.modes:
            patch["numModes"] = int(args.modes)
        if args.target_freq is not None:
            patch["targetFreqGHz"] = float(args.target_freq)
        if args.no_post:
            patch["runPost"] = False
        if patch:
            project.set_analysis(**patch)
        if args.cond is not None or args.beta is not None:
            project.set_post(**{k: v for k, v in (("cond", args.cond), ("beta", args.beta)) if v is not None})
        if args.lc is not None or args.order is not None:
            project.set_mesh(**{k: v for k, v in (("size", args.lc), ("order", args.order)) if v is not None})
        if args.info:
            _print_info(project)
            return 0
        if args.save_as:
            written = project.save_as(args.save_as)
            print(f"saved   : {written}")
        if args.no_run:
            return 0
        # 進捗は「今の」標準出力に書く（解析中は sys.stdout がログへ付け替えられるので、print のままだと
        # ログ → print → ログ … と無限に再帰する）
        console = sys.stdout

        def log(message: str) -> None:
            if not args.quiet and console is not None:
                print(message, file=console, flush=True)

        result = project.run(out_dir=args.out, register=args.register, log=log)
    except (ValueError, KeyError, FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    _print_result(result)
    if args.json_path:
        Path(args.json_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_path).write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"json    : {args.json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
