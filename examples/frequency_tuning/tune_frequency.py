"""周波数自動調整の例: エクスポートした build_model() を外部から呼ぶ。

Multi-Region Editor の [Export Python Script] で書き出したスクリプトは

    def build_model(out_msh=..., mesh_order=..., **user_vars):
        ...
        return out_msh

という形になっており、変数表の変数を `user_vars` で上書きできる。
これを使い「空洞半径 a を変えて TM010 共振周波数を目標値に合わせる」二分法を回す。

実行:
    python tune_frequency.py            # デモ形状 (ピルボックス) を自動生成して調整
    python tune_frequency.py cavity_mesh.py --var a --target 2.856

GUI から書き出したスクリプトを使う場合は、その相対パスを第 1 引数に渡す。
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from axicavity_fem.fem_tm0.solver import solve_tm0_standing
from axicavity_fem.shared.gmsh_export_occ import export_python_script_multi_region
from axicavity_fem.shared.mesh_loader import load_mesh
from axicavity_fem.shared.multi_region_model import (
    Loop,
    MultiRegionGeometry,
    Region,
    Segment,
)


# ---------------------------------------------------------------------------
# デモ形状: 半径 a・長さ L のピルボックス (変数 a, L を持つ)
# ---------------------------------------------------------------------------
def make_demo_script(path: Path) -> Path:
    """変数 a (半径) / L (長さ) を持つピルボックスの build_model スクリプトを生成。"""
    a0, L0 = 40.0, 50.0            # mm
    geom = MultiRegionGeometry(unit="mm", mesh_size=3.0)
    geom.variables = [("a", "40"), ("L", "50")]
    geom.append_point(0.0, 0.0)                 # 軸上・左
    geom.append_point(L0, 0.0, "L", None)       # 軸上・右
    geom.append_point(L0, a0, "L", "a")         # 外周・右
    geom.append_point(0.0, a0, None, "a")       # 外周・左
    bcs = ["None", "PEC", "PEC", "PEC"]          # 軸 (r=0) は None
    for i, bc in enumerate(bcs):
        geom.segments.append(Segment(id=i, type="line",
                                     point_indices=[i, (i + 1) % 4],
                                     bc_name=bc))
    geom.loops.append(Loop(id=0, segment_ids=[0, 1, 2, 3]))
    geom.regions.append(Region(id=0, name="Vacuum", outer_loop_id=0,
                              material_tag="vacuum", eps_r=1.0))
    export_python_script_multi_region(geom, path, mesh_order=2,
                                      msh_output="tuned.msh")
    return path


def load_build_model(script_path: Path):
    """エクスポートされたスクリプトから build_model 関数を取り出す。"""
    spec = importlib.util.spec_from_file_location("exported_mesh", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)        # __main__ ガードがあるので実行されない
    return module.build_model


def lowest_frequency_ghz(msh_path: Path, elem_order: int = 2) -> float:
    """メッシュを解いて最低次 TM0 モード (TM010) の周波数 [GHz] を返す。"""
    mesh = load_mesh(str(msh_path), elem_order)
    res = solve_tm0_standing(mesh, num_modes=4)
    return float(min(res.frequencies))


def tune(build_model, var_name: str, target_ghz: float,
         lo: float, hi: float, *, workdir: Path,
         tol_ghz: float = 1e-4, max_iter: int = 15) -> float:
    """二分法で `var_name` を調整し、TM010 を `target_ghz` に合わせる。

    半径を大きくすると周波数は下がる（単調減少）ことを仮定する。
    """
    msh = workdir / "tuned.msh"

    def f_of(x: float) -> float:
        build_model(out_msh=str(msh), **{var_name: x})
        return lowest_frequency_ghz(msh)

    f_lo, f_hi = f_of(lo), f_of(hi)
    print(f"  {var_name}={lo:.4f} -> {f_lo:.6f} GHz")
    print(f"  {var_name}={hi:.4f} -> {f_hi:.6f} GHz")
    if (f_lo - target_ghz) * (f_hi - target_ghz) > 0:
        raise SystemExit(
            f"目標 {target_ghz} GHz が [{f_hi:.4f}, {f_lo:.4f}] の外です。"
            "探索範囲 lo/hi を広げてください。")

    for it in range(1, max_iter + 1):
        mid = 0.5 * (lo + hi)
        f_mid = f_of(mid)
        print(f"  [{it:2d}] {var_name}={mid:.6f} -> {f_mid:.6f} GHz "
              f"(誤差 {f_mid - target_ghz:+.2e})")
        if abs(f_mid - target_ghz) < tol_ghz:
            return mid
        # 単調減少: f_mid > target なら半径を大きく (lo=mid)
        if f_mid > target_ghz:
            lo, f_lo = mid, f_mid
        else:
            hi, f_hi = mid, f_mid
    print("  警告: max_iter に達しました。")
    return 0.5 * (lo + hi)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("script", nargs="?", default=None,
                    help="エクスポートした Python スクリプト (省略でデモ生成)")
    ap.add_argument("--var", default="a", help="調整する変数名 (既定: a)")
    ap.add_argument("--target", type=float, default=2.856,
                    help="目標周波数 [GHz] (既定: 2.856)")
    ap.add_argument("--lo", type=float, default=30.0, help="探索下限")
    ap.add_argument("--hi", type=float, default=60.0, help="探索上限")
    args = ap.parse_args()

    workdir = Path(__file__).resolve().parent
    if args.script:
        script = Path(args.script).resolve()
    else:
        script = workdir / "demo_pillbox_mesh.py"
        print(f"デモ形状スクリプトを生成: {script}")
        make_demo_script(script)

    build_model = load_build_model(script)
    print(f"目標 TM010 = {args.target} GHz、変数 {args.var} を "
          f"[{args.lo}, {args.hi}] で二分探索します。")
    best = tune(build_model, args.var, args.target, args.lo, args.hi,
                workdir=workdir)
    print(f"\n調整結果: {args.var} = {best:.6f} (単位は形状の Unit)")
    print(f"メッシュ: {workdir / 'tuned.msh'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
