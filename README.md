# AxiCavity-FEM

**A 2-D finite-element solver for the resonant modes of axisymmetric RF cavities — with a
parametric, sketch-based GUI, mesh generation, post-processing, 3-D field views and reporting.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-635%20passing-brightgreen.svg)](tests/)
[![Release](https://img.shields.io/github/v/release/TakuyaNatsui/AxiCavity-FEM)](https://github.com/TakuyaNatsui/AxiCavity-FEM/releases)

日本語版 README は [README.ja.md](README.ja.md) にあります。

YouTube explanatory video (version 2): https://youtu.be/U27KQ3pV1wE

![TM0 mode of a 5-cell accelerating structure](docs/images/tm0_accelerating_structure.png)

AxiCavity-FEM solves the Helmholtz equation on the (z, r) half-plane of a body of
revolution and gives you the resonant frequencies, the field distributions, and the
engineering quantities you actually design with — Q, R/Q, effective voltage, stored
energy, wall loss, power flow, group velocity and attenuation.

It covers both the **axisymmetric TM0 modes** used for acceleration and the
**higher-order azimuthal modes (HOM, n ≥ 1)** that cause beam instabilities, as
standing waves or as traveling waves with a periodic (cell-to-cell) phase advance.

**New in version 3:** a completely new graphical interface — a parametric 2-D sketcher with
geometric constraints and dimensions, projects that keep every analysis as a history, a 3-D
field view, batch runs from the command line and from Python — and a **Windows executable**
that needs no Python installation. The solver core is the same code as version 2.3.0, so the
numbers are unchanged.

---

## Features

**Physics**

- Axisymmetric **TM0** modes (H<sub>φ</sub> formulation, nodal elements) — standing and traveling wave
- **Higher-order modes** of any azimuthal order n ≥ 1 (E-field formulation, edge elements)
- **Periodic boundary conditions** with an arbitrary phase advance, and phase scans
  (`0:180:20`) for dispersion curves
- **Dielectrics**: relative permittivity per region, plus **loss tangent tanδ** giving
  the dielectric Q (`1/Q = 1/Q_wall + 1/Q_diel`)
- 1st and 2nd order triangular elements

**What you get out**

| Quantity | |
|---|---|
| f, Q, Q_wall, Q_diel | resonant frequency and quality factors |
| R/Q, V_eff | shunt impedance and effective accelerating voltage (with transit-time factor for a given β) |
| U, P_loss, P_diel | stored energy, wall loss, dielectric loss |
| P_flow, v_g, α | power flow through a port, group velocity, attenuation (traveling wave) |

**Workflow**

- **Ribbon GUI** (PySide6) that follows the work: modeling → physics → mesh → analysis → results
- **Parametric sketcher** — lines, rectangles, circles, arcs, polygons and fillets; geometric
  constraints (coincident, horizontal, tangent, concentric, symmetric, …) and dimensions solved
  by [planegcs](https://github.com/Salusoft89/planegcs); a parameter table (`a = 50`, `L = c/f/2`)
  that drives dimensions and point coordinates; the z and r axes as reference geometry
- **Closed regions are found automatically**; boundary conditions (PEC / E-short / M-short / None)
  are assigned automatically on the axis and on internal interfaces and can be overridden per curve;
  per-region material (ε<sub>r</sub>, tanδ)
- **Projects** (`.axiproj`) keep the geometry, settings and every mesh and result as a history;
  version 2 geometries (`.gmshproj`) and **Superfish `.af`** files can be imported and exported
- **Results**: field maps, mode tables with all engineering quantities, field values by double-click,
  animated GIFs of traveling waves, field data export (area / line / axis), self-contained HTML reports
- **3-D view** (pyvista): the revolved cavity with wall, meridian planes, cross-sections and a cutaway,
  field arrows, TM0 field lines and HOM cos(nφ) patterns, time-phase animation, PNG / GIF
- **Batch runs** — change parameters and analyse a project without the GUI, from the command line
  (`axicavity-fem-run`) or from Python, for parameter scans and frequency tuning
- **Command line** for every step (`axicavity-fem solve | post | report | export | plot | info`)

| Modeling (parametric sketch) | Physics (boundary conditions and materials) |
|---|---|
| ![Modeling](docs/images/v3_modeling.png) | ![Physics](docs/images/v3_physics.png) |
| **Results** | **3-D view** |
| ![Results](docs/images/v3_results.png) | ![3-D view](docs/images/v3_view3d.png) |

| Higher-order dipole mode (n = 1) | Mesh, boundary conditions and dielectric interfaces |
|---|---|
| ![HOM n=1](docs/images/hom_dipole_n1.png) | ![Mesh overview](docs/images/mesh_overview.png) |

---

## Installation

### Windows: the executable (no Python needed)

Download `AxiCavity-FEM-3.0.0-win64.zip` from
[Releases](https://github.com/TakuyaNatsui/AxiCavity-FEM/releases), unzip it anywhere, and run
`AxiCavity-FEM.exe`. Keep the whole folder together. The executable is not code-signed, so Windows
SmartScreen may ask once ("More info" → "Run anyway"). It includes Intel MKL PARDISO for large
meshes and the 3-D view; meshes are generated by the bundled `mesher` (Gmsh). The same executable
also runs the command line (`AxiCavity-FEM.exe solve …`, `AxiCavity-FEM.exe run …`); see the
`README.txt` in the zip.

### With pip (Windows, Linux, macOS)

Requires Python 3.10 or newer.

```bash
git clone https://github.com/TakuyaNatsui/AxiCavity-FEM.git
cd AxiCavity-FEM
pip install -e ".[gui,viz3d]"     # solver + command line + GUI + 3-D view
```

Or straight from GitHub, without cloning:

```bash
pip install "axicavity-fem[gui,viz3d] @ git+https://github.com/TakuyaNatsui/AxiCavity-FEM.git"
```

| Extra | Adds |
|---|---|
| *(none)* | solver and command line (`axicavity-fem`) |
| `gui` | the GUI (`axicavity-fem-gui`) and batch runs (`axicavity-fem-run`): PySide6, planegcs |
| `viz3d` | the 3-D view: pyvista, pyvistaqt |
| `accel` | Intel MKL PARDISO (`pypardiso`): large meshes (above ~20,000 degrees of freedom) solve several times faster, with identical results |
| `dev` | pytest, pytest-qt |

Gmsh comes in as a pip dependency — no separate installation is needed. On Linux you
may also need the usual OpenGL runtime for it (`libglu1-mesa`).

**Upgrading from 2.3.0:** the package name is the same (`axicavity-fem`), so installing version 3
replaces it. The command-line tool `axicavity-fem` is unchanged; `axicavity-fem-gui` now starts the
new GUI (wxPython is no longer used). Old `.gmshproj` geometries open through *File → Import*.

---

## Quick start

### Graphical interface

```bash
axicavity-fem-gui          # or AxiCavity-FEM.exe
```

1. **Modeling** — draw a rectangle from (0, 0) to (100, 50) mm (or *File → Import* a sample from
   [`samples/`](samples/)). Corners on the axis are constrained to it automatically.
2. **Physics** — the axis is set to `None` and the other walls to `PEC` automatically.
3. **Analysis** — press **Run analysis**. The project is saved first; results are kept in it.
4. **Results** — the mode table shows f = 2.294851 GHz for TM<sub>010</sub>, the analytic
   `j₀₁c/2πa` to six decimals; open the **3-D view** from the same tab.

The [user manual](USER_MANUAL.md) (Japanese) walks through three tutorials.

### Command line

The repository ships with sample geometries in [`samples/`](samples/). This one is a
pillbox cavity, 50 mm in radius and 100 mm long. One command meshes it, finds the modes
and adds the engineering parameters:

```bash
axicavity-fem-run samples/cylinder100mm.gmshproj --modes 6 --out pillbox
```

The mesh it wrote (`pillbox/model.msh`) can then go through the individual steps:

```bash
axicavity-fem solve  --type tm0 -m pillbox/model.msh --elem-order 2 --num-modes 6 -o pillbox.h5
axicavity-fem post   --type tm0 -i pillbox.h5 --cond 5.8e7
axicavity-fem report --type tm0 -i pillbox.h5 -o pillbox_report
```

`solve` finds the modes, `post` adds the engineering parameters to the same file (here for copper
walls, σ = 5.8 × 10⁷ S/m), and `report` writes a self-contained HTML page with field maps.
Each step also writes a plain-text summary next to the HDF5 file.

For higher-order modes, name the azimuthal orders you want; for a traveling wave, give
the cell-to-cell phase advance; to search a band, give a target frequency:

```bash
axicavity-fem-run samples/s-band_1cell.gmshproj --no-post --out sband
axicavity-fem solve --type hom -m sband/model.msh --az-order 0 1 2 -o hom.h5
axicavity-fem solve --type tm0 -m sband/model.msh -p 120 -o traveling.h5
axicavity-fem solve --type tm0 -m sband/model.msh --num-modes 4 --target-freq 2.856 -o sband.h5
```

Run `axicavity-fem <command> --help` for the full options. With the Windows executable, use
`AxiCavity-FEM.exe` in place of `axicavity-fem` and `AxiCavity-FEM.exe run` in place of `axicavity-fem-run`.

### Parameter scans from Python

A project saved from the GUI can be re-analysed with new parameter values; the sketch
constraints and dimensions are solved again for each value:

```python
from axicavity_fem.gui.batch import Project

p = Project.open("pillbox.axiproj")
for a in (40, 45, 50):
    p.set_params(a=a)
    r = p.run(post=False)
    print(a, r.frequencies()[0], "GHz")
```

---

## Accuracy

- **Spherical cavity vs. the spherical-Bessel analytic solution**: relative error down
  to ~1e-8 on the finest mesh, converging at order ≈ 4 for both TM0 and HOM
  ([`examples/accuracy_verification/`](examples/accuracy_verification/)).
- **Pillbox TM010**: agrees with `j₀₁c/2πa` to six decimal places.
- **Dielectric loss**: `Q_diel = 1/tanδ` holds to machine precision for a uniformly
  filled cavity, and the frequency matches the analytic TM010 to 1.4e-7
  ([`examples/dielectric_loss/`](examples/dielectric_loss/)).
- **Quadrature**: TM0 integrates with a 7-point (degree-5) rule by default. A
  convergence study ([`examples/quadrature_convergence/`](examples/quadrature_convergence/))
  showed the older 4-point rule under-integrates the 2nd-order mass term and injects
  spurious modes; the 7-point rule produces none at any mesh density.
- Every returned eigenpair is **residual checked**, so unconverged modes never appear
  in the results as fake resonant frequencies.

Use 2nd-order elements (`--elem-order 2`, the GUI default) for design work — they are one to two
orders of magnitude more accurate than 1st-order elements at the same mesh size.

---

## Documentation

| | |
|---|---|
| [USER_MANUAL.md](USER_MANUAL.md) | Full walkthrough of the GUI, every CLI option and the Python batch API (Japanese) |
| [PHYSICS_AND_CONVENTIONS.md](PHYSICS_AND_CONVENTIONS.md) | Time convention, normalization, power flow, periodic-BC sign |
| [docs/BC_NAMING.md](docs/BC_NAMING.md) | What PEC / E-short / M-short / None mean physically |
| [docs/HDF5_SCHEMA.md](docs/HDF5_SCHEMA.md) | Layout of the output files |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md) | For contributors |
| [packaging/README.md](packaging/README.md) | Building the Windows executable |
| [examples/](examples/) | Runnable verification, frequency-tuning and parameter-scan scripts |
| [CHANGELOG.md](CHANGELOG.md) | Release notes |

---

## Earlier versions

Version 3 replaces the wxPython interface of version 2.3.0 with the new GUI described above.
The solver core (`shared`, `physics_core`, `fem_tm0`, `fem_hom`, `reports`, `cli`) is the same
code — only the version string differs — so results agree with 2.3.0 to the last digit for the
same mesh. Version 2.3.0 remains available at tag
[`v2.3.0`](https://github.com/TakuyaNatsui/AxiCavity-FEM/releases/tag/v2.3.0), and version 1
at tag [`v1.0`](https://github.com/TakuyaNatsui/AxiCavity-FEM/releases/tag/v1.0).

---

## Citation

If this code contributes to published work, please cite it:

```bibtex
@software{AxiCavityFEM,
  author  = {Natsui, Takuya},
  title   = {{AxiCavity-FEM}: a 2-D axisymmetric finite-element solver for RF cavity resonant modes},
  version = {3.0.0},
  year    = {2026},
  url     = {https://github.com/TakuyaNatsui/AxiCavity-FEM}
}
```

## License

MIT — see [LICENSE](LICENSE).

Runtime dependencies keep their own licenses: **Gmsh** is GPL-2.0-or-later, **PySide6 / Qt** is
LGPL-3.0, **planegcs** is LGPL-2.1, pyvista and VTK are MIT / BSD, and the optional **Intel MKL**
(through pypardiso) is under the Intel Simplified Software License. Installing them with pip does
not change the license of this code.

The **Windows executable** bundles Qt / PySide6 and planegcs (dynamically linked, replaceable) and
Intel MKL. Meshing is done by the bundled `mesher` folder, a separate program that contains Gmsh,
is distributed under the GPL together with its source code, and only exchanges files with the main
program. The zip contains the full notices (`THIRD_PARTY_NOTICES.txt`).

## Contributing

Bug reports and pull requests are welcome through
[GitHub Issues](https://github.com/TakuyaNatsui/AxiCavity-FEM/issues).
Please run the test suite before opening a pull request:

```bash
pip install -e ".[gui,viz3d,dev]"
pytest
```

Some tests compare against reference results that are not in the repository; they are skipped
when the files are absent. The developer documentation is in Japanese; the code is organised as
described in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
