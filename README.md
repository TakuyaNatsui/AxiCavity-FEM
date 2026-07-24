# AxiCavity-FEM

**A 2-D finite-element solver for the resonant modes of axisymmetric RF cavities —
with a built-in geometry editor, mesh generation, post-processing and reporting.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-373%20passing-brightgreen.svg)](tests/)

日本語版 README は [README.ja.md](README.ja.md) にあります。

![TM0 mode of a 5-cell accelerating structure](docs/images/tm0_accelerating_structure.png)

AxiCavity-FEM solves the Helmholtz equation on the (z, r) half-plane of a body of
revolution and gives you the resonant frequencies, the field distributions, and the
engineering quantities you actually design with — Q, R/Q, effective voltage, stored
energy, wall loss, power flow, group velocity and attenuation.

It covers both the **axisymmetric TM0 modes** used for acceleration and the
**higher-order azimuthal modes (HOM, n ≥ 1)** that cause beam instabilities, as
standing waves or as traveling waves with a periodic (cell-to-cell) phase advance.

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

- **Geometry editor GUI** — points, lines and arcs; boundary conditions per segment;
  several regions with holes; per-region material; mesh with one click
- **Parametric geometry** — define variables (`a = 100`, `c = a + b`) and type expressions
  into coordinate fields; change a variable and the shape follows
- **Superfish `.af` import / export** — bring existing geometries in and take them back out
- **Outputs**: HDF5 (documented schema), plain-text summaries, field maps as PNG,
  field data as HDF5/text along a line, over an area or on axis, self-contained HTML
  reports, and animated GIFs of traveling-wave modes
- **Result Viewer** — step through modes, switch field components, read values off the plot

<!--
GUI screenshots go here once captured, e.g.

| Geometry editor | Result viewer |
|---|---|
| ![Multi-Region Editor](docs/images/gui_editor.png) | ![Result Viewer](docs/images/gui_result_viewer.png) |
-->

| Higher-order dipole mode (n = 1) | Mesh, boundary conditions and dielectric interfaces |
|---|---|
| ![HOM n=1](docs/images/hom_dipole_n1.png) | ![Mesh overview](docs/images/mesh_overview.png) |

---

## Installation

Requires Python 3.10 or newer.

```bash
git clone https://github.com/TakuyaNatsui/AxiCavity-FEM.git
cd AxiCavity-FEM
pip install -e .          # solver + command line
pip install -e ".[gui]"   # ... and the graphical interface (wxPython)
```

Or straight from GitHub, without cloning:

```bash
pip install "git+https://github.com/TakuyaNatsui/AxiCavity-FEM.git"
```

Gmsh comes in as a pip dependency — no separate installation is needed. On Linux you
may also need the usual OpenGL runtime for it (`libglu1-mesa`).

Optional: `pip install pypardiso` switches the sparse factorization to Intel MKL
PARDISO. Results are identical; large meshes (above roughly 20,000 free degrees of
freedom) solve several times faster.

---

## Quick start

### Command line

The repository ships with sample geometries in [`samples/`](samples/). This one is a
pillbox cavity, 50 mm in radius and 100 mm long:

```bash
axicavity-fem solve  --type tm0 -m samples/cylinder100mm.msh --elem-order 2 --num-modes 6 -o pillbox.h5
axicavity-fem post   --type tm0 -i pillbox.h5 --cond 5.8e7
axicavity-fem report --type tm0 -i pillbox.h5 -o pillbox_report
```

`solve` finds the modes, `post` adds the engineering parameters (here for copper walls,
σ = 5.8 × 10⁷ S/m), and `report` writes a self-contained HTML page with field maps.
Each step also writes a plain-text summary next to the HDF5 file. The first mode comes
out at 2.294851 GHz — the analytic pillbox TM<sub>010</sub> value `j₀₁c/2πa` to six decimals.

To search a specific band instead of starting from the lowest mode:

```bash
axicavity-fem solve --type tm0 -m samples/s-band_1cell.msh --num-modes 4 --target-freq 2.856 -o sband.h5
```

For higher-order modes, name the azimuthal orders you want; for a traveling wave, give
the cell-to-cell phase advance:

```bash
axicavity-fem solve --type hom -m samples/s-band_1cell.msh --az-order 0 1 2 -o hom.h5
axicavity-fem solve --type tm0 -m samples/s-band_1cell.msh -p 120 -o traveling.h5
```

Run `axicavity-fem <command> --help` for the full options, or see the
[user manual](USER_MANUAL.md).

### Graphical interface

```bash
axicavity-fem-gui
```

1. **Multi-Region Editor** — open a sample from `samples/` (or draw a new outline),
   set the boundary condition of each segment, and export a `.msh`.
2. **FEM Analysis** — the mesh path is filled in for you; press **Run Solver**, then
   **Run Post-Process**, then **Create Report** or **View Results**.

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

Use 2nd-order elements (`--elem-order 2`) for design work — they are one to two orders
of magnitude more accurate than 1st-order elements at the same mesh size.

---

## Documentation

| | |
|---|---|
| [USER_MANUAL.md](USER_MANUAL.md) | Full walkthrough of the GUI and every CLI option |
| [PHYSICS_AND_CONVENTIONS.md](PHYSICS_AND_CONVENTIONS.md) | Time convention, normalization, power flow, periodic-BC sign |
| [docs/BC_NAMING.md](docs/BC_NAMING.md) | What PEC / E-short / M-short / None mean physically |
| [docs/HDF5_SCHEMA.md](docs/HDF5_SCHEMA.md) | Layout of the output files |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md) | For contributors |
| [examples/](examples/) | Runnable verification and parameter-sweep scripts |
| [CHANGELOG.md](CHANGELOG.md) | Release notes |

---

## Relation to version 1

Version 2 is a complete rewrite of version 1: an installable package with a unified
command line, a new geometry editor, multi-region and dielectric support, an automated
test suite, and a documented file format. The physics is the same and the numbers agree
with version 1 (checked mode by mode during development).

Version 1 remains available in this repository at tag
[`v1.0`](https://github.com/TakuyaNatsui/AxiCavity-FEM/releases/tag/v1.0) and branch
[`v1`](https://github.com/TakuyaNatsui/AxiCavity-FEM/tree/v1).

---

## Citation

If this code contributes to published work, please cite it:

```bibtex
@software{AxiCavityFEM,
  author  = {Natsui, Takuya},
  title   = {{AxiCavity-FEM}: a 2-D axisymmetric finite-element solver for RF cavity resonant modes},
  version = {2.3.0},
  year    = {2026},
  url     = {https://github.com/TakuyaNatsui/AxiCavity-FEM}
}
```

## License

MIT — see [LICENSE](LICENSE).

Runtime dependencies keep their own licenses: **Gmsh** is GPL (with a linking
exception) and **wxPython** is under the wxWindows Library Licence (LGPL-like).
Installing them with pip does not change the license of this code, but redistributing
a bundle that includes them does carry their terms.

## Contributing

Bug reports and pull requests are welcome through
[GitHub Issues](https://github.com/TakuyaNatsui/AxiCavity-FEM/issues).
Please run the test suite before opening a pull request:

```bash
pip install -e ".[dev]"
pytest
```

The GUI layout is generated with [wxGlade](https://wxglade.sourceforge.net/) from
`main_frame_ui.wxg` / `result_viewer_ui.wxg`. Edit the `.wxg` file and regenerate
`*_ui.py` rather than editing the generated files by hand; the behaviour lives in the
hand-written subclasses (`main_frame.py`, `result_viewer.py`).
