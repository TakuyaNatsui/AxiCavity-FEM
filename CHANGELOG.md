# Changelog

All notable changes to this project are documented here.
Versions follow [semantic versioning](https://semver.org/).

## [2.3.0] — 2026-07-24 — first public release of the version 2 line

Version 2 is a **complete rewrite** of version 1. It keeps the same physics and
reproduces version 1's numbers, but is packaged, tested and documented as a tool
other people can install and use.

Versions 2.0, 2.1 and 2.2 were internal development milestones and were never
released; everything they introduced is included here.

### Added

- **Installable Python package** (`pip install -e .`) with a single unified
  command line: `axicavity-fem solve | post | info | export | plot | report`.
  Version 1 required running separate scripts from inside the source tree.
- **Multi-Region Editor GUI** — draw the cavity outline with points, lines and
  arcs, assign boundary conditions per segment, mesh it with Gmsh, and hand the
  mesh straight to the solver. Supports several regions, holes, and per-region
  materials.
- **Dielectrics.** Per-region relative permittivity `eps_r` for both TM0 and HOM.
- **Dielectric loss.** Per-region loss tangent `tan_delta` gives `Q_diel`,
  `P_diel` and a total `Q` (`1/Q = 1/Q_wall + 1/Q_diel`), computed as a
  perturbation so the eigenproblem stays real and the run time is unchanged.
  For a uniformly filled cavity `Q_diel = 1/tanδ` holds to machine precision.
- **Variables and expressions** in the geometry editor. Define `a = 100`,
  `b = 10`, `c = a + b`, type expressions into coordinate fields, and the shape
  follows when a variable changes. Evaluated by a safe `ast`-based evaluator
  (never `eval`), standard library only.
- **Superfish `.af` import / export**, so existing Poisson/Superfish geometries
  can be brought in and taken back out.
- **HTML reports** (`report`), **field map images** (`plot`), **field data
  export** (`export`, HDF5 / text, area / line / on-axis), and animated GIFs for
  traveling-wave modes.
- **Interactive Result Viewer** — step through modes, switch field components,
  and read field values by double-clicking the plot.
- **Plain-text result summaries** written next to every `.h5` output, so
  frequencies and engineering parameters can be read without opening HDF5.
- **Target frequency search** (`--target-freq`, or *Predicted frequency* in the
  GUI). Sets the shift-invert shift from a frequency you give, returning the
  modes nearest that frequency instead of always starting from the lowest one.
- **Documented HDF5 schema** (`docs/HDF5_SCHEMA.md`) shared by TM0 and HOM.
- **Automated test suite** (pytest) covering the solvers, post-processing, file
  I/O round-trips, geometry model, mesh export, CLI wiring and GUI construction.
- **Optional PARDISO acceleration** (`pip install pypardiso`) for the sparse
  factorization in standing-wave analyses. Results are unchanged; large meshes
  (above roughly 20k free degrees of freedom) solve several times faster.

### Changed

- **7-point Gauss quadrature is now the default for TM0** (was 4-point). The
  2nd-order mass term `r·G_i·G_j` is a degree-5 polynomial that 4-point
  quadrature under-integrates, which made the mass matrix numerically indefinite
  and injected spurious eigenvalues. On a convergence study 4-point produced
  2–11 spurious modes per solve at every mesh density; 7-point produced none,
  with identical accuracy and convergence rate for the physical modes.
- **Boundary condition naming** was revised to be explicit about the physics:
  `PEC`, `E-short`, `M-short`, and `None` for the symmetry axis and internal
  boundaries. See `docs/BC_NAMING.md`.
- **`Q` now means the total Q** including dielectric loss. `Q_wall` carries the
  previous wall-loss-only value, and the two agree bit-for-bit when `tanδ = 0`.

### Fixed

- **Spurious modes are filtered out.** Every returned eigenpair is residual
  checked; unconverged modes are dropped and the solve is retried with a larger
  request to refill the count. In version 1 an unconverged mode could silently
  appear in the result list as a fake resonant frequency.

### Validation

- Spherical cavity against the spherical-Bessel analytic solution: relative
  error ~1e-8 on the finest mesh, convergence order ≈ 4 (`examples/accuracy_verification/`).
- Pillbox TM010 against `j01·c/(2πa)`: agreement to 6 decimal places.
- Dielectric loss against the analytic `Q_diel = 1/tanδ` and against a Superfish
  cross-check (`examples/dielectric_loss/`).

## 1.x

Version 1 lives on in this repository at tag [`v1.0`](https://github.com/TakuyaNatsui/AxiCavity-FEM/releases/tag/v1.0)
and branch [`v1`](https://github.com/TakuyaNatsui/AxiCavity-FEM/tree/v1).
