# Changelog

All notable changes to this project are documented here.
Versions follow [semantic versioning](https://semver.org/).

## [3.0.1] — 2026-11-07 — corrected post-processing, HOM boundary conditions and target wall loss

**Solve again these HOM results: traveling waves with n ≥ 1 on cells with an iris, a coaxial inner conductor or a
magnetic outer wall; traveling waves at any n on cells split into regions along z (a dielectric window, for example);
traveling waves on a mesh finer than about 40 µm; and standing or traveling waves on geometries with a corner below
90°, a gap one element wide or a wall meeting the axis between its ends. Re-run `post` on traveling-wave TM0
results, on TM0 results with a dielectric on the axis (standing waves too), on TM0 results with 1st-order elements and
such a corner or gap, on standing-wave HOM results and on HOM results with n ≥ 1. Export again any field files written
with a target wall loss, and HOM exports with n ≥ 1 that contain points in the elements touching the axis.**
Quantities listed below were wrong in 3.0.0 and all earlier releases. Apart from those HOM cells and the reused-mesh cases under Existing results, the
eigenmodes and frequencies were correct, so no new solve is needed (except for files in the version 1
format, which `post` refuses). Some results need only one of these steps or neither (for example traveling
HOM n = 0 on a cell that is one region): the table in section 13 of the [user manual](USER_MANUAL.md) lists
them by result type.

### Fixed

- **Traveling-wave TM0 post-processing depended on the overall phase of the eigenvector.** U, P_loss,
  P_diel, P_flow, V_eff, V_acc, R/Q, and `rq_apparent`, `r_shunt_m`, `norm_factor` and
  `normalized_eigenvectors` (Python API) were affected. The on-axis scan took the real part (one
  instant) of E_z for the normalization and the voltage integrals, but the phase of a complex
  eigenvector is arbitrary and set by ARPACK's start vector. These values depended on that phase, and
  R/Q was usually far too low (for some modes too high). With SciPy 1.16 the first solve in a new
  process always has the same phase, so **getting the same number twice did not mean it was right**;
  solving again in the same process gave other numbers. An S-band cell at 120° could give anything
  from 3.5 to 52.5 Ω depending on the phase (3.5–49 Ω over 12 solves in one process) and now gives
  82.9 Ω every time on the 5 mm mesh of the sample (83.2 Ω on 2.5 mm, 83.3 Ω when converged). `post` now uses the complex amplitude of
  E_z, as `PHYSICS_AND_CONVENTIONS.md` specifies (`max_z |E_z(r=0)| = 1 V/m`). R/Q now agrees with the
  analytic value for a hollow waveguide (TM01, within 1%) and with the standing-wave solver (the same
  value at θ = 0° and 180°, and at θ = 120° two thirds of the value for the three-cell standing wave).
  Meshes without the axis were not affected.
- **HOM post-processing had three errors in its formulas**, all present since version 1:
  - *Standing waves, every n including 0:* U, Q_wall and P_diel were twice the correct value (the
    coefficient of U was 1 instead of 1/2). So was Q when n = 0 and tanδ = 0. With tanδ > 0 the total Q
    follows 1/Q = 1/Q_wall + 1/Q_diel and Q_diel did not change, so the error of Q is smaller.
  - *n ≥ 1:* the wall loss had an extra factor j in two terms of curl E, so P_loss, Q, Q_wall and α
    were wrong. In a hollow waveguide cell Q was off by +38% (TE11), +63% (TE21), −2.9% (TM11) and
    −4.6% (TM21). With the error above, Q of a standing TE mode was more than twice the correct value
    (2.33 times for TE211 in a pillbox of R = 40 mm, L = 35 mm, and 3.3 times at R = 50 mm, L = 100 mm).
  - *Traveling waves with n ≥ 1 that have E_z:* the second term of P_flow took the real part instead
    of the imaginary part, so P_flow, v_g and α were wrong. In the same waveguide v_g was off by −25%
    (TM11) and −41% (TM21). In a periodic structure some modes even had the wrong sign of v_g.

  With the last two errors together, α was off by +38% (TM11), +76% (TM21), −28% (TE11) and −39%
  (TE21). Q, v_g and α now agree with the analytic values to within 0.4% for the low modes of a
  waveguide cell on a 2 mm mesh (TM01, TM11, TM21, TE11, TE21; higher modes are limited by the mesh,
  for example TM12 at 0.8% on 2 mm and 0.2% on 1 mm), and Q to within 0.3% for the low TM and TE modes
  of a pillbox on a 1 mm mesh (higher modes reach 0.5%, for example TE231). v_g agrees with L dω/dθ,
  which uses the frequencies only.
- **Traveling HOM modes on a cell split into regions along z had completely wrong frequencies, at every n
  including 0.** The sign that relates an edge on one periodic plane to its partner assumed that the triangles
  next to the two planes have the same orientation, but the GUI's regions can have opposite orientations. No
  warning appeared. Two vacuum regions in a row (L = 30 mm, R = 40 mm, θ = 120°, h = 2.5 mm): instead of 3.990 GHz
  and v_g = 0.836c, n = 1 gave a lowest mode between 1.26 and 3.06 GHz depending on how many modes were requested
  (1.791 GHz for three), and v_g between 0.19c and 1.06c, faster than light; n = 0 gave 3.381 GHz instead of 4.396 GHz.
  The sign now comes from the direction of the edges along r. The results now agree with the same cell as one
  region (frequencies within 2×10⁻⁵, Q within 0.2%) and with analytic values, and on `samples/diel_simple1` (a
  dielectric filling half of the cell along z) n = 0 agrees with the TM0 solver within 1×10⁻⁵. Regions split
  along r were not affected. **Solve these results again.**
- **HOM: interior edges between two wall nodes were treated as walls, in standing and traveling waves and at
  every n.** An edge counted as a wall if both of its ends were on a wall, so E_t = 0 was forced on an interior
  edge that cuts a corner below 90° or crosses a gap one element wide, and the same edge was
  counted in the wall loss. Whether such an edge exists depends on the mesh: at 85° it appeared on 1.25 mm and not on
  2.5 mm; at 70° on every mesh; `samples/acc2_pipe_flat` has three on its 5 mm mesh (+0.6% in f at n = 0). At a 70° corner on a
  2.5 mm mesh the lowest mode was +0.36% in f and +4.5% in Q
  against the TM0 solver (+0.09% and +2.2% on 1.25 mm). A traveling cell with such a corner gave frequencies that
  depended on the label of the end planes (up to 0.78% between None and E-short, the setting the manual and
  the GUI recommended), and in a thin cell a spurious mode appeared. In a re-entrant cavity with a 2 mm gap
  on a 2.5 mm mesh, n = 0 lost its fundamental mode. An edge now counts as a wall only if it lies on a wall line:
  with 2nd-order elements its midpoint node must be on the wall too, and with 1st-order elements the mesh's line elements decide.
  These are now stored in the `.h5` file (`/mesh/physical_edges`); for an older 1st-order file `post` counts only the
  outer edges of the domain and warns. At the 70° corner, f now agrees with TM0 within 1.5×10⁻⁴ and Q within 0.6%, the end label no longer
  matters (within 10⁻⁶ with 2nd-order elements; with 1st-order elements up to 3×10⁻⁵, because the corner nodes on the axis are fixed only with a wall label), and the re-entrant cavity is within 0.6% of TM0 at h = 2.5 mm. The end points of an interior PEC
  sheet that reaches the periodic plane now also get E_θ = 0 (+9×10⁻⁶ against the analytic value on a 2 mm
  mesh, was −5×10⁻⁴), and a wall touching only one periodic plane also fixes its image on the other plane (the same
  structure cut elsewhere: within 5×10⁻⁴, was −0.8% on 1.25 mm). A wall that meets the axis between its ends is now
  fixed at n = 0 (a change of up to 6×10⁻⁴ on 2.5 mm). Cells without such edges do not
  change, for example the S-band cell and cylinders at any mesh size and spheres on meshes finer than about a sixth of the radius
  (a 50 mm sphere at h = 10 mm was +4%). A thin traveling cell, about one element long, with E-short or PEC end planes has such edges too. **Solve these results again.**
- **TM0 with 1st-order elements counted the same interior edges in the wall loss**, so Q, Q_wall and α were off at
  corners below 90° and at gaps one element wide (Q −5.6% at 70° on 2.5 mm, now −0.18%; −4.6% and −6.1% for the
  lowest modes of `samples/acc2_pipe_flat` on 5 mm). Re-run `post`. 2nd-order elements were not affected.
- **Traveling HOM on a very fine mesh** (below about 40 µm) stopped with an error or, without any warning, gave
  wrong values (f +1.2×10⁻³ and v_g −6.9% at 34 µm), because the periodic planes were found with a fixed tolerance of
  10 µm. The tolerance now follows the size of the mesh on the planes, and an outer node just outside it, or an edge that nearly lies on the plane, stops the run instead of
  leaving a free node. `post` uses the same tolerance for P_flow. Down to 20 µm the frequency agrees with the analytic value within 3×10⁻⁷. **Solve these results again.**
- **HOM with n = 1: E_θ exactly on the axis, and H_z and H_r very close to the axis.** On the axis E_θ was the
  average of the neighboring nodes (3% away from E_r, which it must equal, on a 2 mm mesh and 8% on 5 mm for TE111); it now equals E_r. That average entered the (n/r) terms in all
  elements touching the axis (r up to about h): H_z was off by about 33% of the peak H there, and close to the axis
  (n/r) terms divided rounding-sized differences: on a 2 mm mesh H_z reached over 200% of the peak H at r = 0.1 mm and over
  300 times the peak at 1 µm, and a double-click near the axis showed this. Within 5% of the element
  size from the axis, in elements touching the axis, H_z and H_r now follow regularity, as on the axis. Beyond that
  (n = 1, 2nd-order elements, 2 mm) H_z is within about 6% of the peak H (1% at r ≈ h), and H_r within 7% just outside the zone and 0.7% in amplitude beyond a few tenths of a mm in a waveguide cell;
  as a complex number H_r is off by up to about 11% just outside that zone (3% at 0.3 mm). The same zone applies to n ≥ 2, where H_z and H_r go to 0.
  This changes exports and field maps only; Q, U, P_loss, P_flow, v_g and α do not use these values.
- **Traveling HOM modes with n ≥ 1: a wall reaching the periodic end plane below the outer radius did not
  get E_θ = 0.** Only the outer corner (r = r_max) was left to the PEC condition. The tip of an iris and
  the inner conductor of a coaxial cell were tied by the periodic condition instead, which relates the
  values on the two planes but does not make them zero, so the frequency came out too low. All phases
  were affected, including θ = 0° and 180°; standing waves, n = 0, TM0, and cells whose only wall on the
  end plane is at r_max were not. On the S-band cell (the six lowest modes of n = 1 and 2, all phases) with a
  2.5 mm mesh the frequency was low by up to 0.35% (0.18% at θ = 120°; with a 1.25 mm mesh 0.08% at θ = 0° and
  0.04% at θ = 120°). Q was low by up to 3.7%, and this does not shrink with the mesh (3.7% on 1.25 mm too).
  v_g moved by up to 0.007c, and α by about 16–22% at θ = 120–150° and by more where v_g is small (93% for one
  mode at θ = 30°). The frequencies now agree with references that do not use this condition: three-cell
  standing waves at θ = 60° and 120° (2.6×10⁻⁶ on 2.5 mm, up to about 1.6×10⁻⁵ on similar meshes), one-cell
  standing waves at θ = 0° and 180° (within 1×10⁻⁷), and the analytic values of a coaxial cell (within 4×10⁻⁷).
  **`post` alone does not fix these results; solve them again.** The same change corrects a cell with a
  magnetic (M-short) outer wall, whose corner lost the periodic condition: the frequency was off by up to
  1.8×10⁻⁴ with open end planes, and with E-short end planes a spurious low mode appeared and the real modes
  were off by 2–3.5%. It is now within 2×10⁻⁶ of the analytic value, whatever the label of the end planes. An outer wall with no label behaves the same way. With a wall label on the end planes, the nodes where the axis meets those planes are now fixed as well (a change below 1×10⁻⁶, also on plain cylinders).
- **H_r of HOM modes exactly on the axis (r = 0) was wrong** because the (n/r)E_z term was dropped: about 0
  for a pillbox TM110, about 0.3 of the true value for a TM11 waveguide cell, and up to several times too
  high in hybrid modes. For n = 1 it now follows from regularity (the field on the axis is a uniform
  transverse field, so H_φ = jH_r), and for n ≥ 2 it is 0. With 2nd-order elements (the default) it agrees with
  the analytic value within 0.35% in amplitude and about 2% as a complex number (TM110, TM11, TE11 on a 2 mm
  mesh). With 1st-order elements it does not converge (see Known limitations). This affected
  `export --shape axis`, a line starting at r = 0, the r = 0 row of an area export (H_r and H_abs), and the
  on-axis nodes of the field maps, the 3D view (where the field now has one direction on the axis) and the
  report plots; Q, P_loss, P_flow, v_g and α do not use it.
- **`post -o` lost the stored values when it failed.** It copied the input over the output before reading
  it, so a failed computation, an input in the version 1 format, a read-only input, or an output held open by
  another program left an output without the earlier `post` values (a read-only input also made the output
  read-only, so `post` kept failing). `post` now writes to a temporary file next to the output and replaces
  the output only when that is complete, also when input and output are the same file. It checks the output
  path before computing and refuses a folder, a `.txt` name, a missing folder and a read-only output. An
  input that is not HDF5 or is in the version 1 format is refused with a clear message. If only the `.txt`
  summary cannot be written (open in a spreadsheet, for example), `post` warns and succeeds, because the
  `.h5` already holds the new values.
- **Exit codes and error messages.** `python -m axicavity_fem.cli` always returned 0; it now returns the code
  of the command. `axicavity-fem-run` printed a traceback for file errors; it now prints an error line.
- **TM0 with a dielectric on the axis: E_z on the axis came from the vacuum formula.** `post` did not divide
  it by ε_r, so V_eff, V_acc and R/Q were wrong, and through the normalization to 1 V/m on the axis so were
  the absolute values of U, P_loss, P_diel and P_flow, and `r_shunt_*` in the Python API, for standing and
  traveling waves. A uniform fill of ε_r = 4 gave R/Q 16 times too high and U 16 times too low. Frequencies,
  Q, v_g and α were not affected, nor was a dielectric away from the axis. U, V and R/Q now agree with analytic
  values for a waveguide cell filled with ε_r = 4 (within 1%), and for the lowest standing mode with a
  dielectric slab across the axis (E_z jumps at the interface) and a dielectric rod on the axis (both within
  0.3% on a 2 mm mesh, converging as h²). Traveling waves and higher modes need a finer mesh: a rod in a cell
  at θ = 120° had U low by 1.2% and 3.2% for its two lowest modes on 2 mm, and 0.3% and 0.8% on 1 mm. The warnings about this
  in `solve` and in the GUI's check are gone.
- **The warnings of `solve` for dielectrics were too broad.** They said that U, P_loss, Q, P_flow and v_g
  could be wrong. These values include ε_r: for a waveguide cell filled with ε_r = 4, Q, v_g and α agree
  with analytic values within 0.5% (TM01 with TM0, and TM01, TM11, TE11, TE21 with HOM), and v_g agrees with
  L dω/dθ with a dielectric on the end plane. HOM now warns only that the shift does not follow ε_r, so a
  low mode can be missed.
- **TM0 could miss the lowest modes without saying so.** With no predicted frequency, the search starts at
  the automatic shift c/(4 r_max), and in standing waves a mode below it is never returned, however many modes you ask for: a
  re-entrant cavity gave 4.43 GHz as its lowest mode instead of 0.86 GHz, and a long shorted coaxial line lost
  its lowest TEM modes. Dielectrics do the same (a uniform ε_r = 4). Traveling waves return the modes nearest the shift, which can skip lower ones too. The solver now counts the modes
  below the lowest returned one, without solving for them, and warns with their number; the frequencies do not change.
  Give a low predicted frequency to get those modes. A cavity with no axis and no M-short has a static solution H_φ ∝ 1/r
  whose discrete frequency is not 0 (0.46 GHz on a coarse 1st-order mesh). It is not counted as a missed mode, and if a returned mode overlaps it, which can happen
  on a coarse mesh or with a thin inner conductor, the solver says the inner conductor is not resolved (some of these are real modes polluted by it; refine the mesh and see whether the frequency goes to 0). The GUI shows these warnings in the results panel, also after a cancel or a failed `post`.
- **TM0 traveling: end-plane nodes that do not pair stayed walls without a warning.** The nodes of the two end planes are paired by r, and a node without a
  partner keeps its natural boundary condition. If the planes are split at different r or differ by rounding, the frequency was wrong
  (4.05 GHz gave 2.87 GHz when r differed by 0.1 µm; −2.9% to −5% with mismatched splits, depending on the mesh size), and a cell with no pairs at all gave a result independent of the phase. `solve` now warns.
- **GUI: checks on the end planes of a traveling cell.** A wall (PEC, E-short or M-short) that covers only part of a periodic end
  plane is overwritten by the periodic condition, so it does not enter the calculation, for TM0 and HOM. The check before a run now warns about this,
  and about end planes split at different r, and a cell with no straight end plane. For traveling HOM, end
  planes covering different r ranges (a wall on one end plane only) are stopped with a clear message; `solve`
  on such a mesh also explains it now. TM0 solves such a cell correctly in f and Q if the planes are split at the same r where they overlap and the part
  without a partner is all wall (it matches the same structure cut elsewhere within 9×10⁻⁵ on 1.5 mm and 6×10⁻⁶ on 0.4 mm); otherwise it warns. A line between two regions with a BC label gets a note that says what it does for the current solver: PEC or E-short is not a wall for TM0, and M-short is a wall for TM0 and has no effect for HOM.
  These analysis-only errors do not stop meshing or exporting the geometry.
- **GUI and `Project.run(mesh_file=...)`: a reused mesh kept the old materials.** Changing ε_r or tanδ leaves the mesh up to date, so the analysis reused it
  together with the old `model.materials.json`, and solved with the old values (a +22% frequency error, no warning). The materials now
  come from the current document. A mesh whose material tags are not in the document still uses the `.materials.json` next to it, and without one the run stops with a message. A region tag missing from the material table in `solve` now warns that it counts as vacuum.
- **`axicavity-fem-run` with a reused `--out` folder reused the old mesh**, so changing the geometry or the mesh
  size gave the numbers of the old mesh, with no warning. It now builds a new mesh and replaces the mesh and results of that kind in the folder,
  after checking the input, so a mistyped option or `mesh_file` does not remove them. A folder holding a `model.msh` that no run wrote stops the run instead.
  `Project.check` and `Project.run` now apply the same input checks as the GUI, so these stop instead of running: a conductivity or predicted frequency ≤ 0, β outside 0 < β ≤ 1, a negative azimuthal order, phases that are all 0, and modes < 1.
  A phase list starting with a negative value now works in `Project.run` and as `--phases=-60,30`, and an argument error marks the result as failed instead of leaving it running. A `--json` path that cannot be written gives an error line.
- **Reading meshes.** In the MSH 2.2 format with mixed labels, the node shared by two curves belonged only to the curve with the lower number, so wall edges at corners stopped being walls (Q +5–8%, and with M-short a spurious mode); line elements now add those nodes. A mesh whose order differs from `--elem-order` now gives a clear error instead of "Factor is exactly singular". Coarse traveling meshes no longer fail with a `TypeError` when the number of modes reaches the matrix size.
- **Field export at a target wall loss (`--scale-to-power`, also in the GUI's export dialog) gave far
  too large TM0 fields.** The factor was computed from P_loss, which is normalized to 1 V/m on the
  axis, and then applied to the stored eigenvector, which is not. A target of 1 W typically gave 10⁹ to
  10¹⁰ W and, over all sample geometries and modes, roughly 10⁵ to 3×10¹¹ W, for standing and traveling
  waves. Meshes without the axis were not affected. The wall integral of the exported H now equals the
  target. HOM exports use P_loss directly, so they were off only through the wall-loss error above
  (n ≥ 1): a target of 1000 W gave 957 to 1633 W for n = 1, 2 in a pillbox and a waveguide cell, up to
  1789 W for TE41, and 371 W for one n = 1 mode of a cell without the axis.
- **TM0 `post` stopped with an error on NumPy 2.4 and later**, which removed `np.trapz`. A new
  installation gets such a version. `post` now uses `np.trapezoid` where it exists (run on NumPy 2.2
  and 2.5).
- **GUI: Re-run post and Create report followed the analysis type in the panel, not the type of the
  result.** For a TM0 result with the panel set to HOM, β was ignored and report creation failed.
  Both now follow the result. They and Open report are also disabled while an external `.h5` is
  displayed; before, they acted on another result of the project without saying so. (With nothing
  displayed they still act on the selected result, or else the newest finished one.) Re-run post also
  works on a displayed result whose `post` failed; before, it acted on another result or stayed
  disabled. When `post` fails or is cancelled during an analysis, that result now becomes the displayed
  one, without changing the tab, so Re-run post acts on it.
- **GUI: a failed Re-run post left the result marked as finished**, although the values in the mode
  table were the old ones (and no longer in the `_processed.h5` file; see `post -o` above). A failed or
  cancelled Re-run post now marks the result as such, and the file and the table keep the earlier values
  (a cancel stops between stages, so they can also be the new ones); Re-run post can repair it, also
  after reopening the project.
- **GUI: selecting a result during a mesh job** left Run, Re-run post, Create report and Export field
  disabled after the job ended. These buttons are now disabled during a mesh job and come back after it.
- **GUI: closing the window during a job** recorded a crash in the result; it is now recorded as cancelled.
- **The hint for a result without `post` values** told you to use Re-run post also for an `.h5` file
  outside the project, where the button is disabled. It now points to the command line. The text of
  Force stop now says that a finished solve is kept.

### Changed

- New application icon (window, taskbar and the Windows EXE): a computed field of the S-band sample cell
  (traveling TM0, 2π/3 mode; H_φ in color, electric field lines in purple), drawn by `tools/make_app_icon.py`.
  Sizes of 32 px and below use a version with fewer, thicker field lines.

### Unchanged

- The HDF5 schema, except for the new optional group `/mesh/physical_edges` (the line elements, used for walls with 1st-order elements); and the frequencies and eigenvectors except for the HOM cells above.
- TM0 with 2nd-order elements (or no interior edge between walls) and without a dielectric on the axis: all `post` values of
  standing waves. All TM0 with such a mesh: f, Q, Q_wall, Q_diel, v_g and α. All TM0: f, Q_diel and v_g.
- HOM, for the same eigenvector: all `post` values of traveling waves at n = 0; Q_diel; U and P_diel of
  traveling waves; P_loss at n = 0; P_flow and v_g of standing waves (0), and of pure TE modes of a
  uniform hollow waveguide (E_z = 0). Hybrid modes of iris-loaded cells do change, including TE-like ones.
- The overall phase itself is still arbitrary and changes when a mode is solved again. Engineering
  quantities no longer depend on it, but these do, by a time shift (a difference between solves, not
  between versions): field maps at time phase 0°, the GUI's double-click values, instantaneous values
  and the Re / Im columns in field exports, the on-axis E_z plot in the report, and field exports of
  traveling HOM modes. With SciPy 1.18 ARPACK's start vector is random, so the phase also changes
  between runs and standing-wave fields can change sign; frequencies differ by about 10⁻¹⁴ and `post`
  values by about 10⁻⁸ or less (except P_flow, v_g and α at θ = 0° and 180°, which are rounding noise).
  With SciPy 1.16 repeated runs are identical.

### Existing results

- 1st-order results solved before 3.0.1 have no line elements, so `post` counts only the outer edges as walls and warns if an interior edge lies between wall nodes: solve again if you have an interior PEC line or a gap one element wide.
- Run the analysis again for: traveling HOM results with n ≥ 1 on cells with an iris or a coaxial inner
  conductor, or with an outer wall that is M-short or has no label; traveling HOM results at any n on cells split into regions along z or on a
  mesh finer than about 40 µm; HOM results on geometries with a corner below 90°, a gap one element wide, a cell about one element long with E-short or PEC ends, or a wall meeting
  the axis between its ends (it depends on the mesh, so if in doubt, solve again). The stored eigenvectors are the old
  ones, so `post` cannot correct them, and nothing warns when you re-run it.
- Also solve again, for TM0 as well: results where ε_r or tanδ changed after the mesh was made in the GUI (or with `mesh_file`), which used the old materials; check `/materials` in the `.h5` file or `summary.materials` in `result.json`. Results from `axicavity-fem-run` in a reused `--out` folder after changing the geometry or mesh size. Results from an external MSH 2.2 mesh with mixed labels.
- All other old files keep the old numbers until `post` is repeated. In the GUI, select the result and click
  **Re-run post**, then **Create report** (`report` reads the values stored in the file, and Open
  report shows the old report until then). The re-run uses the conductivity and β currently in the
  Analysis panel. On the command line, give the `--cond` and `--beta` of the original analysis; the
  values below are examples, and leaving them out means 5.8e7 and 1. They are recorded only in a
  project (`command.log`; `job.json` keeps only the last run, so Re-run post overwrites it), not in the
  `.h5` file:
  `axicavity-fem post --type tm0 -i model_TW_TM0.h5 -o model_TW_TM0_processed.h5 --cond 5.8e7 --beta 1`,
  or for HOM `axicavity-fem post --type hom -i model_SW_HOM.h5 -o model_SW_HOM_processed.h5 --cond 5.8e7`.
- Field files exported with a target wall loss keep the old amplitudes; export them again. For TM0 the
  old `post` result is enough for this, except with 1st-order elements and an interior edge between walls. For HOM with n ≥ 1, repeat `post` first. Exports without a
  target wall loss did not change, except at points in the elements touching the axis, r up to about one element size (HOM with n = 1: E_θ, H_z, H_r and
  the absolute values; n ≥ 2: H_z and H_r), and for results that must be solved again (solve first, then export).
- `axicavity_fem_version` in an `.h5` file records the version that ran `solve`, so it does not show
  whether `post` was repeated. Compare the `generated` time in the `.txt` file written by `post` (and
  `command.log` in a project) with the date you updated. For a standing HOM mode U also tells: about
  2.78×10⁻¹¹ J after the fix, 5.56×10⁻¹¹ J before.
- Version 2.3.0 stays available at its tag and still has all of these errors.

### Known limitations

- On curved walls the HOM wall loss converges only to first order in the mesh size, even with
  2nd-order elements, and the error of Q depends on the shape. On a 50 mm sphere (TM101, n = 0) Q was
  0.904 / 0.953 / 0.975 / 0.988 of the analytic value at h = 8 / 4 / 2 / 1 mm. On the S-band cell, n = 0
  against TM0, the three lowest modes were up to 3% above at h = 2.5 mm and 1% above at h = 0.6 mm with
  PEC-labeled ends, and 4.5% and 1.5% above with E-short ends; over 12 modes the difference reached 14%
  and 5%. The frequency is not affected
  (within 3×10⁻⁵ on the sphere). The TM0 solver is within 0.1% on the sphere meshes, so use it for n = 0.
  For n ≥ 1, check the convergence by refining the mesh. This is not new in 3.0.1.
- With 1st-order elements the fields on and near the axis are not reliable. H_r on the axis follows H_θ, which is
  constant in each element, so it does not converge with the mesh (at 1 mm typically 16% and up to 34% of the peak H;
  for TE modes this is worse than 3.0.0, for TM modes better), and just outside the zone next to the axis H_z and H_r can reach several
  times the peak H (up to 40% at r = 1 mm on a 1 mm mesh). Use 2nd-order elements (the default) for fields near the axis.
- With n ≥ 2, H_z and H_r just outside that zone are noise amplified by n/r (TE21: 30–40% of the peak H on a 2 mm mesh and 60–90%
  on 5 mm at r ≈ 0.05 h, 12–18% on 2 mm at 0.2 mm, 2–3% at 0.6 mm). The second row of a default area export can land there. On the axis, for n ≠ 1, H_θ and E_r are not forced to 0
  (about 1% of the peak on a 2 mm mesh).
- `E_abs` and `H_abs` in exports are the root sum of squares of the real parts of the core variables at φ = 0, not the physical
  magnitude: with n ≥ 1 they mix components that peak at different φ (at the peak of a TE111 mode they are √2 times the 3D view's |E| and |H|), and in a traveling wave they change with the time
  phase. Read peak fields from the 3D view or from the largest component.
- A partial wall on a periodic end plane cannot be represented (see the check above), and the check cannot see it
  when the wall and the opening have the same label. Traveling HOM needs end planes with the same r range. On a TM0 cell with a wall on one end plane, f and Q
  are correct but v_g and α are 1–2% off and do not converge, because the plane cuts through the wall's corner; use a cut that keeps the wall off the plane, or L dω/dθ.
- TM0 cannot represent a PEC line inside the domain: H_φ passes through it, so the frequencies are those without it. Use HOM with n = 0, which treats it as a wall.
- HOM can also miss a mode below its automatic shift (a quarter-wave resonator, a cavity with much dielectric). Only
  TM0 counts missed modes. The automatic shift of HOM is c/(2 r_max) (TM0: c/(4 r_max)) and HOM returns the modes nearest it, so with few modes requested the lowest can be left out (n = 0 of `samples/acc2_pipe_flat`: one mode gives 2.90 GHz, five give 2.76 GHz). Ask for more modes or give a predicted frequency. HOM standing waves also discard any mode below c/(20 r_max), and traveling waves below
  c/(2√1000 r_max), as spurious, so there a predicted frequency does not help (a re-entrant cavity with a 0.2 mm gap, 0.30 GHz at r_max = 40 mm). Use TM0 for such n = 0 modes.
- TM0 v_g and α are unreliable when a periodic plane passes through the sharp tip of a wall, on both planes as well as on one: a knife-edge iris (5°) was −16 to −32% and a 0.04 mm foil −16 to −18% on 1–2.5 mm meshes, shrinking only as about h^0.5, while f and Q agree with the same structure cut at mid cell within 3×10⁻⁴ and 1×10⁻³. A rounded iris converges (S-band cell +2.1% at 5 mm, +0.1% at 0.3 mm). Cut the cell so that walls stay off the plane, use L dω/dθ, or HOM with n = 0.
- HOM Q with n ≥ 1 on a very thin foil does not settle with the mesh (6–9% between cuts at h = 0.5–1 mm, for a 0.04 mm foil), probably because of the sharp tip.
- With a mesh much larger than the cell length (h/L of several times), a traveling cell can return values that are not modes (below the light line βc/2π). Refine the mesh.
- The check before a run and `solve` warn about end planes only as far as the labels and nodes show. The wall drawn in the mesh preview and in field maps follows the node labels, so an edge between two PEC walls across one element is drawn as PEC even if it is not one. Unknown boundary names in an external mesh are ignored without a warning, and with no PEC at all Q is 0.
- With a thin inner conductor on a coarse mesh (radius below about the element size), HOM wall loss is too high because rE_θ/r varies strongly inside the
  elements on the conductor: Q of TE11 is −49%, −17% and −4.5% at h = 5, 2 and 1 mm for a 2 mm radius in a 40 mm cell, and −1.4% at h = 2 mm for 5 mm. Check the convergence.
- The voltage integrals use a fixed scan grid that is not split at a dielectric interface on the axis: under 0.5%
  for ordinary materials, and more for a high ε_r with a thin gap (ε_r = 100 with a 0.5 mm gap: −0.9%).
- A point on the axis of an external, very anisotropic mesh may not be found, and then E_z there counts as 0. Meshes made
  by the GUI do not have this.
- TM0 treats nodes within 1 µm of r = 0 as on the axis, so a mesh finer than about 3 µm gets wrong frequencies (+9.5×10⁻⁴ for R = 40 µm, h = 1 µm). HOM is not affected. TM0 also uses
  a fixed 10 nm tolerance on the periodic planes.
- Phases closer than 0.1° in one run cannot be saved (the run fails at the end).
- At θ = 0° and 180° P_flow is zero up to rounding, so v_g is about 0 and α has no meaning: its value
  is rounding noise (up to about 10¹⁵ /m). This already held for TM0 and n = 0 and now also holds for
  n ≥ 1. For a backward wave (v_g < 0) α is negative by its definition, α = P_loss/(2 P_flow L).

This is the first intended change to the solver core since 2.3.0; `tests/core_files.sha256` was
regenerated.

Regression tests: `tests/test_tm0_traveling_phase_invariance.py`, `tests/test_hom_post_analytic.py`,
`tests/test_hom_periodic_wall.py`, `tests/test_export_scale_to_power.py`, `tests/test_cmd_post_output.py`,
`tests/test_fem_tm0_dielectric.py`, `tests/test_fem_hom_dielectric.py`, `tests/test_tm0_low_modes.py`,
`tests/test_numpy_compat.py`.

## [3.0.0] — 2026-10-03 — new GUI and a Windows executable

Version 3 replaces the wxPython GUI with a new PySide6 interface and adds a Windows
executable. The solver core is the same code as 2.3.0 (only `__version__` changed), so
results are identical for the same mesh.

### Added

- **Ribbon GUI** (PySide6) organised as Modeling → Physics → Mesh → Analysis → Results → View,
  in Japanese and English.
- **Parametric sketcher** with geometric constraints and driving dimensions, solved by
  planegcs: coincident, horizontal, vertical, parallel, perpendicular, tangent, equal,
  concentric, fix, symmetric and point-on-curve; distance, point–line distance,
  circumference–line distance, radius and angle dimensions. A parameter table with
  expressions drives dimensions and point coordinates. The origin and the z / r axes are
  reference geometry; points drawn on an axis are constrained to it.
- The editing operations of the version 2 Multi-Region Editor: insert / delete a point with
  reconnection, line ⇄ arc (with flipping), close the outline, fillets, and automatic
  splitting at intersections.
- **Automatic region detection** and **automatic boundary conditions** (the axis and internal
  interfaces become `None`, walls `PEC`), overridable per curve; per-region materials with
  expressions for ε<sub>r</sub> and tanδ; geometry validation with warnings.
- **Projects** (`.axiproj` with a `.axiproj.data/` folder) holding the geometry, the settings,
  the meshes and the full history of results; older results are marked when the geometry or
  settings changed since. Import / export of version 2 `.gmshproj` and Superfish `.af`;
  export of `.msh` with `.materials.json`, Gmsh `.geo` and a self-contained Python script.
- Meshing and analyses run in **child processes** with cancel and force stop. A confirmation
  dialog shows the exact command lines, and `command.log` keeps them for re-running.
- **Results tab**: field maps, a mode table with every engineering quantity for the selected
  phase, field values by double-click, PNG, traveling-wave GIFs, field data export, and
  opening result files from outside a project.
- **3-D view** (pyvista, `viz3d` extra): the revolved cavity with wall, meridian planes,
  cross-sections and a cutaway, arrows on a Cartesian grid, TM0 field lines repeated in φ,
  HOM cos(nφ) patterns, time-phase animation, PNG and GIF.
- **Batch runs**: `axicavity-fem-run` and the Python API `axicavity_fem.gui.batch.Project`
  re-analyse a project with changed parameters (the sketch is re-solved), for parameter
  scans and frequency tuning.
- **Windows executable** `AxiCavity-FEM.exe` (Nuitka): the GUI and the command line in one
  program, with Intel MKL PARDISO. Meshing runs in a bundled separate program (`mesher`,
  containing Gmsh, distributed under the GPL with its source); the main program reads `.msh`
  files with a pure-Python reader that reproduces Gmsh's node and element ordering.

### Changed

- The wxPython GUI and the wxGlade files are removed. `axicavity-fem-gui` starts the new
  GUI; the command-line tool `axicavity-fem` is unchanged.
- Geometry lives in `.axiproj` projects; `.gmshproj` is an import / export format.
- `samples/` holds input files only; meshes and results are produced on demand.
- pyvista is optional: without it the GUI starts and the 3-D button is disabled.

### Fixed (issues of the version 2 GUI)

- Exit did nothing; it now asks to save unsaved work.
- Ctrl+A was bound to Superfish export; it now selects all.
- *New* left the previous variables behind.
- Changing units only relabelled the values; values can now be converted.
- Region loops were not validated before meshing.
- The `.geo` export ignored the element order, and the mesh order and the solver's element
  order could disagree; there is now a single setting.
- Reports and `command.log` were looked for in guessed places.
- Traveling-wave results showed the post-processed values of the first phase only.
- R/Q and V_eff were missing from the result table; E-wall was offered for HOM results.
- GIF export could not be cancelled; field export lacked instantaneous / scaled values.

Regression tests: `tests/test_known_bugs_v23.py`.

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

Version 1 lives on in this repository at tag
[`v1.0`](https://github.com/TakuyaNatsui/AxiCavity-FEM/releases/tag/v1.0).
