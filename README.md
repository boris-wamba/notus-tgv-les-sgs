# Notus – DNS and LES of the 3D Taylor–Green vortex

Results and post-processing tools from my Master 2 research internship at I2M Bordeaux (SIMFI team,
17 March – 30 September 2025), supervised by Jérôme Jansen and Stéphane Glockner. The internship evaluated the
open-source CFD code [Notus](https://doc.notus-cfd.org) on the 3D Taylor–Green vortex (TGV) benchmark, in DNS and LES.
The Notus source code is not included in this repository.

## What was done

- **DNS mesh convergence** at Re = 1600 on 64³ to 512³ grids with second-order centred schemes, BDF2, CFL = 0.3,
  compared with the spectral reference of Rees et al. (2010): kinetic-energy dissipation rate and 3D energy spectra.
  A 512³ grid is needed to approach the reference.
- **LES with two subgrid-scale models**: Smagorinsky (C_s = 0.1) and mixed-scale model (C_M = 0.06), grids from 32³ to 256³.
  I implemented the Smagorinsky model in Notus. Only the mixed-scale model was available at the start of the internship.
  At 128³ the mixed-scale model predicts the dissipation peak better than Smagorinsky.
- **Numerical-scheme study**: 6 advection/diffusion combinations (centred order 2 and 4, HOUC3) at 128³, Re = 400 and 1600.
  Second-order centred schemes were the most accurate in the tested configuration.
- **Filters**: trapezoid and Simpson top-hat filters of width 2Δx, 4Δx and 8Δx (1D/2D/3D) for the mixed-scale model's test filter
  and for a constant filter width under grid refinement, to separate modelling from discretisation error.
- **Resolved vs. modelled dissipation**: subgrid viscosity compared with an "ideal" value from a filtered DNS.

## The `tgvtools` package

```bash
pip install -e .            # numpy, scipy, matplotlib, sympy
pip install -e ".[adios]"   # optional: read Notus .bp files
pytest                      # 26 tests
```

| Module | Content |
|---|---|
| `tgvtools.filters` | Top-hat filter kernels of trapezoid / Simpson, width 2Δx, 4Δx, 8Δx…, vectorised periodic filtering in 1D–3D, transfer functions, the Notus `[1, c, 1]` high-pass filter |
| `tgvtools.filters.symbolic` | SymPy generator of the discrete stencils: `python -m tgvtools.filters.symbolic --kind simpson --width 8 --ndim 3` (prefactors 1/24, 1/576, 1/13824) |
| `tgvtools.spectra` | Shell-averaged 3D energy spectrum (Parseval-consistent) and a vectorised version of the legacy spectrum used in the report |
| `tgvtools.sgs` | Ideal subgrid viscosity ν(ε/ε̄ − 1), exact subgrid dissipation, peak error |
| `tgvtools.io` | Column layouts of Notus output files, time-series sorting, ADIOS reader |

The tests check, among others: Simpson exact on cubic polynomials, measured convergence order of the filters
(2 for the trapezoid rule, 4 for Simpson), equality of the vectorised spectrum with the original triple loop, and Parseval's identity.

## Command-line scripts (`scripts/`)

| Script | Replaces | Use |
|---|---|---|
| `plot_series.py` | `plot_MeanKEDR.py` | Dissipation rate vs. time for any set of runs and a reference curve |
| `plot_sgs_viscosity.py` | `plot-sgs-viscosity.py`, `nu_exact_N256fdns.py` | Modelled ν_sgs or production vs. the ideal value from a filtered DNS |
| `spectrum3d.py` | `spectrum3D_Re1600.py`, `routines.py` | 3D energy spectra from `.bp` files |
| `prepare_data.py` | `sorted_data.py`, `convert_comma_to_point.py` | Sort restarted time series, convert decimal commas |
| `filter_demo.py` | `test_filter.py` | Compare filter transfer functions on a noisy field |

No path is hard-coded: file locations are passed on the command line.

## Reproduce (outline)

1. Run the TGV case with `config/taylor_green_vortex_3D.nts`; extract the logs with `config/get_data_from_log_TGV.sh`
   (`notus_log_pp_TGV.awk`).
2. `python scripts/prepare_data.py sort <file>` for restarted runs.
3. `python scripts/plot_series.py …` for the dissipation curves, `python scripts/spectrum3d.py …` for spectra,
   `python scripts/plot_sgs_viscosity.py …` for the subgrid-viscosity comparison.

## Limitations

Only the TGV case was studied. Higher-order schemes (HOUC3, order 4) were under development in Notus at the time and not fully
validated. Channel flow and compressible cases were left as perspectives. The filtered-DNS reference uses a 256³ simulation instead of the planned 512³.

## Author

Boris Keubou Wamba

## License

Code in `src/`, `scripts/` and `tests/`: MIT (see `LICENSE`). `config/taylor_green_vortex_3D.nts` is an input
file for Notus (CeCILL license, © Bordeaux INP, Université de Bordeaux, CNRS); the report is © the author.
