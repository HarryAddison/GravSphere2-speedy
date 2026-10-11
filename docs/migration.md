# Migration from the scripts

The refactoring keeps the numerics of the original release. For the three shipped examples the
log-likelihood of the new package equals the original `lnprob` to machine precision (tested on random
parameter vectors including models that are rejected), and all helper functions were compared one by one with the
originals. `tests/test_likelihood.py::test_matches_original_implementation` encodes reference values from the original code.

## Workflow

| Before                                                                   | Now |
| ------------------------------------------------------------------------ | --- |
| Edit `initialise_<Galaxy>.py` (Python, executed on import)               | Write a YAML input file (see `examples/`) |
| Edit the `from initialise_X import *` line in `gravsphere2.py`           | `gravsphere2 -i input.yaml` |
| `run_example_notebook.ipynb` / `run_multcpu_example.py`                  | `gravsphere2 -i input.yaml --ncpus N` (or `--resume`) |
| `plots_example_notebook_vf.ipynb`                                        | `gravsphere2 -i input.yaml --plots-only` (`gravsphere2.plotting` / `analysis`) |
| `Data/Gaia Challenge/…`, `Data/Fornax Simulation/…`                      | `data/gaia_challenge/…`, `data/fornax_simulation/…` (no spaces) |

Settings in the old initialise files map to the YAML as

| `initialise_*.py`                              | YAML key |
| ---------------------------------------------- | -------- |
| `dirf`, `vdata`                                | `data.directory`, `data.kinematics_file` |
| `propermotion`                                 | `data.proper_motions` |
| `individual`, `R` / `surfden`, `surfdenerr`    | `data.photometry.mode`, `.file`, `.source` |
| `Rhalf`                                        | `data.half_light_radius` (`median` for `np.median(R)`) |
| `esc_check`                                    | `data.escape_velocity_check` |
| `rmin`, `rmax`, `rcn`                          | `grids.rmin`, `grids.rmax`, `grids.nonnegative_radius` |
| `barrad_min/max`, `bar_pnts`                   | `grids.tracer_rmin/tracer_rmax`, `grids.tracer_points` |
| `Mstar`, `Mstar_err`                           | `stellar_mass.value`, `stellar_mass.fractional_error` |
| `pfits`, `tracertol`                           | `priors.tracer.reference_parameters`, `.tolerance` |
| `betr0min/max`, `betnmin/max`, `bet0…`, `betinf…` | `priors.anisotropy.log10_r_beta`, `.eta`, `.beta0`, `.beta_inf` |
| `logM200low/high`, `clow/chigh`, `rclow/rchigh`, `nlow/nhigh`, `rtlow/rthigh`, `dellow/delhigh` | `priors.mass.log10_M200`, `.c200`, `.rc`, `.n`, `.rt`, `.delta` |
| `diro`, `name`                                 | `output.directory`, `name` |
| `nlive`, `ncpus`, `fsname`, `use_stop`         | `sampler.nlive`, `.ncpus`, `.checkpoint_name`, `.use_stop` |

`baryonmass_follows_tracer = 'yes'` is no longer an option: the baryonic mass always follows the tracer profile
(the original code did not use the flag either).

## Function names

| Original (`gs2_functions.py`)                        | New |
| ---------------------------------------------------- | --- |
| `sigp_prop_k`, `sigp_k`                              | `jeans.solve_jeans(…, proper_motions=True/False)` |
| `lnlike`, `lnlike_k`, `lnlike_prop_k`, `lnprob`, `ptform` | `GravSphere2Model.log_likelihood`, `.prior_transform` |
| `beta`, `dbeta`, `betaf`                             | `models.anisotropy.beta`, `dbeta_dr`, `integrating_factor` |
| `osipkov`, `gfunc_osipkov`                           | `models.anisotropy.osipkov`, `osipkov_integrating_factor` |
| `corenfw_tides_den/mass/dlnrhodlnr`                  | `models.mass.corenfw_tides_density/mass/dlnrho_dlnr` |
| `escv`                                               | `models.mass.escape_velocity` |
| `alpbetgamden/mass/surf/dlnrhodlnr`                  | `models.tracers.alpbetgam_density/mass/surface/dlnrho_dlnr` |
| `multiplumden/surf/mass/dlnrhodlnr`, `multimode`     | `models.tracers.multiplum_*` with `mode=` argument |
| `lukpdf`, `llkpdf`, `lnpdfj`                         | `velocity_pdf.ln_uniform_kernel_pdf`, `ln_laplace_kernel_pdf`, `ln_velocity_pdf` |
| `velpdf*`, `vztwo_calc`, `kurt_calc`, …              | `generalised_gaussian` module |
| `binthedata`, `calcmedquartnine`, `Sig_addpnts`      | `stats.bin_data`, `confidence_intervals`, `add_surface_density_anchor_points` |
| `cosmo_cfunc`, `cosmo_cfunc_WDM`                     | `cosmology.concentration_cdm`, `concentration_wdm` |
| `get_J`, `get_D`, `sidm_novel`                       | `dark_matter.j_factor`, `d_factor`, `sidm_cross_section` |
| `alpbetgamvsp`, `richfair_vsp`, `vsppdf_calc`, `vsp_pdf` | `vsp` module |
| `constants.py` names (`G`, `Msun`, `kpc`, …)         | `constants.G`, `MSUN`, `KPC`, … (upper-case) |

## Changes and fixes

* **Speed**: the Jeans solver and the tracer surface-density projection are vectorised over radius. On the
  shipped examples a likelihood evaluation drops from about 0.12 s to about 0.011 s with identical values.
* No `from … import *`, module-level state or work at import time (the old initialise files created directories
  when imported).
* Fixed: `np.float` (removed from NumPy 1.24) in the Monte-Carlo integrator; `nfw_dlnrhodlnr` called the
  commented-out `scipy.misc.derivative` (removed from SciPy); `get_J` / `get_D` called `sys.exit` without
  importing `sys` (they now raise `FloatingPointError`), and their integrands return scalars as newer SciPy
  requires; `escv` could not be called with its default mass function.
* `calcmedquartnine` (now `confidence_intervals`) gives correct results for multi-dimensional input, not only 1D.
* The global `multimode` switch of the multi-Plummer profiles is a function argument.
* The original `warnings.simplefilter("ignore")` is gone: floating-point warnings are silenced only inside the
  likelihood, where overflows in extreme prior corners are expected and map to `-inf`.
* Plots no longer need LaTeX by default (`output.use_latex`).
* Credible bands use `numpy.percentile`; the original notebooks sorted samples and took `int(frac × n)`,
  so band edges can differ in the last digits.
