# Changelog

## 2.0.0 — refactored package

* Installable Python package (`pip install -e .`) with `src/` layout, `pyproject.toml` and a
  `gravsphere2` console script (also `python -m gravsphere2`).
* All user settings moved from `initialise_*.py` into a validated YAML input file
  (`gravsphere2 -i input.yaml`), with command-line overrides for output directory, `nlive`, `ncpus`, seed,
  `--resume`, `--dry-run`, `--plots-only`.
* Code split into modules (config, data, priors, likelihood, Jeans solver, velocity PDFs, models, sampler,
  analysis, plotting) with classes replacing module-level globals and `import *`.
* Vectorised Jeans solver and tracer projection (about 10x faster likelihood, identical values).
* 140+ unit tests, including regression values from the original implementation.
* Documentation (MkDocs) and a migration guide from the original scripts.
* Fixed `np.float`, missing `derivative`/`sys` imports, `escv` default mass function, N-d
  `calcmedquartnine`, global `multimode` and global warning suppression (see `docs/migration.md`).

## 1.0 — original release

Scripts `gravsphere2.py`, `gs2_functions.py`, `initialise_*.py`, example notebooks.
