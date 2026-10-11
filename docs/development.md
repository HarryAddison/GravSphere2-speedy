# Development

```bash
pip install -e ".[dev,docs]"
```

## Tests

```bash
pytest                      # whole suite (about 20 s)
pytest tests/test_jeans.py  # one file
pytest --cov=gravsphere2    # coverage report (with pytest-cov)
```

The tests are written with `unittest`, so `python -m unittest discover -s tests -t .` works without
pytest as well. They cover

* analytic limits of every model (Plummer limits, enclosed mass versus numerical integrals of the density,
  NFW and coreNFWtides limits, isotropic Plummer Jeans solution compared with its closed form,
  normalisation and moments of the velocity PDFs),
* input-file loading and validation, data readers and their error messages,
* the prior space and parameter vector, the likelihood (rejected models, escape-velocity check,
  stellar-mass path, picklability for `multiprocessing`) and **regression values from the original code**,
* the sampler wiring (with a small stand-in for dynesty, so no sampling is needed), the command-line
  interface, post-processing and plotting.

Add tests next to the module you change; helpers for building synthetic data and configurations are in
`tests/helpers.py`.

## Documentation

```bash
mkdocs serve      # live preview at http://127.0.0.1:8000
mkdocs build      # static site in site/
```

`tests/test_docs.py` checks that every input-file key appears in `docs/input-file.md`, so the reference
stays complete when options are added.

## Adding an input-file option

1. Add the field (with a default and docstring) to the right dataclass in `gravsphere2/config.py`
   and validate it in `Config.validate`.
2. Use it in the code (`likelihood.py`, `priors.py`, `sampler.py`, …).
3. Document it in `docs/input-file.md` and `examples/template_full.yaml`, and add a test.

## Using a different mass model or priors

The model is assembled in `likelihood.py` (`TotalMass`, `GravSphere2Model.build_profiles`) and the parameter
vector in `priors.py`. A new halo profile needs a density/mass function pair in `gravsphere2/models/mass.py`,
a few extra entries in `build_prior_space` and a branch in `GravSphere2Model.unpack`.

## Code style

Docstrings use NumPy style. Line length is about 100 characters. Public functions have descriptive snake_case
names; units are documented in docstrings (kpc, Msun, km/s unless stated otherwise).
