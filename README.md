# GravSphere2

GravSphere2 is a solver of the spherical higher-order Jeans equations, designed to mass-model systems of
stellar tracers in equilibrium, such as galaxies and star clusters, with minimal bias and maximal information
content. It improves on previous versions of GravSphere and other methods by removing the need for binning, by
using self-consistent non-Gaussian velocity distribution modelling and by a fully general treatment of higher
moments, solving the fourth-order Jeans equations.

This repository is a refactoring of the original release into an installable Python package with a command-line
interface, a YAML input file, unit tests and documentation. Numerical results are unchanged
(see [docs/migration.md](docs/migration.md)).

## Install

```bash
git clone https://github.com/dadams42/GravSphere2.git && cd GravSphere2
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[plots]"          # numpy, scipy, pyyaml, dynesty (+ matplotlib, corner)
```

Python 3.9+ is required. Use `".[dev]"` for the tests and `".[docs]"` for the documentation tools.

## Run

```bash
gravsphere2 -i examples/gaia_challenge_plumcusp.yaml --dry-run     # check data/setup in seconds
gravsphere2 -i examples/gaia_challenge_plumcusp.yaml --ncpus 8     # run the sampler
gravsphere2 -i examples/gaia_challenge_plumcusp.yaml --plots-only  # re-make the plots
python3 -m gravsphere2 -i examples/fornax_simulation.yaml          # equivalent module form
```

| Option | Meaning |
| ------ | ------- |
| `-i FILE` | YAML input file (required) |
| `-o DIR` | override the output directory |
| `--nlive N`, `--ncpus N`, `--seed N` | override the sampler settings |
| `--resume` | continue from the checkpoint file |
| `--dry-run` | validate, load data and evaluate the likelihood once |
| `--plots-only` | plots from the saved samples of a finished run |
| `--no-plots`, `-v`, `--version` | skip plots, debug logging, version |

## Input file

Everything the user can set (data files, priors, radial grids, stellar mass, sampler options, output) is in one
YAML file. See `examples/` for the three examples of the release paper (Gaia Challenge cusp and core mocks and a
Fornax-like simulation), `examples/template_full.yaml` for every option with its default, and
[docs/input-file.md](docs/input-file.md) for the reference.

```yaml
name: my_dwarf
data:
  directory: ../data/my_dwarf
  kinematics_file: stars.txt        # rows: R, vLOS, err, vPMt, err, vPMR, err
  proper_motions: true
  photometry: {mode: individual, file: positions.txt}
  half_light_radius: median
stellar_mass: {value: 4.3e7}
sampler: {nlive: 500, ncpus: 8}
```

## Repository layout

```text
src/gravsphere2/   the package (see docs/method.md for a module overview)
tests/             unit tests (pytest / unittest)
examples/          input files, template, Python API example
data/              Gaia Challenge mocks and simulated Fornax data
docs/, mkdocs.yml  documentation (mkdocs serve)
```

## Tests and docs

```bash
pytest                 # or: python -m unittest discover -s tests -t .
mkdocs serve           # needs the [docs] extra
```

## Citations

If you use this code, please acknowledge it and cite the release paper
[Bañares-Hernández et al. (2025)](https://ui.adsabs.harvard.edu/abs/2025arXiv250924103B/abstract) and the
[GitHub page](https://github.com/dadams42/GravSphere2). The complete list of papers to cite (previous GravSphere
papers, the velocity-PDF method, dynesty, the J-factor integral and the example data) is in
[docs/index.md](docs/index.md#citing).

## Notes

The example notebooks of the original release (`run_example_notebook.ipynb`,
`plots_example_notebook_vf.ipynb`) are replaced by the command line and by `gravsphere2.plotting` /
`gravsphere2.analysis`. Former versions of GravSphere:
[GravSphere + binulator](https://github.com/justinread/gravsphere) and
[pyGravSphere](https://github.com/AnnaGenina/pyGravSphere).

Licensed under the MIT license (see `LICENSE`). Bugs and questions are welcome.

Andrés Bañares Hernández, Justin Read & Mariana P. Júlio
