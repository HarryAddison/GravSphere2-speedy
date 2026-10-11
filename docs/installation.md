# Installation

GravSphere2 requires Python 3.9 or newer. The original release was verified with Python 3.10,
NumPy 1.26, SciPy 1.15, dynesty 2.1.5, matplotlib 3.9 and corner 2.2.

```bash
git clone https://github.com/dadams42/GravSphere2.git
cd GravSphere2
python3 -m venv .venv && source .venv/bin/activate     # recommended
pip install -e .                  # core: numpy, scipy, pyyaml, dynesty
pip install -e ".[plots]"         # + matplotlib and corner for the plots
pip install -e ".[dev]"           # + pytest for running the tests
pip install -e ".[docs]"          # + mkdocs to build this documentation
```

After installation the `gravsphere2` command is available, and the package can also be run as
`python3 -m gravsphere2`:

```bash
gravsphere2 --version
python3 -m gravsphere2 --help
```

!!! note "Why not `python3 gravsphere2 -i file.yaml`?"
    Running `python3 gravsphere2` would execute a *directory* called `gravsphere2` in the
    current folder. An installed package keeps its code in `src/gravsphere2/`, so use
    either `gravsphere2 -i file.yaml` or `python3 -m gravsphere2 -i file.yaml`; both are
    identical.

## Dependencies

| Package    | Needed for                                     | Required |
| ---------- | ---------------------------------------------- | -------- |
| numpy, scipy | all numerics                                 | yes      |
| pyyaml     | reading the input file                         | yes      |
| dynesty    | nested sampling (`gravsphere2 -i …` without `--dry-run`/`--plots-only`) | yes |
| matplotlib | plots                                          | optional |
| corner     | corner plot of the posterior                   | optional |
