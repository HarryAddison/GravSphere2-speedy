# Quick start

Three examples from the release paper ship in `examples/`, with their data in `data/`:

| Input file                               | System                                                         |
| ---------------------------------------- | -------------------------------------------------------------- |
| `gaia_challenge_plumcusp.yaml`           | Gaia Challenge mock, cuspy halo, 1000 tracers (LOS + PMs), binned photometry |
| `gaia_challenge_plumcore.yaml`           | Gaia Challenge mock, cored halo                                 |
| `fornax_simulation.yaml`                 | simulated Fornax-like dwarf, 100 tracers (LOS + PMs), bin-free photometry, free stellar mass |

## 1. Check the setup (seconds)

```bash
gravsphere2 -i examples/gaia_challenge_plumcusp.yaml --dry-run
```

This validates the input file, loads the data, builds the model and evaluates the likelihood once:

```text
run name            : gaia_challenge_plumcusp
tracers (kinematic) : 1000  (proper motions: True)
photometry          : binned profile (99 entries)
half-light radius   : 0.116023 kpc
Jeans grid          : 0.0004773 - 11.6 kpc
free parameters     : 19
ln L at prior centre: -11121.4  (11.0 ms per evaluation)
```

## 2. Run the sampler

```bash
gravsphere2 -i examples/gaia_challenge_plumcusp.yaml --ncpus 8
```

A full run takes from hours to a couple of days depending on the number of tracers, `nlive` and
CPUs. For a quick trial reduce `--nlive` (for example `--nlive 100`) or set `sampler.maxiter`.
Progress is printed by dynesty and a checkpoint is written regularly.

Useful options (see `gravsphere2 --help`):

| Option                 | Meaning                                                              |
| ---------------------- | -------------------------------------------------------------------- |
| `-i FILE`              | YAML input file (required)                                           |
| `-o DIR`               | override `output.directory`                                          |
| `--nlive N`, `--ncpus N`, `--seed N` | override the matching `sampler` entries               |
| `--resume`             | continue an interrupted run from its checkpoint                      |
| `--dry-run`            | validate and evaluate the likelihood once, then exit                 |
| `--plots-only`         | only make plots from the saved samples of a finished run             |
| `--no-plots`           | skip plotting after sampling                                         |
| `-v`                   | debug logging                                                        |

Exit status is `0` on success, `2` for input-file or data errors and `1` for sampler errors.

## 3. Inspect the results

Results are written to `<output.directory>/<name>/`, see [Outputs](outputs.md). Re-create the plots
at any time with

```bash
gravsphere2 -i examples/gaia_challenge_plumcusp.yaml --plots-only
```

## Your own system

Copy `examples/template_full.yaml`, point it at your data files and edit the priors, as explained in
[Input file reference](input-file.md) and [Data formats](data-formats.md).
