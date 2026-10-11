# GravSphere2

GravSphere2 is a solver of the spherical **higher-order Jeans equations** for mass-modelling
systems of stellar tracers in equilibrium, such as dwarf galaxies and star clusters. It improves
on earlier GravSphere versions by

* removing the need for **binning** (individual stellar positions and velocities are fitted),
* using self-consistent **non-Gaussian velocity distributions** (Sanders & Evans 2020), and
* a fully general treatment of higher moments through the **fourth-order Jeans equations**.

Line-of-sight velocities and, optionally, proper motions are modelled jointly. The dark-matter
halo is a *coreNFWtides* profile, optionally with a stellar mass component that follows the tracer
light. Parameters are sampled with dynamic nested sampling ([dynesty](https://dynesty.readthedocs.io/)).

This version of the code is a refactoring of the original scripts into an installable Python
package with a command-line interface, a YAML input file, unit tests and documentation. The
numerical results are unchanged; see [Migration from the scripts](migration.md).

## Where to go next

| I want to…                                  | Read                                   |
| ------------------------------------------- | -------------------------------------- |
| install the package                         | [Installation](installation.md)        |
| run an example in five minutes              | [Quick start](quickstart.md)           |
| set up my own galaxy                        | [Input file reference](input-file.md) and [Data formats](data-formats.md) |
| understand the model and the sampler vector | [Method](method.md)                    |
| know what a run produces                    | [Outputs](outputs.md)                  |
| use the code from Python / a notebook       | [Python API](python-api.md)            |
| move from the old `initialise_*.py` files   | [Migration](migration.md)              |
| contribute or run the tests                 | [Development](development.md)          |

## Citing

If you use this code please cite the release paper
[Bañares-Hernández et al. (2025)](https://ui.adsabs.harvard.edu/abs/2025arXiv250924103B/abstract)
and the GitHub page <https://github.com/dadams42/GravSphere2>. Also cite:

* earlier GravSphere papers (ADS):
  [2017MNRAS.471.4541R](https://ui.adsabs.harvard.edu/abs/2017MNRAS.471.4541R/abstract),
  [2018MNRAS.481..860R](https://ui.adsabs.harvard.edu/abs/2018MNRAS.481..860R/abstract),
  [2020MNRAS.498..144G](https://ui.adsabs.harvard.edu/abs/2020MNRAS.498..144G/abstract),
  [2021arXiv210211890C](https://ui.adsabs.harvard.edu/abs/2021arXiv210211890C/abstract),
  [2025A&A...693A.104B](https://ui.adsabs.harvard.edu/abs/2025A%26A...693A.104B/);
* the velocity PDF modelling:
  [Sanders & Evans 2020](https://ui.adsabs.harvard.edu/abs/2020MNRAS.499.5806S/abstract)
  (code: <https://github.com/jls713/gh_alternative>);
* the sampler, dynesty:
  [2020MNRAS.493.3132S](https://ui.adsabs.harvard.edu/abs/2020MNRAS.493.3132S/abstract); dynesty's own
  [reference list](https://dynesty.readthedocs.io/en/latest/references.html) gives the related papers;
* the J-factor integral, if used:
  [2020JCAP...09..004A](https://ui.adsabs.harvard.edu/abs/2020JCAP...09..004A/abstract);
* the Gaia Challenge mock data, if used:
  [2021MNRAS.501..978R](https://ui.adsabs.harvard.edu/abs/2021MNRAS.501..978R/abstract),
  [2011ApJ...742...20W](https://ui.adsabs.harvard.edu/abs/2011ApJ...742...20W/abstract) and
  the [Gaia Challenge wiki](http://astrowiki.ph.surrey.ac.uk/dokuwiki/doku.php?id=start);
* the simulated dwarf galaxy data, if used:
  [2025arXiv250418617T](https://ui.adsabs.harvard.edu/abs/2025arXiv250418617T/abstract).

Former versions of GravSphere (binned, with virial shape parameters) are
[GravSphere + binulator](https://github.com/justinread/gravsphere) and
[pyGravSphere](https://github.com/AnnaGenina/pyGravSphere).
