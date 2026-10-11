"""Shared helpers for the test-suite: synthetic data and configurations."""

import tempfile
from pathlib import Path

import numpy as np

from gravsphere2.config import Config

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples"


def write_synthetic_data(directory, n=60, seed=0, proper_motions=True, individual=False):
    """Write a small synthetic data set; return the config dictionary for it."""
    directory = Path(directory)
    rng = np.random.default_rng(seed)
    r = np.sort(rng.uniform(0.05, 1.5, n))
    v = rng.normal(0, 8.0, n)
    err = np.full(n, 1.0)
    if proper_motions:
        rows = [r, v, err, rng.normal(0, 8, n), err, rng.normal(0, 8, n), err]
    else:
        rows = [r, v, err]
    np.savetxt(directory / "kin.txt", np.array(rows))

    if individual:
        np.savetxt(directory / "phot.txt", np.sort(rng.uniform(0.02, 2.0, 400)))
        phot = {"mode": "individual", "file": "phot.txt"}
        rhalf = "median"
        tracer = {}
    else:
        rb = np.logspace(-1.3, 0.4, 15)
        sd = 100.0 * (1 + (rb / 0.3) ** 2) ** -2
        np.savetxt(directory / "phot.txt", np.array([rb, sd / sd.max(), 0.05 * sd / sd.max()]))
        phot = {"mode": "binned", "file": "phot.txt"}
        rhalf = 0.3
        tracer = {"reference_parameters": [1.0, 0.3, 2, 5, 0.1], "tolerance": 0.5}

    return {
        "name": "synthetic",
        "output": {"directory": str(directory / "out"), "make_plots": False},
        "data": {
            "directory": str(directory),
            "kinematics_file": "kin.txt",
            "proper_motions": proper_motions,
            "photometry": phot,
            "half_light_radius": rhalf,
        },
        "priors": {"tracer": tracer},
        "sampler": {"nlive": 20, "ncpus": 1},
    }


def make_config(directory, **kwargs):
    return Config.from_dict(write_synthetic_data(directory, **kwargs), base_dir=directory)


class TempDirMixin:
    """unittest mixin providing ``self.tmp`` (a pathlib.Path)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        super().setUp()

    def tearDown(self):
        self._tmp.cleanup()
        super().tearDown()
