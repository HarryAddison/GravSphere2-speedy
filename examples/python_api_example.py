"""Using GravSphere2 as a library: evaluate the likelihood and plot profiles.

Run from the repository root:  python examples/python_api_example.py
"""

from pathlib import Path

import numpy as np

from gravsphere2 import Config, GravSphere2Model, load_dataset
from gravsphere2 import analysis

cfg = Config.load(Path(__file__).with_name("gaia_challenge_plumcusp.yaml"))
dataset = load_dataset(cfg)
model = GravSphere2Model(cfg, dataset)

print(f"{model.ndims} free parameters:")
for name, lo, hi in zip(model.priors.names, model.priors.lower, model.priors.upper):
    print(f"  {name:>14s}  [{lo:.4g}, {hi:.4g}]")

# Evaluate the log-likelihood for a few random points of the prior volume.
rng = np.random.default_rng(1)
for _ in range(3):
    theta = model.prior_transform(rng.uniform(0.35, 0.65, model.ndims))
    print("ln L =", model.log_likelihood(theta))

# Dark-matter density bands from (here: prior) samples.
samples = np.array([model.prior_transform(rng.uniform(0.4, 0.6, model.ndims)) for _ in range(20)])
radii = analysis.default_radii(model)
bands = analysis.credible_bands(analysis.dm_density_profiles(model, samples, radii))
print("median DM density at Rhalf:", np.interp(model.rhalf, radii, bands["median"]))
