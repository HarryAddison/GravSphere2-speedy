# Outputs

For a run named `<name>` with `output.directory: <out>` the files are

```text
<out>/<name>/
├── config_used.yaml                  # the effective configuration (all defaults filled in)
├── Sampler Chains/
│   ├── <checkpoint_name>             # dynesty checkpoint (resume with --resume)
│   ├── <name>_results.pkl            # pickled dynesty Results
│   └── <name>_samples_equal.npy      # equally weighted posterior samples (n_samples x ndims)
└── Plots/                            # (if output.make_plots)
    ├── dm_density.pdf  dm_slope.pdf  anisotropy.pdf
    ├── sigma_los.pdf  kurtosis_los.pdf
    ├── sigma_pm_t.pdf sigma_pm_r.pdf kurtosis_pm_t.pdf kurtosis_pm_r.pdf   # with proper motions
    ├── corner.pdf                    # needs `corner`
    └── dynesty_run.pdf dynesty_trace.pdf   # after a run, needs dynesty
```

Profile plots show the median and the 68 % and 95 % credible bands computed from
`output.profile_samples` random posterior draws; the dashed line marks the half-light radius.

## Using the samples

The columns of `*_samples_equal.npy` follow the [parameter vector](method.md#parameter-vector). Use
`GravSphere2Model.unpack` to turn a row into physical parameters (and `model.priors.names` for labels):

```python
import numpy as np
from gravsphere2 import Config, GravSphere2Model, load_dataset
from gravsphere2.analysis import parameter_summary

cfg = Config.load("examples/gaia_challenge_plumcusp.yaml")
model = GravSphere2Model(cfg, load_dataset(cfg))
samples = np.load(cfg.chains_dir / f"{cfg.name}_samples_equal.npy")
for name, (med, lo, hi) in parameter_summary(samples, model.priors.names).items():
    print(f"{name:>14s} = {med:.3g} (+{hi-med:.2g} -{med-lo:.2g})")
```

!!! warning
    Keep results files that you care about: dynesty checkpoints can be incompatible between dynesty or Python
    versions, so the `.npy` samples file is the portable record of a run.
