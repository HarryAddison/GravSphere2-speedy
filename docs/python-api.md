# Python API

The command line covers the standard workflow; the same building blocks are available as a library, for
example in notebooks.

```python
import numpy as np
from gravsphere2 import Config, GravSphere2Model, load_dataset

cfg = Config.load("examples/fornax_simulation.yaml")      # validated configuration
dataset = load_dataset(cfg)                                 # data arrays
model = GravSphere2Model(cfg, dataset)                      # likelihood + priors

theta = model.prior_transform(np.full(model.ndims, 0.5))    # centre of the prior volume
print(model.log_likelihood(theta))                          # same as model(theta)
```

`model.prior_transform` and `model.log_likelihood` are exactly what `dynesty` receives, so running the
sampler yourself is

```python
from dynesty import DynamicNestedSampler

sampler = DynamicNestedSampler(model.log_likelihood, model.prior_transform, model.ndims, nlive=500)
sampler.run_nested(checkpoint_file="chk", use_stop=True)
samples = sampler.results.samples_equal()
```

Configurations can also be built in code, with the same validation as for files:

```python
cfg = Config.from_dict({"name": "test", "data": {...}}, base_dir=".")
cfg.sampler.nlive = 100
cfg.validate()
```

## Main modules

| Module | Contents |
| ------ | -------- |
| `gravsphere2.config` | `Config` and its sections (`DataConfig`, `PriorsConfig`, `SamplerConfig`, …), `ConfigError`. |
| `gravsphere2.data` | `Dataset`, `load_dataset`, `DataError`. |
| `gravsphere2.likelihood` | `GravSphere2Model` (`prior_transform`, `log_likelihood`, `unpack`, `build_profiles`). |
| `gravsphere2.jeans` | `solve_jeans` → `JeansResult`; `radial_dispersion_squared`. |
| `gravsphere2.models.mass` | `corenfw_tides_density/mass/dlnrho_dlnr`, NFW helpers, `escape_velocity`. |
| `gravsphere2.models.tracers` | alpha-beta-gamma and multi-Plummer profiles, projections. |
| `gravsphere2.models.anisotropy` | `beta`, `dbeta_dr`, `integrating_factor`, symmetrised conversions. |
| `gravsphere2.velocity_pdf` | `ln_velocity_pdf` (variance, kurtosis and error convolution). |
| `gravsphere2.analysis` | `credible_bands`, profile functions for posterior samples, `parameter_summary`. |
| `gravsphere2.plotting` | `make_all_plots`, `plot_bands`, `plot_corner`. |
| `gravsphere2.dark_matter` | `j_factor`, `d_factor`, `sidm_cross_section`. |
| `gravsphere2.stats` | `bin_data`, `confidence_intervals`, … |

Every public function has a docstring: use `help(gravsphere2.jeans.solve_jeans)` or your editor.
A runnable example is in `examples/python_api_example.py`.
