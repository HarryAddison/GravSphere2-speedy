# Input file reference

Everything the user can set lives in one YAML file passed with `-i`. Unknown keys are rejected with a
message listing the valid ones, and values are validated before any expensive work starts. **Relative
paths are resolved relative to the directory containing the input file.** A complete template with all
defaults is `examples/template_full.yaml`; the effective configuration of each run is saved as
`config_used.yaml` in its output directory.

!!! tip "Scientific notation"
    YAML 1.1 (PyYAML) reads `4.3e7` as text, not a number. GravSphere2 converts such values
    automatically, so `4.3e7`, `4.3e+7` and `43000000` are all accepted.

## Top level

| Key       | Default            | Description |
| --------- | ------------------ | ----------- |
| `name`    | `gravsphere2_run`  | Run name; also names the output sub-directory. No `/` or `\`. |
| `output`  |                    | Output location and plotting, see below. |
| `data`    |                    | Input data. |
| `grids`   |                    | Radial grids. |
| `stellar_mass` |               | Optional baryonic mass. |
| `priors`  |                    | Prior ranges. |
| `sampler` |                    | Nested-sampling options. |

## `output`

| Key                 | Default  | Description |
| ------------------- | -------- | ----------- |
| `directory`         | `output` | Base output directory (`<directory>/<name>/…`). |
| `make_plots`        | `true`   | Create posterior plots after sampling. |
| `use_latex`         | `false`  | Typeset plot text with LaTeX. |
| `profile_samples`   | `300`    | Posterior draws used for profile credible bands. |
| `n_radii`           | `100`    | Radii per profile plot. |
| `plot_radius_range` | `null`   | `[rmin, rmax]` in kpc; default `[Rhalf/10, 10 Rhalf]` limited to the Jeans grid. |

## `data`

| Key                     | Default | Description |
| ----------------------- | ------- | ----------- |
| `directory`             | `.`     | Folder containing the data files. |
| `kinematics_file`       | —       | **Required.** See [Data formats](data-formats.md). |
| `proper_motions`        | `true`  | Fit proper motions (7-row file) or line-of-sight velocities only (3-row file). |
| `photometry.mode`       | `binned`| `binned` (profile with errors) or `individual` (bin-free stellar positions). |
| `photometry.file`       | `null`  | Photometry file (required when `source: file`). |
| `photometry.source`     | `file`  | `file`, or `kinematic_tracers` to reuse the kinematic radii (individual mode). |
| `half_light_radius`     | `null`  | Reference scale (kpc). Required for binned data; `median` (default) for individual data. It need not be exact. |
| `escape_velocity_check` | `false` | Reject models in which any tracer exceeds the escape velocity. Useful for few tracers or when individual velocities differ greatly from the dispersion; it assumes all stars are members. |

## `grids`

| Key                  | Default | Description |
| -------------------- | ------- | ----------- |
| `rmin`               | `null`  | Inner radius of the Jeans grid (kpc). `null`: `min(Rhalf/100, min(R)/10)`. |
| `rmax`               | `null`  | Outer radius. `null`: `max(100 Rhalf, 2 max(R))`. |
| `nonnegative_radius` | `null`  | Fourth moments must be non-negative inside this radius. `null`: `rmax/2`. |
| `tracer_rmin`, `tracer_rmax` | `null` | Grid of the tracer profile; default equals the Jeans grid. |
| `tracer_points`      | `300`   | Number of points in the tracer-profile grid. |

## `stellar_mass`

| Key                | Default | Description |
| ------------------ | ------- | ----------- |
| `value`            | `0.0`   | Stellar mass in Msun. `0`: massless tracers (dark matter only). Otherwise the mass is a free parameter with a flat prior. The baryonic mass follows the tracer profile. |
| `fractional_error` | `0.25`  | Prior half-width as a fraction of `value`. |

## `priors`

### `priors.tracer` (binned photometry)

| Key                    | Default | Description |
| ---------------------- | ------- | ----------- |
| `reference_parameters` | `null`  | Pre-fitted alpha-beta-gamma parameters `[rho0, r0, alpha, beta, gamma]`, e.g. from a direct fit, [binulator](https://github.com/justinread/gravsphere), or a Plummer profile matched to the central density and ~Rhalf. |
| `tolerance`            | `0.5`   | Flat priors span `± tolerance` (fractional) around the reference. `alpha > 0`, `beta > 3`, `gamma < 3` are enforced. |

For individual photometry the tracer priors are fixed: `log10 r0` within one dex of `Rhalf`,
`alpha ∈ [0.1, 4]`, `beta ∈ [3.1, 7]`, `gamma ∈ [0, 2.9]`. For broader (for example logarithmic) priors
edit `gravsphere2/priors.py`.

### `priors.anisotropy`

Flat ranges `[min, max]` of the **symmetrised** anisotropy
`beta_tilde = beta / (2 - beta)` and its profile parameters; the same ranges apply to the second-order
profile and to its fourth-order counterpart.

| Key            | Default       | Description |
| -------------- | ------------- | ----------- |
| `beta0`        | `[-1, 1]`     | Central anisotropy. |
| `beta_inf`     | `[-1, 1]`     | Outer anisotropy. |
| `log10_r_beta` | `[-2, 0]`     | log10 of the transition radius (kpc). |
| `eta`          | `[1, 3]`      | Transition sharpness `n`. |

### `priors.mass` (coreNFWtides)

| Key          | Default        | Description |
| ------------ | -------------- | ----------- |
| `log10_M200` | `[7.5, 11.5]`  | log10 of M200 in Msun. |
| `c200`       | `[1, 100]`     | Concentration. |
| `rc`         | `[0.01, 10]`   | Core radius in kpc (sampled in log10). |
| `n`          | `[0, 1]`       | Core strength (0 cusp, 1 core). |
| `rt`         | `[1, 20]`      | Tidal radius in kpc (sampled in log10). |
| `delta`      | `[3.01, 5]`    | Outer slope beyond `rt`; must exceed 3. |

## `sampler`

| Key               | Default      | Description |
| ----------------- | ------------ | ----------- |
| `nlive`           | `500`        | Live points. Use of order a few times the number of parameters squared, and at least ~50 per mode. More is more reliable and slower. |
| `ncpus`           | `1`          | Parallel likelihood evaluations through `multiprocessing`. |
| `use_stop`        | `true`       | Stop when dynesty's stopping criterion is met. |
| `maxiter`         | `null`       | Optional iteration cap. |
| `checkpoint_name` | `checkpoint` | File name in `Sampler Chains`. |
| `seed`            | `null`       | Integer for reproducible runs (new runs only). |
| `print_progress`  | `true`       | dynesty progress output. |
