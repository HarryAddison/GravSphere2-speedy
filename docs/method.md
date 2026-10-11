# Method

This page summarises what the code computes. The full derivation and validation are in the
release paper (see [Citing](index.md#citing)).

## Jeans modelling

For a spherical, non-rotating tracer population with density ν(r), enclosed mass M(r) and velocity
anisotropy β(r) = 1 − σ_θ²/σ_r², the second- and fourth-order Jeans equations are solved for the radial
moments ⟨v_r²⟩ and ⟨v_r⁴⟩, which are then projected along the line of sight to give, as a function of
projected radius, the second and fourth moments of the line-of-sight (LOS), radial proper-motion and
tangential proper-motion velocities. The kurtosis κ = ⟨v⁴⟩/⟨v²⟩² of each component follows.
The implementation is in `gravsphere2.jeans.solve_jeans`.

## Velocity distribution and likelihood

Each stellar velocity is modelled with the Sanders & Evans (2020) family of non-Gaussian PDFs, parametrised
by the model variance and kurtosis (uniform kernel for κ < 3, Laplace kernel for κ > 3), convolved analytically with the
Gaussian measurement error of that star (`gravsphere2.velocity_pdf`). The kinematic log-likelihood
is the sum of the log-PDFs over all stars and over LOS, radial-PM and tangential-PM velocities.

The tracer profile enters through either

* **individual positions** (`photometry.mode: individual`): `Σ ln[R Σ(R)] − N ln M_tot`, the probability of the observed
  projected radii under the model surface density; or
* **a binned profile** (`photometry.mode: binned`): a Gaussian χ² term on the surface density.

Models with negative fourth moments inside `grids.nonnegative_radius`, undefined kurtosis, or (optionally) tracers above
the escape velocity have likelihood `-inf`.

## Models

* **Dark matter**: *coreNFWtides* (Read et al. 2016, 2018) — an NFW halo of mass `M200` and concentration `c200`
  with a core of radius `rc` and strength `n` (0 = cusp, 1 = core) and tidal truncation beyond `rt` with outer
  slope `delta` (`gravsphere2.models.mass`).
* **Tracers**: an alpha-beta-gamma profile
  ρ(r) = ρ₀ (r/r₀)^(−γ) [1 + (r/r₀)^α]^((γ−β)/α) (`gravsphere2.models.tracers`).
* **Stars** (optional): a baryonic mass with the tracer profile and a free total mass.
* **Anisotropy**: β(r) = β₀ + (β∞ − β₀) / [1 + (r_β/r)ⁿ], with a separate set of parameters (primed) for the
  fourth-order moment. Sampling uses the symmetrised β̃ = β/(2 − β) ∈ (−1, 1) (`gravsphere2.models.anisotropy`).

## Parameter vector

The sampler explores the following vector (`model.priors.names` lists the labels):

| index                        | parameter(s)                                                     | note |
| ---------------------------- | ---------------------------------------------------------------- | ---- |
| 0–3                          | β̃₀, β̃∞, log₁₀ r_β, η                                           | second-order anisotropy |
| 4–7                          | β̃₀′, β̃∞′, log₁₀ r_β′, η′                                       | fourth-order anisotropy |
| 8–12 (binned) / 8–11 (individual) | ρ₀, r₀, α, β, γ / log₁₀ r₀, α, β, γ                       | tracer profile; ρ₀ is fixed by normalisation in individual mode |
| next 6                       | log₁₀ M200, c200, log₁₀ rc, n, log₁₀ rt, δ                       | coreNFWtides |
| last (optional)              | M★                                                               | only if `stellar_mass.value > 0` |

Parameters sampled in log₁₀ are exponentiated, and symmetrised anisotropies converted to β, inside
`GravSphere2Model.unpack` before use.

## Numerical choices

* All projections use Simpson integration over an angle grid; radial grids are logarithmic with 150 points
  (`intpnts` of `solve_jeans`), the tracer profile uses `grids.tracer_points` points.
* Density, mass and surface-density profiles are tabulated once per likelihood call and interpolated.
* The radius loops of the original implementation are vectorised in this version, which makes a likelihood
  evaluation roughly ten times faster with identical results.

## Package layout

```text
src/gravsphere2/
├── cli.py                 command-line interface (entry point `gravsphere2`)
├── config.py              YAML schema (dataclasses), loading and validation
├── data.py                Dataset and file readers
├── priors.py              prior hyper-rectangle and parameter vector layout
├── likelihood.py          GravSphere2Model: prior transform + log-likelihood
├── jeans.py               second/fourth-order Jeans solver
├── velocity_pdf.py        Sanders & Evans PDFs with error convolution
├── line_profiles.py       vendored special functions of Sanders & Evans (2020)
├── sampler.py             dynesty wrapper, checkpoints, saving results
├── analysis.py            posterior profiles and credible bands
├── plotting.py            matplotlib plots
├── models/                anisotropy.py, mass.py, tracers.py
├── constants.py           units and cosmology
└── (optional extras)      cosmology, dark_matter (J/D factors, SIDM), vsp, stats,
                           acceleration, generalised_gaussian
```
