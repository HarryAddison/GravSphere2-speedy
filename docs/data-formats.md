# Data formats

All files are plain-text tables readable by `numpy.loadtxt`. **Each row holds one quantity** and each
column one star (or bin), exactly as in the original release. Distances are in **kpc** and velocities
in **km/s**. Velocities (including proper-motion velocities) are relative to the system, so subtract
the global averages beforehand.

## Kinematics (`data.kinematics_file`)

With `proper_motions: true` the file has **7 rows**:

| row | quantity                                  |
| --- | ----------------------------------------- |
| 1   | projected radius `R` of each tracer       |
| 2   | line-of-sight velocity `vLOS`             |
| 3   | error of `vLOS`                           |
| 4   | tangential proper-motion velocity `vPMt`  |
| 5   | error of `vPMt`                           |
| 6   | radial proper-motion velocity `vPMR`      |
| 7   | error of `vPMR`                           |

With `proper_motions: false` only the first **3 rows** (`R`, `vLOS`, `vLOS_err`) are read. Errors must be
positive; non-finite values are rejected. The proper-motion measurements share the radii of the line-of-sight
sample.

## Photometry (`data.photometry`)

=== "mode: binned"

    A file with **3 rows**: bin radius `R`, surface density and its error. Binned tracer parameters are
    constrained by a Gaussian likelihood. You must give pre-fitted reference parameters in
    `priors.tracer.reference_parameters` and a numeric `data.half_light_radius`.

=== "mode: individual"

    A file with a single row (or column) of projected radii of the photometric tracers
    (bin-free fit; the most information-preserving option). Set `source: kinematic_tracers` to reuse the
    kinematic radii instead of a file. `half_light_radius: median` uses the median radius.

!!! tip
    You can create a binned profile from positions with `gravsphere2.stats.bin_data(R, weights, n_per_bin)`.
