"""Prior space and prior transform for the nested sampler.

Parameter vector layout (``theta``)
-----------------------------------
=========================  ============================================
indices                    meaning
=========================  ============================================
``0..3``                   symmetrised anisotropy ``beta0, beta_inf,
                           log10 r_beta, eta`` (second order)
``4..7``                   same, fourth-order counterpart (``beta'``)
``8 .. 8+n_tracer-1``      tracer alpha-beta-gamma parameters
                           (``rho0, r0, alpha, beta, gamma`` when binned,
                           ``log10 r0, alpha, beta, gamma`` when fitting
                           individual positions)
next 6                     ``log10 M200, c200, log10 rc, n, log10 rt,
                           delta`` (coreNFWtides)
last (optional)            stellar mass ``Mstar`` (Msun)
=========================  ============================================

Parameters sampled in log10 are listed in :attr:`PriorSpace.log_indices`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config
from .data import Dataset

N_BETA_PARAMS = 4
N_MASS_PARAMS = 6

BETA_NAMES = ["beta0", "beta_inf", "log10_r_beta", "eta"]
TRACER_NAMES_BINNED = ["rho_p", "r_p", "alpha_p", "beta_p", "gamma_p"]
TRACER_NAMES_INDIVIDUAL = ["log10_r_p", "alpha_p", "beta_p", "gamma_p"]
MASS_NAMES = ["log10_M200", "c200", "log10_rc", "n", "log10_rt", "delta"]


@dataclass
class PriorSpace:
    """Flat prior hyper-rectangle and bookkeeping for the parameter vector."""

    lower: np.ndarray
    upper: np.ndarray
    log_indices: list
    n_tracer: int
    has_stellar_mass: bool
    individual: bool
    names: list

    @property
    def ndims(self) -> int:
        return len(self.lower)

    @property
    def anisotropy_slice(self) -> slice:
        return slice(0, 2 * N_BETA_PARAMS)

    @property
    def tracer_slice(self) -> slice:
        start = 2 * N_BETA_PARAMS
        return slice(start, start + self.n_tracer)

    @property
    def mass_slice(self) -> slice:
        """Dark-matter parameters plus stellar mass if it is a free parameter."""
        start = 2 * N_BETA_PARAMS + self.n_tracer
        return slice(start, start + N_MASS_PARAMS + int(self.has_stellar_mass))

    def transform(self, u):
        """Map the unit hypercube ``u`` to parameter space (for dynesty)."""
        return self.lower + (self.upper - self.lower) * np.asarray(u)


def _tracer_bounds(cfg: Config, dataset: Dataset, rhalf: float):
    tol = cfg.priors.tracer.tolerance
    if not dataset.individual:
        pfits = np.asarray(cfg.priors.tracer.reference_parameters, dtype=float)
        nu_min = np.zeros(len(pfits))
        nu_max = np.zeros(len(pfits))
        for i in range(len(pfits)):
            if pfits[i] > 0:
                nu_min[i] = pfits[i] * (1.0 - tol)
                nu_max[i] = pfits[i] * (1.0 + tol)
            else:
                nu_min[i] = pfits[i] * (1.0 + tol)
                nu_max[i] = pfits[i] * (1.0 - tol)
        # Positivity and convergence require alpha > 0, beta > 3, gamma < 3.
        nu_min[2:4] = np.maximum(nu_min[2:4], np.array([0.0, 3.0]))
        nu_max[4] = np.min(np.array([nu_max[4], 3.0]))
        return nu_min, nu_max
    # Individual positions: only the shape parameters r0, alpha, beta, gamma
    # matter (rho0 is fixed by the probability normalisation).
    nu_min = np.array([np.log10(rhalf) - 1.0, 0.1, 3.1, 0.0])
    nu_max = np.array([np.log10(rhalf) + 1.0, 4.0, 7.0, 2.9])
    return nu_min, nu_max


def build_prior_space(cfg: Config, dataset: Dataset, rhalf: float) -> PriorSpace:
    """Create the :class:`PriorSpace` implied by a configuration and dataset."""
    ani = cfg.priors.anisotropy
    mass = cfg.priors.mass
    has_stars = cfg.stellar_mass.value > 0.0

    nu_min, nu_max = _tracer_bounds(cfg, dataset, rhalf)
    n_tracer = len(nu_min)
    nb = N_BETA_PARAMS
    base = 2 * nb + n_tracer
    ndims = base + N_MASS_PARAMS + int(has_stars)

    lower = np.zeros(ndims)
    upper = np.zeros(ndims)

    beta_ranges = [ani.beta0, ani.beta_inf, ani.log10_r_beta, ani.eta]
    for offset in (0, nb):  # second-order and fourth-order
        for j, (lo, hi) in enumerate(beta_ranges):
            lower[offset + j] = lo
            upper[offset + j] = hi

    lower[2 * nb : base] = nu_min
    upper[2 * nb : base] = nu_max

    mass_ranges = [
        mass.log10_M200,
        mass.c200,
        (np.log10(mass.rc[0]), np.log10(mass.rc[1])),
        mass.n,
        (np.log10(mass.rt[0]), np.log10(mass.rt[1])),
        mass.delta,
    ]
    for j, (lo, hi) in enumerate(mass_ranges):
        lower[base + j] = lo
        upper[base + j] = hi

    if has_stars:
        frac = cfg.stellar_mass.fractional_error
        mstar = cfg.stellar_mass.value
        lower[base + N_MASS_PARAMS] = mstar - mstar * frac
        upper[base + N_MASS_PARAMS] = mstar + mstar * frac

    # Parameters that are sampled in log10 and exponentiated before use.
    log_indices = [2, 6, base, base + 2, base + 4]
    if dataset.individual:
        log_indices.append(2 * nb)

    names = [f"{n}" for n in BETA_NAMES] + [f"{n}_prime" for n in BETA_NAMES]
    names += TRACER_NAMES_INDIVIDUAL if dataset.individual else TRACER_NAMES_BINNED
    names += MASS_NAMES
    if has_stars:
        names.append("Mstar")

    return PriorSpace(
        lower=lower,
        upper=upper,
        log_indices=log_indices,
        n_tracer=n_tracer,
        has_stellar_mass=has_stars,
        individual=dataset.individual,
        names=names,
    )
