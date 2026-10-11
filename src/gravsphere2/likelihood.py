"""The GravSphere2 likelihood.

:class:`GravSphere2Model` ties together the data, the priors and the physical
models.  An instance is a picklable callable pair ``(prior_transform,
log_likelihood)`` that can be handed directly to ``dynesty`` (including with a
``multiprocessing`` pool).

The likelihood of the kinematic data is the product, over tracers, of the
non-Gaussian velocity PDF of Sanders & Evans (2020) with the model variance
and kurtosis from the fourth-order Jeans solver.  The tracer-density term is
either the Gaussian likelihood of a binned surface-density profile or the
probability of the individual stellar positions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .config import Config
from .data import Dataset
from .jeans import solve_jeans
from .models.anisotropy import symmetrised_to_beta
from .models.mass import corenfw_tides_mass, escape_velocity
from .models.tracers import (
    alpbetgam_density,
    alpbetgam_mass,
    alpbetgam_surface,
    alpbetgam_total_mass,
)
from .priors import N_BETA_PARAMS, N_MASS_PARAMS, PriorSpace, build_prior_space
from .velocity_pdf import ln_velocity_pdf

logger = logging.getLogger(__name__)

#: Indices of the symmetrised anisotropy parameters (beta0, beta_inf and primes).
_SYMMETRISED = (0, 1, 4, 5)


@dataclass
class ModelParameters:
    """Physical parameters decoded from a sampler vector."""

    beta_pars: np.ndarray  #: second-order anisotropy ``[beta0, beta_inf, r_beta, n]``
    betap_pars: np.ndarray  #: fourth-order anisotropy
    tracer_pars: np.ndarray  #: ``[rho0, r0, alpha, beta, gamma]`` (rho0 = 1 if individual)
    mass_pars: np.ndarray  #: coreNFWtides parameters (+ stellar mass if free)


class InterpolatedProfile:
    """Piecewise-linear profile on a fixed radial grid (callable, picklable)."""

    def __init__(self, r, values):
        self.r = r
        self.values = values

    def __call__(self, r):
        return np.interp(r, self.r, self.values, right=1e-30)


class TotalMass:
    """Enclosed mass: coreNFWtides dark matter plus optional stellar component.

    The stellar mass follows the tracer light, normalised to its total mass.
    """

    def __init__(self, tracer_pars=None, normmass=None):
        self.tracer_pars = tracer_pars
        self.normmass = normmass

    @property
    def has_stars(self) -> bool:
        return self.tracer_pars is not None

    def __call__(self, r, mass_pars):
        if self.has_stars:
            m200, c, rc, n, rt, delta, mstar = mass_pars
            stars = alpbetgam_mass(r, *self.tracer_pars) * mstar / self.normmass
            return corenfw_tides_mass(r, m200, c, rc, n, rt, delta) + stars
        m200, c, rc, n, rt, delta = mass_pars
        return corenfw_tides_mass(r, m200, c, rc, n, rt, delta)


class GravSphere2Model:
    """Likelihood and prior transform for one stellar system.

    Parameters
    ----------
    cfg : Config
        Validated configuration.
    dataset : Dataset
        Observed data.

    Attributes
    ----------
    priors : PriorSpace
    ndims : int
    rmin, rmax, nonneg_radius : float
        Jeans-solver radial range and non-negativity radius (kpc).
    """

    def __init__(self, cfg: Config, dataset: Dataset):
        self.dataset = dataset
        self.individual = dataset.individual
        self.proper_motions = dataset.proper_motions
        self.escape_check = cfg.data.escape_velocity_check
        self.use_stars = cfg.stellar_mass.value > 0.0
        self.intpnts_tracer = cfg.grids.tracer_points

        rhalf = dataset.half_light_radius
        self.rhalf = rhalf
        R = dataset.R
        g = cfg.grids
        self.rmin = g.rmin if g.rmin is not None else np.min([rhalf / 100, np.min(R) / 10])
        self.rmax = g.rmax if g.rmax is not None else np.max([rhalf * 100, np.max(R) * 2])
        self.nonneg_radius = (
            g.nonnegative_radius if g.nonnegative_radius is not None else self.rmax / 2.0
        )
        tracer_rmin = g.tracer_rmin if g.tracer_rmin is not None else self.rmin
        tracer_rmax = g.tracer_rmax if g.tracer_rmax is not None else self.rmax
        if tracer_rmin == 0:
            tracer_rmin = 1.0e-3
        self.tracer_grid = np.logspace(
            np.log10(tracer_rmin), np.log10(tracer_rmax), self.intpnts_tracer
        )

        self.priors: PriorSpace = build_prior_space(cfg, dataset, rhalf)
        self.ndims = self.priors.ndims
        logger.debug("GravSphere2Model with %d free parameters", self.ndims)

    # -- sampler interface -------------------------------------------------

    def prior_transform(self, u):
        """Unit hypercube -> parameter vector (flat priors)."""
        return self.priors.transform(u)

    def __call__(self, theta):
        return self.log_likelihood(theta)

    # -- parameter handling ------------------------------------------------

    def unpack(self, theta) -> ModelParameters:
        """Decode a sampler vector into physical parameters.

        Converts symmetrised anisotropies to ``beta`` and exponentiates the
        log-sampled parameters.
        """
        theta = np.copy(theta)
        for i in _SYMMETRISED:
            theta[i] = symmetrised_to_beta(theta[i])
        for i in self.priors.log_indices:
            theta[i] = 10 ** theta[i]
        tracer = theta[self.priors.tracer_slice]
        if self.individual:
            tracer = np.append(np.array([1.0]), tracer)
        return ModelParameters(
            beta_pars=theta[:N_BETA_PARAMS],
            betap_pars=theta[N_BETA_PARAMS : 2 * N_BETA_PARAMS],
            tracer_pars=tracer,
            mass_pars=theta[self.priors.mass_slice],
        )

    def build_profiles(self, params: ModelParameters):
        """Tracer density/surface-density interpolators and the total mass function.

        Returns
        -------
        density, surface, mass_func, normmass
        """
        surc = params.tracer_pars
        rho = alpbetgam_density(self.tracer_grid, *surc)
        sigma = alpbetgam_surface(self.tracer_grid, *surc, self.intpnts_tracer)
        normmass = alpbetgam_total_mass(*surc)
        mass_func = TotalMass(surc, normmass) if self.use_stars else TotalMass()
        return (
            InterpolatedProfile(self.tracer_grid, rho),
            InterpolatedProfile(self.tracer_grid, sigma),
            mass_func,
            normmass,
        )

    # -- likelihood --------------------------------------------------------

    def log_likelihood(self, theta):
        """Natural-log likelihood of the data for parameter vector ``theta``.

        Floating-point overflow or invalid operations in extreme corners of the
        prior volume turn into ``nan`` and hence ``-inf``; the corresponding
        NumPy warnings are silenced locally (not globally).
        """
        with np.errstate(all="ignore"):
            params = self.unpack(theta)
            density, surface, mass_func, normmass = self.build_profiles(params)
            return self._lnlike(density, surface, mass_func, params, normmass)

    def _escapes(self, mass_func, params):
        """``True`` if any tracer exceeds the escape velocity (model rejected)."""
        d = self.dataset
        radii = np.append(d.r_los, d.r_pm) if self.proper_motions else d.r_los
        escs = escape_velocity(
            radii, params.mass_pars, self.rmin, self.rmax, mass_func=mass_func, pnts=10**3
        )
        nlos = len(d.v_los)
        if np.any(np.abs(d.v_los) > escs[:nlos]):
            return True
        if self.proper_motions:
            pm_escs = escs[nlos:]
            if np.any(np.abs(d.v_pm_r) > pm_escs[: len(d.v_pm_r)]):
                return True
            if np.any(np.abs(d.v_pm_t) > pm_escs[: len(d.v_pm_t)]):
                return True
        return False

    def _lnlike(self, density, surface, mass_func, params, normmass):
        d = self.dataset
        if self.escape_check and self._escapes(mass_func, params):
            return -np.inf

        result = solve_jeans(
            d.R,
            d.r_los,
            d.r_pm,
            density,
            surface,
            mass_func,
            params.mass_pars,
            params.beta_pars,
            params.betap_pars,
            self.rmin,
            self.rmax,
            nonn=self.nonneg_radius,
            proper_motions=self.proper_motions,
        )

        # Reject models with unphysical moments.
        if result.negative_moments or np.isnan(np.sum(result.kurtosis_los)):
            return -np.inf
        if self.proper_motions and (
            np.isnan(np.sum(result.kurtosis_pm_t)) or np.isnan(np.sum(result.kurtosis_pm_r))
        ):
            return -np.inf

        lnlike = np.sum(
            ln_velocity_pdf(d.v_los, result.sigma2_los / 1e6, result.kurtosis_los, d.v_los_err)
        )
        if self.proper_motions:
            lnlike += np.sum(
                ln_velocity_pdf(
                    d.v_pm_t, result.sigma2_pm_t / 1e6, result.kurtosis_pm_t, d.v_pm_t_err
                )
            ) + np.sum(
                ln_velocity_pdf(
                    d.v_pm_r, result.sigma2_pm_r / 1e6, result.kurtosis_pm_r, d.v_pm_r_err
                )
            )

        if self.individual:
            # Photometric likelihood of individual stars, no binning.
            lnlike += np.sum(np.log(d.R * result.surface_density)) - len(d.R) * np.log(normmass)
        else:
            # Surface brightness / number density profile likelihood.
            lnlike -= 0.5 * np.sum(
                (d.surfden - result.surface_density) ** 2 / d.surfden_err**2
            )

        if np.isnan(lnlike):
            return -np.inf
        return lnlike
