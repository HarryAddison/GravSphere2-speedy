"""Post-processing of posterior samples: profiles and credible bands."""

from __future__ import annotations

import numpy as np

from .jeans import solve_jeans
from .likelihood import GravSphere2Model
from .models.anisotropy import beta, beta_to_symmetrised
from .models.mass import corenfw_tides_density, corenfw_tides_dlnrho_dlnr

#: Percentile levels of the credible bands (68, 95 and 99.7 per cent).
BAND_LEVELS = {
    "lo997": 0.15,
    "lo95": 2.5,
    "lo68": 16.0,
    "median": 50.0,
    "hi68": 84.0,
    "hi95": 97.5,
    "hi997": 99.85,
}


def credible_bands(values):
    """Percentile bands of ``values`` (shape ``(n_samples, n_radii)``) along axis 0.

    Returns
    -------
    dict
        Keys ``median``, ``lo68``, ``hi68``, ``lo95``, ``hi95``, ``lo997`` and
        ``hi997``, each of shape ``(n_radii,)``.
    """
    values = np.asarray(values, dtype=float)
    return {k: np.nanpercentile(values, q, axis=0) for k, q in BAND_LEVELS.items()}


def default_radii(model: GravSphere2Model, radius_range=None, n=100):
    """Logarithmic radial grid for profile plots (kpc)."""
    if radius_range is None:
        lo = max(model.rhalf / 10.0, model.rmin)
        hi = min(model.rhalf * 10.0, model.rmax)
    else:
        lo, hi = radius_range
    return np.logspace(np.log10(lo), np.log10(hi), n)


def thin_samples(samples, n, seed=None):
    """Randomly select ``n`` rows of ``samples`` (all of them if fewer)."""
    samples = np.asarray(samples)
    if n >= len(samples):
        return samples
    rng = np.random.default_rng(seed)
    return samples[rng.choice(len(samples), size=n, replace=False)]


def _dm_pars(model, theta):
    return model.unpack(theta).mass_pars[:6]


def dm_density_profiles(model, samples, radii):
    """Dark-matter density ``rho(r)`` for each sample, shape ``(n, len(radii))``."""
    return np.array([corenfw_tides_density(radii, *_dm_pars(model, th)) for th in samples])


def dm_slope_profiles(model, samples, radii, dx=1e-5):
    """Logarithmic dark-matter density slope for each sample."""
    return np.array(
        [corenfw_tides_dlnrho_dlnr(radii, *_dm_pars(model, th), dx=dx) for th in samples]
    )


def anisotropy_profiles(model, samples, radii):
    """Symmetrised anisotropy ``beta / (2 - beta)`` for each sample."""
    out = []
    for th in samples:
        be = beta(radii, model.unpack(th).beta_pars)
        out.append(beta_to_symmetrised(be))
    return np.array(out)


def dispersion_profiles(model, samples, radii):
    """Projected dispersion (km/s) and kurtosis profiles for each sample.

    Returns
    -------
    dict
        ``sigma_los`` and ``kurtosis_los`` always; with proper motions also
        ``sigma_pm_t``, ``sigma_pm_r``, ``kurtosis_pm_t`` and ``kurtosis_pm_r``.
        Each value has shape ``(n_samples, len(radii))``.
    """
    keys = ["sigma_los", "kurtosis_los"]
    if model.proper_motions:
        keys += ["sigma_pm_t", "sigma_pm_r", "kurtosis_pm_t", "kurtosis_pm_r"]
    out = {k: [] for k in keys}
    for th in samples:
        params = model.unpack(th)
        density, surface, mass_func, _ = model.build_profiles(params)
        res = solve_jeans(
            radii,
            radii,
            radii,
            density,
            surface,
            mass_func,
            params.mass_pars,
            params.beta_pars,
            params.betap_pars,
            model.rmin,
            model.rmax,
            nonn=False,
            proper_motions=model.proper_motions,
        )
        out["sigma_los"].append(np.sqrt(res.sigma2_los) / 1e3)
        out["kurtosis_los"].append(res.kurtosis_los)
        if model.proper_motions:
            out["sigma_pm_t"].append(np.sqrt(res.sigma2_pm_t) / 1e3)
            out["sigma_pm_r"].append(np.sqrt(res.sigma2_pm_r) / 1e3)
            out["kurtosis_pm_t"].append(res.kurtosis_pm_t)
            out["kurtosis_pm_r"].append(res.kurtosis_pm_r)
    return {k: np.array(v) for k, v in out.items()}


def parameter_summary(samples, names):
    """Median and 68 per cent interval of each sampled parameter.

    Returns
    -------
    dict
        ``{name: (median, lower, upper)}`` using the 16th and 84th percentiles.
    """
    samples = np.asarray(samples)
    if samples.shape[1] != len(names):
        raise ValueError("Number of parameter names does not match the samples")
    lo, med, hi = np.percentile(samples, [16, 50, 84], axis=0)
    return {n: (med[i], lo[i], hi[i]) for i, n in enumerate(names)}
