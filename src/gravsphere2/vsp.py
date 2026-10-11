"""Virial shape parameters (VSPs) and related legacy diagnostics."""

import numpy as np
from scipy.integrate import simpson

from .constants import GUSE
from .models.anisotropy import osipkov, osipkov_integrating_factor
from .models.tracers import alpbetgam_density, alpbetgam_mass


def alpbetgam_sigma_r(r, rho0s, r0s, alps, bets, gams, rho0, r0, alp, bet, gam, ra):
    """Radial dispersion of an alpha-beta-gamma tracer in an alpha-beta-gamma
    potential with Osipkov-Merritt anisotropy of scale ``ra``."""
    nu = alpbetgam_density(r, rho0s, r0s, alps, bets, gams)
    mass = alpbetgam_mass(r, rho0, r0, alp, bet, gam)
    gf = osipkov_integrating_factor(r, ra)
    sigr = np.zeros(len(r))
    for i in range(len(r) - 3):
        sigr[i] = (
            1.0
            / nu[i]
            / gf[i]
            * simpson(GUSE * mass[i:] * nu[i:] / r[i:] ** 2.0 * gf[i:], x=r[i:])
        )
    return sigr


def alpbetgam_vsp(rho0s, r0s, alps, bets, gams, rho0, r0, alp, bet, gam, ra):
    """Virial shape parameters and Richardson & Fairbairn zeta parameters.

    Returns
    -------
    vsp1, vsp2, zeta_a, zeta_b
    """
    intpnts = int(1e4)
    r = np.logspace(np.log10(r0s / 50.0), np.log10(500.0 * r0s), int(intpnts))
    nu = alpbetgam_density(r, rho0s, r0s, alps, bets, gams)
    mass = alpbetgam_mass(r, rho0, r0, alp, bet, gam)
    sigr = alpbetgam_sigma_r(r, rho0s, r0s, alps, bets, gams, rho0, r0, alp, bet, gam, ra)
    bet = osipkov(r, ra)

    sigstar = np.zeros(len(r))
    for i in range(1, len(r) - 3):
        sigstar[i] = 2.0 * simpson(nu[i:] * r[i:] / np.sqrt(r[i:] ** 2.0 - r[i - 1] ** 2.0), x=r[i:])

    # Normalise similarly to the data
    norm = simpson(sigstar * 2.0 * np.pi * r, x=r)
    nu = nu / norm
    sigstar = sigstar / norm

    vsp1 = simpson(2.0 / 5.0 * GUSE * mass * nu * (5.0 - 2.0 * bet) * sigr * r, x=r) / 1.0e12
    vsp2 = simpson(4.0 / 35.0 * GUSE * mass * nu * (7.0 - 6.0 * bet) * sigr * r**3.0, x=r) / 1.0e12

    ntot = simpson(sigstar * r, x=r)
    sigint = simpson(sigstar * r**3.0, x=r)
    zeta_a = (
        9.0
        / 10.0
        * ntot
        * simpson(GUSE * mass * nu * (5.0 - 2.0 * bet) * sigr * r, x=r)
        / (simpson(GUSE * mass * nu * r, x=r)) ** 2.0
    )
    zeta_b = (
        9.0
        / 35.0
        * ntot**2.0
        * simpson(GUSE * mass * nu * (7.0 - 6.0 * bet) * sigr * r**3.0, x=r)
        / ((simpson(GUSE * mass * nu * r, x=r)) ** 2.0 * sigint)
    )
    return vsp1, vsp2, zeta_a, zeta_b


def richardson_fairbairn_vsp(vz, Rkin, mskin):
    """Richardson & Fairbairn VSP estimators from individual velocities."""
    vsp1 = 1.0 / (np.pi * 2.0) * np.sum(vz**4.0 * mskin) / np.sum(mskin)
    vsp2 = 1.0 / (np.pi * 2.0) * np.sum(vz**4.0 * mskin * Rkin**2.0) / np.sum(mskin * Rkin**2.0)
    return vsp1, vsp2


def vsp_pdf_table(vsp, nbins=50):
    """Histogram a VSP sample into a normalised PDF table ``(bins, pdf)``."""
    bins_plus_one = np.linspace(np.min(vsp), np.max(vsp), nbins + 1)
    bins = np.linspace(np.min(vsp), np.max(vsp), nbins)
    pdf, _ = np.histogram(vsp, bins=bins_plus_one)
    pdf = pdf / np.max(pdf)
    return bins[pdf > 0], pdf[pdf > 0]


def vsp_pdf(vsp, bins, pdf):
    """Interpolate a VSP PDF table at ``vsp`` (zero outside the table)."""
    return np.interp(vsp, bins, pdf, left=0, right=0)
