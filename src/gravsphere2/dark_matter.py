"""Dark-matter particle-physics diagnostics.

J- and D-factors for annihilation / decay searches and the SIDM
cross-section estimator of Read et al. (2018).
"""

import numpy as np
from scipy.integrate import dblquad
from scipy.integrate import simpson

from .constants import (
    GUSE,
    KPC_CM,
    MSUN,
    KPC,
    MSUNKPC3_TO_GEVCM3,
    ODEN,
    RHOCRIT,
    YEAR,
)
from .models.mass import nfw_density


def sidm_cross_section(rc, M200, c, oden=ODEN, rhocrit=RHOCRIT):
    """SIDM cross-section per unit particle mass (cm^2 / g).

    Calculated from a pure-core (``n = 1``) coreNFWtides fit; see
    Read et al. (2018).
    """
    gamma_x = 0.005 / (1e9 * YEAR)
    rho_unit = MSUN / KPC**3.0
    rc = np.abs(rc) * 10.0
    gcon = 1.0 / (np.log(1.0 + c) - c / (1.0 + c))
    deltachar = oden * c**3.0 * gcon / 3.0
    rv = (3.0 / 4.0 * M200 / (np.pi * oden * rhocrit)) ** (1.0 / 3.0)
    rs = rv / c
    rhos = rhocrit * deltachar
    rhorc = nfw_density(rc, rhos, rs)
    r = np.logspace(np.log10(rc), np.log10(rs * 5000.0), 50000)
    rho = nfw_density(r, rhos, rs)
    mass = M200 * gcon * (np.log(1.0 + r / rs) - r / rs / (1.0 + r / rs))
    sigvtworc = GUSE / rhorc * simpson(mass * rho / r**2.0, x=r)
    sigvrc = np.sqrt(sigvtworc)
    sigm = np.sqrt(np.pi) * gamma_x / (4.0 * rhorc * rho_unit * sigvrc)
    return sigm * 100.0**2.0 / 1000.0


def los_radius(s, b, distance):
    """Galactocentric radius for line-of-sight distance ``s`` and angle ``b``."""
    return np.sqrt((distance * np.sin(b)) ** 2.0 + s * s)


def _j_integrand(s, b, distance, rho, mass_pars):
    return np.sin(b) * rho(np.array([los_radius(s, b, distance)]), mass_pars)[0] ** 2


def _d_integrand(s, b, distance, rho, mass_pars):
    return np.sin(b) * rho(np.array([los_radius(s, b, distance)]), mass_pars)[0]


def _line_of_sight_integral(integrand, rho, mass_pars, distance, r_max, rmaximum, accuracy):
    b_min = 0.0
    b_max = np.arcsin(r_max / distance)

    def s_min(b):
        return -((rmaximum**2 - (distance * np.sin(b)) ** 2) ** 0.5)

    def s_max(b):
        return (rmaximum**2 - (distance * np.sin(b)) ** 2) ** 0.5

    value = dblquad(
        integrand,
        b_min,
        b_max,
        s_min,
        s_max,
        args=(distance, rho, mass_pars),
        epsabs=accuracy,
        epsrel=accuracy,
    )
    return value[0]


def j_factor(rho, mass_pars, distance, r_max, rmaximum=250.0, accuracy=1.0e-8):
    """Annihilation J-factor in GeV^2 c^-4 cm^-5.

    Parameters
    ----------
    rho : callable
        Density ``rho(r, mass_pars)`` in Msun kpc^-3 (``r`` an array).
    distance : float
        Distance to the system (kpc).
    r_max : float
        Maximum radius of the integration cone (kpc), giving an integration
        angle ``r_max / distance``.
    rmaximum : float
        Outer integration limit along the line of sight.  The default is
        appropriate for dwarf galaxies; reconsider it for massive systems.
    """
    value = _line_of_sight_integral(_j_integrand, rho, mass_pars, distance, r_max, rmaximum, accuracy)
    j_max = value * KPC_CM * 2.0 * np.pi * MSUNKPC3_TO_GEVCM3**2.0
    if np.isinf(j_max) or j_max < 0:
        raise FloatingPointError(f"Invalid J factor: {j_max}")
    return j_max


def d_factor(rho, mass_pars, distance, r_max, rmaximum=250.0, accuracy=1.0e-8):
    """Decay D-factor in GeV c^-2 cm^-2 (arguments as :func:`j_factor`)."""
    value = _line_of_sight_integral(_d_integrand, rho, mass_pars, distance, r_max, rmaximum, accuracy)
    d_max = value * KPC_CM * 2.0 * np.pi * MSUNKPC3_TO_GEVCM3
    if np.isinf(d_max) or d_max < 0:
        raise FloatingPointError(f"Invalid D factor: {d_max}")
    return d_max
