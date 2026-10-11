"""Mass models.

The default dark-matter model is the *coreNFWtides* profile of Read et al.
(2016, 2018): an NFW profile with a core (radius ``rc``, strength ``n``) and
tidal stripping beyond ``rt`` with outer slope ``delta``.  Helpers for plain
NFW haloes, central Plummer masses and escape velocities are also included.
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid

from ..constants import (
    GUSE,
    KMS,
    G,
    KPC,
    MSUN,
    ODEN,
    RHOCRIT,
    VIRIAL_OVERDENSITY,
)


def _nfw_parameters(M200, c):
    """NFW scale parameters ``(gcon, rs, rhos)`` for a given M200 and c200."""
    gcon = 1.0 / (np.log(1.0 + c) - c / (1.0 + c))
    deltachar = ODEN * c**3.0 * gcon / 3.0
    rv = (3.0 / 4.0 * M200 / (np.pi * ODEN * RHOCRIT)) ** (1.0 / 3.0)
    rs = rv / c
    rhos = RHOCRIT * deltachar
    return gcon, rs, rhos


def _tidal_transition(M200, c, rc, n, rt):
    """Quantities at the truncation radius for coreNFWtides."""
    gcon, rs, rhos = _nfw_parameters(M200, c)
    frt = np.tanh(rt / np.abs(rc))
    manal_rt = M200 * gcon * (np.log(1.0 + rt / rs) - rt / rs / (1.0 + rt / rs))
    rho_rt = (
        rhos / ((rt / rs) * (1.0 + (rt / rs)) ** 2.0) * frt**n
        + 1.0
        / (4.0 * np.pi * rt**2.0 * np.abs(rc))
        * manal_rt
        * (1.0 - frt**2.0)
        * n
        * frt ** (n - 1.0)
    )
    return gcon, rs, rhos, frt, manal_rt, rho_rt


def corenfw_tides_density(r, M200, c, rc, n, rt, delta):
    """coreNFWtides density profile (Msun kpc^-3).

    Parameters
    ----------
    r : ndarray
        Radii in kpc.
    M200, c : float
        Virial mass (Msun) and concentration of the underlying NFW halo.
    rc, n : float
        Core radius (kpc) and core strength (0 = cusp, 1 = full core).
    rt, delta : float
        Tidal radius (kpc) and outer logarithmic slope beyond it.
    """
    r = np.asarray(r, dtype=float)
    gcon, rs, rhos, frt, manal_rt, rho_rt = _tidal_transition(M200, c, rc, n, rt)
    rhoanal = rhos / ((r / rs) * (1.0 + (r / rs)) ** 2.0)
    manal = M200 * gcon * (np.log(1.0 + r / rs) - r / rs / (1.0 + r / rs))
    x = r / np.abs(rc)
    f = np.tanh(x)
    my_rhoanal = (
        rhoanal * f**n
        + 1.0 / (4.0 * np.pi * r**2.0 * np.abs(rc)) * manal * (1.0 - f**2.0) * n * f ** (n - 1.0)
    )
    my_rhoanal[r > rt] = rho_rt * (r[r > rt] / rt) ** (-delta)
    return my_rhoanal


def corenfw_tides_mass(r, M200, c, rc, n, rt, delta):
    """coreNFWtides enclosed mass profile (Msun)."""
    r = np.asarray(r, dtype=float)
    gcon, rs, rhos, frt, manal_rt, rho_rt = _tidal_transition(M200, c, rc, n, rt)
    manal = M200 * gcon * (np.log(1.0 + r / rs) - r / rs / (1.0 + r / rs))
    x = r / np.abs(rc)
    f = np.tanh(x)
    my_manal = manal * f**n
    mrt = manal_rt * frt**n
    my_manal[r > rt] = mrt + 4.0 * np.pi * rho_rt * rt**3.0 / (3.0 - delta) * (
        (r[r > rt] / rt) ** (3.0 - delta) - 1.0
    )
    return my_manal


def _corenfw_tides_mass_from_pars(r, pars):
    """coreNFWtides mass taking the parameters as a single sequence."""
    return corenfw_tides_mass(r, *pars)


def corenfw_tides_dlnrho_dlnr(r, M200, c, rc, n, rt, delta, dx=1e-5):
    """Logarithmic density slope of coreNFWtides (forward finite difference)."""
    dden = (
        corenfw_tides_density(r + dx, M200, c, rc, n, rt, delta)
        - corenfw_tides_density(r, M200, c, rc, n, rt, delta)
    ) / dx
    return dden / corenfw_tides_density(r, M200, c, rc, n, rt, delta) * r


def vmax_func(M200, c200, h):
    """Maximum circular velocity (km/s) of an NFW halo (Sigad et al. 2000)."""
    oden = 200.0
    r200 = (3.0 / 4.0 * M200 / (np.pi * oden * RHOCRIT)) ** (1.0 / 3.0)
    vmax = (
        0.465
        * np.sqrt(GUSE * M200 / r200)
        / np.sqrt(1.0 / c200 * np.log(1.0 + c200) - (1.0 + c200) ** (-1.0))
    )
    return vmax / KMS


# ---------------------------------------------------------------------------
# Plain NFW helpers
# ---------------------------------------------------------------------------


def nfw_density(r, rhos, rs):
    """NFW density for a characteristic density and scale radius."""
    return rhos / ((r / rs) * (1.0 + (r / rs)) ** 2.0)


def nfw_delta_c(c):
    """Characteristic overdensity of an NFW halo for the virial convention."""
    return (VIRIAL_OVERDENSITY / 3) * c**3 / (np.log(1 + c) - c / (1 + c))


def _virial_nfw(mvir, cvir):
    rv = (mvir / ((4 / 3) * np.pi * VIRIAL_OVERDENSITY * RHOCRIT)) ** (1 / 3)
    rs = rv / cvir
    rhos = RHOCRIT * nfw_delta_c(cvir)
    return rs, rhos


def nfw_virial_density(r, mvir, cvir):
    """NFW density parameterised by virial mass and concentration."""
    rs, rhos = _virial_nfw(mvir, cvir)
    return nfw_density(r, rhos, rs)


def nfw_virial_mass(r, mvir, cvir):
    """NFW enclosed mass parameterised by virial mass and concentration."""
    rs, rhos = _virial_nfw(mvir, cvir)
    return 4 * np.pi * rhos * rs**3 * (np.log((r + rs) / rs) + rs / (r + rs) - 1)


def nfw_rmax_vmax(mvir, cvir):
    """Return ``[r_max (kpc), v_max (km/s)]`` for an NFW halo."""
    rs, rhos = _virial_nfw(mvir, cvir)
    phis = 4 * np.pi * G * (rhos * MSUN / KPC**3) * (rs * KPC) ** 2
    return np.array([2.16 * rs, 0.465 * np.sqrt(phis) / 1000])


def nfw_dlnrho_dlnr(r, mvir, cvir, dx=1e-6):
    """Logarithmic slope of the virial NFW density (central difference)."""
    dden = (
        nfw_virial_density(r + dx, mvir, cvir) - nfw_virial_density(r - dx, mvir, cvir)
    ) / (2.0 * dx)
    return dden / nfw_virial_density(r, mvir, cvir) * r


# ---------------------------------------------------------------------------
# Central dark mass (e.g. remnants) modelled as a Plummer sphere
# ---------------------------------------------------------------------------


def plummer_density(r, pars):
    """Plummer density; ``pars = [M, a]``."""
    return 3.0 * pars[0] / (4.0 * np.pi * pars[1] ** 3.0) * (1.0 + r**2.0 / pars[1] ** 2.0) ** (-5.0 / 2.0)


def plummer_mass(r, pars):
    """Plummer enclosed mass; ``pars = [M, a]``."""
    return pars[0] * r**3.0 / (r**2.0 + pars[1] ** 2.0) ** (3.0 / 2.0)


# ---------------------------------------------------------------------------
# Escape velocity
# ---------------------------------------------------------------------------


def escape_velocity(r, Mpars, rmin, rmax, mass_func=None, pnts=10**3):
    """Escape velocity (km/s) at radii ``r`` for a given enclosed-mass model.

    The potential is obtained by integrating ``M(r)/r^2`` inwards from
    ``rmax`` on a linear grid of ``pnts`` points.

    Parameters
    ----------
    Mpars : sequence
        Parameters passed to ``mass_func(r, Mpars)``.
    mass_func : callable, optional
        Enclosed mass ``M(r, Mpars)``.  Defaults to coreNFWtides with
        ``Mpars = [M200, c, rc, n, rt, delta]``.
    """
    if mass_func is None:
        mass_func = _corenfw_tides_mass_from_pars
    rint = np.linspace(rmin, rmax, pnts)
    potn = cumulative_trapezoid(mass_func(rint, Mpars) / rint**2, rint)
    pot = potn[-1] - potn
    return np.interp(r, rint[1:], (2 * pot * GUSE) ** 0.5 / 1e3)
