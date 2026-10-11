"""Tracer (stellar) density profiles.

The main tracer model is the alpha-beta-gamma profile

.. math::

    \\rho(r) = \\rho_0 (r/r_0)^{-\\gamma}
               \\left[1 + (r/r_0)^{\\alpha}\\right]^{(\\gamma-\\beta)/\\alpha}

(a Plummer sphere is ``alpha, beta, gamma = 2, 5, 0``).  Multi-Plummer and
double alpha-beta-gamma profiles are also provided.
"""

import numpy as np
from scipy.integrate import simpson
from scipy.special import gamma as gamma_fn
from scipy.special import hyp2f1


def alpbetgam_density(r, rho0, r0, alp, bet, gam):
    """3D density of an alpha-beta-gamma profile."""
    return rho0 * (r / r0) ** (-gam) * (1.0 + (r / r0) ** alp) ** ((gam - bet) / alp)


def alpbetgam_dlnrho_dlnr(r, rho0, r0, alp, bet, gam):
    """Logarithmic density slope of an alpha-beta-gamma profile."""
    return -gam + (gam - bet) * (r / r0) ** alp * (1.0 + (r / r0) ** alp) ** (-1.0)


def alpbetgam_mass(r, rho0, r0, alp, bet, gam):
    """Enclosed mass of an alpha-beta-gamma profile (valid for alp > 0, gam < 3)."""
    x = r / r0
    a = (3.0 - gam) / alp
    b = (bet - gam) / alp
    c = a + 1.0
    hyper = hyp2f1(a, b, c, -(x**alp))
    return -(1 / (gam - 3)) * (4 * np.pi * rho0 * r0**3) * x ** (3 - gam) * hyper


def alpbetgam_total_mass(rho0, r0, alp, bet, gam):
    """Total (r -> infinity) mass of an alpha-beta-gamma profile (bet > 3)."""
    return (
        12.566370614359172
        * rho0
        * ((1 / r0) ** alp) ** ((-3 + gam) / alp)
        * (1 / r0) ** (-gam)
        * gamma_fn((bet - gam) / alp + (-3 + gam) / alp)
        * gamma_fn((3 - gam) / alp)
        / (gamma_fn((bet - gam) / alp) * alp)
    )


def _abel_projection(density, r, intpnts):
    """Project a spherical density to a surface density (Abel integral).

    ``density`` is a callable of the 3D radius.  The integral is evaluated
    with Simpson's rule over the angle ``theta`` for all radii at once.
    """
    theta = np.linspace(0, np.pi / 2.0 - 1.0e-30, num=intpnts)
    cth = np.cos(theta)
    cth2 = cth**2
    r = np.asarray(r, dtype=float)
    q = r[:, None] / cth[None, :]
    y = density(q)
    return 2.0 * r * simpson(y / cth2[None, :], x=theta, axis=-1)


def alpbetgam_surface(r, rho0, r0, alp, bet, gam, intpnts):
    """Projected (surface) density of an alpha-beta-gamma profile.

    Parameters
    ----------
    intpnts : int
        Number of integration points along the line of sight.
    """
    return _abel_projection(
        lambda q: alpbetgam_density(q, rho0, r0, alp, bet, gam), r, intpnts
    )


def double_alpbetgam_density(r, rho01, rho02, r01, r02, alp1, alp2, bet1, bet2, gam1, gam2):
    """Sum of two alpha-beta-gamma density profiles."""
    return alpbetgam_density(r, rho01, r01, alp1, bet1, gam1) + alpbetgam_density(
        r, rho02, r02, alp2, bet2, gam2
    )


def double_alpbetgam_mass(r, rho01, rho02, r01, r02, alp1, alp2, bet1, bet2, gam1, gam2):
    """Enclosed mass of two summed alpha-beta-gamma profiles (numerical)."""
    den = double_alpbetgam_density(
        r, rho01, rho02, r01, r02, alp1, alp2, bet1, bet2, gam1, gam2
    )
    mass = np.zeros(len(r))
    for i in range(3, len(r)):
        mass[i] = simpson(4.0 * np.pi * r[:i] ** 2.0 * den[:i], x=r[:i])
    return mass


def double_alpbetgam_surface(
    r, rho01, rho02, r01, r02, alp1, alp2, bet1, bet2, gam1, gam2, intpnts
):
    """Projected density of two summed alpha-beta-gamma profiles."""
    return _abel_projection(
        lambda q: double_alpbetgam_density(
            q, rho01, rho02, r01, r02, alp1, alp2, bet1, bet2, gam1, gam2
        ),
        r,
        intpnts,
    )


# ---------------------------------------------------------------------------
# Multi-Plummer profiles.  ``pars = [M_1..M_n, a_1..a_n]``.
# ---------------------------------------------------------------------------


def _plummer_scales(pars, mode="normal"):
    half = int(len(pars) / 2.0)
    masses = pars[0:half]
    scales = pars[half:len(pars)]
    if mode == "seq":
        # Each scale is offset by the previous raw scale parameter.
        scales = np.asarray(scales, dtype=float)
        scales = np.concatenate(([scales[0]], scales[1:] + scales[:-1]))
    return masses, scales


def multiplum_density(r, pars, mode="normal"):
    """Density of a sum of Plummer spheres (``mode='seq'`` offsets each scale by the previous one)."""
    masses, scales = _plummer_scales(pars, mode)
    out = np.zeros(len(r))
    for mass, a in zip(masses, scales):
        out = out + 3.0 * mass / (4.0 * np.pi * a**3.0) * (1.0 + r**2.0 / a**2.0) ** (-5.0 / 2.0)
    return out


def multiplum_surface(r, pars, mode="normal"):
    """Surface density of a sum of Plummer spheres."""
    masses, scales = _plummer_scales(pars, mode)
    out = np.zeros(len(r))
    for mass, a in zip(masses, scales):
        out = out + mass * a**2.0 / (np.pi * (a**2.0 + r**2.0) ** 2.0)
    return out


def multiplum_dlnrho_dlnr(r, pars, mode="normal"):
    """Logarithmic density slope of a sum of Plummer spheres."""
    masses, scales = _plummer_scales(pars, mode)
    den = np.zeros(len(r))
    dden = np.zeros(len(r))
    for mass, a in zip(masses, scales):
        den = den + 3.0 * mass / (4.0 * np.pi * a**3.0) * (1.0 + r**2.0 / a**2.0) ** (-5.0 / 2.0)
        dden = dden - 15.0 * mass / (4.0 * np.pi * a**3.0) * r / a**2.0 * (
            1.0 + r**2.0 / a**2.0
        ) ** (-7.0 / 2.0)
    return dden * r / den


def multiplum_mass(r, pars, mode="normal"):
    """Enclosed mass of a sum of Plummer spheres."""
    masses, scales = _plummer_scales(pars, mode)
    out = np.zeros(len(r))
    for mass, a in zip(masses, scales):
        out = out + mass * r**3.0 / (r**2.0 + a**2.0) ** (3.0 / 2.0)
    return out


def threeplum_surface(r, m1, m2, m3, a1, a2, a3):
    """Surface density of three Plummer components."""
    return multiplum_surface(r, [m1, m2, m3, a1, a2, a3])


def threeplum_density(r, m1, m2, m3, a1, a2, a3):
    """Density of three Plummer components."""
    return multiplum_density(r, [m1, m2, m3, a1, a2, a3])


def threeplum_mass(r, m1, m2, m3, a1, a2, a3):
    """Enclosed mass of three Plummer components."""
    return multiplum_mass(r, [m1, m2, m3, a1, a2, a3])


def threeplum_half_light_radius(m1, m2, m3, a1, a2, a3):
    """Projected half-light radius of the three-Plummer model."""
    ranal = np.logspace(-3, 1, int(500))
    surf = threeplum_surface(ranal, m1, m2, m3, a1, a2, a3)
    menc_half = 0.0
    i = 3
    while menc_half < (m1 + m2 + m3) / 2.0:
        menc_half = 2.0 * np.pi * simpson(surf[:i] * ranal[:i], x=ranal[:i])
        i = i + 1
    return ranal[i - 1]
