"""Spherical higher-order Jeans solver.

Given a tracer density, an enclosed-mass profile and anisotropy profiles, the
solver computes, as functions of projected radius, the second and fourth
velocity moments along the line of sight (LOS) and, optionally, for the
radial and tangential proper-motion (PM) components.  The ratio of the fourth
moment to the squared second moment gives the kurtosis used to model the
non-Gaussian velocity distribution.

The solution follows the integral forms of the second- and fourth-order Jeans
equations used in GravSphere2 (Bañares-Hernández et al.).  All line-of-sight
integrals are evaluated on a common ``(radius, angle)`` grid with Simpson's
rule; the radius loop of the original implementation is vectorised here.
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import simpson

from .constants import GUSE
from .models.anisotropy import beta as beta_profile
from .models.anisotropy import dbeta_dr
from .models.anisotropy import integrating_factor

#: Offset from pi/2 of the largest integration angle (avoids cos(theta) = 0).
THETA_EPS = 1.0e-5


@dataclass
class JeansResult:
    """Output of :func:`solve_jeans`.

    Velocity dispersions are *squared* second moments in (m/s)^2; divide by
    ``1e6`` to obtain (km/s)^2.  Proper-motion fields are ``None`` when the
    solver was run in line-of-sight-only mode.
    """

    surface_density: np.ndarray
    sigma2_los: np.ndarray
    kurtosis_los: np.ndarray
    negative_moments: bool
    sigma2_pm_t: "np.ndarray | None" = None
    sigma2_pm_r: "np.ndarray | None" = None
    kurtosis_pm_t: "np.ndarray | None" = None
    kurtosis_pm_r: "np.ndarray | None" = None


def _on_grid(func, rq):
    """Evaluate a 1D-array function on a 2D array of radii."""
    return func(rq.ravel()).reshape(rq.shape)


def non_negative_cut(rint, nonn):
    """Number of outermost grid points excluded from the non-negativity check.

    Reproduces the reference implementation: starting from the outermost grid
    point, step inwards until a radius ``<= nonn`` is found.
    """
    i = 1
    for i in range(1, len(rint) - 1):
        if rint[-i] <= nonn:
            break
    return i


def radial_dispersion_squared(
    rint, theta, tracer_density, mass_func, mass_pars, betpars, *, mass_on_grid=None
):
    """Radial velocity dispersion ``sigma_r^2(r)`` from the second-order Jeans equation.

    Parameters
    ----------
    rint : ndarray
        Radii (kpc) at which to evaluate the dispersion.
    theta : ndarray
        Integration angles; the line-of-sight radius is ``r / cos(theta)``.
    tracer_density : callable
        3D tracer density ``nu(r)``.
    mass_func : callable
        Enclosed mass ``M(r, mass_pars)`` (Msun).
    betpars : sequence
        Anisotropy parameters ``[beta0, beta_inf, r_beta, n]``.
    mass_on_grid : ndarray, optional
        Precomputed ``M(rint[:, None] / cos(theta))`` to avoid recomputation.
    """
    sth = np.sin(theta)
    cth = np.cos(theta)
    rq = rint[:, None] / cth[None, :]
    mq = mass_on_grid if mass_on_grid is not None else _on_grid(
        lambda x: mass_func(x, mass_pars), rq
    )
    nuq = _on_grid(tracer_density, rq)
    fq = integrating_factor(rq, betpars)
    nur = tracer_density(rint)
    f_r = integrating_factor(rint, betpars)
    return (
        1.0 / nur / rint / f_r * simpson(GUSE * mq * nuq * fq * sth[None, :], x=theta, axis=-1)
    )


def solve_jeans(
    R,
    r_los,
    r_pm,
    tracer_density,
    surface_density_func,
    mass_func,
    mass_pars,
    betpars,
    betppars,
    rmin,
    rmax,
    *,
    intpnts=150,
    nonn=False,
    proper_motions=True,
):
    """Solve the second- and fourth-order spherical Jeans equations.

    Parameters
    ----------
    R : ndarray
        Projected radii at which the tracer surface density is returned.
    r_los, r_pm : ndarray
        Projected radii of the LOS and PM velocity measurements.
    tracer_density, surface_density_func : callable
        3D and projected tracer densities as functions of radius.
    mass_func : callable
        Enclosed mass ``M(r, mass_pars)``.
    mass_pars : sequence
        Parameters passed through to ``mass_func``.
    betpars, betppars : sequence
        Second- and fourth-order anisotropy parameters
        ``[beta0, beta_inf, r_beta, n]`` (``beta'`` for ``betppars``).
    rmin, rmax : float
        Radial range (kpc) of the internal logarithmic grid.
    intpnts : int
        Number of grid points in radius and in integration angle.
    nonn : float or False
        If truthy, flag models with negative fourth moments for ``r <= nonn``.
    proper_motions : bool
        Also compute the proper-motion moments.

    Returns
    -------
    JeansResult
    """
    intpnts = int(intpnts)
    theta = np.linspace(0.0, np.pi / 2.0 - THETA_EPS, intpnts)
    cth = np.cos(theta)
    cth2 = cth**2.0
    cth3 = cth**3.0
    cth4 = cth**4.0

    rint = np.logspace(np.log10(rmin), np.log10(rmax), intpnts)
    rcol = rint[:, None]
    rq = rcol / cth[None, :]

    # Quantities on the (radius, angle) grid that are reused several times.
    mq = _on_grid(lambda x: mass_func(x, mass_pars), rq)
    nuq = _on_grid(tracer_density, rq)

    sigr2 = radial_dispersion_squared(
        rint, theta, tracer_density, mass_func, mass_pars, betpars, mass_on_grid=mq
    )
    sig = surface_density_func(rint)

    sigr2q = np.interp(rq.ravel(), rint, sigr2, right=0).reshape(rq.shape)
    betaq = beta_profile(rq, betpars)
    pref = 2.0 * rint / sig

    def project(integrand):
        return pref * simpson(integrand, x=theta, axis=-1)

    sig_los2 = project((1.0 - betaq * cth2) * nuq * sigr2q / cth2)
    if proper_motions:
        sig_pmr2 = project((1.0 - betaq + betaq * cth2) * nuq * sigr2q / cth2)
        sig_pmt2 = project((1.0 - betaq) * nuq * sigr2q / cth2)

    # Fourth-order radial moment.
    f_r = integrating_factor(rint, betppars)
    fq = integrating_factor(rq, betppars)
    sig_r4 = (
        1.0
        / tracer_density(rint)
        / rint
        / f_r
        * simpson(GUSE * (mq * 3.0 * sigr2q) * nuq * fq * np.sin(theta)[None, :], x=theta, axis=-1)
    )

    sigr4q = np.interp(rq.ravel(), rint, sig_r4, right=0).reshape(rq.shape)
    bpq = beta_profile(rq, betppars)
    dbpq = dbeta_dr(rq, betppars)
    bdif = 0.75 * (bpq - betaq) * sigr2q * GUSE * mq / rq**2

    sig_los4 = project(
        (
            (1.0 - 2.0 * bpq * cth2 + 0.5 * bpq * (1.0 + bpq) * cth4 - 0.25 * dbpq * rcol * cth3)
            * sigr4q
            + rcol * cth3 * bdif
        )
        * nuq
        / cth2
    )
    if proper_motions:
        common = (1.0 - bpq) * (2 - bpq) - 0.5 * rq * dbpq
        sig_pmt4 = project((common * 0.5 * sigr4q + bdif * rq) * nuq / cth2)
        sig_pmr4 = project(
            (
                (common * (1.0 - 2.0 * cth2 + cth4) * 0.5 + 2 * (1 - bpq) * cth2 - (1 - 2 * bpq) * cth4)
                * sigr4q
                + bdif * rq * (1.0 - 2.0 * cth2 + cth4)
            )
            * nuq
            / cth2
        )

    # Reject models with negative fourth moments inside r = nonn.
    negative = False
    if nonn != False:  # noqa: E712 - reproduces the reference behaviour (0 disables)
        i = non_negative_cut(rint, nonn)
        moments = [sig_los4[:-i]]
        if proper_motions:
            moments = [sig_los4[:-i], sig_pmr4[:-i], sig_pmt4[:-i]]
        if np.min(np.array(moments)) < 0.0:
            negative = True

    surface_out = np.interp(R, rint, sig, left=0, right=0)
    los2_out = np.interp(r_los, rint, sig_los2, right=0)
    los4_out = np.interp(r_los, rint, sig_los4, right=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        kurt_los = los4_out / los2_out**2

    if not proper_motions:
        return JeansResult(surface_out, los2_out, kurt_los, negative)

    pmr2_out = np.interp(r_pm, rint, sig_pmr2, right=0)
    pmt2_out = np.interp(r_pm, rint, sig_pmt2, right=0)
    pmr4_out = np.interp(r_pm, rint, sig_pmr4, right=0)
    pmt4_out = np.interp(r_pm, rint, sig_pmt4, right=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        kurt_pmt = pmt4_out / pmt2_out**2
        kurt_pmr = pmr4_out / pmr2_out**2
    return JeansResult(
        surface_out,
        los2_out,
        kurt_los,
        negative,
        sigma2_pm_t=pmt2_out,
        sigma2_pm_r=pmr2_out,
        kurtosis_pm_t=kurt_pmt,
        kurtosis_pm_r=kurt_pmr,
    )
