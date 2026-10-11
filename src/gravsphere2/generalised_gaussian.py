"""Generalised-Gaussian velocity distributions (legacy / binned-data tools).

These functions describe a generalised Gaussian plus a Gaussian background,
convolved with Gaussian measurement errors, as used in earlier GravSphere
versions for fitting velocity distributions in radial bins.  They are not
used by the default GravSphere2 likelihood.

``theta = [vz_mean, alpha, beta, back_amp, back_mean, back_sigma]``.
"""

import numpy as np
from scipy.integrate import quad
from scipy.special import gamma


def monte_carlo_integral(func, a, b, n, rng=None):
    """Fast 1D Monte-Carlo integral of ``func`` over ``[a, b]`` with ``n`` samples."""
    rng = np.random.default_rng() if rng is None else rng
    xrand = rng.uniform(a, b, n)
    integral = func(xrand).sum()
    return (b - a) / float(n) * integral


def velpdf_noerr(vz, theta):
    """PDF without measurement errors."""
    vzmean, alp, bet, backamp, backmean, backsig = theta[:6]
    return (1.0 - backamp) * bet / (2.0 * alp * gamma(1.0 / bet)) * np.exp(
        -((np.abs(vz - vzmean) / alp) ** bet)
    ) + backamp / (np.sqrt(2.0 * np.pi) * backsig) * np.exp(
        -0.5 * (vz - backmean) ** 2.0 / backsig**2.0
    )


def velpdf_fast(vz, vzerr, theta):
    """Approximate error-convolved PDF (errors added in quadrature)."""
    vzmean = theta[0]
    bet = theta[2]
    fgamma = gamma(1.0 / bet) / gamma(3.0 / bet)
    alp = np.sqrt(theta[1] ** 2.0 + vzerr**2.0 * fgamma)
    backamp = theta[3]
    backmean = theta[4]
    backsig = np.sqrt(theta[5] ** 2.0 + vzerr**2.0)
    return (1.0 - backamp) * bet / (2.0 * alp * gamma(1.0 / bet)) * np.exp(
        -((np.abs(vz - vzmean) / alp) ** bet)
    ) + backamp / (np.sqrt(2.0 * np.pi) * backsig) * np.exp(
        -0.5 * (vz - backmean) ** 2.0 / backsig**2.0
    )


def velpdf_integrand(vz, vzerr, vzint, theta):
    """Integrand of the error convolution (Gaussian errors)."""
    vzmean, alp, bet, backamp, backmean, backsig = theta[:6]
    gauss_err = 1.0 / (np.sqrt(2.0 * np.pi) * vzerr) * np.exp(-0.5 * (vz - vzint) ** 2.0 / vzerr**2.0)
    return (1.0 - backamp) * bet / (2.0 * alp * gamma(1.0 / bet)) * np.exp(
        -((np.abs(vzint - vzmean) / alp) ** bet)
    ) * gauss_err + backamp / (np.sqrt(2.0 * np.pi) * backsig) * np.exp(
        -0.5 * (vzint - backmean) ** 2.0 / backsig**2.0
    ) * gauss_err


def velocity_second_moment(theta):
    """Return ``<v^2>^(1/2)`` of the generalised Gaussian."""
    alp = theta[1]
    bet = theta[2]
    return np.sqrt(alp**2.0 * gamma(3.0 / bet) / gamma(1.0 / bet))


def kurtosis(theta):
    """Kurtosis of the generalised Gaussian."""
    bet = theta[2]
    return gamma(5.0 / bet) * gamma(1.0 / bet) / (gamma(3.0 / bet)) ** 2.0


def velocity_fourth_moment(theta):
    """Return ``<v^4>`` of the generalised Gaussian."""
    return kurtosis(theta) * velocity_second_moment(theta) ** 4.0


def _convolve(vz, vzerr, theta, integrate):
    vzmean = theta[0]
    sig = velocity_second_moment(theta)
    vzlow = -sig * 10 + vzmean
    vzhigh = sig * 10 + vzmean
    if isinstance(vz, np.ndarray):
        pdf = np.zeros(len(vz))
        for i in range(len(vz)):
            pdf[i] = integrate(
                lambda vzint, i=i: velpdf_integrand(vz[i], vzerr[i], vzint, theta), vzlow, vzhigh
            )
        return pdf
    return integrate(lambda vzint: velpdf_integrand(vz, vzerr, vzint, theta), vzlow, vzhigh)


def velpdf(vz, vzerr, theta):
    """Error-convolved PDF using full numerical quadrature (slow, accurate)."""
    return _convolve(vz, vzerr, theta, lambda f, a, b: quad(f, a, b)[0])


def velpdf_monte(vz, vzerr, theta, npnts=500, rng=None):
    """Error-convolved PDF using Monte-Carlo integration."""
    return _convolve(vz, vzerr, theta, lambda f, a, b: monte_carlo_integral(f, a, b, int(npnts), rng))


def interpolate_fourth_moment(ranal, rbin, vzfourbin):
    """Interpolate (and extrapolate) the fourth moment over a radial grid."""
    return np.interp(ranal, rbin, vzfourbin)
