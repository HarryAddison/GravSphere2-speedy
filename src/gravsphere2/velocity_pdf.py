"""Velocity probability distribution functions.

GravSphere2 models individual stellar velocities with the non-Gaussian
kernel-convolved PDFs of Sanders & Evans (2020), whose shape is controlled by
the variance and kurtosis (1.8 < kurtosis < 6).  Gaussian measurement errors
are convolved analytically.

For kurtosis below 3 a uniform kernel is used, above 3 a Laplace kernel.  The
pre-tabulated maps from kurtosis to kernel parameter are built once at import.

Link: https://arxiv.org/abs/2009.07858 (J. Sanders' ``gh_alternative``).
"""

import numpy as np

from .line_profiles import lnalpha, lnerfc, lnerfcx, log_ndtr, logdiffexp

# Kurtosis -> uniform-kernel parameter map (1.8 < kurtosis < 3.0)
_AAU = np.linspace(1e-6, 100, 10**4)
_KKU = 3.0 - (2 * _AAU**4 / 15) * (1 + _AAU**2 / 3) ** (-2)

# Kurtosis -> Laplace-kernel parameter map (3.0 < kurtosis < 6.0)
_AAL = np.linspace(1e-6, 100, 10**4)
_KKL = 3.0 + 12 * _AAL**4 * (2 * _AAL**2 + 1) ** (-2)


def asolve_uniform(kurt):
    """Map kurtosis (1.8 < k < 3) to the uniform-kernel parameter."""
    return np.interp(kurt, _KKU[::-1], _AAU[::-1])


def asolve_laplace(kurt):
    """Map kurtosis (3 < k < 6) to the Laplace-kernel parameter."""
    return np.interp(kurt, _KKL, _AAL)


def ln_uniform_kernel_pdf(x, var, kurt, err):
    """Log-PDF for the uniform kernel (1.8 < kurt < 3.0).

    Parameters
    ----------
    x : array_like
        Velocities relative to the system mean.
    var : array_like
        Velocity variance.
    kurt : array_like
        Kurtosis of the intrinsic distribution.
    err : array_like
        Gaussian measurement error convolved with the intrinsic PDF.
    """
    sigma = var**0.5
    w = x / sigma
    werr = err / sigma
    a = asolve_uniform(kurt)
    b = ((1 + a**2 / 3)) ** 0.5
    t = np.sqrt(1.0 + b * b * werr * werr)
    w0 = 0.0
    it = 1.0 / t
    bw = b * (w - w0)
    ln_pdf = np.log(0.5 * b / a) + np.where(
        (b * w + a) * it < 0.0,
        logdiffexp(log_ndtr((bw + a) * it), log_ndtr((bw - a) * it)),
        logdiffexp(log_ndtr(-(bw - a) * it), log_ndtr(-(bw + a) * it)),
    )
    ln_pdf -= np.log(sigma)
    return ln_pdf


def ln_laplace_kernel_pdf(x, var, kurt, err):
    """Log-PDF for the Laplace kernel (3.0 < kurt < 6.0); arguments as above."""
    sigma = var**0.5
    w = x / sigma
    werr = err / sigma
    a = asolve_laplace(kurt)
    b = (2 * a**2 + 1) ** 0.5
    mean_w = 0.0
    t = np.sqrt(1.0 + b * b * werr * werr)
    ap = a
    am = a

    arg_u = t * t - 2 * ap * b * (w - mean_w)
    positive_term = np.zeros_like(x)
    prefactor = np.log(b / (4.0 * ap))
    if isinstance(kurt, np.ndarray):
        prefactor = prefactor[arg_u < 0.0]
    positive_term[arg_u < 0.0] = (
        prefactor
        + (arg_u / 2.0 / ap**2)[arg_u < 0.0]
        + lnerfc(((t * t - ap * b * (w - mean_w)) / np.sqrt(2) / t / ap)[arg_u < 0.0])
    )
    prefactor = np.log(b / ap)
    if isinstance(kurt, np.ndarray):
        prefactor = prefactor[arg_u > 0.0]
    positive_term[arg_u > 0.0] = (
        0.5 * np.log(np.pi / 8.0)
        + prefactor
        + lnalpha((b * (w - mean_w) / t)[arg_u > 0.0])
        + lnerfcx(((t * t - ap * b * (w - mean_w)) / np.sqrt(2) / t / ap)[arg_u > 0.0])
    )

    arg_u = t * t + 2 * am * b * (w - mean_w)
    negative_term = np.zeros_like(x)
    prefactor = np.log(b / (4.0 * am))
    if isinstance(kurt, np.ndarray):
        prefactor = prefactor[arg_u < 0.0]
    negative_term[arg_u < 0.0] = (
        prefactor
        + (arg_u / 2.0 / am**2)[arg_u < 0.0]
        + lnerfc(((t * t + am * b * (w - mean_w)) / np.sqrt(2) / t / am)[arg_u < 0.0])
    )
    prefactor = np.log(b / am)
    if isinstance(kurt, np.ndarray):
        prefactor = prefactor[arg_u > 0.0]
    negative_term[arg_u > 0.0] = (
        0.5 * np.log(np.pi / 8.0)
        + prefactor
        + lnalpha((b * (w - mean_w) / t)[arg_u > 0.0])
        + lnerfcx(((t * t + am * b * (w - mean_w)) / np.sqrt(2) / t / am)[arg_u > 0.0])
    )
    return np.logaddexp(positive_term, negative_term) - np.log(sigma)


def ln_velocity_pdf(x, var, kurt, err):
    """Log-PDF for a given variance, kurtosis (1.8 < kurt < 6.0) and Gaussian error.

    Selects the uniform kernel for ``kurt < 3`` and the Laplace kernel
    otherwise.
    """
    return np.select(
        [kurt < 3.0, kurt >= 3.0],
        [ln_uniform_kernel_pdf(x, var, kurt, err), ln_laplace_kernel_pdf(x, var, kurt, err)],
    )
