"""Velocity-anisotropy models.

GravSphere2 uses a generalised Osipkov-Merritt-like profile

.. math::

    \\beta(r) = \\beta_0 + (\\beta_\\infty - \\beta_0)
                \\left[1 + (r_\\beta / r)^n\\right]^{-1}

for the second-order anisotropy and a second copy of the same functional
form for the fourth-order counterpart ``beta'``.  The sampler works with the
*symmetrised* anisotropy ``beta_tilde = beta / (2 - beta)``.

Anisotropy parameters are passed as a length-4 array
``[beta0, beta_inf, r_beta, n]``.
"""

import numpy as np

# Indices into the anisotropy parameter array.
BETA0, BETA_INF, R_BETA, N_BETA = 0, 1, 2, 3


def symmetrised_to_beta(beta_tilde):
    """Convert symmetrised anisotropy ``beta_tilde`` to ``beta``."""
    return 2.0 * beta_tilde / (1.0 + beta_tilde)


def beta_to_symmetrised(beta):
    """Convert anisotropy ``beta`` to its symmetrised counterpart."""
    return beta / (2.0 - beta)


def beta(r, betpars):
    """Anisotropy profile ``beta(r)``.

    Parameters
    ----------
    r : array_like
        Radii (kpc).
    betpars : sequence of float
        ``[beta0, beta_inf, r_beta, n]``.
    """
    rt = betpars[R_BETA]
    n = betpars[N_BETA]
    bet0 = betpars[BETA0]
    betinf = betpars[BETA_INF]
    return bet0 + (betinf - bet0) * (1.0 / (1.0 + (rt / r) ** n))


def dbeta_dr(r, betpars):
    """Radial derivative ``d beta / d r`` of :func:`beta`."""
    rt = betpars[R_BETA]
    n = betpars[N_BETA]
    bet0 = betpars[BETA0]
    betinf = betpars[BETA_INF]
    return (
        n
        * rt**n
        * r ** (-(1.0 + n))
        * (betinf - bet0)
        * (1.0 / (1.0 + (rt / r) ** n) ** 2)
    )


def integrating_factor(r, betpars):
    """Integrating factor ``f(r) = exp(2 int beta/r dr)`` of the Jeans equation."""
    rt = betpars[R_BETA]
    n = betpars[N_BETA]
    bet0 = betpars[BETA0]
    betinf = betpars[BETA_INF]
    return r ** (2.0 * betinf) * ((rt / r) ** n + 1.0) ** (2.0 / n * (betinf - bet0))


def osipkov(r, r0):
    """Osipkov-Merritt anisotropy ``r^2 / (r^2 + r0^2)``."""
    return r**2.0 / (r**2.0 + r0**2.0)


def osipkov_integrating_factor(r, r0):
    """Integrating factor for the Osipkov-Merritt profile (generalised form, n=2)."""
    n0 = 2.0
    bet0 = 0.0
    betinf = 1.0
    return r ** (2.0 * betinf) * ((r0 / r) ** n0 + 1.0) ** (2.0 / n0 * (betinf - bet0))
