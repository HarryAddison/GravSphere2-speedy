"""Cosmological priors for the coreNFWtides parameters."""

import numpy as np


def concentration_cdm(M200, h):
    """Concentration-mass relation of Dutton & Macciò (2014).

    Parameters
    ----------
    M200 : float or ndarray
        Halo mass defined with respect to 200 times the critical density (Msun).
    h : float
        Dimensionless Hubble parameter.
    """
    return 10.0 ** (0.905 - 0.101 * (np.log10(M200 * h) - 12.0))


def concentration_wdm(M200, h, omega_m, rhocrit, m_wdm):
    """Warm-dark-matter modification of the c-M relation.

    Uses the formula of https://arxiv.org/pdf/1112.0330.pdf.

    Parameters
    ----------
    m_wdm : float
        Warm-dark-matter particle mass in keV.
    rhocrit : float
        Critical density in Msun kpc^-3.
    """
    c_cdm = concentration_cdm(M200, h)
    gamma1 = 15.0
    gamma2 = 0.3
    lamfseff = (
        0.049 * (m_wdm) ** (-1.11) * (omega_m / 0.25) ** (0.11) * (h / 0.7) ** (1.22) * 1000.0
    )
    lamhm = 13.93 * lamfseff
    mhm = 4.0 / 3.0 * np.pi * rhocrit * (lamhm / 2.0) ** 3.0
    return c_cdm * (1.0 + gamma1 * mhm / M200) ** (-gamma2)
