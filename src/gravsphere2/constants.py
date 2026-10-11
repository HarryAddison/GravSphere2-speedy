"""Physical constants, unit conversions and cosmology used by GravSphere2.

Internal units
--------------
* lengths in kpc, masses in solar masses, densities in Msun kpc^-3
* velocities in km/s at the interface (the Jeans solver works in SI-like
  units through :data:`GUSE` and converts at the end)

The numerical values are identical to those in the original
``constants.py`` of the GravSphere2 release.
"""

import numpy as np

# Fundamental
G = 6.67e-11  # m^3 kg^-1 s^-2

# Units
MSUN = 1.989e30  # kg
KPC = 3.086e19  # m
KMS = 1000.0  # m/s per km/s

# Conversions
DEG = 360.0 / 2.0 / np.pi
ARCSEC = 360.0 / 2.0 / np.pi * 60.0 * 60.0
ARCMIN = ARCSEC / 60.0
MSUNKPC3_TO_GEVCM3 = 3.7973132915271756e-08
KPC_CM = 1e3 * 3.0856775 * 1e18
YEAR = 365.0 * 24.0 * 60.0 * 60.0  # s

# Cosmology [Msun kpc^-3]
OMEGA_M_H2 = 0.12
RHOCRIT = 135.05
ODEN = 200  # overdensity defining M200
H = 0.7
OMEGA_M = OMEGA_M_H2 / H**2.0

# Specific to GravSphere: G * Msun / kpc, so that G*M/r with M in Msun and
# r in kpc has units of (m/s)^2.
GUSE = G * MSUN / KPC

# Virial overdensity used by the NFW helper functions (Bryan & Norman-like).
VIRIAL_OVERDENSITY = 102
