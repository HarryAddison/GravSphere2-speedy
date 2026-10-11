"""Line-of-sight acceleration bounds (e.g. for pulsar timing constraints)."""

import numpy as np

from .constants import KPC


def max_los_acceleration(R, mass_func, central_mass_func, mass_pars, star_radii, star_profile,
                         mstar, G, rmin, rmax, intpnts=150):
    """Maximum line-of-sight acceleration (SI units) at projected radii ``R``.

    Along each line of sight the acceleration component towards the observer
    is ``a(r) sin(theta)``; the maximum over the line of sight is returned.

    Parameters
    ----------
    mass_func, central_mass_func : callable
        Enclosed mass of the main component and of a central dark mass,
        each called as ``f(r, mass_pars)``.
    star_radii, star_profile : ndarray
        Enclosed-mass profile of the stellar component (normalised to 1).
    mstar : float
        Stellar mass (Msun); ``0`` disables the stellar component.
    G : float
        Gravitational constant in the units of ``mass`` / ``r`` (``GUSE``).
    """
    intpnts = int(intpnts)
    rq = np.logspace(np.log10(rmin), np.log10(rmax), intpnts)
    mq = mass_func(rq, mass_pars) + central_mass_func(rq, mass_pars)
    if mstar > 0:
        mq = mq + mstar * np.interp(rq, star_radii, star_profile)
    aq = G * mq / rq**2.0 / KPC

    theta = np.linspace(0.0, np.pi / 2.0 - 1.0e-5, intpnts)
    sth = np.sin(theta)
    cth = np.cos(theta)

    amax = np.zeros(len(R))
    for i in range(len(R)):
        aqq = np.interp(R[i] / cth, rq, aq)
        amax[i] = np.max(np.abs(aqq * sth))
    return amax
