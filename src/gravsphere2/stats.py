"""Data-preparation and summary-statistics helpers."""

import numpy as np
from scipy.integrate import simpson


def surface_renormalisation(rbin, surfden):
    """Total projected integral and projected half-light radius of a profile.

    Returns
    -------
    rhalf, menc_tot : float
        Half-light radius and the integral of the surface density.
    """
    ranal = np.linspace(0, 10, int(5000))
    surfden_ranal = np.interp(ranal, rbin, surfden, left=0, right=0)
    menc_tot = 2.0 * np.pi * simpson(surfden_ranal * ranal, x=ranal)
    menc_half = 0.0
    i = 3
    while menc_half < menc_tot / 2.0:
        menc_half = 2.0 * np.pi * simpson(surfden_ranal[:i] * ranal[:i], x=ranal[:i])
        i = i + 1
    return ranal[i - 1], menc_tot


def bin_data(R, ms, nbin):
    """Bin tracer positions into radial bins with ``nbin`` tracers each.

    Parameters
    ----------
    R : ndarray
        Projected radii.
    ms : ndarray
        Weight of each tracer (use ones for unweighted counts).
    nbin : float
        Total weight per bin.

    Returns
    -------
    rbin, surfden, surfdenerr, rhalf
        Bin centres, normalised surface density and error, and the
        projected half-light radius.
    """
    index = np.argsort(R)
    right_bin_edge = np.zeros(len(R))
    norm = np.zeros(len(R))
    cnt = 0
    jsum = 0
    for i in range(len(R)):
        if jsum < nbin:
            norm[cnt] = norm[cnt] + ms[index[i]]
            right_bin_edge[cnt] = R[index[i]]
            jsum = jsum + ms[index[i]]
        if jsum >= nbin:
            jsum = 0.0
            cnt = cnt + 1
    right_bin_edge = right_bin_edge[:cnt]
    norm = norm[:cnt]

    surfden = np.zeros(cnt)
    rbin = np.zeros(cnt)
    for i in range(len(rbin)):
        if i == 0:
            surfden[i] = norm[i] / (np.pi * right_bin_edge[i] ** 2.0)
            rbin[i] = right_bin_edge[i] / 2.0
        else:
            surfden[i] = norm[i] / (np.pi * right_bin_edge[i] ** 2.0 - np.pi * right_bin_edge[i - 1] ** 2.0)
            rbin[i] = (right_bin_edge[i] + right_bin_edge[i - 1]) / 2.0
    surfdenerr = surfden / np.sqrt(nbin)

    rhalf, menc_tot = surface_renormalisation(rbin, surfden)
    surfden = surfden / menc_tot
    surfdenerr = surfdenerr / menc_tot
    return rbin, surfden, surfdenerr, rhalf


def confidence_intervals(array):
    """Median and 68 / 95 / 99.7 per cent intervals along axis 0.

    Parameters
    ----------
    array : ndarray
        Samples along axis 0 (any number of trailing dimensions).

    Returns
    -------
    median, low68, high68, low95, high95, low997, high997
        Order statistics ``sorted[int(q * n)]`` for q = 0.5, 0.16, 0.84,
        0.025, 0.975, 0.0015 and 0.9985.
    """
    ordered = np.sort(np.asarray(array), axis=0)
    n = len(ordered)
    pick = lambda q: ordered[int(q / 100.0 * n)]
    return (
        ordered[int(n / 2.0)],
        pick(16.0),
        pick(84.0),
        pick(2.5),
        pick(97.5),
        pick(0.15),
        pick(99.85),
    )


def add_surface_density_anchor_points(x, y, yerr):
    """Add pseudo data at large and small radii to a surface-density profile.

    When using a negative Plummer component, extra "data points" bounded by
    zero and the outermost point disfavour models with globally negative
    tracer density.
    """
    addpnts = len(x)
    xouter = np.max(x)
    youter = np.min(y)
    xinner = np.min(x)
    yinner = np.max(y)

    xadd_right = np.logspace(np.log10(xouter), np.log10(xouter * 1000), addpnts)
    yadd_right = np.zeros(addpnts) + youter / 2.0
    yerradd_right = yadd_right

    xadd_left = np.logspace(np.log10(xinner), np.log10(xinner / 1000), addpnts)
    yadd_left = np.zeros(addpnts) + yinner
    yerradd_left = yadd_left / 2.0

    x = np.concatenate((x, xadd_right))
    y = np.concatenate((y, yadd_right))
    yerr = np.concatenate((yerr, yerradd_right))
    x = np.concatenate((xadd_left, x))
    y = np.concatenate((yadd_left, y))
    yerr = np.concatenate((yerradd_left, yerr))
    return x, y, yerr
