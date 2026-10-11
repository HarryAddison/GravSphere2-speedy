"""Standard diagnostic and result plots (matplotlib).

``matplotlib`` is imported lazily; the ``corner`` and ``dynesty`` packages are
optional and only needed for the corner plot and the run diagnostics.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from . import analysis
from .likelihood import GravSphere2Model

logger = logging.getLogger(__name__)

BAND_COLOUR = "mediumseagreen"


def _pyplot(use_latex=False):
    import matplotlib.pyplot as plt

    plt.rc("font", family="serif")
    plt.rc("text", usetex=bool(use_latex))
    return plt


def plot_bands(radii, bands, xlabel, ylabel, rhalf=None, path=None, logy=False, use_latex=False):
    """Plot 68 and 95 per cent credible bands and the median of a profile.

    Returns the matplotlib figure (and saves it if ``path`` is given).
    """
    plt = _pyplot(use_latex)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.fill_between(radii, bands["lo95"], bands["hi95"], color=BAND_COLOUR, alpha=0.35, label="95%")
    ax.fill_between(radii, bands["lo68"], bands["hi68"], color=BAND_COLOUR, alpha=0.7, label="68%")
    ax.plot(radii, bands["median"], color="darkslategray", lw=1.5, label="median")
    if rhalf is not None:
        ax.axvline(rhalf, ls="--", color="gray", lw=1.5, zorder=0)
    ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path)
    return fig


def plot_corner(samples, names, path=None, use_latex=False):
    """Corner plot of the posterior (requires the ``corner`` package)."""
    import corner

    _pyplot(use_latex)
    fig = corner.corner(
        samples,
        labels=names,
        quantiles=(0.16, 0.5, 0.84),
        show_titles=True,
        title_fmt=".2f",
        bins=50,
        smooth=1.0,
        plot_datapoints=False,
        plot_density=False,
        fill_contours=True,
        color="darkslategray",
    )
    if path is not None:
        fig.savefig(path)
    return fig


def plot_run_diagnostics(results, plots_dir, use_latex=False):
    """Save dynesty run and trace plots (requires ``dynesty``)."""
    from dynesty import plotting as dyplot

    plt = _pyplot(use_latex)
    plots_dir = Path(plots_dir)
    fig, _ = dyplot.runplot(results)
    fig.savefig(plots_dir / "dynesty_run.pdf")
    plt.close(fig)
    fig, _ = dyplot.traceplot(results)
    fig.savefig(plots_dir / "dynesty_trace.pdf")
    plt.close(fig)


def make_all_plots(model: GravSphere2Model, samples, plots_dir, *, n_samples=300, n_radii=100,
                   radius_range=None, results=None, use_latex=False, seed=0):
    """Generate the standard set of result plots.

    Writes dark-matter density and slope, anisotropy, projected dispersion
    and kurtosis profiles, and (if ``corner`` is installed) a corner plot into
    ``plots_dir``.  Returns the list of files written.
    """
    plt = _pyplot(use_latex)
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    written = []

    def save(fig, name):
        path = plots_dir / name
        fig.savefig(path)
        plt.close(fig)
        written.append(path)

    radii = analysis.default_radii(model, radius_range, n_radii)
    draws = analysis.thin_samples(samples, n_samples, seed=seed)
    rh = model.rhalf

    bands = analysis.credible_bands(analysis.dm_density_profiles(model, draws, radii))
    save(plot_bands(radii, bands, r"$r$ [kpc]", r"$\rho_{\rm DM}$ [M$_\odot$ kpc$^{-3}$]", rh, logy=True,
                    use_latex=use_latex), "dm_density.pdf")

    bands = analysis.credible_bands(analysis.dm_slope_profiles(model, draws, radii))
    save(plot_bands(radii, bands, r"$r$ [kpc]", r"$d\ln\rho / d\ln r$", rh, use_latex=use_latex),
         "dm_slope.pdf")

    bands = analysis.credible_bands(analysis.anisotropy_profiles(model, draws, radii))
    save(plot_bands(radii, bands, r"$r$ [kpc]", r"$\tilde\beta$", rh, use_latex=use_latex),
         "anisotropy.pdf")

    profiles = analysis.dispersion_profiles(model, draws, radii)
    labels = {
        "sigma_los": r"$\sigma_{\rm LOS}$ [km s$^{-1}$]",
        "sigma_pm_t": r"$\sigma_{\rm PM,t}$ [km s$^{-1}$]",
        "sigma_pm_r": r"$\sigma_{\rm PM,R}$ [km s$^{-1}$]",
        "kurtosis_los": r"$\kappa_{\rm LOS}$",
        "kurtosis_pm_t": r"$\kappa_{\rm PM,t}$",
        "kurtosis_pm_r": r"$\kappa_{\rm PM,R}$",
    }
    for key, values in profiles.items():
        bands = analysis.credible_bands(values)
        save(plot_bands(radii, bands, r"$R$ [kpc]", labels[key], rh, use_latex=use_latex), f"{key}.pdf")

    try:
        fig = plot_corner(np.asarray(samples), model.priors.names, use_latex=use_latex)
        save(fig, "corner.pdf")
    except ImportError:
        logger.warning("corner is not installed; skipping the corner plot")

    if results is not None:
        try:
            plot_run_diagnostics(results, plots_dir, use_latex)
            written += [plots_dir / "dynesty_run.pdf", plots_dir / "dynesty_trace.pdf"]
        except ImportError:
            logger.warning("dynesty is not installed; skipping run diagnostics")
    return written
