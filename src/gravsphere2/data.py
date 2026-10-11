"""Loading and validation of kinematic and photometric data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import Config, ConfigError


class DataError(ValueError):
    """Raised when data files are missing or have the wrong shape."""


@dataclass
class Dataset:
    """Observed data for one stellar system.

    Distances are in kpc, velocities in km/s (relative to the system).

    Attributes
    ----------
    r_los, v_los, v_los_err
        Projected radius, line-of-sight velocity and error of each tracer.
    v_pm_t, v_pm_t_err, v_pm_r, v_pm_r_err
        Tangential and radial proper-motion velocities and errors (``None``
        without proper motions).  They share the radii ``r_los``.
    R
        Photometric tracer positions (individual mode) or bin radii (binned).
    surfden, surfden_err
        Binned tracer surface density and its error (``None`` in individual
        mode).
    half_light_radius
        Reference scale radius (kpc).
    """

    r_los: np.ndarray
    v_los: np.ndarray
    v_los_err: np.ndarray
    R: np.ndarray
    half_light_radius: float
    v_pm_t: Optional[np.ndarray] = None
    v_pm_t_err: Optional[np.ndarray] = None
    v_pm_r: Optional[np.ndarray] = None
    v_pm_r_err: Optional[np.ndarray] = None
    surfden: Optional[np.ndarray] = None
    surfden_err: Optional[np.ndarray] = None

    @property
    def proper_motions(self) -> bool:
        return self.v_pm_t is not None

    @property
    def individual(self) -> bool:
        """``True`` when individual stellar positions are fitted."""
        return self.surfden is None

    @property
    def r_pm(self) -> np.ndarray:
        """Radii of the proper-motion measurements (same as LOS)."""
        return self.r_los

    def validate(self) -> None:
        """Check array lengths and finiteness."""
        n = len(self.r_los)
        if n == 0:
            raise DataError("The kinematic sample is empty")
        same = [self.v_los, self.v_los_err]
        if self.proper_motions:
            same += [self.v_pm_t, self.v_pm_t_err, self.v_pm_r, self.v_pm_r_err]
        for arr in same:
            if len(arr) != n:
                raise DataError("All kinematic arrays must have the same length")
        if not self.individual:
            if not (len(self.R) == len(self.surfden) == len(self.surfden_err)):
                raise DataError("Binned photometry arrays must have the same length")
        for name in ("r_los", "v_los", "v_los_err", "R"):
            if not np.all(np.isfinite(getattr(self, name))):
                raise DataError(f"Non-finite values found in '{name}'")
        if np.any(self.v_los_err <= 0):
            raise DataError("LOS velocity errors must be positive")
        if self.proper_motions and (np.any(self.v_pm_t_err <= 0) or np.any(self.v_pm_r_err <= 0)):
            raise DataError("Proper-motion velocity errors must be positive")
        if np.any(np.asarray(self.R) <= 0):
            raise DataError("Photometric radii must be positive")


def _load_rows(path, nrows, what):
    if not path.is_file():
        raise DataError(f"{what} file not found: {path}")
    try:
        arr = np.loadtxt(path)
    except ValueError as exc:
        raise DataError(f"Could not read {what} file {path}: {exc}") from exc
    arr = np.atleast_2d(arr)
    if arr.shape[0] != nrows:
        raise DataError(
            f"{what} file {path} must have {nrows} rows (one per quantity), found {arr.shape[0]}"
        )
    return arr


def load_dataset(cfg: Config) -> Dataset:
    """Read the data described by ``cfg.data`` into a :class:`Dataset`."""
    d = cfg.data
    ddir = cfg.data_dir

    if d.proper_motions:
        rows = _load_rows(ddir / d.kinematics_file, 7, "Kinematics")
        r_los, v_los, v_los_err, v_pm_t, v_pm_t_err, v_pm_r, v_pm_r_err = rows
    else:
        rows = _load_rows(ddir / d.kinematics_file, 3, "Kinematics")
        r_los, v_los, v_los_err = rows
        v_pm_t = v_pm_t_err = v_pm_r = v_pm_r_err = None

    ph = d.photometry
    surfden = surfden_err = None
    if ph.mode == "binned":
        R, surfden, surfden_err = _load_rows(ddir / ph.file, 3, "Photometry")
    elif ph.source == "kinematic_tracers":
        R = r_los
    else:
        path = ddir / ph.file
        if not path.is_file():
            raise DataError(f"Photometry file not found: {path}")
        R = np.atleast_1d(np.loadtxt(path))
        if R.ndim != 1:
            raise DataError(f"Individual photometry file {path} must hold a single column/row of radii")

    if isinstance(d.half_light_radius, str):  # 'median'
        rhalf = float(np.median(R))
    else:
        rhalf = float(d.half_light_radius)

    data = Dataset(
        r_los=r_los,
        v_los=v_los,
        v_los_err=v_los_err,
        R=R,
        half_light_radius=rhalf,
        v_pm_t=v_pm_t,
        v_pm_t_err=v_pm_t_err,
        v_pm_r=v_pm_r,
        v_pm_r_err=v_pm_r_err,
        surfden=surfden,
        surfden_err=surfden_err,
    )
    data.validate()
    return data
