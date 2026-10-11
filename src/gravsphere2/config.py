"""Configuration handling for GravSphere2.

All user-adjustable settings (data files, priors, grids, sampler options)
live in a YAML *input file*.  This module defines the schema as dataclasses,
loads and validates the YAML, and resolves relative paths.

Relative paths in the input file are interpreted relative to the directory
containing the input file.

Example
-------
>>> from gravsphere2.config import Config
>>> cfg = Config.load("examples/gaia_challenge_plumcusp.yaml")   # doctest: +SKIP
>>> cfg.sampler.nlive                                           # doctest: +SKIP
500
"""

from __future__ import annotations

import dataclasses
import re
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import yaml

Range = tuple  # (lower, upper)


class ConfigError(ValueError):
    """Raised when the input file is malformed or inconsistent."""


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@dataclass
class OutputConfig:
    """Where results are written.

    Outputs go to ``<directory>/<name>/`` with ``Plots`` and
    ``Sampler Chains`` sub-directories (as in the original code).

    ``profile_samples``
        Number of posterior draws used for the profile credible bands.
    ``n_radii`` / ``plot_radius_range``
        Radial grid of the plots; the range ``[min, max]`` (kpc) defaults to
        ``[Rhalf / 10, 10 Rhalf]`` clipped to the Jeans grid.
    """

    directory: str = "output"
    make_plots: bool = True
    use_latex: bool = False
    profile_samples: int = 300
    n_radii: int = 100
    plot_radius_range: Optional[list] = None


@dataclass
class PhotometryConfig:
    """Tracer-density (photometric) data.

    ``mode``
        ``"binned"``: a file with three rows ``R, surface_density,
        surface_density_error``; or ``"individual"``: fit individual stellar
        positions (most accurate when available).
    ``source``
        ``"file"`` to read ``file`` or ``"kinematic_tracers"`` to use the
        projected radii of the kinematic sample as photometric positions
        (individual mode only).
    """

    mode: str = "binned"
    file: Optional[str] = None
    source: str = "file"


@dataclass
class DataConfig:
    """Input data.

    The kinematics file holds one *row per quantity*.  With proper motions
    the rows are ``R, vLOS, vLOS_err, vPMt, vPMt_err, vPMR, vPMR_err``;
    without them ``R, vLOS, vLOS_err``.  Distances are in kpc and velocities
    in km/s, relative to the system (subtract global averages beforehand).
    """

    directory: str = "."
    kinematics_file: str = ""
    proper_motions: bool = True
    photometry: PhotometryConfig = field(default_factory=PhotometryConfig)
    half_light_radius: Union[float, str, None] = None
    escape_velocity_check: bool = False


@dataclass
class GridConfig:
    """Radial grids used by the Jeans solver and the tracer profile.

    Entries left as ``null`` take the defaults of the reference code:
    ``rmin = min(Rhalf/100, min(R)/10)``, ``rmax = max(100 Rhalf, 2 max(R))``,
    ``nonnegative_radius = rmax/2``, with the tracer-profile grid equal to
    the Jeans grid.
    """

    rmin: Optional[float] = None
    rmax: Optional[float] = None
    nonnegative_radius: Optional[float] = None
    tracer_rmin: Optional[float] = None
    tracer_rmax: Optional[float] = None
    tracer_points: int = 300


@dataclass
class StellarMassConfig:
    """Baryonic (stellar) mass that follows the tracer light.

    ``value = 0`` gives a massless-tracer, dark-matter-only model.  When
    positive, the mass is a free parameter with a flat prior of
    ``value * (1 +/- fractional_error)``.
    """

    value: float = 0.0
    fractional_error: float = 0.25


@dataclass
class TracerPriorConfig:
    """Priors on the alpha-beta-gamma tracer profile (binned mode).

    ``reference_parameters`` are pre-fitted ``[rho0, r0, alpha, beta, gamma]``;
    flat priors extend ``tolerance`` (fractional) around them.
    """

    reference_parameters: Optional[list] = None
    tolerance: float = 0.5


@dataclass
class AnisotropyPriorConfig:
    """Flat prior ranges ``[min, max]`` for the symmetrised anisotropy.

    The same ranges are used for the second-order profile and its
    fourth-order counterpart.
    """

    beta0: Range = (-1.0, 1.0)
    beta_inf: Range = (-1.0, 1.0)
    log10_r_beta: Range = (-2.0, 0.0)
    eta: Range = (1.0, 3.0)


@dataclass
class MassPriorConfig:
    """Flat priors for the coreNFWtides dark-matter profile.

    ``rc`` and ``rt`` (kpc) are given linearly but sampled in log10.
    """

    log10_M200: Range = (7.5, 11.5)
    c200: Range = (1.0, 100.0)
    rc: Range = (1e-2, 10.0)
    n: Range = (0.0, 1.0)
    rt: Range = (1.0, 20.0)
    delta: Range = (3.01, 5.0)


@dataclass
class PriorsConfig:
    """All prior settings."""

    tracer: TracerPriorConfig = field(default_factory=TracerPriorConfig)
    anisotropy: AnisotropyPriorConfig = field(default_factory=AnisotropyPriorConfig)
    mass: MassPriorConfig = field(default_factory=MassPriorConfig)


@dataclass
class SamplerConfig:
    """Dynamic nested sampling (dynesty) options."""

    nlive: int = 500
    ncpus: int = 1
    use_stop: bool = True
    maxiter: Optional[int] = None
    checkpoint_name: str = "checkpoint"
    seed: Optional[int] = None
    print_progress: bool = True


@dataclass
class Config:
    """Top-level configuration (the contents of the YAML input file)."""

    name: str = "gravsphere2_run"
    output: OutputConfig = field(default_factory=OutputConfig)
    data: DataConfig = field(default_factory=DataConfig)
    grids: GridConfig = field(default_factory=GridConfig)
    stellar_mass: StellarMassConfig = field(default_factory=StellarMassConfig)
    priors: PriorsConfig = field(default_factory=PriorsConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    base_dir: Path = field(default_factory=Path.cwd, metadata={"yaml": False})

    # -- construction ------------------------------------------------------

    @classmethod
    def load(cls, path) -> "Config":
        """Read, build and validate a configuration from a YAML file."""
        path = Path(path)
        if not path.is_file():
            raise ConfigError(f"Input file not found: {path}")
        with open(path, "r", encoding="utf-8") as handle:
            try:
                raw = yaml.safe_load(handle)
            except yaml.YAMLError as exc:
                raise ConfigError(f"Could not parse YAML in {path}: {exc}") from exc
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ConfigError(f"{path}: the top level of the input file must be a mapping")
        cfg = cls.from_dict(raw, base_dir=path.resolve().parent)
        return cfg

    @classmethod
    def from_dict(cls, raw: dict, base_dir=None) -> "Config":
        """Build and validate a configuration from a plain dictionary."""
        raw = dict(raw)
        if "base_dir" in raw:
            raise ConfigError("'base_dir' is not a valid input-file key")
        cfg = _build(cls, raw, "<root>")
        cfg.base_dir = Path(base_dir) if base_dir is not None else Path.cwd()
        cfg.validate()
        return cfg

    # -- helpers -----------------------------------------------------------

    def resolve(self, path) -> Path:
        """Resolve ``path`` relative to the input-file directory."""
        path = Path(path).expanduser()
        return path if path.is_absolute() else (self.base_dir / path)

    @property
    def data_dir(self) -> Path:
        return self.resolve(self.data.directory)

    @property
    def run_dir(self) -> Path:
        """Output directory of this run: ``<output.directory>/<name>``."""
        return self.resolve(self.output.directory) / self.name

    @property
    def plots_dir(self) -> Path:
        return self.run_dir / "Plots"

    @property
    def chains_dir(self) -> Path:
        return self.run_dir / "Sampler Chains"

    def to_dict(self) -> dict:
        """Serialisable dictionary of the configuration (without ``base_dir``)."""
        return _to_plain(self)

    # -- validation --------------------------------------------------------

    def validate(self) -> None:
        """Check internal consistency; raises :class:`ConfigError`."""
        if not self.name or any(ch in self.name for ch in "/\\"):
            raise ConfigError("'name' must be a non-empty string without path separators")

        d = self.data
        if not d.kinematics_file:
            raise ConfigError("data.kinematics_file is required")
        ph = d.photometry
        if ph.mode not in ("binned", "individual"):
            raise ConfigError("data.photometry.mode must be 'binned' or 'individual'")
        if ph.source not in ("file", "kinematic_tracers"):
            raise ConfigError("data.photometry.source must be 'file' or 'kinematic_tracers'")
        if ph.source == "file" and not ph.file:
            raise ConfigError("data.photometry.file is required when source is 'file'")
        if ph.mode == "binned" and ph.source != "file":
            raise ConfigError("binned photometry must be read from a file")

        rh = d.half_light_radius
        if rh is None:
            if ph.mode == "binned":
                raise ConfigError("data.half_light_radius is required for binned photometry")
            d.half_light_radius = "median"
        elif isinstance(rh, str):
            if rh != "median":
                raise ConfigError("data.half_light_radius must be a positive number or 'median'")
            if ph.mode == "binned":
                raise ConfigError("'median' half-light radius needs individual photometry")
        elif rh <= 0:
            raise ConfigError("data.half_light_radius must be positive")

        if ph.mode == "binned":
            ref = self.priors.tracer.reference_parameters
            if ref is None or len(ref) != 5:
                raise ConfigError(
                    "priors.tracer.reference_parameters must list 5 values "
                    "[rho0, r0, alpha, beta, gamma] for binned photometry"
                )
        if self.priors.tracer.tolerance < 0:
            raise ConfigError("priors.tracer.tolerance must be non-negative")

        if self.stellar_mass.value < 0:
            raise ConfigError("stellar_mass.value must be non-negative")
        if not 0 <= self.stellar_mass.fractional_error < 1:
            raise ConfigError("stellar_mass.fractional_error must be in [0, 1)")

        for section_name, section in (
            ("priors.anisotropy", self.priors.anisotropy),
            ("priors.mass", self.priors.mass),
        ):
            for f in dataclasses.fields(section):
                rng = getattr(section, f.name)
                _check_range(f"{section_name}.{f.name}", rng)
                setattr(section, f.name, (float(rng[0]), float(rng[1])))
        for name in ("rc", "rt"):
            if self.priors.mass.__dict__[name][0] <= 0:
                raise ConfigError(f"priors.mass.{name} must be positive (it is sampled in log10)")
        if self.priors.mass.delta[0] <= 3:
            raise ConfigError("priors.mass.delta must exceed 3 for a finite enclosed mass")
        if self.priors.anisotropy.eta[0] <= 0:
            raise ConfigError("priors.anisotropy.eta must be positive")

        g = self.grids
        for name in ("rmin", "rmax", "nonnegative_radius", "tracer_rmin", "tracer_rmax"):
            val = getattr(g, name)
            if val is not None and val < 0:
                raise ConfigError(f"grids.{name} must be non-negative")
        if g.rmin is not None and g.rmax is not None and g.rmin >= g.rmax:
            raise ConfigError("grids.rmin must be smaller than grids.rmax")
        if g.tracer_points < 10:
            raise ConfigError("grids.tracer_points must be at least 10")

        o = self.output
        if o.profile_samples < 1 or o.n_radii < 3:
            raise ConfigError("output.profile_samples must be >= 1 and output.n_radii >= 3")
        if o.plot_radius_range is not None:
            _check_range("output.plot_radius_range", o.plot_radius_range)
            if o.plot_radius_range[0] <= 0:
                raise ConfigError("output.plot_radius_range must be positive")

        s = self.sampler
        if s.nlive < 2:
            raise ConfigError("sampler.nlive must be at least 2")
        if s.ncpus < 1:
            raise ConfigError("sampler.ncpus must be at least 1")
        if s.maxiter is not None and s.maxiter < 1:
            raise ConfigError("sampler.maxiter must be positive")


# ---------------------------------------------------------------------------
# Generic dataclass <-> dict helpers
# ---------------------------------------------------------------------------


def _check_range(label, rng):
    if not isinstance(rng, (list, tuple)) or len(rng) != 2:
        raise ConfigError(f"{label} must be a two-element list [min, max]")
    try:
        lo, hi = float(rng[0]), float(rng[1])
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{label} must contain numbers") from exc
    if not lo < hi:
        raise ConfigError(f"{label}: min ({lo}) must be smaller than max ({hi})")


# PyYAML follows YAML 1.1, where ``4.3e7`` (no dot, no sign) is a *string*.
_SCI_NOTATION = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)[eE][-+]?\d+$")


def _coerce_numbers(value):
    """Convert YAML-1.1 scientific-notation strings (``1e5``) to floats."""
    if isinstance(value, str) and _SCI_NOTATION.match(value.strip()):
        return float(value)
    if isinstance(value, (list, tuple)):
        return [_coerce_numbers(v) for v in value]
    return value


def _build(cls, raw, where):
    """Recursively construct dataclass ``cls`` from mapping ``raw``."""
    if not isinstance(raw, dict):
        raise ConfigError(f"Section '{where}' must be a mapping, got {type(raw).__name__}")
    hints = typing.get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls) if f.metadata.get("yaml", True)}
    unknown = sorted(set(raw) - set(fields))
    if unknown:
        raise ConfigError(
            f"Unknown key(s) {unknown} in section '{where}'. Valid keys: {sorted(fields)}"
        )
    kwargs = {}
    for name, value in raw.items():
        hint = hints[name]
        if dataclasses.is_dataclass(hint):
            kwargs[name] = _build(hint, {} if value is None else value, f"{where}.{name}")
        elif hint is str:
            kwargs[name] = value
        else:
            kwargs[name] = _coerce_numbers(value)
    return cls(**kwargs)


def _to_plain(obj):
    if dataclasses.is_dataclass(obj):
        return {
            f.name: _to_plain(getattr(obj, f.name))
            for f in dataclasses.fields(obj)
            if f.metadata.get("yaml", True)
        }
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def dump_config(cfg: Config, path) -> None:
    """Write the effective configuration to ``path`` as YAML."""
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg.to_dict(), handle, sort_keys=False)
