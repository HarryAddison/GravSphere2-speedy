"""Command-line interface.

Run ``gravsphere2 --help`` (or ``python -m gravsphere2 --help``) for usage.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import List, Optional

import numpy as np

from . import __version__
from .config import Config, ConfigError
from .data import DataError, load_dataset
from .likelihood import GravSphere2Model
from .sampler import SamplerError, load_samples, run_sampler

logger = logging.getLogger("gravsphere2")

EXIT_OK = 0
EXIT_RUNTIME_ERROR = 1
EXIT_USAGE_ERROR = 2


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="gravsphere2",
        description=(
            "GravSphere2: higher-order Jeans mass modelling of spherical stellar "
            "systems. All model, data and sampler settings are read from a YAML "
            "input file."
        ),
        epilog="Example: gravsphere2 -i examples/gaia_challenge_plumcusp.yaml --ncpus 4",
    )
    parser.add_argument("-i", "--input", required=True, metavar="FILE.yaml", help="YAML input file")
    parser.add_argument("-o", "--output-dir", metavar="DIR",
                        help="override output.directory of the input file")
    parser.add_argument("--nlive", type=int, help="override sampler.nlive")
    parser.add_argument("--ncpus", type=int, help="override sampler.ncpus (parallel likelihood evaluations)")
    parser.add_argument("--seed", type=int, help="override sampler.seed (random-number seed)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true", help="continue a run from its checkpoint file")
    mode.add_argument("--dry-run", action="store_true",
                      help="validate the input, load the data and evaluate the likelihood once; do not sample")
    mode.add_argument("--plots-only", action="store_true",
                      help="only make plots from the saved samples of a finished run")
    parser.add_argument("--no-plots", action="store_true", help="skip plotting after the run")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="debug-level logging")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _setup_logging(verbosity: int) -> None:
    """INFO-level progress messages by default, DEBUG with ``-v``."""
    level = logging.DEBUG if verbosity >= 1 else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")


def _apply_overrides(cfg: Config, args) -> None:
    if args.output_dir is not None:
        # command-line paths are relative to the current directory
        from pathlib import Path

        cfg.output.directory = str(Path(args.output_dir).expanduser().resolve())
    if args.nlive is not None:
        cfg.sampler.nlive = args.nlive
    if args.ncpus is not None:
        cfg.sampler.ncpus = args.ncpus
    if args.seed is not None:
        cfg.sampler.seed = args.seed
    if args.no_plots:
        cfg.output.make_plots = False
    cfg.validate()


def _dry_run(model: GravSphere2Model, cfg: Config) -> None:
    d = model.dataset
    print(f"run name            : {cfg.name}")
    print(f"output directory    : {cfg.run_dir}")
    print(f"tracers (kinematic) : {len(d.r_los)}  (proper motions: {d.proper_motions})")
    mode = "individual positions" if d.individual else "binned profile"
    print(f"photometry          : {mode} ({len(d.R)} entries)")
    print(f"half-light radius   : {model.rhalf:.6g} kpc")
    print(f"Jeans grid          : {model.rmin:.4g} - {model.rmax:.4g} kpc")
    print(f"free parameters     : {model.ndims}")
    theta = model.prior_transform(np.full(model.ndims, 0.5))
    start = time.perf_counter()
    lnl = model.log_likelihood(theta)
    elapsed = time.perf_counter() - start
    print(f"ln L at prior centre: {lnl:.6g}  ({elapsed * 1e3:.1f} ms per evaluation)")


def _make_plots(model, cfg, samples, results=None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from .plotting import make_all_plots

    files = make_all_plots(
        model,
        samples,
        cfg.plots_dir,
        n_samples=cfg.output.profile_samples,
        n_radii=cfg.output.n_radii,
        radius_range=cfg.output.plot_radius_range,
        results=results,
        use_latex=cfg.output.use_latex,
    )
    logger.info("Wrote %d plot(s) to %s", len(files), cfg.plots_dir)


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point of the ``gravsphere2`` command; returns the exit status."""
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    try:
        cfg = Config.load(args.input)
        _apply_overrides(cfg, args)
        dataset = load_dataset(cfg)
        model = GravSphere2Model(cfg, dataset)

        if args.dry_run:
            _dry_run(model, cfg)
            return EXIT_OK
        if args.plots_only:
            samples = load_samples(cfg)
            _make_plots(model, cfg, samples)
            return EXIT_OK

        results = run_sampler(model, cfg, resume=args.resume)
        if cfg.output.make_plots:
            _make_plots(model, cfg, np.asarray(results.samples_equal()), results)
        return EXIT_OK
    except (ConfigError, DataError) as exc:
        print(f"gravsphere2: error: {exc}", file=sys.stderr)
        return EXIT_USAGE_ERROR
    except SamplerError as exc:
        print(f"gravsphere2: error: {exc}", file=sys.stderr)
        return EXIT_RUNTIME_ERROR
    except KeyboardInterrupt:
        print("gravsphere2: interrupted (a checkpoint, if written, can be resumed with --resume)",
              file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
