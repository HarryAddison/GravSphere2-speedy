"""Running the nested sampler and persisting its results.

``dynesty`` is imported lazily so that the rest of the package (data loading,
likelihood, tests) works without it installed.
"""

from __future__ import annotations

import logging
import pickle
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .config import Config, dump_config
from .likelihood import GravSphere2Model

logger = logging.getLogger(__name__)

CONFIG_COPY_NAME = "config_used.yaml"


class SamplerError(RuntimeError):
    """Raised when the sampler cannot be run (e.g. missing checkpoint)."""


def prepare_output_dirs(cfg: Config) -> None:
    """Create the output tree and save a copy of the effective configuration."""
    for path in (cfg.plots_dir, cfg.chains_dir):
        path.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, cfg.run_dir / CONFIG_COPY_NAME)


def checkpoint_path(cfg: Config) -> Path:
    """Path of the dynesty checkpoint file."""
    return cfg.chains_dir / cfg.sampler.checkpoint_name


def samples_path(cfg: Config) -> Path:
    """Path of the saved equal-weight posterior samples."""
    return cfg.chains_dir / f"{cfg.name}_samples_equal.npy"


def _import_dynesty():
    try:
        import dynesty
    except ImportError as exc:  # pragma: no cover - exercised via mocking
        raise SamplerError(
            "dynesty is required to run the sampler: pip install dynesty"
        ) from exc
    return dynesty


@contextmanager
def _pool(ncpus):
    """Yield a multiprocessing pool (or ``None`` for a single CPU)."""
    if ncpus <= 1:
        yield None
        return
    from multiprocessing import Pool

    pool = Pool(ncpus)
    try:
        yield pool
    finally:
        pool.close()
        pool.join()


def run_sampler(model: GravSphere2Model, cfg: Config, resume: bool = False):
    """Run dynamic nested sampling and save the results.

    Parameters
    ----------
    model : GravSphere2Model
    cfg : Config
    resume : bool
        Restore the sampler from the checkpoint file and continue.

    Returns
    -------
    dynesty.results.Results
    """
    dynesty = _import_dynesty()
    prepare_output_dirs(cfg)
    chk = checkpoint_path(cfg)
    s = cfg.sampler

    run_kwargs = {"checkpoint_file": str(chk), "use_stop": s.use_stop, "print_progress": s.print_progress}
    if s.maxiter is not None:
        run_kwargs["maxiter"] = s.maxiter

    with _pool(s.ncpus) as pool:
        if resume:
            if not chk.exists():
                raise SamplerError(f"Cannot resume: checkpoint file not found: {chk}")
            logger.info("Restoring sampler from %s", chk)
            sampler = dynesty.DynamicNestedSampler.restore(str(chk), pool=pool)
            run_kwargs["resume"] = True
        else:
            kwargs = {"nlive": s.nlive}
            if pool is not None:
                kwargs.update(pool=pool, queue_size=s.ncpus)
            if s.seed is not None:
                kwargs["rstate"] = np.random.default_rng(s.seed)
            sampler = dynesty.DynamicNestedSampler(
                model.log_likelihood, model.prior_transform, model.ndims, **kwargs
            )
        logger.info("Running dynesty (%d parameters, nlive=%d, ncpus=%d)", model.ndims, s.nlive, s.ncpus)
        sampler.run_nested(**run_kwargs)

    results = sampler.results
    save_results(results, cfg)
    return results


def save_results(results, cfg: Config) -> None:
    """Persist the dynesty results object and the equal-weight samples."""
    pkl = cfg.chains_dir / f"{cfg.name}_results.pkl"
    with open(pkl, "wb") as handle:
        pickle.dump(results, handle)
    np.save(samples_path(cfg), np.asarray(results.samples_equal()))
    logger.info("Saved results to %s", cfg.chains_dir)


def load_samples(cfg: Config) -> np.ndarray:
    """Load equal-weight posterior samples of a finished run.

    Falls back to restoring the dynesty checkpoint if the ``.npy`` file does
    not exist.
    """
    path = samples_path(cfg)
    if path.exists():
        return np.load(path)
    chk = checkpoint_path(cfg)
    if chk.exists():
        dynesty = _import_dynesty()
        return np.asarray(dynesty.DynamicNestedSampler.restore(str(chk)).results.samples_equal())
    raise SamplerError(f"No samples or checkpoint found for run '{cfg.name}' in {cfg.chains_dir}")
