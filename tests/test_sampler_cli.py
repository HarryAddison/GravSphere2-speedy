import contextlib
import io
import sys
import unittest
from unittest import mock

import numpy as np
import yaml

from gravsphere2 import cli
from gravsphere2.config import Config
from gravsphere2.data import load_dataset
from gravsphere2.likelihood import GravSphere2Model
from gravsphere2.sampler import (
    CONFIG_COPY_NAME, SamplerError, checkpoint_path, load_samples, run_sampler, samples_path,
)

from . import fake_dynesty
from .helpers import EXAMPLES, TempDirMixin, write_synthetic_data


class SamplerTestCase(TempDirMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        fake_dynesty.CALLS.clear()
        patcher = mock.patch.dict(sys.modules, {"dynesty": fake_dynesty.make_module()})
        patcher.start()
        self.addCleanup(patcher.stop)

    def config_file(self, **overrides):
        d = write_synthetic_data(self.tmp)
        for section, values in overrides.items():
            d.setdefault(section, {}).update(values)
        path = self.tmp / "input.yaml"
        path.write_text(yaml.safe_dump(d))
        return path


class TestRunSampler(SamplerTestCase):
    def build(self, **overrides):
        cfg = Config.load(self.config_file(**overrides))
        return cfg, GravSphere2Model(cfg, load_dataset(cfg))

    def test_fresh_run_writes_outputs(self):
        cfg, model = self.build(sampler={"nlive": 33, "maxiter": 5, "seed": 4})
        results = run_sampler(model, cfg)
        self.assertEqual(results.samples_equal().shape[1], model.ndims)
        init = dict(fake_dynesty.CALLS)["init"]
        self.assertEqual(init["nlive"], 33)
        self.assertNotIn("pool", init)
        run = dict(fake_dynesty.CALLS)["run"]
        self.assertEqual(run["maxiter"], 5)
        self.assertTrue(run["use_stop"])
        self.assertTrue((cfg.run_dir / CONFIG_COPY_NAME).exists())
        self.assertTrue(cfg.plots_dir.is_dir())
        self.assertTrue(samples_path(cfg).exists())
        self.assertTrue(checkpoint_path(cfg).exists())
        self.assertEqual(load_samples(cfg).shape, results.samples_equal().shape)

    def test_resume_requires_checkpoint(self):
        cfg, model = self.build()
        with self.assertRaisesRegex(SamplerError, "checkpoint"):
            run_sampler(model, cfg, resume=True)

    def test_resume_restores_and_continues(self):
        cfg, model = self.build()
        run_sampler(model, cfg)
        fake_dynesty.CALLS.clear()
        run_sampler(model, cfg, resume=True)
        run = dict(fake_dynesty.CALLS)["run"]
        self.assertTrue(run["resume"])
        self.assertIn("restored_from", dict(fake_dynesty.CALLS)["init"])

    def test_parallel_run_uses_pool(self):
        cfg, model = self.build(sampler={"ncpus": 2})
        pool = mock.MagicMock()
        with mock.patch("multiprocessing.Pool", return_value=pool):
            run_sampler(model, cfg)
        init = dict(fake_dynesty.CALLS)["init"]
        self.assertIs(init["pool"], pool)
        self.assertEqual(init["queue_size"], 2)
        pool.close.assert_called_once()
        pool.join.assert_called_once()

    def test_load_samples_without_anything_fails(self):
        cfg, _ = self.build()
        with self.assertRaisesRegex(SamplerError, "No samples"):
            load_samples(cfg)

    def test_missing_dynesty_gives_clear_error(self):
        cfg, model = self.build()
        with mock.patch.dict(sys.modules, {"dynesty": None}):
            with self.assertRaisesRegex(SamplerError, "pip install dynesty"):
                run_sampler(model, cfg)


class TestCli(SamplerTestCase):
    def run_cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_dry_run(self):
        code, out, _ = self.run_cli("-i", str(self.config_file()), "--dry-run")
        self.assertEqual(code, 0)
        self.assertIn("free parameters     : 19", out)
        self.assertIn("ln L at prior centre", out)
        self.assertEqual(fake_dynesty.CALLS, [])

    def test_dry_run_on_shipped_examples(self):
        for path in sorted(EXAMPLES.glob("*.yaml")):
            with self.subTest(example=path.name):
                code, out, _ = self.run_cli("-i", str(path), "--dry-run")
                self.assertEqual(code, 0)

    def test_missing_input_file_exit_code(self):
        code, _, err = self.run_cli("-i", str(self.tmp / "missing.yaml"))
        self.assertEqual(code, cli.EXIT_USAGE_ERROR)
        self.assertIn("not found", err)

    def test_bad_config_exit_code(self):
        path = self.config_file(sampler={"nlive": 1})
        code, _, err = self.run_cli("-i", str(path))
        self.assertEqual(code, cli.EXIT_USAGE_ERROR)
        self.assertIn("nlive", err)

    def test_bad_data_exit_code(self):
        path = self.config_file()
        (self.tmp / "kin.txt").write_text("1 2 3\n")
        code, _, err = self.run_cli("-i", str(path))
        self.assertEqual(code, cli.EXIT_USAGE_ERROR)
        self.assertIn("rows", err)

    def test_input_is_required(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as ctx:
            cli.main([])
        self.assertEqual(ctx.exception.code, 2)

    def test_modes_are_mutually_exclusive(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli.main(["-i", "x.yaml", "--resume", "--dry-run"])

    def test_command_line_overrides(self):
        out_dir = self.tmp / "elsewhere"
        code, _, _ = self.run_cli(
            "-i", str(self.config_file()), "-o", str(out_dir), "--nlive", "11", "--ncpus", "1",
            "--seed", "5", "--no-plots",
        )
        self.assertEqual(code, 0)
        init = dict(fake_dynesty.CALLS)["init"]
        self.assertEqual(init["nlive"], 11)
        self.assertTrue((out_dir / "synthetic" / "Sampler Chains").is_dir())

    def test_full_run_with_plots(self):
        path = self.config_file(output={"make_plots": True, "profile_samples": 3, "n_radii": 8})
        code, _, _ = self.run_cli("-i", str(path))
        self.assertEqual(code, 0)
        plots = self.tmp / "out" / "synthetic" / "Plots"
        names = {p.name for p in plots.glob("*.pdf")}
        self.assertTrue({"dm_density.pdf", "dm_slope.pdf", "anisotropy.pdf", "sigma_los.pdf",
                         "kurtosis_pm_r.pdf"} <= names)

    def test_plots_only_uses_saved_samples(self):
        path = self.config_file(output={"make_plots": False, "profile_samples": 2, "n_radii": 6})
        self.assertEqual(self.run_cli("-i", str(path))[0], 0)
        plots = self.tmp / "out" / "synthetic" / "Plots"
        self.assertFalse(list(plots.glob("*.pdf")))
        self.assertEqual(self.run_cli("-i", str(path), "--plots-only")[0], 0)
        self.assertTrue((plots / "dm_density.pdf").exists())

    def test_resume_without_checkpoint_is_runtime_error(self):
        code, _, err = self.run_cli("-i", str(self.config_file()), "--resume")
        self.assertEqual(code, cli.EXIT_RUNTIME_ERROR)
        self.assertIn("checkpoint", err)

    def test_version(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit) as ctx:
            cli.main(["--version"])
        self.assertEqual(ctx.exception.code, 0)
        self.assertIn("2.0.0", out.getvalue())

    def test_python_dash_m_entry_point(self):
        import runpy

        with mock.patch.object(sys, "argv", ["gravsphere2", "--version"]), \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as ctx:
            runpy.run_module("gravsphere2", run_name="__main__")
        self.assertEqual(ctx.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
