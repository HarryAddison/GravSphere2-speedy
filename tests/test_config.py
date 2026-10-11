import tempfile
import unittest
from pathlib import Path

import yaml

from gravsphere2.config import Config, ConfigError, dump_config

from .helpers import EXAMPLES, TempDirMixin, write_synthetic_data


class TestConfigLoading(TempDirMixin, unittest.TestCase):
    def write(self, d, name="cfg.yaml"):
        path = self.tmp / name
        path.write_text(yaml.safe_dump(d))
        return path

    def test_examples_load(self):
        for path in sorted(EXAMPLES.glob("*.yaml")):
            with self.subTest(example=path.name):
                cfg = Config.load(path)
                self.assertTrue(cfg.data_dir.is_dir())

    def test_defaults_and_resolution(self):
        d = write_synthetic_data(self.tmp)
        d["data"]["directory"] = "."  # relative to the input file
        d["output"]["directory"] = "results"
        cfg = Config.load(self.write(d))
        self.assertEqual(cfg.data_dir.resolve(), self.tmp.resolve())
        self.assertEqual(cfg.run_dir, cfg.base_dir / "results" / "synthetic")
        self.assertEqual(cfg.sampler.nlive, 20)
        self.assertEqual(cfg.priors.mass.delta, (3.01, 5.0))

    def test_scientific_notation_without_dot_is_numeric(self):
        d = write_synthetic_data(self.tmp)
        d["stellar_mass"] = {"value": "4.3e7"}  # what PyYAML yields for 4.3e7
        path = self.tmp / "sci.yaml"
        path.write_text(yaml.safe_dump(d).replace("'4.3e7'", "4.3e7"))
        cfg = Config.load(path)
        self.assertEqual(cfg.stellar_mass.value, 4.3e7)

    def test_unknown_key_is_rejected_with_hint(self):
        d = write_synthetic_data(self.tmp)
        d["sampler"]["nlives"] = 3
        with self.assertRaisesRegex(ConfigError, "nlives.*sampler"):
            Config.load(self.write(d))

    def test_missing_file(self):
        with self.assertRaisesRegex(ConfigError, "not found"):
            Config.load(self.tmp / "nope.yaml")

    def test_invalid_yaml(self):
        path = self.tmp / "bad.yaml"
        path.write_text("a: [1, 2")
        with self.assertRaisesRegex(ConfigError, "parse"):
            Config.load(path)

    def test_non_mapping_top_level(self):
        path = self.tmp / "list.yaml"
        path.write_text("- 1\n- 2\n")
        with self.assertRaisesRegex(ConfigError, "mapping"):
            Config.load(path)

    def test_round_trip(self):
        cfg = Config.from_dict(write_synthetic_data(self.tmp), base_dir=self.tmp)
        out = self.tmp / "dump.yaml"
        dump_config(cfg, out)
        again = Config.load(out)
        self.assertEqual(cfg.to_dict(), again.to_dict())


class TestConfigValidation(TempDirMixin, unittest.TestCase):
    def cfg(self, **mods):
        d = write_synthetic_data(self.tmp)
        for dotted, value in mods.items():
            keys = dotted.split("__")
            node = d
            for k in keys[:-1]:
                node = node.setdefault(k, {})
            node[keys[-1]] = value
        return Config.from_dict(d, base_dir=self.tmp)

    def assertInvalid(self, pattern, **mods):
        with self.assertRaisesRegex(ConfigError, pattern):
            self.cfg(**mods)

    def test_valid_baseline(self):
        self.cfg()

    def test_bad_photometry_mode(self):
        self.assertInvalid("mode", data__photometry__mode="rings")

    def test_binned_needs_reference_parameters(self):
        self.assertInvalid("reference_parameters", priors__tracer__reference_parameters=[1, 2])

    def test_binned_needs_numeric_rhalf(self):
        self.assertInvalid("half_light_radius", data__half_light_radius=None)
        self.assertInvalid("median", data__half_light_radius="median")

    def test_negative_rhalf(self):
        self.assertInvalid("positive", data__half_light_radius=-1.0)

    def test_kinematics_required(self):
        self.assertInvalid("kinematics_file", data__kinematics_file="")

    def test_range_order(self):
        self.assertInvalid("min .* smaller", priors__mass__c200=[10, 5])

    def test_range_shape(self):
        self.assertInvalid("two-element", priors__mass__c200=[1, 2, 3])

    def test_log_sampled_radii_positive(self):
        self.assertInvalid("positive", priors__mass__rc=[0.0, 1.0])

    def test_delta_must_exceed_three(self):
        self.assertInvalid("delta", priors__mass__delta=[2.0, 5.0])

    def test_sampler_limits(self):
        self.assertInvalid("nlive", sampler__nlive=1)
        self.assertInvalid("ncpus", sampler__ncpus=0)

    def test_stellar_mass_limits(self):
        self.assertInvalid("non-negative", stellar_mass__value=-1.0)
        self.assertInvalid("fractional_error", stellar_mass__fractional_error=1.5)

    def test_grid_order(self):
        self.assertInvalid("rmin", grids__rmin=5.0, grids__rmax=1.0)

    def test_name_without_path_separators(self):
        self.assertInvalid("name", name="a/b")

    def test_individual_defaults_to_median(self):
        d = write_synthetic_data(self.tmp, individual=True)
        d["data"].pop("half_light_radius")
        self.assertEqual(Config.from_dict(d, base_dir=self.tmp).data.half_light_radius, "median")

    def test_binned_cannot_use_kinematic_positions(self):
        self.assertInvalid("binned", data__photometry__source="kinematic_tracers")


if __name__ == "__main__":
    unittest.main()
