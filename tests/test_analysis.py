import unittest

import numpy as np

from gravsphere2 import analysis

from .helpers import TempDirMixin, make_config
from gravsphere2.data import load_dataset
from gravsphere2.likelihood import GravSphere2Model


class TestCredibleBands(unittest.TestCase):
    def test_percentiles_of_uniform_sample(self):
        values = np.tile(np.linspace(0, 100, 10001)[:, None], (1, 2))
        b = analysis.credible_bands(values)
        self.assertAlmostEqual(b["median"][0], 50.0, places=6)
        self.assertAlmostEqual(b["lo68"][1], 16.0, places=6)
        self.assertAlmostEqual(b["hi95"][0], 97.5, places=6)
        self.assertEqual(set(b), {"median", "lo68", "hi68", "lo95", "hi95", "lo997", "hi997"})

    def test_ordering_and_nan_tolerance(self):
        rng = np.random.default_rng(0)
        values = rng.normal(size=(500, 4))
        values[0, 0] = np.nan
        b = analysis.credible_bands(values)
        for lo, hi in (("lo997", "lo95"), ("lo95", "lo68"), ("lo68", "median"), ("median", "hi68"),
                       ("hi68", "hi95"), ("hi95", "hi997")):
            self.assertTrue(np.all(b[lo] <= b[hi]))


class TestSampleHelpers(unittest.TestCase):
    def test_thin_samples(self):
        s = np.arange(100.0)[:, None]
        self.assertEqual(len(analysis.thin_samples(s, 10, seed=1)), 10)
        np.testing.assert_array_equal(analysis.thin_samples(s, 1000), s)
        np.testing.assert_array_equal(analysis.thin_samples(s, 10, seed=1), analysis.thin_samples(s, 10, seed=1))

    def test_parameter_summary(self):
        s = np.column_stack([np.linspace(0, 1, 1001), np.linspace(10, 20, 1001)])
        out = analysis.parameter_summary(s, ["a", "b"])
        self.assertAlmostEqual(out["a"][0], 0.5)
        self.assertAlmostEqual(out["b"][1], 11.6)
        with self.assertRaises(ValueError):
            analysis.parameter_summary(s, ["only_one"])


class TestProfiles(TempDirMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        cfg = make_config(self.tmp)
        self.model = GravSphere2Model(cfg, load_dataset(cfg))
        rng = np.random.default_rng(2)
        self.samples = np.array([self.model.prior_transform(rng.uniform(0.4, 0.6, self.model.ndims))
                                 for _ in range(4)])
        self.radii = analysis.default_radii(self.model, n=12)

    def test_default_radii(self):
        r = analysis.default_radii(self.model, n=12)
        self.assertEqual(len(r), 12)
        self.assertTrue(np.all(np.diff(np.log(r)) > 0))
        self.assertAlmostEqual(r[0], max(self.model.rhalf / 10, self.model.rmin))
        r2 = analysis.default_radii(self.model, radius_range=[0.1, 1.0], n=5)
        self.assertAlmostEqual(r2[0], 0.1)
        self.assertAlmostEqual(r2[-1], 1.0)

    def test_profile_shapes_and_signs(self):
        rho = analysis.dm_density_profiles(self.model, self.samples, self.radii)
        self.assertEqual(rho.shape, (4, 12))
        self.assertTrue(np.all(rho > 0))
        slope = analysis.dm_slope_profiles(self.model, self.samples, self.radii)
        self.assertTrue(np.all(slope < 1) and np.all(np.isfinite(slope)))
        beta_t = analysis.anisotropy_profiles(self.model, self.samples, self.radii)
        self.assertTrue(np.all(np.abs(beta_t) <= 1.0 + 1e-9))

    def test_dispersion_profiles(self):
        out = analysis.dispersion_profiles(self.model, self.samples, self.radii)
        self.assertEqual(set(out), {"sigma_los", "kurtosis_los", "sigma_pm_t", "sigma_pm_r",
                                    "kurtosis_pm_t", "kurtosis_pm_r"})
        self.assertEqual(out["sigma_los"].shape, (4, 12))
        self.assertTrue(np.all(out["sigma_los"] > 0))

    def test_dispersion_profiles_los_only(self):
        cfg = make_config(self.tmp, proper_motions=False)
        model = GravSphere2Model(cfg, load_dataset(cfg))
        out = analysis.dispersion_profiles(model, self.samples[:2], self.radii)
        self.assertEqual(set(out), {"sigma_los", "kurtosis_los"})


if __name__ == "__main__":
    unittest.main()
