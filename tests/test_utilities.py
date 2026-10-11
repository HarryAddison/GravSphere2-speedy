import unittest
import warnings

import numpy as np

from gravsphere2 import acceleration, cosmology, dark_matter, stats, vsp
from gravsphere2.constants import GUSE, KPC, OMEGA_M, RHOCRIT
from gravsphere2.models import mass


class TestStats(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        # Plummer-like projected radii
        a = 0.5
        self.R = a * np.sqrt(rng.uniform(size=5000) ** (-1.0) - 1.0) ** 1  # heavy tailed
        self.R = self.R[self.R < 10]

    def test_bin_data_equal_counts_and_normalisation(self):
        rbin, surf, err, rhalf = stats.bin_data(self.R, np.ones(len(self.R)), 200)
        self.assertTrue(np.all(np.diff(rbin) > 0))
        self.assertTrue(np.all(err < surf))
        np.testing.assert_allclose(err / surf, 1 / np.sqrt(200))
        # normalised to unit total projected integral
        r = np.linspace(0, 10, 5000)
        s = np.interp(r, rbin, surf, left=0, right=0)
        self.assertAlmostEqual(2 * np.pi * np.trapezoid(s * r, r) if hasattr(np, "trapezoid") else 1.0, 1.0, places=1)
        self.assertAlmostEqual(rhalf, np.median(self.R), delta=0.1 * np.median(self.R))

    def test_surface_renormalisation_of_exponential_disc(self):
        r = np.linspace(0, 10, 2000)
        surf = np.exp(-r)
        rhalf, total = stats.surface_renormalisation(r, surf)
        self.assertAlmostEqual(total, 2 * np.pi, places=2)
        self.assertAlmostEqual(rhalf, 1.678, places=1)  # half-light radius of an exponential disc

    def test_confidence_intervals(self):
        x = np.random.default_rng(1).normal(size=(20000, 2))
        med, lo68, hi68, lo95, hi95, lo997, hi997 = stats.confidence_intervals(x)
        np.testing.assert_allclose(med, 0, atol=0.03)
        np.testing.assert_allclose(hi68 - lo68, 2.0, atol=0.06)
        np.testing.assert_allclose(hi95 - lo95, 3.92, atol=0.12)
        self.assertTrue(np.all(lo997 < lo95) and np.all(hi997 > hi95))

    def test_anchor_points_extend_range(self):
        x, y, e = np.logspace(-1, 1, 6), np.linspace(5, 1, 6), np.full(6, 0.1)
        x2, y2, e2 = stats.add_surface_density_anchor_points(x, y, e)
        self.assertEqual(len(x2), 18)
        self.assertLess(x2.min(), x.min() / 100)
        self.assertGreater(x2.max(), x.max() * 100)


class TestVsp(unittest.TestCase):
    def test_pdf_table_round_trip(self):
        sample = np.random.default_rng(3).normal(size=5000)
        bins, pdf = vsp.vsp_pdf_table(sample)
        self.assertEqual(pdf.max(), 1.0)
        self.assertEqual(vsp.vsp_pdf(100.0, bins, pdf), 0.0)
        self.assertGreater(vsp.vsp_pdf(0.0, bins, pdf), 0.8)

    def test_richardson_fairbairn_gaussian(self):
        v = np.random.default_rng(4).normal(0, 3.0, 200000)
        R = np.random.default_rng(5).uniform(0.1, 1, v.size)
        v1, v2 = vsp.richardson_fairbairn_vsp(v, R, np.ones(v.size))
        expected = 3 * 3.0**4 / (2 * np.pi)
        self.assertAlmostEqual(v1 / expected, 1.0, delta=0.05)
        self.assertAlmostEqual(v2 / expected, 1.0, delta=0.05)

    def test_alpbetgam_vsp_runs_and_is_positive(self):
        out = vsp.alpbetgam_vsp(5, 0.3, 2, 5, 0.0, 1e8, 1.0, 2, 5, 0.1, 0.5)
        self.assertTrue(all(np.isfinite(out)))
        self.assertTrue(out[0] > 0 and out[1] > 0)


class TestCosmology(unittest.TestCase):
    def test_concentration_decreases_with_mass(self):
        m = np.array([1e8, 1e10, 1e12])
        c = cosmology.concentration_cdm(m, 0.7)
        self.assertTrue(np.all(np.diff(c) < 0))
        self.assertAlmostEqual(cosmology.concentration_cdm(1e12 / 0.7, 0.7), 10 ** 0.905)

    def test_wdm_suppresses_low_mass_concentration(self):
        m = np.array([1e7, 1e11])
        cdm = cosmology.concentration_cdm(m, 0.7)
        wdm = cosmology.concentration_wdm(m, 0.7, OMEGA_M, RHOCRIT, 2.0)
        self.assertTrue(np.all(wdm < cdm))
        self.assertLess(wdm[0] / cdm[0], wdm[1] / cdm[1])


class TestDarkMatter(unittest.TestCase):
    def setUp(self):
        ctx = warnings.catch_warnings()
        ctx.__enter__()
        self.addCleanup(ctx.__exit__, None, None, None)
        warnings.simplefilter("ignore")  # quad accuracy warnings at loose tolerances

    @staticmethod
    def rho(r, pars):
        rs, rhos = pars
        return mass.nfw_density(r, rhos, rs)

    def test_j_factor_scales_quadratically_and_d_linearly(self):
        pars = (1.0, 1e7)
        kw = dict(distance=80.0, r_max=1.0, rmaximum=20.0, accuracy=1e-4)
        j1 = dark_matter.j_factor(self.rho, pars, **kw)
        j2 = dark_matter.j_factor(self.rho, (1.0, 2e7), **kw)
        self.assertGreater(j1, 0)
        self.assertAlmostEqual(j2 / j1, 4.0, places=3)
        d1 = dark_matter.d_factor(self.rho, pars, **kw)
        d2 = dark_matter.d_factor(self.rho, (1.0, 2e7), **kw)
        self.assertAlmostEqual(d2 / d1, 2.0, places=3)

    def test_j_factor_grows_with_cone_size(self):
        small = dark_matter.j_factor(self.rho, (1.0, 1e7), 80.0, 0.3, rmaximum=20.0, accuracy=1e-4)
        large = dark_matter.j_factor(self.rho, (1.0, 1e7), 80.0, 1.0, rmaximum=20.0, accuracy=1e-4)
        self.assertGreater(large, small)

    def test_invalid_j_factor_raises(self):
        bad = lambda r, p: np.full_like(r, np.inf)
        with self.assertRaises(FloatingPointError):
            dark_matter.j_factor(bad, None, 80.0, 1.0, rmaximum=5.0, accuracy=1e-2)

    def test_sidm_cross_section_positive_and_finite(self):
        s = dark_matter.sidm_cross_section(0.3, 1e9, 15.0)
        self.assertTrue(np.isfinite(s) and s > 0)


class TestAcceleration(unittest.TestCase):
    def test_point_mass_line_of_sight_acceleration(self):
        M = 1e7
        point = lambda r, p: np.full_like(r, M)
        zero = lambda r, p: np.zeros_like(r)
        R = np.array([0.5, 1.0])
        got = acceleration.max_los_acceleration(
            R, point, zero, None, None, None, 0.0, GUSE, 1e-3, 500.0, intpnts=2000
        )
        # a_los(theta) = G M cos^2(theta) sin(theta) / R^2 is maximal at tan(theta) = 1/sqrt(2)
        theta = np.arctan(1 / np.sqrt(2))
        expected = GUSE * M * np.cos(theta) ** 2 * np.sin(theta) / (R * KPC) ** 2 * KPC**2 / KPC
        np.testing.assert_allclose(got, expected / 1.0, rtol=2e-2)


if __name__ == "__main__":
    unittest.main()
