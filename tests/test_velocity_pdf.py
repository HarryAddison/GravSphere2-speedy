import unittest

import numpy as np
from scipy.integrate import trapezoid

from gravsphere2 import generalised_gaussian as gg
from gravsphere2.velocity_pdf import asolve_laplace, asolve_uniform, ln_velocity_pdf


class TestKernelPdf(unittest.TestCase):
    x = np.linspace(-80, 80, 160001)

    def pdf(self, var, kurt, err):
        ones = np.ones_like(self.x)
        return np.exp(ln_velocity_pdf(self.x, var * ones, kurt * ones, err * ones))

    def test_normalised_and_correct_moments(self):
        for kurt in (2.2, 2.9, 3.4, 5.0):
            with self.subTest(kurtosis=kurt):
                var, err = 9.0, 0.5
                p = self.pdf(var, kurt, err)
                self.assertAlmostEqual(trapezoid(p, self.x), 1.0, places=4)
                m2 = trapezoid(p * self.x**2, self.x)
                self.assertAlmostEqual(m2, var + err**2, delta=2e-3 * m2)
                m4 = trapezoid(p * self.x**4, self.x)
                # intrinsic kurtosis recovered after removing the Gaussian error
                intrinsic4 = m4 - 6 * var * err**2 - 3 * err**4
                self.assertAlmostEqual(intrinsic4 / var**2, kurt, delta=0.03 * kurt)

    def test_symmetric(self):
        p = self.pdf(4.0, 3.5, 0.3)
        np.testing.assert_allclose(p, p[::-1], rtol=1e-8, atol=1e-14)

    def test_gaussian_limit(self):
        p = self.pdf(4.0, 3.0001, 0.0001)
        gauss = np.exp(-self.x**2 / 8.0) / np.sqrt(8 * np.pi)
        self.assertLess(np.max(np.abs(p - gauss)), 2e-3)

    def test_error_broadens_distribution(self):
        narrow = self.pdf(4.0, 4.0, 0.1)
        broad = self.pdf(4.0, 4.0, 3.0)
        self.assertGreater(narrow[len(self.x) // 2], broad[len(self.x) // 2])

    def test_parameter_maps_are_monotonic(self):
        self.assertTrue(np.all(np.diff(asolve_laplace(np.linspace(3.1, 5.9, 20))) > 0))
        # the uniform-kernel parameter shrinks towards the Gaussian limit (kurtosis 3)
        self.assertTrue(np.all(np.diff(asolve_uniform(np.linspace(1.9, 2.9, 20))) < 0))


class TestGeneralisedGaussian(unittest.TestCase):
    theta = [0.0, 2.0, 2.0, 0.0, 0.0, 5.0]

    def test_gaussian_special_case(self):
        # beta = 2 is a Gaussian with sigma = alpha / sqrt(2)
        self.assertAlmostEqual(gg.velocity_second_moment(self.theta), 2.0 / np.sqrt(2) * 1.0, places=6)
        self.assertAlmostEqual(gg.kurtosis(self.theta), 3.0, places=6)
        self.assertAlmostEqual(gg.velocity_fourth_moment(self.theta), 3 * 2.0**2, places=6)

    def test_pdf_normalised(self):
        v = np.linspace(-30, 30, 20001)
        self.assertAlmostEqual(trapezoid(gg.velpdf_noerr(v, self.theta), v), 1.0, places=5)

    def test_quadrature_with_tiny_error_matches_noerr(self):
        v = np.array([-1.0, 0.5, 2.0])
        conv = gg.velpdf(v, np.full(3, 0.05), self.theta)
        np.testing.assert_allclose(conv, gg.velpdf_noerr(v, self.theta), rtol=1e-2)

    def test_monte_carlo_is_reproducible_and_close(self):
        v = np.array([0.5])
        e = np.array([0.5])
        a = gg.velpdf_monte(v, e, self.theta, npnts=20000, rng=np.random.default_rng(1))
        b = gg.velpdf_monte(v, e, self.theta, npnts=20000, rng=np.random.default_rng(1))
        np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(a, gg.velpdf(v, e, self.theta), rtol=0.05)


if __name__ == "__main__":
    unittest.main()
