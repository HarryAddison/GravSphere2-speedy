import unittest

import numpy as np
from scipy.integrate import quad

from gravsphere2.constants import GUSE
from gravsphere2.jeans import JeansResult, non_negative_cut, radial_dispersion_squared, solve_jeans
from gravsphere2.models.tracers import multiplum_density, multiplum_mass, multiplum_surface

TOTAL, A = 1e6, 0.4  # Plummer mass (Msun) and scale radius (kpc)
ISOTROPIC = [0.0, 0.0, 1.0, 2.0]


def tracer(r):
    return multiplum_density(r, [TOTAL, A])


def surface(r):
    return multiplum_surface(r, [TOTAL, A])


def mass(r, pars):
    return multiplum_mass(r, [TOTAL, A])


def analytic_sigr2(r):
    """Isotropic Plummer in its own potential: sigma_r^2 = G M / (6 sqrt(r^2 + a^2))."""
    return GUSE * TOTAL / (6.0 * np.sqrt(r**2 + A**2))


class TestRadialDispersion(unittest.TestCase):
    def test_isotropic_plummer_matches_analytic(self):
        theta = np.linspace(0, np.pi / 2 - 1e-5, 300)
        r = np.array([0.05, 0.2, 0.4, 1.0])
        got = radial_dispersion_squared(r, theta, tracer, mass, None, ISOTROPIC)
        np.testing.assert_allclose(got, analytic_sigr2(r), rtol=3e-3)


class TestSolveJeans(unittest.TestCase):
    R = np.array([0.1, 0.3, 0.6])

    def solve(self, **kwargs):
        defaults = dict(
            R=self.R, r_los=self.R, r_pm=self.R, tracer_density=tracer, surface_density_func=surface,
            mass_func=mass, mass_pars=None, betpars=ISOTROPIC, betppars=ISOTROPIC,
            rmin=1e-3, rmax=50.0, intpnts=200,
        )
        defaults.update(kwargs)
        return solve_jeans(**defaults)

    def test_returns_result_object(self):
        res = self.solve()
        self.assertIsInstance(res, JeansResult)
        self.assertEqual(res.sigma2_los.shape, self.R.shape)

    def test_surface_density_is_passed_through(self):
        res = self.solve()
        # linear interpolation on a 200-point log grid
        np.testing.assert_allclose(res.surface_density, surface(self.R), rtol=1e-2)

    def test_los_dispersion_matches_direct_projection(self):
        res = self.solve()
        for R, got in zip(self.R, res.sigma2_los):
            integrand = lambda s: tracer(np.array([np.sqrt(R**2 + s**2)]))[0] * analytic_sigr2(np.sqrt(R**2 + s**2))
            expected = 2 * quad(integrand, 0, np.inf)[0] / surface(np.array([R]))[0]
            self.assertAlmostEqual(got / expected, 1.0, delta=1e-2)

    def test_isotropic_components_agree(self):
        res = self.solve()
        np.testing.assert_allclose(res.sigma2_pm_t, res.sigma2_los, rtol=1e-6)
        np.testing.assert_allclose(res.sigma2_pm_r, res.sigma2_los, rtol=1e-6)
        np.testing.assert_allclose(res.kurtosis_pm_t, res.kurtosis_los, rtol=1e-6)

    def test_isotropic_kurtosis_is_physical(self):
        res = self.solve()
        self.assertTrue(np.all(res.kurtosis_los > 2.0) and np.all(res.kurtosis_los < 4.0))
        self.assertFalse(res.negative_moments)

    def test_radial_anisotropy_raises_los_dispersion_less_than_pm_radial_split(self):
        radial = [0.7, 0.7, 1.0, 2.0]
        res = self.solve(betpars=radial, betppars=radial)
        self.assertTrue(np.all(res.sigma2_pm_r > res.sigma2_pm_t))

    def test_los_only_mode_matches_full_mode(self):
        full = self.solve()
        los = self.solve(proper_motions=False)
        self.assertIsNone(los.sigma2_pm_t)
        np.testing.assert_allclose(los.sigma2_los, full.sigma2_los)
        np.testing.assert_allclose(los.kurtosis_los, full.kurtosis_los)

    def test_dispersion_scales_with_root_mass(self):
        lo = self.solve()
        hi = self.solve(mass_func=lambda r, p: 4.0 * mass(r, p))
        np.testing.assert_allclose(hi.sigma2_los / lo.sigma2_los, 4.0, rtol=1e-6)

    def test_zero_outside_grid(self):
        res = self.solve(r_los=np.array([1e4]), r_pm=np.array([1e4]))
        self.assertEqual(res.sigma2_los[0], 0.0)


class TestNonNegativeCut(unittest.TestCase):
    def test_cut_index(self):
        rint = np.logspace(-2, 1, 10)
        i = non_negative_cut(rint, rint[5])
        self.assertEqual(rint[-i], rint[5])

    def test_no_match_returns_last_loop_index(self):
        rint = np.logspace(-2, 1, 10)
        self.assertEqual(non_negative_cut(rint, 0.0), len(rint) - 2)


if __name__ == "__main__":
    unittest.main()
