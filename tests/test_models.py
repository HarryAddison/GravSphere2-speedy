import unittest

import numpy as np
from scipy.integrate import quad

from gravsphere2 import constants
from gravsphere2.models import anisotropy, mass, tracers


class TestConstants(unittest.TestCase):
    def test_guse_value(self):
        self.assertAlmostEqual(constants.GUSE, 6.67e-11 * 1.989e30 / 3.086e19)

    def test_omega_m(self):
        self.assertAlmostEqual(constants.OMEGA_M, 0.12 / 0.7**2)


class TestAnisotropy(unittest.TestCase):
    pars = [0.1, 0.8, 0.5, 2.0]

    def test_limits(self):
        self.assertAlmostEqual(anisotropy.beta(1e-6, self.pars), 0.1, places=6)
        self.assertAlmostEqual(anisotropy.beta(1e6, self.pars), 0.8, places=6)

    def test_midpoint_is_average(self):
        self.assertAlmostEqual(anisotropy.beta(0.5, self.pars), 0.45)

    def test_symmetrised_round_trip(self):
        b = np.linspace(-1, 0.9, 20)
        tilde = anisotropy.beta_to_symmetrised(b)
        np.testing.assert_allclose(anisotropy.symmetrised_to_beta(tilde), b)

    def test_derivative_matches_finite_difference(self):
        r = np.array([0.1, 0.5, 2.0])
        h = 1e-6
        fd = (anisotropy.beta(r + h, self.pars) - anisotropy.beta(r - h, self.pars)) / (2 * h)
        np.testing.assert_allclose(anisotropy.dbeta_dr(r, self.pars), fd, rtol=1e-5)

    def test_integrating_factor_satisfies_defining_ode(self):
        # d ln f / d ln r = 2 beta
        r = np.array([0.2, 1.0, 3.0])
        h = 1e-6
        f = anisotropy.integrating_factor
        dlnf = (np.log(f(r * (1 + h), self.pars)) - np.log(f(r * (1 - h), self.pars))) / (2 * h)
        np.testing.assert_allclose(dlnf, 2 * anisotropy.beta(r, self.pars), rtol=1e-4)

    def test_osipkov(self):
        self.assertAlmostEqual(anisotropy.osipkov(1.0, 1.0), 0.5)


class TestTracers(unittest.TestCase):
    def test_plummer_special_case(self):
        r = np.logspace(-2, 1, 10)
        a, total = 0.4, 1e6
        rho0 = 3 * total / (4 * np.pi * a**3)
        np.testing.assert_allclose(
            tracers.alpbetgam_density(r, rho0, a, 2, 5, 0),
            tracers.multiplum_density(r, [total, a]),
        )

    def test_mass_matches_numerical_integral(self):
        rho0, r0, alp, bet, gam = 5.0, 0.3, 2.0, 5.0, 0.5
        r = 1.7
        numeric = quad(lambda x: 4 * np.pi * x**2 * tracers.alpbetgam_density(x, rho0, r0, alp, bet, gam), 0, r)[0]
        self.assertAlmostEqual(tracers.alpbetgam_mass(r, rho0, r0, alp, bet, gam) / numeric, 1.0, places=6)

    def test_total_mass_is_large_radius_limit(self):
        pars = (5.0, 0.3, 2.0, 5.0, 0.1)
        self.assertAlmostEqual(
            tracers.alpbetgam_mass(1e4, *pars) / tracers.alpbetgam_total_mass(*pars), 1.0, places=5
        )

    def test_projection_of_plummer_is_analytic(self):
        a, total = 0.4, 1e6
        rho0 = 3 * total / (4 * np.pi * a**3)
        R = np.array([0.05, 0.2, 0.8])
        surf = tracers.alpbetgam_surface(R, rho0, a, 2, 5, 0, 400)
        np.testing.assert_allclose(surf, tracers.multiplum_surface(R, [total, a]), rtol=2e-3)

    def test_slope_limits(self):
        s = tracers.alpbetgam_dlnrho_dlnr(np.array([1e-6, 1e6]), 1.0, 1.0, 2, 5, 0.5)
        np.testing.assert_allclose(s, [-0.5, -5.0], atol=1e-6)

    def test_seq_mode_offsets_scales(self):
        r = np.array([0.1, 1.0])
        normal = tracers.multiplum_mass(r, [1.0, 1.0, 0.2, 0.3], "normal")
        seq = tracers.multiplum_mass(r, [1.0, 1.0, 0.2, 0.3], "seq")
        explicit = tracers.multiplum_mass(r, [1.0, 1.0, 0.2, 0.5], "normal")
        self.assertFalse(np.allclose(normal, seq))
        np.testing.assert_allclose(seq, explicit)

    def test_multiplum_slope_matches_numeric(self):
        r = np.array([0.1, 0.5, 2.0])
        p = [1.0, 2.0, 0.2, 0.9]
        h = 1e-6
        d = lambda x: tracers.multiplum_density(x, p)
        fd = (np.log(d(r * (1 + h))) - np.log(d(r * (1 - h)))) / (np.log(1 + h) - np.log(1 - h))
        np.testing.assert_allclose(tracers.multiplum_dlnrho_dlnr(r, p), fd, rtol=1e-5)


class TestMass(unittest.TestCase):
    pars = (1e9, 15.0, 0.5, 0.7, 5.0, 4.0)
    r = np.logspace(-2, 1.5, 60)

    def test_mass_is_monotonic(self):
        m = mass.corenfw_tides_mass(self.r, *self.pars)
        self.assertTrue(np.all(np.diff(m) > 0))

    def test_density_integrates_to_mass(self):
        rho = lambda x: mass.corenfw_tides_density(np.array([x]), *self.pars)[0]
        for rr in (0.3, 3.0, 20.0):  # inside and outside the tidal radius
            numeric = quad(lambda x: 4 * np.pi * x**2 * rho(x), 1e-6, rr, limit=200, points=[5.0] if rr > 5 else None)[0]
            analytic = mass.corenfw_tides_mass(np.array([rr]), *self.pars)[0]
            self.assertAlmostEqual(numeric / analytic, 1.0, places=3)

    def test_n_zero_recovers_nfw_inside_rt(self):
        m200, c = 1e9, 15.0
        r = np.array([0.1, 0.5, 2.0])
        cored = mass.corenfw_tides_mass(r, m200, c, 0.5, 0.0, 50.0, 4.0)
        gcon = 1 / (np.log(1 + c) - c / (1 + c))
        rv = (3 / 4 * m200 / (np.pi * constants.ODEN * constants.RHOCRIT)) ** (1 / 3)
        rs = rv / c
        nfw = m200 * gcon * (np.log(1 + r / rs) - r / rs / (1 + r / rs))
        np.testing.assert_allclose(cored, nfw)

    def test_core_lowers_central_density(self):
        cusp = mass.corenfw_tides_density(np.array([0.01]), 1e9, 15, 0.5, 0.0, 5, 4)
        core = mass.corenfw_tides_density(np.array([0.01]), 1e9, 15, 0.5, 1.0, 5, 4)
        self.assertLess(core[0], cusp[0])

    def test_outer_slope_beyond_rt_is_minus_delta(self):
        s = mass.corenfw_tides_dlnrho_dlnr(np.array([8.0, 12.0]), *self.pars)
        np.testing.assert_allclose(s, -self.pars[-1], atol=1e-3)

    def test_virial_nfw_consistency(self):
        r = np.array([0.5, 5.0])
        rs, rhos = mass._virial_nfw(1e10, 12.0)
        numeric = quad(lambda x: 4 * np.pi * x**2 * mass.nfw_density(x, rhos, rs), 1e-8, 5.0)[0]
        self.assertAlmostEqual(mass.nfw_virial_mass(np.array([5.0]), 1e10, 12.0)[0] / numeric, 1.0, places=5)

    def test_plummer_mass_limit(self):
        self.assertAlmostEqual(mass.plummer_mass(1e6, [3e5, 0.2]) / 3e5, 1.0, places=6)

    def test_escape_velocity_decreases_outwards(self):
        v = mass.escape_velocity(np.array([0.1, 1.0, 5.0]), self.pars, 0.01, 10.0)
        self.assertTrue(np.all(np.diff(v) < 0))
        self.assertGreater(v[0], 0)

    def test_escape_velocity_custom_mass_function(self):
        point = lambda r, pars: np.full_like(r, pars[0])  # point mass
        v = mass.escape_velocity(np.array([1.0]), [1e8], 0.5, 200.0, mass_func=point, pnts=20000)
        # potential relative to rmax: v^2 = 2 G M (1/r - 1/rmax)
        expected = np.sqrt(2 * constants.GUSE * 1e8 * (1.0 - 1 / 200.0)) / 1e3
        self.assertAlmostEqual(v[0] / expected, 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
