import pickle
import unittest

import numpy as np

from gravsphere2.config import Config
from gravsphere2.data import load_dataset
from gravsphere2.likelihood import GravSphere2Model, InterpolatedProfile, TotalMass
from gravsphere2.models.mass import corenfw_tides_mass
from gravsphere2.models.tracers import alpbetgam_mass
from gravsphere2.priors import build_prior_space

from .helpers import EXAMPLES, TempDirMixin, write_synthetic_data


def build(tmp, stellar=0.0, escape=False, **data_kwargs):
    d = write_synthetic_data(tmp, **data_kwargs)
    d["stellar_mass"] = {"value": stellar}
    d["data"]["escape_velocity_check"] = escape
    cfg = Config.from_dict(d, base_dir=tmp)
    return cfg, load_dataset(cfg)


class TestPriorSpace(TempDirMixin, unittest.TestCase):
    def test_dimensions(self):
        cases = [
            (dict(), 8 + 5 + 6),
            (dict(individual=True), 8 + 4 + 6),
            (dict(stellar=1e6), 8 + 5 + 6 + 1),
        ]
        for kwargs, ndims in cases:
            with self.subTest(**kwargs):
                cfg, ds = build(self.tmp, **kwargs)
                space = build_prior_space(cfg, ds, ds.half_light_radius)
                self.assertEqual(space.ndims, ndims)
                self.assertEqual(len(space.names), ndims)
                self.assertTrue(np.all(space.lower < space.upper))

    def test_transform_maps_unit_cube_to_bounds(self):
        cfg, ds = build(self.tmp)
        space = build_prior_space(cfg, ds, 0.3)
        np.testing.assert_allclose(space.transform(np.zeros(space.ndims)), space.lower)
        np.testing.assert_allclose(space.transform(np.ones(space.ndims)), space.upper)
        np.testing.assert_allclose(space.transform(np.full(space.ndims, 0.5)), (space.lower + space.upper) / 2)

    def test_binned_tracer_priors_follow_reference_with_physical_limits(self):
        cfg, ds = build(self.tmp)
        space = build_prior_space(cfg, ds, 0.3)
        t = space.tracer_slice
        lo, hi = space.lower[t], space.upper[t]
        np.testing.assert_allclose(lo[:2], [0.5, 0.15])
        np.testing.assert_allclose(hi[:2], [1.5, 0.45])
        self.assertGreaterEqual(lo[3], 3.0)  # outer slope beta > 3
        self.assertLessEqual(hi[4], 3.0)  # inner slope gamma < 3

    def test_negative_reference_value_keeps_ordering(self):
        cfg, ds = build(self.tmp)
        cfg.priors.tracer.reference_parameters = [1.0, 0.3, 2, 5, -0.2]
        space = build_prior_space(cfg, ds, 0.3)
        self.assertLess(space.lower[space.tracer_slice][4], space.upper[space.tracer_slice][4])

    def test_log_parameters_and_rc_rt_limits(self):
        cfg, ds = build(self.tmp)
        space = build_prior_space(cfg, ds, 0.3)
        m = space.mass_slice
        self.assertAlmostEqual(space.lower[m][2], np.log10(cfg.priors.mass.rc[0]))
        self.assertAlmostEqual(space.upper[m][4], np.log10(cfg.priors.mass.rt[1]))
        self.assertEqual(space.log_indices[:2], [2, 6])

    def test_individual_adds_log_radius(self):
        cfg, ds = build(self.tmp, individual=True)
        space = build_prior_space(cfg, ds, 0.5)
        self.assertIn(8, space.log_indices)
        self.assertAlmostEqual(space.lower[8], np.log10(0.5) - 1.0)

    def test_stellar_mass_prior(self):
        cfg, ds = build(self.tmp, stellar=1e6)
        cfg.stellar_mass.fractional_error = 0.25
        space = build_prior_space(cfg, ds, 0.3)
        self.assertAlmostEqual(space.lower[-1], 0.75e6)
        self.assertAlmostEqual(space.upper[-1], 1.25e6)


class TestModel(TempDirMixin, unittest.TestCase):
    def test_unpack_converts_and_exponentiates(self):
        cfg, ds = build(self.tmp)
        m = GravSphere2Model(cfg, ds)
        theta = m.prior_transform(np.full(m.ndims, 0.5))
        theta[0], theta[2] = 0.5, -1.0
        p = m.unpack(theta)
        self.assertAlmostEqual(p.beta_pars[0], 2 * 0.5 / 1.5)  # symmetrised -> beta
        self.assertAlmostEqual(p.beta_pars[2], 0.1)  # 10**-1
        self.assertEqual(len(p.tracer_pars), 5)
        self.assertEqual(len(p.mass_pars), 6)
        self.assertAlmostEqual(p.mass_pars[0], 10 ** theta[13])  # M200
        # input untouched
        self.assertEqual(theta[2], -1.0)

    def test_unpack_individual_prepends_unit_density(self):
        cfg, ds = build(self.tmp, individual=True)
        m = GravSphere2Model(cfg, ds)
        p = m.unpack(m.prior_transform(np.full(m.ndims, 0.5)))
        self.assertEqual(len(p.tracer_pars), 5)
        self.assertEqual(p.tracer_pars[0], 1.0)

    def test_default_grids(self):
        cfg, ds = build(self.tmp)
        m = GravSphere2Model(cfg, ds)
        self.assertAlmostEqual(m.rmin, min(0.3 / 100, ds.R.min() / 10))
        self.assertAlmostEqual(m.rmax, max(30.0, ds.R.max() * 2))
        self.assertAlmostEqual(m.nonneg_radius, m.rmax / 2)

    def test_grid_overrides(self):
        cfg, ds = build(self.tmp)
        cfg.grids.rmin, cfg.grids.rmax, cfg.grids.nonnegative_radius = 1e-3, 10.0, 4.0
        m = GravSphere2Model(cfg, ds)
        self.assertEqual((m.rmin, m.rmax, m.nonneg_radius), (1e-3, 10.0, 4.0))

    def _finite_theta(self, m, tries=50):
        rng = np.random.default_rng(3)
        for _ in range(tries):
            theta = m.prior_transform(rng.uniform(0.3, 0.7, m.ndims))
            if np.isfinite(m(theta)):
                return theta
        self.fail("no finite likelihood found")

    def test_likelihood_finite_and_deterministic(self):
        for kwargs in (dict(), dict(individual=True), dict(proper_motions=False), dict(stellar=1e6)):
            with self.subTest(**kwargs):
                cfg, ds = build(self.tmp, **kwargs)
                m = GravSphere2Model(cfg, ds)
                theta = self._finite_theta(m)
                self.assertEqual(m(theta), m.log_likelihood(theta))
                self.assertEqual(m(theta), m(theta))

    def test_likelihood_responds_to_mass(self):
        cfg, ds = build(self.tmp)
        m = GravSphere2Model(cfg, ds)
        theta = self._finite_theta(m)
        other = theta.copy()
        other[13] += 0.2  # log10 M200
        self.assertNotEqual(m(theta), m(other))

    def test_stellar_mass_enters_likelihood(self):
        cfg, ds = build(self.tmp, stellar=1e8)
        m = GravSphere2Model(cfg, ds)
        theta = self._finite_theta(m)
        heavier = theta.copy()
        heavier[-1] = m.priors.upper[-1]
        self.assertNotEqual(m(theta), m(heavier))

    def test_escape_velocity_check_rejects_unbound_tracers(self):
        cfg, ds = build(self.tmp, escape=True)
        m = GravSphere2Model(cfg, ds)
        theta = self._finite_theta(m)
        ds.v_los = ds.v_los.copy()
        ds.v_los[0] = 1e5  # km/s, far above any escape velocity
        self.assertEqual(m(theta), -np.inf)
        # Without the check the same data are not rejected on escape grounds.
        cfg2, ds2 = build(self.tmp, escape=False)
        ds2.v_los[0] = 1e5
        self.assertTrue(np.isfinite(GravSphere2Model(cfg2, ds2)(theta)))

    def test_unphysical_parameters_give_minus_inf_not_exceptions(self):
        cfg, ds = build(self.tmp)
        m = GravSphere2Model(cfg, ds)
        rng = np.random.default_rng(0)
        for _ in range(15):
            val = m(m.prior_transform(rng.uniform(size=m.ndims)))
            self.assertFalse(np.isnan(val))

    def test_model_is_picklable_for_multiprocessing(self):
        cfg, ds = build(self.tmp)
        m = GravSphere2Model(cfg, ds)
        theta = self._finite_theta(m)
        clone = pickle.loads(pickle.dumps(m))
        self.assertEqual(clone(theta), m(theta))
        pickle.loads(pickle.dumps(m.log_likelihood))

    def test_matches_original_implementation(self):
        # Reference values were produced by the original (pre-refactor)
        # GravSphere2 scripts at the first of 12 parameter vectors drawn as
        # u = default_rng(7).uniform(0.35, 0.65, (12, ndims)).
        references = {
            "gaia_challenge_plumcusp.yaml": -10978.32083877,
            "gaia_challenge_plumcore.yaml": -12195.26094407,
            "fornax_simulation.yaml": -30085.97325923,
        }
        for name, expected in references.items():
            with self.subTest(example=name):
                cfg = Config.load(EXAMPLES / name)
                m = GravSphere2Model(cfg, load_dataset(cfg))
                u = np.random.default_rng(7).uniform(0.35, 0.65, size=(12, m.ndims))[0]
                self.assertAlmostEqual(m(m.prior_transform(u)), expected, places=6)


class TestHelpers(unittest.TestCase):
    def test_interpolated_profile_extrapolation(self):
        prof = InterpolatedProfile(np.array([1.0, 2.0]), np.array([10.0, 20.0]))
        self.assertEqual(prof(1.5), 15.0)
        self.assertEqual(prof(0.5), 10.0)  # constant on the left
        self.assertEqual(prof(5.0), 1e-30)  # negligible on the right

    def test_total_mass_without_and_with_stars(self):
        r = np.array([0.1, 1.0])
        dm = (1e9, 15.0, 0.5, 0.7, 5.0, 4.0)
        np.testing.assert_allclose(TotalMass()(r, dm), corenfw_tides_mass(r, *dm))
        tr = (2.0, 0.3, 2.0, 5.0, 0.1)
        total = TotalMass(tr, normmass=4.0)(r, (*dm, 1e6))
        expected = corenfw_tides_mass(r, *dm) + alpbetgam_mass(r, *tr) * 1e6 / 4.0
        np.testing.assert_allclose(total, expected)


if __name__ == "__main__":
    unittest.main()
