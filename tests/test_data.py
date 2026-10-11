import unittest

import numpy as np

from gravsphere2.config import Config
from gravsphere2.data import DataError, Dataset, load_dataset

from .helpers import EXAMPLES, TempDirMixin, make_config, write_synthetic_data


class TestLoadDataset(TempDirMixin, unittest.TestCase):
    def test_binned_with_proper_motions(self):
        ds = load_dataset(make_config(self.tmp, n=40))
        self.assertEqual(len(ds.r_los), 40)
        self.assertTrue(ds.proper_motions)
        self.assertFalse(ds.individual)
        self.assertEqual(len(ds.R), len(ds.surfden))
        self.assertEqual(ds.half_light_radius, 0.3)
        np.testing.assert_array_equal(ds.r_pm, ds.r_los)

    def test_los_only(self):
        ds = load_dataset(make_config(self.tmp, proper_motions=False))
        self.assertFalse(ds.proper_motions)
        self.assertIsNone(ds.v_pm_t)

    def test_individual_uses_median_radius(self):
        ds = load_dataset(make_config(self.tmp, individual=True))
        self.assertTrue(ds.individual)
        self.assertAlmostEqual(ds.half_light_radius, float(np.median(ds.R)))

    def test_kinematic_tracers_as_photometry(self):
        d = write_synthetic_data(self.tmp, individual=True)
        d["data"]["photometry"] = {"mode": "individual", "source": "kinematic_tracers"}
        ds = load_dataset(Config.from_dict(d, base_dir=self.tmp))
        np.testing.assert_array_equal(ds.R, ds.r_los)

    def test_wrong_number_of_rows(self):
        d = write_synthetic_data(self.tmp, proper_motions=False)
        d["data"]["proper_motions"] = True  # file only has 3 rows
        with self.assertRaisesRegex(DataError, "7 rows"):
            load_dataset(Config.from_dict(d, base_dir=self.tmp))

    def test_missing_file(self):
        cfg = make_config(self.tmp)
        (self.tmp / "kin.txt").unlink()
        with self.assertRaisesRegex(DataError, "not found"):
            load_dataset(cfg)

    def test_unparsable_file(self):
        cfg = make_config(self.tmp)
        (self.tmp / "kin.txt").write_text("a b c\n")
        with self.assertRaisesRegex(DataError, "Could not read"):
            load_dataset(cfg)

    def test_nonpositive_error_rejected(self):
        cfg = make_config(self.tmp, proper_motions=False)
        arr = np.loadtxt(self.tmp / "kin.txt")
        arr[2, 0] = 0.0
        np.savetxt(self.tmp / "kin.txt", arr)
        with self.assertRaisesRegex(DataError, "errors must be positive"):
            load_dataset(cfg)

    def test_nan_rejected(self):
        cfg = make_config(self.tmp, proper_motions=False)
        arr = np.loadtxt(self.tmp / "kin.txt")
        arr[1, 3] = np.nan
        np.savetxt(self.tmp / "kin.txt", arr)
        with self.assertRaisesRegex(DataError, "Non-finite"):
            load_dataset(cfg)

    def test_shipped_example_data(self):
        cfg = Config.load(EXAMPLES / "gaia_challenge_plumcusp.yaml")
        ds = load_dataset(cfg)
        self.assertEqual(len(ds.r_los), 1000)
        cfg = Config.load(EXAMPLES / "fornax_simulation.yaml")
        ds = load_dataset(cfg)
        self.assertEqual((len(ds.r_los), len(ds.R)), (100, 10000))


class TestDatasetValidation(unittest.TestCase):
    def test_length_mismatch(self):
        ds = Dataset(np.ones(3), np.ones(2), np.ones(3), np.ones(3), 1.0)
        with self.assertRaisesRegex(DataError, "same length"):
            ds.validate()

    def test_empty(self):
        e = np.array([])
        with self.assertRaisesRegex(DataError, "empty"):
            Dataset(e, e, e, np.ones(1), 1.0).validate()


if __name__ == "__main__":
    unittest.main()
