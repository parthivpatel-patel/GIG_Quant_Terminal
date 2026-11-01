"""Native categorical projection against independent NumPy OLS references."""
from __future__ import annotations

import importlib
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

module = importlib.import_module("gig.factors.neutralize")


class NativeAdmissionTests(unittest.TestCase):
    def test_explicit_missing_native_never_substitutes(self):
        data = pd.DataFrame(np.ones((1, 12)))
        with patch.object(module, "_CPP_NEUTRALIZE", None), self.assertRaisesRegex(RuntimeError, "unavailable"):
            module.neutralize(data, pd.Series("a", index=data.columns), use_cpp=True)

    def test_explicit_native_error_is_not_hidden(self):
        data = pd.DataFrame(np.ones((1, 12)))
        with patch.object(module, "_CPP_NEUTRALIZE", side_effect=ValueError("unsupported")), self.assertRaisesRegex(ValueError, "unsupported"):
            module.neutralize(data, pd.Series("a", index=data.columns), use_cpp=True)


@unittest.skipIf(module._CPP_NEUTRALIZE is None, "Optional native extension is not built")
class NativeNumericalReferences(unittest.TestCase):
    def setUp(self):
        self.kernel = module._CPP_NEUTRALIZE
        self.columns = [f"S{i}" for i in range(60)]
        self.sectors = pd.Series(["A"] * 20 + ["B"] * 20 + ["C"] * 20, index=self.columns)
        self.data = pd.DataFrame(np.random.default_rng(719).normal(size=(8, 60)), columns=self.columns)

    def parity(self, mask=None, *, demean=True):
        before = self.data.copy(deep=True)
        py = module.neutralize(self.data, self.sectors, mask, demean=demean, use_cpp=False)
        native = module.neutralize(self.data, self.sectors, mask, demean=demean, use_cpp=True)
        np.testing.assert_allclose(native, py, rtol=2e-11, atol=2e-12, equal_nan=True)
        pd.testing.assert_frame_equal(self.data, before)
        return native

    def test_full_sector_ols_parity(self):
        for demean in (False, True):
            with self.subTest(demean=demean):
                self.parity(demean=demean)

    def test_exact_shared_sample_threshold(self):
        mask = pd.DataFrame(False, index=self.data.index, columns=self.columns)
        mask.iloc[:, :10] = True  # n_sector_dummies + 8
        native = self.parity(mask)
        self.assertEqual(int(np.isfinite(native.to_numpy()).sum()), 80)
        mask.iloc[:, 9] = False
        self.assertTrue(self.parity(mask).isna().all().all())

    def test_absent_baseline_and_zero_sector_columns_are_rank_safe(self):
        for eligible in (slice(20, 60), slice(20, 40), slice(0, 20)):
            with self.subTest(eligible=eligible):
                mask = pd.DataFrame(False, index=self.data.index, columns=self.columns)
                mask.iloc[:, eligible] = True
                native = self.parity(mask)
                self.assertTrue(np.isfinite(native.to_numpy()[mask.to_numpy()]).all())

    def test_missing_sectors_are_explicit_unknown_group(self):
        self.sectors.iloc[:20] = None
        self.parity()

    def test_random_masks_nonfinite_and_independent_reference(self):
        self.data.iloc[0, :3] = [np.nan, np.inf, -np.inf]
        mask = pd.DataFrame(np.random.default_rng(17).random(self.data.shape) > .25,
                            index=self.data.index, columns=self.columns)
        result = self.parity(mask)
        # Direct within-sector centering is an independent scalar reference.
        for i, row in enumerate(self.data.to_numpy()):
            eligible = mask.iloc[i].to_numpy() & np.isfinite(row)
            for sector in ("A", "B", "C"):
                group = eligible & (self.sectors.to_numpy() == sector)
                expected = row[group] - row[group].mean()
                np.testing.assert_allclose(result.iloc[i].to_numpy()[group], expected, atol=2e-12)
        self.assertTrue(np.isnan(result.iloc[0, :3]).all())

    def test_direct_kernel_rejects_dense_or_overlapping_design(self):
        values = np.ones((2, 12))
        mask = np.ones(values.shape, dtype=np.uint8)
        for dummy in (np.full((12, 2), .5), np.ones((12, 2)), np.full((12, 1), np.nan)):
            with self.subTest(dummy=dummy[0]), self.assertRaises(ValueError):
                self.kernel(values, dummy, mask, True)

    def test_direct_shapes_and_mask_domain_fail_closed(self):
        for values, dummy, mask in (
            (np.ones(12), np.empty((12, 0)), np.ones((1, 12))),
            (np.ones((1, 12)), np.empty((11, 0)), np.ones((1, 12))),
            (np.ones((1, 12)), np.empty((12, 0)), np.full((1, 12), 2)),
        ):
            with self.subTest(shape=values.shape), self.assertRaises(ValueError):
                self.kernel(values, dummy, mask, True)

    def test_input_arrays_are_unchanged_and_output_detached(self):
        values = self.data.to_numpy(copy=True)
        dummy = pd.get_dummies(self.sectors, drop_first=True).to_numpy(dtype=float)
        mask = np.ones(values.shape, dtype=np.uint8)
        before = [x.copy() for x in (values, dummy, mask)]
        result = self.kernel(values, dummy, mask, False)
        for actual, expected in zip((values, dummy, mask), before, strict=True):
            np.testing.assert_array_equal(actual, expected)
        result[:] = 99
        np.testing.assert_array_equal(values, before[0])

    def test_large_constant_categorical_projection_is_finite_and_exact(self):
        values = np.full((1, 12), 1e308)
        result = self.kernel(values, np.empty((12, 0)), np.ones((1, 12), dtype=np.uint8), True)
        np.testing.assert_array_equal(result, np.zeros_like(values))

    def test_unrepresentable_residuals_fail_instead_of_returning_infinity(self):
        values = np.array([[1.7e308] * 11 + [-1.7e308]])
        with self.assertRaises(OverflowError):
            self.kernel(values, np.empty((12, 0)), np.ones((1, 12), dtype=np.uint8), False)

    def test_empty_dimensions_and_all_masked_return_explicit_missing(self):
        result = self.kernel(np.empty((0, 12)), np.empty((12, 0)), np.empty((0, 12), dtype=np.uint8), True)
        self.assertEqual(result.shape, (0, 12))
        values = np.ones((2, 12))
        result = self.kernel(values, np.empty((12, 0)), np.zeros((2, 12), dtype=np.uint8), True)
        self.assertTrue(np.isnan(result).all())


if __name__ == "__main__":
    unittest.main()
