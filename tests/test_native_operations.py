"""Native parity and operational recovery tests; no network or broker calls."""
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from gig.native import available, rolling_stats
from gig.ops.maintenance import create_backup, restore_backup, startup_report, verify_backup


class RollingTests(unittest.TestCase):
    def test_reference_hand_computed(self):
        means, deviations = rolling_stats([1, 2, 3, 4], 3)
        np.testing.assert_allclose(means, [np.nan, np.nan, 2, 3], equal_nan=True)
        np.testing.assert_allclose(deviations, [np.nan, np.nan, 1, 1], equal_nan=True)

    @unittest.skipUnless(available(), "optional C++ library not built")
    def test_native_parity_multiple_shapes_and_offsets(self):
        rng = np.random.default_rng(171)
        for data in [rng.normal(size=5200), np.ones(1200), np.arange(1300.) + 1e12]:
            for window in [1, 2, 31, 252]:
                for ddof in [0, min(1, window-1)]:
                    reference = rolling_stats(data, window, ddof=ddof)
                    actual = rolling_stats(data, window, ddof=ddof, backend="native")
                    for expected, found in zip(reference, actual):
                        np.testing.assert_allclose(found, expected, rtol=2e-10, atol=2e-8, equal_nan=True)

    def test_invalid_input_rejected(self):
        for values, window, ddof in [([], 1, 0), ([1, np.nan], 2, 1),
                                     ([1, np.inf], 2, 1), ([1, 2], True, 0),
                                     ([1, 2], 3, 1), ([1, 2], 2, 2),
                                     ([[1, 2]], 1, 0), ([1e151], 1, 0)]:
            with self.assertRaises(ValueError):
                rolling_stats(values, window, ddof=ddof)

    def test_input_not_mutated(self):
        values = np.array([1., 2., 3.])
        old = values.copy()
        rolling_stats(values, 2)
        np.testing.assert_array_equal(values, old)

    def test_no_silent_backend_substitution(self):
        with self.assertRaises(ValueError):
            rolling_stats([1, 2], 2, backend="invented")


class MaintenanceTests(unittest.TestCase):
    def test_backup_restore_roundtrip_secret_exclusion_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source = root / "state"
            source.mkdir()
            (source / "orders.sqlite").write_bytes(b"immutable test state")
            (source / ".env").write_text("secret sentinel")
            (source / "credentials.json").write_text("secret sentinel")
            archive = root / "state.tar.gz"
            with self.assertRaises(ValueError):
                create_backup(source, archive)
            report = create_backup(source, archive, quiesced=True)
            self.assertEqual(report["file_count"], 1)
            self.assertFalse(report["durable_external_backup"])
            self.assertEqual(len(report["sha256"]), 64)
            verify_backup(archive)
            restore_backup(archive, root / "restored")
            self.assertEqual((root / "restored/orders.sqlite").read_bytes(), b"immutable test state")
            self.assertFalse((root / "restored/.env").exists())
            with self.assertRaises(ValueError):
                restore_backup(archive, root / "restored")

    def test_traversal_and_symlink_archives_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            for name in ["../escape", "/absolute", "a\\escape", ".", "a/./state", "a//state"]:
                archive = root / "evil.tar.gz"
                with tarfile.open(archive, "w:gz") as tar:
                    info = tarfile.TarInfo(name)
                    info.size = 1
                    tar.addfile(info, io.BytesIO(b"x"))
                with self.assertRaises(ValueError):
                    restore_backup(archive, root / "destination")
                self.assertFalse((root / "destination").exists())
            source = root / "state"
            source.mkdir()
            (source / "link").symlink_to(root / "evil.tar.gz")
            with self.assertRaises(ValueError):
                create_backup(source, root / "new.tar.gz", quiesced=True)

    def test_corrupt_digest_rejected_before_restore(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            archive = root / "corrupt.tar.gz"
            manifest = {"schema_version": 1, "quiesced": True,
                        "files": [{"path": "orders", "size": 1, "sha256": "wrong"}]}
            with tarfile.open(archive, "w:gz") as tar:
                for name, data in [("orders", b"x"), ("manifest.json", json.dumps(manifest).encode())]:
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
            with self.assertRaises(ValueError):
                restore_backup(archive, root / "new")
            self.assertFalse((root / "new").exists())

    def test_startup_rejects_broker_gate(self):
        with patch.dict("os.environ", {"GIG_BROKER_CONNECTIVITY_ENABLED": "true"}):
            report = startup_report()
        self.assertEqual(report["state"], "UNAVAILABLE")
        self.assertFalse(report["broker_connectivity_disabled"])
        self.assertFalse(report["trading_authorized"])


if __name__ == "__main__":
    unittest.main()
