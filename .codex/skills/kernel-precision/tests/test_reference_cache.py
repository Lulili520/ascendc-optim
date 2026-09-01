import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/validate_precision.py"
SPEC = importlib.util.spec_from_file_location("validate_precision", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FailingTorch:
    class Tensor:
        pass

    @staticmethod
    def save(payload, path):
        path.write_bytes(b"partial")
        raise OSError("disk full")


class LargeTensor:
    def numel(self):
        return 1024

    def element_size(self):
        return 4


class FakeTorch:
    Tensor = LargeTensor

    @staticmethod
    def save(payload, path):
        raise AssertionError("save must not run without capacity")


class ReferenceCacheTest(unittest.TestCase):
    def test_tmp_space_failure_is_environmental(self):
        error = OSError("Not enough space left in TMPDIR (0 KB)")
        with tempfile.TemporaryDirectory() as directory:
            self.assertTrue(MODULE.is_environment_error(error, Path(directory) / "missing.log"))

    def test_partial_cache_is_removed_when_save_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "operator.hash.pt"
            saved = MODULE.save_reference_cache(cache, {"value": 1}, FailingTorch)
            self.assertFalse(saved)
            self.assertFalse(cache.with_suffix(".pt.tmp").exists())
            self.assertFalse(cache.exists())

    def test_cache_is_skipped_when_free_space_is_insufficient(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "operator.hash.pt"
            usage = type("Usage", (), {"free": 4096})()
            with mock.patch.object(MODULE.shutil, "disk_usage", return_value=usage):
                saved = MODULE.save_reference_cache(cache, {"input": LargeTensor()}, FakeTorch)
            self.assertFalse(saved)
            self.assertFalse(cache.exists())


if __name__ == "__main__":
    unittest.main()
