"""Fast import-surface tests for untextre.detector."""

import importlib
import sys
import types
from pathlib import Path
from unittest.mock import Mock, patch

import pytest


def test_detector_import_does_not_construct_easyocr_reader(monkeypatch):
    sys.modules.pop("untextre.detector", None)

    sentinel = types.ModuleType("easyocr")

    def fail_reader(*_args, **_kwargs):
        raise AssertionError("easyocr.Reader should be lazy and only used during EasyOCR init")

    sentinel.Reader = fail_reader
    monkeypatch.setitem(sys.modules, "easyocr", sentinel)

    detector = importlib.import_module("untextre.detector")

    assert detector._easyocr_reader is None



def test_get_yolo11x_model_path_is_in_untextre_cache():
    from untextre.detector import _get_yolo11x_model_path

    path = _get_yolo11x_model_path()

    assert path.parent == Path.home() / ".untextre" / "models"
    assert path.name == "yolo11x-train28-best.pt"


def test_validate_yolo11x_rejects_small_file(tmp_path):
    from untextre.detector import _validate_yolo11x_model_file

    small = tmp_path / "yolo11x-train28-best.pt"
    small.write_bytes(b"x" * 1024)

    with pytest.raises(RuntimeError, match="too small"):
        _validate_yolo11x_model_file(small)


def test_download_yolo11x_writes_file_atomically(tmp_path):
    from untextre import detector as det

    target = tmp_path / "yolo11x-train28-best.pt"
    tmp_target = Path(f"{target}.tmp")
    total_size = det.YOLO11X_MODEL_MIN_BYTES + 1

    class FakeResponse:
        def __init__(self):
            self.remaining = total_size

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, size):
            assert not target.exists()
            if self.remaining <= 0:
                return b""
            chunk_size = min(size, self.remaining)
            self.remaining -= chunk_size
            return b"Y" * chunk_size

    fake_urlopen = Mock(return_value=FakeResponse())

    det._download_yolo11x_model(target, urlopen=fake_urlopen)

    fake_urlopen.assert_called_once_with(
        det.YOLO11X_MODEL_URL,
        timeout=det.YOLO11X_DOWNLOAD_TIMEOUT_SECONDS,
    )
    assert target.exists()
    assert target.stat().st_size == total_size
    assert not tmp_target.exists()


def test_get_yolo11x_model_returns_singleton():
    from untextre import detector as det

    fake_model = object()
    with patch.object(det, "_yolo11x_model", None):
        with patch.object(det, "_load_yolo11x_model", return_value=fake_model) as load_model:
            first = det.get_yolo11x_model()
            second = det.get_yolo11x_model()

    assert first is fake_model
    assert second is fake_model
    load_model.assert_called_once_with()
