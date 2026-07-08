"""Fast import-surface tests for untextre.detector."""

import importlib
import sys
import types


def test_detector_import_does_not_construct_easyocr_reader(monkeypatch):
    sys.modules.pop("untextre.detector", None)

    sentinel = types.ModuleType("easyocr")

    def fail_reader(*_args, **_kwargs):
        raise AssertionError("easyocr.Reader should be lazy and only used during EasyOCR init")

    sentinel.Reader = fail_reader
    monkeypatch.setitem(sys.modules, "easyocr", sentinel)

    detector = importlib.import_module("untextre.detector")

    assert detector._easyocr_reader is None


