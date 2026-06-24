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


def test_initialize_models_imports_easyocr_only_for_easyocr(monkeypatch):
    import untextre.detector as detector

    class FakeReader:
        def __init__(self, langs, verbose=False):
            self.langs = langs
            self.verbose = verbose

    fake_easyocr = types.ModuleType("easyocr")
    fake_easyocr.Reader = FakeReader
    monkeypatch.setitem(sys.modules, "easyocr", fake_easyocr)
    monkeypatch.setattr(detector, "_easyocr_reader", None)

    detector.initialize_models(["easyocr"])

    assert isinstance(detector._easyocr_reader, FakeReader)
    assert detector._easyocr_reader.langs == ["en"]
