"""Model lifecycle tests for long-running bulk processing."""

import sys
import types

import pytest

import untextre.consensus as consensus_mod
import untextre.detector as detector_mod
import untextre.inpaint as inpaint_mod
from untextre.utils import MODEL_CONFIDENCE_FLOOR


def test_initialize_consensus_models_keeps_existing_instances(monkeypatch):
    """Repeated preload calls must not replace already-loaded detector models."""
    existing_doctr = types.SimpleNamespace(
        confidence_threshold=MODEL_CONFIDENCE_FLOOR,
        min_text_size=3,
    )
    existing_easyocr = object()
    existing_east = object()
    monkeypatch.setattr(detector_mod, "_doctr_detector", existing_doctr)
    monkeypatch.setattr(detector_mod, "_easyocr_reader", existing_easyocr)
    monkeypatch.setattr(detector_mod, "_east_net", existing_east)

    def fail_loader(*_args, **_kwargs):
        raise AssertionError("model loader should not be called when instance exists")

    monkeypatch.setattr(detector_mod, "TextDetector", fail_loader)
    monkeypatch.setattr(detector_mod, "_load_east_model", fail_loader)

    consensus_mod.initialize_consensus_models()

    assert detector_mod._doctr_detector is existing_doctr
    assert detector_mod._easyocr_reader is existing_easyocr
    assert detector_mod._east_net is existing_east


def test_initialize_consensus_models_uses_detector_cache(monkeypatch):
    """Consensus preload should share detector.py model instances, not duplicate them."""
    fake_east = object()

    class FakeTextDetector:
        def __init__(self, confidence_threshold, min_text_size):
            self.confidence_threshold = confidence_threshold
            self.min_text_size = min_text_size

    class FakeReader:
        def __init__(self, langs, verbose=False):
            self.langs = langs
            self.verbose = verbose

    fake_easyocr = types.ModuleType("easyocr")
    fake_easyocr.Reader = FakeReader
    monkeypatch.setitem(sys.modules, "easyocr", fake_easyocr)

    monkeypatch.setattr(detector_mod, "_doctr_detector", None)
    monkeypatch.setattr(detector_mod, "_easyocr_reader", None)
    monkeypatch.setattr(detector_mod, "_east_net", None)
    monkeypatch.setattr(detector_mod, "TextDetector", FakeTextDetector)
    monkeypatch.setattr(detector_mod, "_load_east_model", lambda: fake_east)

    def fail_local_loader(*_args, **_kwargs):
        raise AssertionError("consensus should delegate model loading to detector.py")

    monkeypatch.setattr(consensus_mod, "TextDetector", fail_local_loader, raising=False)

    consensus_mod.initialize_consensus_models()

    assert isinstance(detector_mod._doctr_detector, FakeTextDetector)
    assert detector_mod._doctr_detector.confidence_threshold == MODEL_CONFIDENCE_FLOOR
    assert detector_mod._doctr_detector.min_text_size == 3
    assert isinstance(detector_mod._easyocr_reader, FakeReader)
    assert detector_mod._easyocr_reader.langs == ["en"]
    assert detector_mod._east_net is fake_east


def test_reset_lama_model_drops_model_before_cuda_cache_cleanup(monkeypatch):
    """CUDA cache cleanup should happen after the old model reference is gone."""
    class CudaInpainter:
        device = types.SimpleNamespace(type="cuda")

    events = []
    monkeypatch.setattr(inpaint_mod, "_lama_inpainter", CudaInpainter())
    monkeypatch.setattr(inpaint_mod, "_lama_init_failed", True)

    torch = pytest.importorskip("torch")
    monkeypatch.setattr(
        torch.cuda,
        "empty_cache",
        lambda: events.append(inpaint_mod._lama_inpainter is None),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    inpaint_mod.reset_lama_model()

    assert events == [True]
    assert inpaint_mod._lama_inpainter is None
    assert inpaint_mod._lama_init_failed is False
