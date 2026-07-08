"""Model lifecycle tests for long-running bulk processing."""

import sys
import types

import pytest
from unittest.mock import Mock

import untextre.consensus as consensus_mod
import untextre.detector as detector_mod
import untextre.inpaint as inpaint_mod
from untextre.utils import MODEL_CONFIDENCE_FLOOR


def test_initialize_consensus_models_keeps_existing_yolo11x_instances(monkeypatch):
    """Repeated preload calls must not replace already-loaded production detector models."""
    existing_yolo = object()
    existing_easyocr = object()
    existing_east = object()
    monkeypatch.setattr(detector_mod, "_yolo11x_model", existing_yolo)
    monkeypatch.setattr(detector_mod, "_easyocr_reader", existing_easyocr)
    monkeypatch.setattr(detector_mod, "_east_net", existing_east)

    def fail_loader(*_args, **_kwargs):
        raise AssertionError("model loader should not be called when instance exists")

    doctr_getter = Mock(side_effect=AssertionError("DocTR should not be preloaded"))
    monkeypatch.setattr(detector_mod, "get_doctr_detector", doctr_getter)
    monkeypatch.setattr(detector_mod, "_load_yolo11x_model", fail_loader)
    monkeypatch.setattr(detector_mod, "_load_east_model", fail_loader)

    consensus_mod.initialize_consensus_models()

    assert detector_mod._yolo11x_model is existing_yolo
    assert detector_mod._easyocr_reader is existing_easyocr
    assert detector_mod._east_net is existing_east
    assert doctr_getter.call_count == 0


def test_initialize_consensus_models_uses_yolo11x_detector_cache(monkeypatch):
    """Consensus preload should share detector.py production model instances."""
    fake_yolo = object()
    fake_east = object()

    class FakeReader:
        def __init__(self, langs, verbose=False):
            self.langs = langs
            self.verbose = verbose

    fake_easyocr = types.ModuleType("easyocr")
    fake_easyocr.Reader = FakeReader
    monkeypatch.setitem(sys.modules, "easyocr", fake_easyocr)

    monkeypatch.setattr(detector_mod, "_doctr_detector", None)
    monkeypatch.setattr(detector_mod, "_yolo11x_model", None)
    monkeypatch.setattr(detector_mod, "_easyocr_reader", None)
    monkeypatch.setattr(detector_mod, "_east_net", None)
    monkeypatch.setattr(detector_mod, "get_doctr_detector", Mock(side_effect=AssertionError("DocTR should not be preloaded")))
    monkeypatch.setattr(detector_mod, "_load_yolo11x_model", Mock(return_value=fake_yolo))
    monkeypatch.setattr(detector_mod, "_load_east_model", lambda: fake_east)

    consensus_mod.initialize_consensus_models()

    assert detector_mod._doctr_detector is None
    assert detector_mod._yolo11x_model is fake_yolo
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
