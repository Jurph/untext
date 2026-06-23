"""Model lifecycle tests for long-running bulk processing."""

import types

import pytest

import untextre.consensus as consensus_mod
import untextre.inpaint as inpaint_mod


def test_initialize_consensus_models_keeps_existing_instances(monkeypatch):
    """Repeated preload calls must not replace already-loaded detector models."""
    existing_doctr = object()
    existing_easyocr = object()
    existing_east = object()
    monkeypatch.setattr(consensus_mod, "_global_doctr_detector", existing_doctr)
    monkeypatch.setattr(consensus_mod, "_global_easyocr_reader", existing_easyocr)
    monkeypatch.setattr(consensus_mod, "_global_east_model", existing_east)

    def fail_loader(*_args, **_kwargs):
        raise AssertionError("model loader should not be called when instance exists")

    monkeypatch.setattr(consensus_mod, "TextDetector", fail_loader)

    consensus_mod.initialize_consensus_models()

    assert consensus_mod._global_doctr_detector is existing_doctr
    assert consensus_mod._global_easyocr_reader is existing_easyocr
    assert consensus_mod._global_east_model is existing_east


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
