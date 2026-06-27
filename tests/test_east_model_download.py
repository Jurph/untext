"""EAST model download reliability tests."""

from pathlib import Path

import pytest

import untextre.detector as detector


class _FakeResponse:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _size):
        if self._chunks:
            return self._chunks.pop(0)
        return b""


def test_download_east_model_rejects_tiny_response(tmp_path):
    model_path = tmp_path / "frozen_east_text_detection.pb"

    def fake_urlopen(url, timeout):
        assert url == detector.EAST_MODEL_URL
        assert timeout == detector.EAST_DOWNLOAD_TIMEOUT_SECONDS
        return _FakeResponse([b"<html>not a protobuf</html>"])

    with pytest.raises(RuntimeError) as excinfo:
        detector._download_east_model(model_path, urlopen=fake_urlopen)

    message = str(excinfo.value)
    assert "EAST model download failed" in message
    assert detector.EAST_MODEL_URL in message
    assert str(model_path) in message
    assert "docs/detector-models.md" in message
    assert not model_path.exists()
    assert not Path(f"{model_path}.tmp").exists()


def test_download_east_model_installs_valid_sized_response(tmp_path, monkeypatch):
    model_path = tmp_path / "frozen_east_text_detection.pb"
    monkeypatch.setattr(detector, "EAST_MODEL_MIN_BYTES", 4)

    def fake_urlopen(_url, timeout):
        assert timeout == detector.EAST_DOWNLOAD_TIMEOUT_SECONDS
        return _FakeResponse([b"abcd", b"efgh"])

    detector._download_east_model(model_path, urlopen=fake_urlopen)

    assert model_path.read_bytes() == b"abcdefgh"
    assert not Path(f"{model_path}.tmp").exists()


def test_load_east_model_rejects_tiny_cached_file(tmp_path, monkeypatch):
    model_path = tmp_path / "frozen_east_text_detection.pb"
    model_path.write_bytes(b"<html>not a protobuf</html>")
    monkeypatch.setattr(detector, "_get_east_model_path", lambda: model_path)

    def fail_read_net(_path):
        raise AssertionError("tiny cached model should not be passed to OpenCV")

    monkeypatch.setattr(detector.cv2.dnn, "readNet", fail_read_net)

    with pytest.raises(RuntimeError) as excinfo:
        detector._load_east_model()

    assert "EAST model file is too small" in str(excinfo.value)
    assert str(model_path) in str(excinfo.value)
