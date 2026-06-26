"""Tests for untextre.known_mask processing."""

import numpy as np

import untextre.known_mask as known_mask_mod
from untextre.known_mask import process_with_known_mask

class TestProcessWithKnownMask:
    def test_no_match_saves_original_and_skips_inpaint(self, monkeypatch, tmp_path):
        image = np.ones((30, 30, 3), dtype=np.uint8) * 127
        image_path = tmp_path / "input.png"
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        known_mask = np.zeros((10, 10, 4), dtype=np.uint8)

        monkeypatch.setattr(known_mask_mod, "load_image", lambda _p: image.copy())
        monkeypatch.setattr(known_mask_mod, "find_known_mask_in_image", lambda *_args, **_kwargs: None)
        saved = {}

        def fake_save(arr, path, **kwargs):
            saved[str(path)] = (arr.copy(), kwargs.get("source_path"))

        monkeypatch.setattr(known_mask_mod, "save_image", fake_save)

        timings = process_with_known_mask(
            image_path=image_path,
            output_dir=output_dir,
            known_mask_rgba=known_mask,
            keep_masks=True,
            method="telea",
        )

        assert timings is not None
        assert timings["mask_found"] is False
        assert any(name.endswith("input_clean.png") for name in saved)
        saved_image, source_path = next(
            data for name, data in saved.items() if name.endswith("input_clean.png")
        )
        np.testing.assert_array_equal(saved_image, image)
        assert source_path == image_path
        assert not any(name.endswith("input_mask.png") for name in saved)

    def test_match_path_saves_mask_and_inpainted_output(self, monkeypatch, tmp_path):
        image = np.ones((40, 40, 3), dtype=np.uint8) * 180
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[8:20, 12:26] = 255
        bbox = (12, 8, 14, 12)
        known_mask = np.zeros((10, 10, 4), dtype=np.uint8)
        image_path = tmp_path / "photo.png"
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        monkeypatch.setattr(known_mask_mod, "load_image", lambda _p: image.copy())
        monkeypatch.setattr(
            known_mask_mod,
            "find_known_mask_in_image",
            lambda *_args, **_kwargs: (mask, bbox, 10),
        )

        import untextre.inpaint as inpaint_mod

        inpaint_calls = {}

        def fake_inpaint(img, inpaint_mask, bbox=None, method="lama"):
            inpaint_calls["bbox"] = bbox
            inpaint_calls["method"] = method
            np.testing.assert_array_equal(inpaint_mask, mask)
            return np.zeros_like(img)

        monkeypatch.setattr(inpaint_mod, "inpaint_image", fake_inpaint)

        saved_paths = []

        def fake_save(_arr, p, **kwargs):
            saved_paths.append((p.name, kwargs.get("source_path")))

        monkeypatch.setattr(known_mask_mod, "save_image", fake_save)

        timings = process_with_known_mask(
            image_path=image_path,
            output_dir=output_dir,
            known_mask_rgba=known_mask,
            keep_masks=True,
            method="telea",
        )

        assert timings is not None
        assert timings["mask_found"] is True
        assert inpaint_calls["bbox"] == bbox
        assert inpaint_calls["method"] == "telea"
        assert ("photo_mask.png", None) in saved_paths
        assert ("photo_clean.png", image_path) in saved_paths

