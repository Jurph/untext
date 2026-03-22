import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from untextre.discovery import bucket_images_by_size

def test_bucket_images_by_size_groups_correctly(tmp_path):
    # Create fake image files with known sizes
    paths = [tmp_path / f"img{i}.png" for i in range(4)]
    for p in paths:
        p.touch()

    # Mock load_image to return arrays of known shapes
    shapes = {
        paths[0]: np.zeros((100, 200, 3), dtype=np.uint8),  # 200x100
        paths[1]: np.zeros((100, 200, 3), dtype=np.uint8),  # 200x100
        paths[2]: np.zeros((300, 400, 3), dtype=np.uint8),  # 400x300
        paths[3]: np.zeros((200, 100, 3), dtype=np.uint8),  # 100x200 (portrait)
    }
    with patch("untextre.discovery.load_image", side_effect=lambda p: shapes[p]):
        buckets = bucket_images_by_size(paths)

    assert (200, 100) in buckets
    assert len(buckets[(200, 100)]) == 2
    assert (400, 300) in buckets
    assert (100, 200) in buckets  # portrait is a separate bucket

def test_bucket_images_skips_unreadable(tmp_path):
    paths = [tmp_path / "bad.png"]
    paths[0].touch()
    with patch("untextre.discovery.load_image", side_effect=ValueError("bad")):
        buckets = bucket_images_by_size(paths)
    assert buckets == {}
