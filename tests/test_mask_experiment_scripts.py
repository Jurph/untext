import json
import subprocess
import sys

import cv2
import numpy as np


def _write_case(root, case_id="case-1"):
    images = root / "images"
    masks = root / "masks"
    originals = root / "originals"
    images.mkdir(parents=True)
    masks.mkdir(parents=True)
    originals.mkdir(parents=True)

    clean = np.ones((24, 24, 3), dtype=np.uint8) * 220
    watermarked = clean.copy()
    truth = np.zeros((24, 24), dtype=np.uint8)
    truth[8:12, 8:16] = 255
    watermarked[truth > 0] = 0

    cv2.imwrite(str(images / f"{case_id}.png"), watermarked)
    cv2.imwrite(str(masks / f"{case_id}_mask.png"), truth)
    cv2.imwrite(str(originals / f"{case_id}.png"), clean)
    manifest = {
        "cases": [
            {
                "id": case_id,
                "image": f"images/{case_id}.png",
                "truth_mask": f"masks/{case_id}_mask.png",
                "clean": f"originals/{case_id}.png",
                "truth_bbox": [8, 8, 8, 4],
            }
        ]
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root / "manifest.json"


def test_run_mask_grid_writes_one_row_per_fixture_config(tmp_path):
    manifest = _write_case(tmp_path / "fixtures")
    out = tmp_path / "grid.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_mask_grid.py",
            "--preset",
            "local-cleanup",
            "--manifest",
            str(manifest),
            "--out",
            str(out),
            "--limit-configs",
            "2",
        ],
        check=True,
        cwd=".",
        text=True,
        capture_output=True,
    )

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert result.returncode == 0
    assert len(rows) == 2
    assert {row["case_id"] for row in rows} == {"case-1"}
    assert rows[0]["config_id"] == "local-cleanup-000000"
    assert "target_recall" in rows[0]


def test_summarize_mask_grid_writes_ranked_csv_and_top_json(tmp_path):
    jsonl = tmp_path / "grid.jsonl"
    jsonl.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "config_id": "winner",
                        "preset": "local-cleanup",
                        "target_recall": 1.0,
                        "weighted_precision": 0.9,
                        "coverage": 0.02,
                        "overmask_ratio": 1.1,
                        "score": 0.8,
                        "config": {"config_id": "winner"},
                    }
                ),
                json.dumps(
                    {
                        "config_id": "filtered",
                        "preset": "local-cleanup",
                        "target_recall": 0.5,
                        "weighted_precision": 1.0,
                        "coverage": 0.01,
                        "overmask_ratio": 1.0,
                        "score": 0.5,
                        "config": {"config_id": "filtered"},
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    csv_out = tmp_path / "summary.csv"
    top_json = tmp_path / "top.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/summarize_mask_grid.py",
            str(jsonl),
            "--out",
            str(csv_out),
            "--top-json",
            str(top_json),
        ],
        check=True,
        cwd=".",
    )

    assert "winner" in csv_out.read_text(encoding="utf-8")
    top = json.loads(top_json.read_text(encoding="utf-8"))
    assert top[0]["config_id"] == "winner"


def test_run_inpaint_eval_accepts_top_config_json(tmp_path):
    manifest = _write_case(tmp_path / "fixtures")
    configs = tmp_path / "top.json"
    configs.write_text(
        json.dumps([{"config_id": "oracle", "config": {"config_id": "oracle"}}]),
        encoding="utf-8",
    )
    out = tmp_path / "inpaint.jsonl"

    subprocess.run(
        [
            sys.executable,
            "scripts/run_inpaint_eval.py",
            "--configs",
            str(configs),
            "--manifest",
            str(manifest),
            "--method",
            "telea",
            "--out",
            str(out),
        ],
        check=True,
        cwd=".",
    )

    row = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert row["method"] == "telea"
    assert "ssim_gain_ratio" in row
    assert "delta_e_gain_ratio" in row
