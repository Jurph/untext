"""Reporting helpers for CLI and watermark discovery outputs."""

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .orb_prep import (
    CandidateOrbVariant,
    build_candidate_orb_variants,
    prepare_candidate_bgra_for_orb,
)
from .utils import setup_logger

logger = setup_logger(__name__)

@dataclass(frozen=True)
class WatermarkTemplate:
    name: str
    rgba: np.ndarray
    orb_variants: tuple[CandidateOrbVariant, ...]

    def __iter__(self):
        yield self.name
        yield self.rgba

    def __getitem__(self, index: int):
        return (self.name, self.rgba)[index]


def _make_watermark_template(name: str, rgba: np.ndarray) -> WatermarkTemplate:
    return WatermarkTemplate(name, rgba, tuple(build_candidate_orb_variants(rgba)))


def _save_discovered_watermark_candidates(
    output_path: Path,
    candidates: List[np.ndarray],
) -> List[WatermarkTemplate]:
    """Save discovered candidates in the same orb-prepped form that -K will use."""
    watermark_templates: List[WatermarkTemplate] = []

    for i, bgra in enumerate(candidates):
        suffix = "" if i == 0 else f"_{i + 1}"
        candidate_path = output_path / f"watermark_candidate{suffix}.png"
        prepared_bgra = prepare_candidate_bgra_for_orb(bgra)
        if candidate_path.exists():
            logger.warning(f"Overwriting existing candidate: {candidate_path.name}")
        cv2.imwrite(str(candidate_path), prepared_bgra)
        logger.info(f"Saved watermark candidate: {candidate_path.name}")
        watermark_templates.append(_make_watermark_template(candidate_path.name, prepared_bgra))

    written_names = {
        f"watermark_candidate{'' if j == 0 else f'_{j+1}'}.png"
        for j in range(len(candidates))
    }
    stale = [
        p for p in sorted(output_path.glob("watermark_candidate*.png"))
        if p.name not in written_names
    ]
    if stale:
        logger.warning(
            "Stale candidate file(s) from prior run still present: "
            + ", ".join(p.name for p in stale)
        )

    return watermark_templates


def _save_clean_timing_report(detailed_timings: list, total_time: float, avg_time: float, timing_file: Path, method: str, confidence_threshold: float, target_color: Optional[tuple], forced_bbox: Optional[tuple]) -> None:
    """Save a clean timing report to file without duplicate logging."""
    with open(timing_file, 'w') as f:
        f.write("=" * 74 + "\n")
        f.write("CONSENSUS DETECTION + SPATIAL TF-IDF TIMING REPORT\n")
        f.write("=" * 74 + "\n")
        f.write(f"Confidence threshold: {confidence_threshold}\n")
        f.write("TF-IDF granularity: g=4 (auto-retry with g=8 if needed)\n")
        f.write(f"Inpainting method: {method}\n")
        if target_color:
            f.write(f"Target color (deprecated): {target_color}\n")
        if forced_bbox:
            f.write(f"Forced bbox: {forced_bbox}\n")
        
        # Count retries and expansions
        retried_count = sum(1 for t in detailed_timings if t.get('retried_with_g8', False))
        expanded_count = sum(t.get('bboxes_expanded', 0) for t in detailed_timings)
        if retried_count > 0:
            f.write(f"Images retried with g=8: {retried_count}\n")
        if expanded_count > 0:
            f.write(f"Total bboxes expanded: {expanded_count}\n")
        
        f.write("\nColumns: MP=Megapixels, Det=Detection, Msk=Mask, Inp=Inpaint, Tot=Total, Failover=R/T/G/W/B (Rotation/Target/Gray/White/Baseline)\n")
        
        # Header with wider format
        f.write(f"{'Image Name':<25} {'MP':>4} {'Det':>4} {'TF-IDF':>6} {'Msk':>4} {'Inp':>5} {'Tot':>5} {'Boxes':>5} {'Fail':>4}\n")
        f.write("-" * 74 + "\n")
        
        # Individual rows (support both consensus path keys and template-match path keys)
        for timing in detailed_timings:
            name = (timing.get('image_name') or timing.get('image') or '?')[:25]

            # Map failover type to marker
            failover_type = timing.get('failover_type', 'none')
            if timing.get('matched_template'):
                failover_marker = "Tpl"  # Template match
            elif failover_type == 'rotation':
                failover_marker = "R"
            elif failover_type == 'target_color':
                failover_marker = "T"
            elif failover_type == 'gray_enhancement':
                failover_marker = "G"
            elif failover_type == 'white_enhancement':
                failover_marker = "W"
            elif failover_type == 'watermark':
                failover_marker = "B"  # Baseline watermark regions
            else:
                failover_marker = ""

            # Handle None/missing values (template-match and skipped entries lack some keys)
            color_time = timing.get('color_time')
            mask_time = timing.get('mask_time')
            inpaint_time = timing.get('inpaint_time')
            color_time_str = "N/A" if color_time is None else f"{color_time:>6.1f}"
            mask_time_str = "N/A" if mask_time is None else f"{mask_time:>4.1f}"
            inpaint_time_str = "N/A" if inpaint_time is None else f"{inpaint_time:>5.1f}"

            image_mp = timing.get('image_mp', 0)
            detection_time = timing.get('detection_time', 0)
            total_time = timing.get('total_time', 0)
            consensus_boxes_count = timing.get('consensus_boxes_count', 0)

            row = (f"{name:<25} "
                   f"{image_mp:>4.1f} "
                   f"{detection_time:>4.1f} "
                   f"{color_time_str:>6} "
                   f"{mask_time_str:>4} "
                   f"{inpaint_time_str:>5} "
                   f"{total_time:>5.1f} "
                   f"{consensus_boxes_count:>5d} "
                   f"{failover_marker:>4}\n")
            f.write(row)
        
        if len(detailed_timings) > 1:
            f.write("-" * 74 + "\n")

            # Statistics (use .get() so template-match/skipped entries are included)
            det_times = [t.get('detection_time', 0) for t in detailed_timings]
            col_times = [t['color_time'] for t in detailed_timings if t.get('color_time') is not None]
            msk_times = [t['mask_time'] for t in detailed_timings if t.get('mask_time') is not None]
            inp_times = [t['inpaint_time'] for t in detailed_timings if t.get('inpaint_time') is not None]
            tot_times = [t.get('total_time', 0) for t in detailed_timings]
            box_counts = [t.get('consensus_boxes_count', 0) for t in detailed_timings]
            
            # Handle cases where all values might be None
            col_median = statistics.median(col_times) if col_times else 0.0
            col_mean = statistics.mean(col_times) if col_times else 0.0
            msk_median = statistics.median(msk_times) if msk_times else 0.0
            msk_mean = statistics.mean(msk_times) if msk_times else 0.0
            inp_median = statistics.median(inp_times) if inp_times else 0.0
            inp_mean = statistics.mean(inp_times) if inp_times else 0.0
            
            f.write(f"{'MEDIAN':<25} {'':>4} {statistics.median(det_times):>4.1f} "
                   f"{col_median:>6.1f} {msk_median:>4.1f} "
                   f"{inp_median:>5.1f} {statistics.median(tot_times):>5.1f} "
                   f"{statistics.median(box_counts):>5.1f}\n")
            
            f.write(f"{'AVERAGE':<25} {'':>4} {statistics.mean(det_times):>4.1f} "
                   f"{col_mean:>6.1f} {msk_mean:>4.1f} "
                   f"{inp_mean:>5.1f} {statistics.mean(tot_times):>5.1f} "
                   f"{statistics.mean(box_counts):>5.1f}\n")
        
        f.write("-" * 70 + "\n")
        f.write(f"Total processing time: {total_time:.1f} seconds\n")
        f.write(f"Average time per image: {avg_time:.1f} seconds\n")
        f.write(f"Images processed: {len(detailed_timings)}\n")
        
        # Consensus statistics
        total_boxes = sum(t.get('consensus_boxes_count', 0) for t in detailed_timings)
        images_with_consensus = sum(1 for t in detailed_timings if t.get('consensus_boxes_count', 0) > 0)
        f.write(f"Total consensus boxes: {total_boxes}\n")
        f.write(f"Images with consensus: {images_with_consensus}/{len(detailed_timings)} ({100*images_with_consensus/len(detailed_timings):.1f}%)\n")
        
        # Failover statistics
        failover_counts = {}
        template_match_count = 0
        for timing in detailed_timings:
            if timing.get('matched_template'):
                template_match_count += 1
            else:
                failover_type = timing.get('failover_type', 'none')
                failover_counts[failover_type] = failover_counts.get(failover_type, 0) + 1

        f.write("\nFailover usage:\n")
        if template_match_count > 0:
            f.write(f"  Template match: {template_match_count}\n")
        f.write(f"  Normal consensus: {failover_counts.get('none', 0)}\n")
        f.write(f"  Rotation failover: {failover_counts.get('rotation', 0)}\n")
        f.write(f"  Target color enhancement: {failover_counts.get('target_color', 0)}\n")
        f.write(f"  Gray enhancement: {failover_counts.get('gray_enhancement', 0)}\n")
        f.write(f"  White enhancement: {failover_counts.get('white_enhancement', 0)}\n")
        f.write(f"  Baseline watermark regions: {failover_counts.get('watermark', 0)}\n")
