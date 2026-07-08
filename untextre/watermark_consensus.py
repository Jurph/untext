from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import time

import cv2
import numpy as np
from .utils import setup_logger

logger = setup_logger(__name__)


@dataclass(frozen=True)
class CandidateMetadata:
    family_key: tuple[int, int]
    zone: tuple[int, int]
    candidate_index: int
    source_kind: str


@dataclass
class CandidateGeometry:
    alpha_soft: np.ndarray
    support_mask: np.ndarray
    edge_field: np.ndarray
    distance_field: np.ndarray
    full_components: tuple["ComponentDescriptor", ...]
    filtered_components: tuple["ComponentDescriptor", ...]
    score_alpha_soft: np.ndarray
    score_support_mask: np.ndarray
    score_distance_field: np.ndarray
    score_scale: float


@dataclass(frozen=True)
class ComponentDescriptor:
    view_name: str
    bbox: tuple[int, int, int, int]
    area: int
    long_side: float
    short_side: float
    hypotenuse: float
    sqrt_area: float
    aspect_ratio: float
    fill_ratio: float
    solidity: float
    circularity: float


@dataclass(frozen=True)
class ScaleProposal:
    scale: float
    weight: float
    view_name: str
    feature_name: str
    ref_bbox: tuple[int, int, int, int] | None
    mov_bbox: tuple[int, int, int, int] | None


@dataclass
class CandidateRecord:
    bgra: np.ndarray
    geometry: CandidateGeometry
    metadata: CandidateMetadata


@dataclass(frozen=True)
class PairwiseScore:
    compatibility: float
    shared_core_score: float
    unmatched_a_penalty: float
    unmatched_b_penalty: float
    scale: float
    tx: float
    ty: float
    dice: float
    translation_response: float
    scale_origin: str = "fallback"
    scale_proposals: tuple[ScaleProposal, ...] = ()


@dataclass
class CandidateGraph:
    records: tuple[CandidateRecord, ...]
    weights: np.ndarray
    directed_scores: dict


@dataclass(frozen=True)
class ClusterRecord:
    member_indices: tuple[int, ...]
    anchor_index: int
    mean_weight: float
    family_count: int


@dataclass
class ConsensusTemplate:
    label: str
    bgra: np.ndarray
    alpha_mass: float
    member_indices: tuple[int, ...]
    family_count: int


def alpha_to_soft_mask(alpha_u8: np.ndarray) -> np.ndarray:
    if alpha_u8.ndim != 2:
        raise ValueError("alpha_to_soft_mask expects HxW alpha")
    return (alpha_u8.astype(np.float32) / 255.0).clip(0.0, 1.0)


def _alpha_to_support_mask(alpha_u8: np.ndarray) -> np.ndarray:
    soft = alpha_to_soft_mask(alpha_u8)
    if not np.any(soft > 0):
        return np.zeros_like(alpha_u8, dtype=np.uint8)
    thresh = float(np.quantile(soft[soft > 0], 0.25))
    mask = (soft >= thresh).astype(np.uint8) * 255
    return mask


def build_edge_field(alpha_u8: np.ndarray) -> np.ndarray:
    support = _alpha_to_support_mask(alpha_u8)
    if not np.any(support):
        return np.zeros_like(alpha_u8, dtype=np.float32)
    edges = cv2.Canny(support, 50, 150)
    return edges.astype(np.float32) / 255.0


def build_distance_field(alpha_u8: np.ndarray) -> np.ndarray:
    edge = build_edge_field(alpha_u8)
    if not np.any(edge > 0):
        return np.zeros_like(edge, dtype=np.float32)
    edge_u8 = (edge * 255).astype(np.uint8)
    dist = cv2.distanceTransform(255 - edge_u8, cv2.DIST_L2, 3)
    return np.exp(-dist / 4.0).astype(np.float32)


def _build_scoring_view(
    img: np.ndarray,
    longest_side: int = 256,
    binary: bool = False,
) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    current_longest = max(h, w)
    if current_longest <= longest_side:
        return img.copy(), 1.0

    scale = float(longest_side) / float(current_longest)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    interp = cv2.INTER_NEAREST if binary else cv2.INTER_LINEAR
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    return resized, scale


def _filter_scoring_alpha(
    bgra: np.ndarray,
    area_ratio: float = 0.05,
    min_area: int = 24,
    max_components: int = 24,
    color_distance_floor: float = 8.0,
) -> np.ndarray:
    """Filter scoring alpha components using area and Lab color consistency.

    EMPIRICAL: `color_distance_floor=8.0` was selected from PB/SG same-logo
    subset experiments. It closely matched the previous BGR<=24 behavior while
    using Lab distance, where Euclidean distance has perceptual meaning.
    """
    alpha_u8 = bgra[:, :, 3]
    mask = (alpha_u8 > 0).astype(np.uint8)
    if not np.any(mask):
        return alpha_u8

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return alpha_u8

    component_areas = [int(stats[label, cv2.CC_STAT_AREA]) for label in range(1, num_labels)]
    if not component_areas:
        return alpha_u8

    largest_area = max(component_areas)
    area_floor = max(min_area, int(round(largest_area * area_ratio)))
    ranked_labels = sorted(
        range(1, num_labels),
        key=lambda label: int(stats[label, cv2.CC_STAT_AREA]),
        reverse=True,
    )
    candidate_labels = [
        label
        for label in ranked_labels
        if int(stats[label, cv2.CC_STAT_AREA]) >= area_floor
    ][:max_components]
    if not candidate_labels:
        candidate_labels = ranked_labels[:1]

    seed_labels = candidate_labels[: min(8, len(candidate_labels))]
    weighted_lab_sum = np.zeros(3, dtype=np.float64)
    total_weight = 0.0
    component_labs: dict[int, np.ndarray] = {}

    for label in candidate_labels:
        ys, xs = np.where(labels == label)
        if ys.size == 0:
            continue
        color_bgr = bgra[ys, xs, :3].astype(np.float32).mean(axis=0)
        color_lab = _bgr_color_to_lab(color_bgr)
        component_labs[label] = color_lab
        if label in seed_labels:
            area = float(stats[label, cv2.CC_STAT_AREA])
            weighted_lab_sum += color_lab.astype(np.float64) * area
            total_weight += area

    if total_weight <= 0.0:
        keep_labels = candidate_labels
    else:
        reference_lab = (weighted_lab_sum / total_weight).astype(np.float32)
        keep_labels = []
        for label in candidate_labels:
            color_distance = float(np.linalg.norm(component_labs[label] - reference_lab))
            area = int(stats[label, cv2.CC_STAT_AREA])
            if color_distance <= color_distance_floor or area >= int(round(largest_area * 0.75)):
                keep_labels.append(label)

    if not keep_labels:
        keep_labels = candidate_labels[:1]

    filtered = np.zeros_like(alpha_u8)
    for label in keep_labels:
        filtered[labels == label] = alpha_u8[labels == label]
    return filtered


def _bgr_color_to_lab(color_bgr: np.ndarray) -> np.ndarray:
    """Convert one BGR color to OpenCV Lab float coordinates."""
    pixel = (color_bgr.reshape(1, 1, 3).astype(np.float32) / 255.0)
    return cv2.cvtColor(pixel, cv2.COLOR_BGR2LAB).reshape(3)


def _merge_bbox_group(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    xs = [box[0] for box in boxes]
    ys = [box[1] for box in boxes]
    x2s = [box[0] + box[2] for box in boxes]
    y2s = [box[1] + box[3] for box in boxes]
    x0 = min(xs)
    y0 = min(ys)
    x1 = max(x2s)
    y1 = max(y2s)
    return (x0, y0, x1 - x0, y1 - y0)


def _expanded_bbox_for_orientation(
    box: tuple[int, int, int, int],
    canvas_shape: tuple[int, int],
    major_ratio: float,
    minor_ratio: float,
    min_major_padding_px: int,
    min_minor_padding_px: int,
) -> tuple[int, int, int, int]:
    x, y, w, h = box
    canvas_h, canvas_w = canvas_shape
    is_horizontal = w >= h

    if is_horizontal:
        pad_x = max(int(round(w * major_ratio)), min_major_padding_px)
        pad_y = max(int(round(h * minor_ratio)), min_minor_padding_px)
    else:
        pad_x = max(int(round(w * minor_ratio)), min_minor_padding_px)
        pad_y = max(int(round(h * major_ratio)), min_major_padding_px)

    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(canvas_w, x + w + pad_x)
    y1 = min(canvas_h, y + h + pad_y)
    return (x0, y0, x1 - x0, y1 - y0)


def _bboxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (
        ax + aw < bx
        or bx + bw < ax
        or ay + ah < by
        or by + bh < ay
    )


def _crop_group_with_ratio_padding(
    bgra: np.ndarray,
    group_mask: np.ndarray,
    padding_ratio: float,
) -> np.ndarray:
    ys, xs = np.where(group_mask)
    if ys.size == 0:
        return bgra[:1, :1].copy()

    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    cropped = np.zeros((y1 - y0, x1 - x0, 4), dtype=bgra.dtype)
    cropped[:, :, :3] = bgra[y0:y1, x0:x1, :3]
    cropped[:, :, 3] = np.where(group_mask[y0:y1, x0:x1], bgra[y0:y1, x0:x1, 3], 0)

    pad_x = max(1, int(round(cropped.shape[1] * padding_ratio)))
    pad_y = max(1, int(round(cropped.shape[0] * padding_ratio)))
    out = np.zeros(
        (cropped.shape[0] + pad_y * 2, cropped.shape[1] + pad_x * 2, 4),
        dtype=bgra.dtype,
    )
    out[pad_y:pad_y + cropped.shape[0], pad_x:pad_x + cropped.shape[1]] = cropped
    return out


def split_candidate_bgra(
    bgra: np.ndarray,
    major_merge_ratio: float = 0.25,
    minor_merge_ratio: float = 0.05,
    min_major_padding_px: int = 3,
    min_minor_padding_px: int = 1,
    crop_padding_ratio: float = 0.05,
) -> tuple[np.ndarray, ...]:
    alpha = bgra[:, :, 3]
    mask = (alpha > 0).astype(np.uint8)
    if not np.any(mask):
        return (_crop_bgra_to_alpha(bgra, padding=0),)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return (_crop_group_with_ratio_padding(bgra, mask.astype(bool), crop_padding_ratio),)

    component_boxes = {
        label: (
            int(stats[label, cv2.CC_STAT_LEFT]),
            int(stats[label, cv2.CC_STAT_TOP]),
            int(stats[label, cv2.CC_STAT_WIDTH]),
            int(stats[label, cv2.CC_STAT_HEIGHT]),
        )
        for label in range(1, num_labels)
    }
    component_areas = {
        label: int(stats[label, cv2.CC_STAT_AREA])
        for label in range(1, num_labels)
    }

    groups: list[set[int]] = [{label} for label in range(1, num_labels)]
    canvas_shape = bgra.shape[:2]
    changed = True
    while changed:
        changed = False
        group_boxes = {
            index: _merge_bbox_group([component_boxes[label] for label in group])
            for index, group in enumerate(groups)
        }
        expanded_boxes = {
            index: _expanded_bbox_for_orientation(
                group_boxes[index],
                canvas_shape,
                major_ratio=major_merge_ratio,
                minor_ratio=minor_merge_ratio,
                min_major_padding_px=min_major_padding_px,
                min_minor_padding_px=min_minor_padding_px,
            )
            for index in range(len(groups))
        }

        merged_groups: list[set[int]] = []
        seen = [False] * len(groups)
        for start in range(len(groups)):
            if seen[start]:
                continue
            stack = [start]
            seen[start] = True
            merged = set(groups[start])
            while stack:
                current = stack.pop()
                for other in range(len(groups)):
                    if seen[other]:
                        continue
                    if _bboxes_overlap(expanded_boxes[current], expanded_boxes[other]):
                        seen[other] = True
                        stack.append(other)
                        merged.update(groups[other])
            merged_groups.append(merged)

        normalized_before = sorted([tuple(sorted(group)) for group in groups])
        normalized_after = sorted([tuple(sorted(group)) for group in merged_groups])
        if normalized_after != normalized_before:
            changed = True
            groups = merged_groups

    outputs: list[tuple[int, np.ndarray]] = []
    for group in groups:
        group_mask = np.isin(labels, list(group))
        cropped = _crop_group_with_ratio_padding(
            bgra,
            group_mask=group_mask,
            padding_ratio=crop_padding_ratio,
        )
        group_area = int(sum(component_areas[label] for label in group))
        outputs.append((group_area, cropped))

    outputs.sort(key=lambda item: item[0], reverse=True)
    return tuple(crop for _, crop in outputs)


def _describe_components(
    mask_u8: np.ndarray,
    view_name: str,
    max_components: int = 20,
    min_long_side: int = 10,
) -> tuple[ComponentDescriptor, ...]:
    mask = (mask_u8 > 0).astype(np.uint8)
    if not np.any(mask):
        return ()

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return ()

    ranked_labels = sorted(
        range(1, num_labels),
        key=lambda label: int(stats[label, cv2.CC_STAT_AREA]),
        reverse=True,
    )

    descriptors: list[ComponentDescriptor] = []
    for label in ranked_labels:
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        long_side = float(max(w, h))
        short_side = float(max(1, min(w, h)))
        if long_side < float(min_long_side):
            continue

        component_mask = (labels[y:y + h, x:x + w] == label).astype(np.uint8)
        component_mask_u8 = component_mask * 255
        contours, _ = cv2.findContours(
            component_mask_u8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        contour = max(contours, key=cv2.contourArea) if contours else None
        if contour is None:
            hull_area = 0.0
            perimeter = 0.0
        else:
            hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
            perimeter = float(cv2.arcLength(contour, closed=True))

        fill_ratio = float(area) / max(float(w * h), 1.0)
        solidity = float(area) / max(hull_area, 1.0) if hull_area > 0.0 else fill_ratio
        circularity = 0.0
        if perimeter > 1e-6:
            circularity = float((4.0 * np.pi * area) / (perimeter * perimeter))
        circularity = max(0.0, min(circularity, 1.0))

        descriptors.append(
            ComponentDescriptor(
                view_name=view_name,
                bbox=(x, y, w, h),
                area=area,
                long_side=long_side,
                short_side=short_side,
                hypotenuse=float(np.hypot(w, h)),
                sqrt_area=float(np.sqrt(area)),
                aspect_ratio=float(long_side / max(short_side, 1.0)),
                fill_ratio=fill_ratio,
                solidity=max(0.0, min(solidity, 1.0)),
                circularity=circularity,
            )
        )
        if len(descriptors) >= max_components:
            break

    return tuple(descriptors)


def build_candidate_record(bgra: np.ndarray, metadata: CandidateMetadata) -> CandidateRecord:
    bgra = _crop_bgra_to_alpha(bgra, padding=2)
    alpha = bgra[:, :, 3]
    alpha_soft = alpha_to_soft_mask(alpha)
    support_mask = _alpha_to_support_mask(alpha)
    edge_field = build_edge_field(alpha)
    distance_field = build_distance_field(alpha)

    scoring_alpha_u8 = _filter_scoring_alpha(bgra)
    score_alpha_soft_full = alpha_to_soft_mask(scoring_alpha_u8)
    score_support_mask_full = _alpha_to_support_mask(scoring_alpha_u8)
    score_distance_field_full = build_distance_field(scoring_alpha_u8)
    full_components = _describe_components(support_mask, view_name="full")
    filtered_components = _describe_components(score_support_mask_full, view_name="filtered")

    score_alpha_soft, score_scale = _build_scoring_view(score_alpha_soft_full.astype(np.float32), binary=False)
    score_support_mask, _ = _build_scoring_view(score_support_mask_full, binary=True)
    score_distance_field, _ = _build_scoring_view(score_distance_field_full.astype(np.float32), binary=False)
    geometry = CandidateGeometry(
        alpha_soft=alpha_soft,
        support_mask=support_mask,
        edge_field=edge_field,
        distance_field=distance_field,
        full_components=full_components,
        filtered_components=filtered_components,
        score_alpha_soft=score_alpha_soft,
        score_support_mask=score_support_mask,
        score_distance_field=score_distance_field,
        score_scale=score_scale,
    )
    return CandidateRecord(bgra=bgra, geometry=geometry, metadata=metadata)


def _resize_image(img: np.ndarray, scale: float, binary: bool = False) -> np.ndarray:
    h, w = img.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_NEAREST if binary else cv2.INTER_LINEAR
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


def _center_on_canvas(img: np.ndarray, canvas_shape: tuple[int, int]) -> tuple[np.ndarray, int, int]:
    canvas_h, canvas_w = canvas_shape
    h, w = img.shape[:2]
    if h > canvas_h or w > canvas_w:
        raise ValueError(
            f"canvas {canvas_shape} is too small for image shape {img.shape[:2]}"
        )
    if img.ndim == 2:
        out = np.zeros((canvas_h, canvas_w), dtype=img.dtype)
    else:
        out = np.zeros((canvas_h, canvas_w, img.shape[2]), dtype=img.dtype)
    y0 = (canvas_h - h) // 2
    x0 = (canvas_w - w) // 2
    out[y0:y0 + h, x0:x0 + w] = img
    return out, x0, y0


def _build_alignment_canvas_shape(
    ref_shape: tuple[int, int],
    mov_shape: tuple[int, int],
    expansion_factor: float = 3.0,
) -> tuple[int, int]:
    max_h = max(ref_shape[0], mov_shape[0], 64)
    max_w = max(ref_shape[1], mov_shape[1], 64)
    return (
        max(64, int(np.ceil(max_h * expansion_factor))),
        max(64, int(np.ceil(max_w * expansion_factor))),
    )


@lru_cache(maxsize=128)
def _get_hann_window(canvas_shape: tuple[int, int]) -> np.ndarray:
    canvas_h, canvas_w = canvas_shape
    return cv2.createHanningWindow((canvas_w, canvas_h), cv2.CV_32F)


def _translate_image(img: np.ndarray, tx: float, ty: float, binary: bool = False) -> np.ndarray:
    interp = cv2.INTER_NEAREST if binary else cv2.INTER_LINEAR
    h, w = img.shape[:2]
    matrix = np.array([[1.0, 0.0, -tx], [0.0, 1.0, -ty]], dtype=np.float32)
    return cv2.warpAffine(
        img,
        matrix,
        (w, h),
        flags=interp,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _alignment_metrics(
    ref_support: np.ndarray,
    mov_support: np.ndarray,
    ref_alpha: np.ndarray,
    mov_alpha: np.ndarray,
) -> tuple[float, float, float, float]:
    ref_mask = ref_support > 0
    mov_mask = mov_support > 0

    ref_area = float(ref_mask.sum())
    mov_area = float(mov_mask.sum())
    inter = float(np.logical_and(ref_mask, mov_mask).sum())

    if ref_area == 0.0 or mov_area == 0.0:
        return 0.0, 1.0, 1.0, 0.0

    dice = (2.0 * inter) / max(ref_area + mov_area, 1.0)
    shared_core = inter / max(min(ref_area, mov_area), 1.0)
    unmatched_a = max(ref_area - inter, 0.0) / ref_area
    unmatched_b = max(mov_area - inter, 0.0) / mov_area

    # Soft overlap supplements binary overlap, helping semi-transparent edges.
    soft_inter = float(np.minimum(ref_alpha, mov_alpha).sum())
    soft_norm = max(min(float(ref_alpha.sum()), float(mov_alpha.sum())), 1.0)
    soft_shared = soft_inter / soft_norm
    shared_core = 0.7 * shared_core + 0.3 * soft_shared

    return shared_core, unmatched_a, unmatched_b, dice


def _max_scale_from_comparator_source(
    comparator: CandidateRecord,
    moving: CandidateRecord,
    fallback_max_scale: float,
) -> float:
    source_w, source_h = comparator.metadata.family_key
    mov_h, mov_w = moving.bgra.shape[:2]
    if mov_h <= 0 or mov_w <= 0:
        return fallback_max_scale
    physical_cap = min(float(source_h) / float(mov_h), float(source_w) / float(mov_w))
    if not np.isfinite(physical_cap) or physical_cap <= 0.0:
        return fallback_max_scale
    return min(float(fallback_max_scale), physical_cap)


def _component_shape_weight(
    ref_component: ComponentDescriptor,
    mov_component: ComponentDescriptor,
) -> float:
    aspect_delta = abs(
        np.log(max(ref_component.aspect_ratio, 1e-6) / max(mov_component.aspect_ratio, 1e-6))
    )
    # EMPIRICAL — exponential decay scales (0.9, 0.35) not formally validated
    aspect_weight = float(np.exp(-aspect_delta / 0.9))
    fill_weight = float(np.exp(-abs(ref_component.fill_ratio - mov_component.fill_ratio) / 0.35))
    solidity_weight = float(np.exp(-abs(ref_component.solidity - mov_component.solidity) / 0.35))
    circularity_weight = float(
        np.exp(-abs(ref_component.circularity - mov_component.circularity) / 0.35)
    )
    return float(
        max(
            0.15,
            min(
                1.0,
                0.40 * aspect_weight
                + 0.20 * fill_weight
                + 0.20 * solidity_weight
                + 0.20 * circularity_weight,
            ),
        )
    )


def propose_pair_scales(
    candidate_a: CandidateRecord,
    candidate_b: CandidateRecord,
    min_scale: float = 0.25,
    max_scale: float = 4.0,
    max_proposals: int = 16,
) -> tuple[ScaleProposal, ...]:
    feature_weights = (
        ("long_side", 1.00),
        ("short_side", 0.85),
        ("hypotenuse", 0.90),
        ("sqrt_area", 0.75),
    )
    component_views = (
        ("full", candidate_a.geometry.full_components, candidate_b.geometry.full_components),
        ("filtered", candidate_a.geometry.filtered_components, candidate_b.geometry.filtered_components),
    )

    votes: list[tuple[float, float, str, str, tuple[int, int, int, int], tuple[int, int, int, int]]] = []
    for view_name, ref_components, mov_components in component_views:
        if not ref_components or not mov_components:
            continue
        for ref_component in ref_components:
            for mov_component in mov_components:
                shape_weight = _component_shape_weight(ref_component, mov_component)
                prominence_weight = float(np.log1p(max(1, min(ref_component.area, mov_component.area))))
                for feature_name, feature_weight in feature_weights:
                    ref_value = float(getattr(ref_component, feature_name))
                    mov_value = float(getattr(mov_component, feature_name))
                    if ref_value <= 0.0 or mov_value <= 0.0:
                        continue
                    scale = ref_value / mov_value
                    if not np.isfinite(scale) or scale <= 0.0:
                        continue
                    if scale < min_scale or scale > max_scale:
                        continue
                    votes.append(
                        (
                            float(np.log(scale)),
                            float(shape_weight * feature_weight * prominence_weight),
                            view_name,
                            feature_name,
                            ref_component.bbox,
                            mov_component.bbox,
                        )
                    )

    if not votes:
        return ()

    votes.sort(key=lambda vote: vote[0])
    merge_tolerance = float(np.log(1.10))
    clustered_votes: list[list[tuple[float, float, str, str, tuple[int, int, int, int], tuple[int, int, int, int]]]] = []
    for vote in votes:
        if not clustered_votes:
            clustered_votes.append([vote])
            continue
        previous_cluster = clustered_votes[-1]
        previous_weights = np.array([entry[1] for entry in previous_cluster], dtype=np.float64)
        previous_logs = np.array([entry[0] for entry in previous_cluster], dtype=np.float64)
        previous_center = float(np.average(previous_logs, weights=previous_weights))
        if abs(vote[0] - previous_center) <= merge_tolerance:
            previous_cluster.append(vote)
        else:
            clustered_votes.append([vote])

    proposals: list[ScaleProposal] = []
    for cluster in clustered_votes:
        weights = np.array([entry[1] for entry in cluster], dtype=np.float64)
        log_scales = np.array([entry[0] for entry in cluster], dtype=np.float64)
        total_weight = float(weights.sum())
        if total_weight <= 0.0:
            continue
        scale = float(np.exp(np.average(log_scales, weights=weights)))
        representative = max(cluster, key=lambda entry: entry[1])
        proposals.append(
            ScaleProposal(
                scale=scale,
                weight=total_weight,
                view_name=representative[2],
                feature_name=representative[3],
                ref_bbox=representative[4],
                mov_bbox=representative[5],
            )
        )

    proposals.sort(key=lambda proposal: (-proposal.weight, proposal.scale))
    return tuple(proposals[:max_proposals])


def _evaluate_pairwise_scale(
    ref: CandidateGeometry,
    mov: CandidateGeometry,
    working_scale: float,
    aspect_prior: float,
) -> PairwiseScore | None:
    mov_dist = _resize_image(mov.score_distance_field.astype(np.float32), float(working_scale))
    mov_support = _resize_image(mov.score_support_mask, float(working_scale), binary=True)
    mov_alpha = _resize_image(mov.score_alpha_soft.astype(np.float32), float(working_scale))

    canvas_shape = _build_alignment_canvas_shape(
        ref.score_distance_field.shape[:2],
        mov_dist.shape[:2],
    )

    ref_dist_canvas, _, _ = _center_on_canvas(
        ref.score_distance_field.astype(np.float32),
        canvas_shape,
    )
    ref_support_canvas, _, _ = _center_on_canvas(ref.score_support_mask, canvas_shape)
    ref_alpha_canvas, _, _ = _center_on_canvas(
        ref.score_alpha_soft.astype(np.float32),
        canvas_shape,
    )

    mov_dist_canvas, _, _ = _center_on_canvas(mov_dist, canvas_shape)
    mov_support_canvas, _, _ = _center_on_canvas(mov_support, canvas_shape)
    mov_alpha_canvas, _, _ = _center_on_canvas(mov_alpha, canvas_shape)

    if np.max(mov_dist_canvas) < 1e-6:
        return None

    hann = _get_hann_window(canvas_shape)
    (tx, ty), response = cv2.phaseCorrelate(ref_dist_canvas * hann, mov_dist_canvas * hann)
    aligned_support = _translate_image(mov_support_canvas, tx, ty, binary=True)
    aligned_alpha = _translate_image(mov_alpha_canvas, tx, ty, binary=False)

    shared_core, unmatched_a, unmatched_b, dice = _alignment_metrics(
        ref_support_canvas,
        aligned_support,
        ref_alpha_canvas,
        aligned_alpha,
    )
    translation_response = max(0.0, min(float(response), 1.0))

    # EMPIRICAL — weights (0.55, 0.25, 0.10, -0.45, -0.45) not formally validated
    compatibility = (
        0.55 * shared_core
        + 0.25 * dice
        + 0.10 * translation_response
        - 0.45 * unmatched_a
        - 0.45 * unmatched_b
    ) * aspect_prior
    compatibility = max(0.0, min(1.0, compatibility))

    return PairwiseScore(
        compatibility=float(compatibility),
        shared_core_score=float(shared_core),
        unmatched_a_penalty=float(unmatched_a),
        unmatched_b_penalty=float(unmatched_b),
        scale=float(working_scale * mov.score_scale / max(ref.score_scale, 1e-6)),
        tx=float(tx / max(ref.score_scale, 1e-6)),
        ty=float(ty / max(ref.score_scale, 1e-6)),
        dice=float(dice),
        translation_response=float(translation_response),
    )


def _attach_pairwise_metadata(
    score: PairwiseScore,
    scale_origin: str,
    scale_proposals: tuple[ScaleProposal, ...],
) -> PairwiseScore:
    return PairwiseScore(
        compatibility=score.compatibility,
        shared_core_score=score.shared_core_score,
        unmatched_a_penalty=score.unmatched_a_penalty,
        unmatched_b_penalty=score.unmatched_b_penalty,
        scale=score.scale,
        tx=score.tx,
        ty=score.ty,
        dice=score.dice,
        translation_response=score.translation_response,
        scale_origin=scale_origin,
        scale_proposals=scale_proposals,
    )


def _stable_scale_ladder(
    min_scale: float,
    max_scale: float,
    step_ratio: float,
) -> list[float]:
    if max_scale <= min_scale or np.isclose(min_scale, max_scale):
        return [float(max_scale)]

    scales = [float(min_scale)]
    while scales[-1] * step_ratio < max_scale:
        scales.append(float(scales[-1] * step_ratio))
    if not np.isclose(scales[-1], max_scale):
        scales.append(float(max_scale))
    return scales


def score_candidate_pair(
    candidate_a: CandidateRecord,
    candidate_b: CandidateRecord,
    min_scale: float = 0.25,
    max_scale: float = 4.0,
    num_scales: int = 25,
    _precomputed_proposals: "tuple[ScaleProposal, ...] | None" = None,
) -> PairwiseScore:
    ref = candidate_a.geometry
    mov = candidate_b.geometry

    max_scale = _max_scale_from_comparator_source(candidate_a, candidate_b, max_scale)
    if max_scale <= 0.0:
        max_scale = min_scale
    if max_scale < min_scale:
        min_scale = max_scale
    if _precomputed_proposals is not None:
        scale_proposals = _precomputed_proposals
    else:
        scale_proposals = propose_pair_scales(
            candidate_a,
            candidate_b,
            min_scale=min_scale,
            max_scale=max_scale,
        )

    aspect_a = candidate_a.bgra.shape[1] / max(candidate_a.bgra.shape[0], 1)
    aspect_b = candidate_b.bgra.shape[1] / max(candidate_b.bgra.shape[0], 1)
    ratio = min(aspect_a, aspect_b) / max(max(aspect_a, aspect_b), 1e-6)
    aspect_prior = max(0.25, ratio)

    best = PairwiseScore(
        compatibility=0.0,
        shared_core_score=0.0,
        unmatched_a_penalty=1.0,
        unmatched_b_penalty=1.0,
        scale=1.0,
        tx=0.0,
        ty=0.0,
        dice=0.0,
        translation_response=0.0,
        scale_origin="fallback",
        scale_proposals=scale_proposals,
    )

    if np.isclose(min_scale, max_scale):
        score = _evaluate_pairwise_scale(ref, mov, float(min_scale), aspect_prior)
        return _attach_pairwise_metadata(score, "fallback", scale_proposals) if score is not None else best

    coarse_scale_entries: list[tuple[float, str]] = []
    seen_scales: set[float] = set()

    def _append_scale(scale: float, origin: str) -> None:
        rounded = round(float(scale), 6)
        if rounded in seen_scales:
            return
        seen_scales.add(rounded)
        coarse_scale_entries.append((float(scale), origin))

    for proposal in scale_proposals:
        _append_scale(proposal.scale, "proposal")
    for scale in _stable_scale_ladder(min_scale, max_scale, step_ratio=1.08):
        _append_scale(scale, "fallback")

    coarse_results: list[tuple[float, PairwiseScore, str]] = []

    for scale, origin in coarse_scale_entries:
        score = _evaluate_pairwise_scale(ref, mov, float(scale), aspect_prior)
        if score is None:
            continue
        score_with_meta = _attach_pairwise_metadata(score, origin, scale_proposals)
        coarse_results.append((float(scale), score_with_meta, origin))
        if (
            score_with_meta.compatibility > best.compatibility
            or (
                np.isclose(score_with_meta.compatibility, best.compatibility)
                and best.scale_origin == "fallback"
                and origin != "fallback"
            )
        ):
            best = score_with_meta

    if not coarse_results:
        return best

    coarse_results.sort(key=lambda item: item[1].compatibility, reverse=True)
    correction = mov.score_scale / max(ref.score_scale, 1e-6)
    refine_candidates = coarse_results[: min(3, len(coarse_results))]
    refined_scales: set[float] = set()
    refined_scale_origins: dict[float, str] = {}

    for _, score, origin in refine_candidates:
        center = score.scale / max(correction, 1e-6)
        local_min = max(min_scale, center / 1.18)
        local_max = min(max_scale, center * 1.18)
        if np.isclose(local_min, local_max):
            rounded = round(float(local_min), 6)
            refined_scales.add(float(local_min))
            refined_scale_origins[rounded] = "proposal_refine" if origin == "proposal" else "refine"
            continue
        for local_scale in _stable_scale_ladder(local_min, local_max, step_ratio=1.03):
            rounded = round(float(local_scale), 6)
            refined_scales.add(float(local_scale))
            if rounded not in refined_scale_origins:
                refined_scale_origins[rounded] = "proposal_refine" if origin == "proposal" else "refine"

    for scale in sorted(refined_scales):
        score = _evaluate_pairwise_scale(ref, mov, scale, aspect_prior)
        if score is None:
            continue
        origin = refined_scale_origins.get(round(float(scale), 6), "refine")
        score_with_meta = _attach_pairwise_metadata(score, origin, scale_proposals)
        if (
            score_with_meta.compatibility > best.compatibility
            or (
                np.isclose(score_with_meta.compatibility, best.compatibility)
                and best.scale_origin == "fallback"
                and origin != "fallback"
            )
        ):
            best = score_with_meta

    return best


def build_candidate_graph(
    records: list[CandidateRecord] | tuple[CandidateRecord, ...],
    min_weight_floor: float = 0.05,
) -> CandidateGraph:
    graph_records = tuple(records)
    count = len(graph_records)
    weights = np.zeros((count, count), dtype=np.float32)
    directed_scores: dict[tuple[int, int], "PairwiseScore | None"] = {}
    total_pairs = count * (count - 1) // 2

    logger.info(
        "Consensus graph start: %d candidate(s) -> %d unordered pairs / %d directional alignments",
        count,
        total_pairs,
        total_pairs * 2,
    )

    if total_pairs <= 0:
        return CandidateGraph(
            records=graph_records,
            weights=weights,
            directed_scores=directed_scores,
        )

    progress_stride = max(1, total_pairs // 10)
    completed_pairs = 0
    last_logged_pairs = 0
    started_at = time.perf_counter()

    for i in range(count):
        directed_scores[(i, i)] = None
        for j in range(i + 1, count):
            score_ij = score_candidate_pair(graph_records[i], graph_records[j])
            proposals_ji = tuple(
                ScaleProposal(
                    scale=1.0 / p.scale,
                    weight=p.weight,
                    view_name=p.view_name,
                    feature_name=p.feature_name,
                    ref_bbox=p.mov_bbox,
                    mov_bbox=p.ref_bbox,
                )
                for p in score_ij.scale_proposals
                if p.scale > 0
            )
            score_ji = score_candidate_pair(
                graph_records[j], graph_records[i],
                _precomputed_proposals=proposals_ji,
            )
            directed_scores[(i, j)] = score_ij
            directed_scores[(j, i)] = score_ji
            weight = min(score_ij.compatibility, score_ji.compatibility)
            if weight < min_weight_floor:
                weight = 0.0
            weights[i, j] = float(weight)
            weights[j, i] = float(weight)

            completed_pairs += 1
            should_log_progress = (
                completed_pairs == total_pairs
                or completed_pairs - last_logged_pairs >= progress_stride
            )
            if should_log_progress:
                elapsed_seconds = time.perf_counter() - started_at
                elapsed_minutes = int(elapsed_seconds // 60)
                elapsed_remainder = int(elapsed_seconds % 60)
                percent = int(round((completed_pairs / total_pairs) * 100))
                logger.info(
                    "Consensus graph progress: %d/%d pairs (%d%%), elapsed %02d:%02d",
                    completed_pairs,
                    total_pairs,
                    percent,
                    elapsed_minutes,
                    elapsed_remainder,
                )
                last_logged_pairs = completed_pairs

    return CandidateGraph(
        records=graph_records,
        weights=weights,
        directed_scores=directed_scores,
    )


def _connected_components(adjacency: np.ndarray) -> list[tuple[int, ...]]:
    count = adjacency.shape[0]
    seen = np.zeros(count, dtype=bool)
    components: list[tuple[int, ...]] = []

    for start in range(count):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        members: list[int] = []
        while stack:
            node = stack.pop()
            members.append(node)
            neighbors = np.flatnonzero(adjacency[node])
            for neighbor in neighbors:
                if not seen[neighbor]:
                    seen[neighbor] = True
                    stack.append(int(neighbor))
        components.append(tuple(sorted(members)))
    return components


def _prune_component_by_degree(
    component: tuple[int, ...],
    adjacency: np.ndarray,
    min_degree: int,
) -> tuple[int, ...]:
    active = np.array(component, dtype=np.int32)
    while active.size > 0:
        local = adjacency[np.ix_(active, active)]
        degrees = local.sum(axis=1)
        keep = degrees >= min_degree
        if np.all(keep):
            break
        active = active[keep]
        if active.size < max(min_degree + 1, 2):
            break
    return tuple(int(index) for index in active.tolist())


def _component_mean_weight(indices: tuple[int, ...], weights: np.ndarray) -> float:
    if len(indices) < 2:
        return 0.0
    local = weights[np.ix_(indices, indices)]
    tri = local[np.triu_indices(len(indices), k=1)]
    if tri.size == 0:
        return 0.0
    return float(np.mean(tri))


def _component_anchor_index(indices: tuple[int, ...], weights: np.ndarray) -> int:
    local = weights[np.ix_(indices, indices)]
    sums = local.sum(axis=1)
    best_offset = int(np.argmax(sums))
    return indices[best_offset]


def extract_candidate_clusters(
    graph: CandidateGraph,
    edge_threshold: float = 0.5,
    min_cluster_size: int = 2,
    fallback_pair_threshold: float = 0.30,
) -> list[ClusterRecord]:
    adjacency = graph.weights >= float(edge_threshold)
    np.fill_diagonal(adjacency, False)

    raw_components = _connected_components(adjacency)
    clusters: list[ClusterRecord] = []

    for component in raw_components:
        if len(component) < min_cluster_size:
            continue

        refined = component
        if len(component) >= 3:
            refined = _prune_component_by_degree(component, adjacency, min_degree=2)
            if len(refined) < min_cluster_size:
                continue

            local_adj = adjacency[np.ix_(refined, refined)]
            subcomponents = _connected_components(local_adj)
            refined_components = [
                tuple(refined[index] for index in subcomponent)
                for subcomponent in subcomponents
                if len(subcomponent) >= min_cluster_size
            ]
        else:
            refined_components = [refined]

        for refined_component in refined_components:
            families = {
                graph.records[index].metadata.family_key
                for index in refined_component
            }
            clusters.append(
                ClusterRecord(
                    member_indices=tuple(sorted(refined_component)),
                    anchor_index=_component_anchor_index(refined_component, graph.weights),
                    mean_weight=_component_mean_weight(refined_component, graph.weights),
                    family_count=len(families),
                )
            )

    clusters.sort(
        key=lambda cluster: (
            -cluster.family_count,
            -len(cluster.member_indices),
            -cluster.mean_weight,
            cluster.anchor_index,
        )
    )
    if clusters:
        return clusters

    count = len(graph.records)
    if count < 2:
        return clusters

    pair_strengths = np.zeros((count, count), dtype=np.float32)
    for i in range(count):
        for j in range(i + 1, count):
            score_ij = graph.directed_scores[(i, j)]
            score_ji = graph.directed_scores[(j, i)]
            compat_ij = 0.0 if score_ij is None else float(score_ij.compatibility)
            compat_ji = 0.0 if score_ji is None else float(score_ji.compatibility)
            pair_strength = max(compat_ij, compat_ji)
            pair_strengths[i, j] = pair_strength
            pair_strengths[j, i] = pair_strength

    best_partners = np.argmax(pair_strengths, axis=1)
    candidate_pairs: list[tuple[float, int, int]] = []
    for i in range(count):
        j = int(best_partners[i])
        if i == j or j <= i:
            continue
        if best_partners[j] != i:
            continue
        strength = float(pair_strengths[i, j])
        if strength < fallback_pair_threshold:
            continue
        candidate_pairs.append((strength, i, j))

    candidate_pairs.sort(reverse=True)
    used_indices: set[int] = set()
    for strength, i, j in candidate_pairs:
        if i in used_indices or j in used_indices:
            continue
        member_indices = (i, j)
        families = {graph.records[i].metadata.family_key, graph.records[j].metadata.family_key}
        clusters.append(
            ClusterRecord(
                member_indices=member_indices,
                anchor_index=_component_anchor_index(member_indices, pair_strengths),
                mean_weight=strength,
                family_count=len(families),
            )
        )
        used_indices.add(i)
        used_indices.add(j)

    return clusters


def _scaled_shape(shape: tuple[int, int], scale: float) -> tuple[int, int]:
    h, w = shape[:2]
    return (
        max(1, int(round(h * scale))),
        max(1, int(round(w * scale))),
    )


def _cluster_canvas_shape(graph: CandidateGraph, cluster: ClusterRecord) -> tuple[int, int]:
    anchor_shape = graph.records[cluster.anchor_index].bgra.shape[:2]
    max_h, max_w = anchor_shape
    for member_index in cluster.member_indices:
        if member_index == cluster.anchor_index:
            continue
        score = graph.directed_scores[(cluster.anchor_index, member_index)]
        if score is None:
            continue
        scaled_h, scaled_w = _scaled_shape(
            graph.records[member_index].bgra.shape[:2],
            score.scale,
        )
        max_h = max(max_h, scaled_h)
        max_w = max(max_w, scaled_w)
    return _build_alignment_canvas_shape(anchor_shape, (max_h, max_w))


def _align_record_to_anchor(
    graph: CandidateGraph,
    anchor_index: int,
    member_index: int,
    canvas_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    record = graph.records[member_index]
    if member_index == anchor_index:
        scale = 1.0
        tx = 0.0
        ty = 0.0
    else:
        score = graph.directed_scores[(anchor_index, member_index)]
        if score is None:
            raise ValueError(f"missing directed score for anchor {anchor_index} <- member {member_index}")
        scale = score.scale
        tx = score.tx
        ty = score.ty

    scaled_bgra = _resize_image(record.bgra, float(scale), binary=False)
    scaled_alpha = _resize_image(record.geometry.alpha_soft.astype(np.float32), float(scale))
    scaled_support = _resize_image(record.geometry.support_mask, float(scale), binary=True)

    bgra_canvas, _, _ = _center_on_canvas(scaled_bgra, canvas_shape)
    alpha_canvas, _, _ = _center_on_canvas(scaled_alpha, canvas_shape)
    support_canvas, _, _ = _center_on_canvas(scaled_support, canvas_shape)

    return (
        _translate_image(bgra_canvas, tx, ty, binary=False),
        _translate_image(alpha_canvas, tx, ty, binary=False),
        _translate_image(support_canvas, tx, ty, binary=True),
    )


def _crop_bgra_to_alpha(bgra: np.ndarray, padding: int = 0) -> np.ndarray:
    alpha = bgra[:, :, 3]
    nonzero = np.argwhere(alpha > 0)
    if nonzero.size == 0:
        return bgra[:1, :1].copy()
    y0, x0 = nonzero.min(axis=0)
    y1, x1 = nonzero.max(axis=0)
    cropped = bgra[y0:y1 + 1, x0:x1 + 1].copy()
    if padding <= 0:
        return cropped
    out = np.zeros(
        (cropped.shape[0] + padding * 2, cropped.shape[1] + padding * 2, cropped.shape[2]),
        dtype=cropped.dtype,
    )
    out[padding:padding + cropped.shape[0], padding:padding + cropped.shape[1]] = cropped
    return out


def _largest_component_area(mask: np.ndarray) -> int:
    mask_u8 = mask.astype(np.uint8)
    if not np.any(mask_u8):
        return 0
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return 0
    return int(np.max(stats[1:, cv2.CC_STAT_AREA]))


def _has_separated_disputed_component(
    disputed_mask: np.ndarray,
    stingy_mask: np.ndarray,
    area_ratio_floor: float,
    distance_floor: float,
) -> bool:
    disputed_u8 = disputed_mask.astype(np.uint8)
    if not np.any(disputed_u8):
        return False

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(disputed_u8, connectivity=8)
    if num_labels <= 1:
        return False

    stingy_area = max(int(np.count_nonzero(stingy_mask)), 1)
    distance_map = cv2.distanceTransform((~stingy_mask).astype(np.uint8) * 255, cv2.DIST_L2, 3)

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area / stingy_area < area_ratio_floor:
            continue
        ys, xs = np.where(labels == label)
        if ys.size == 0:
            continue
        min_distance = float(distance_map[ys, xs].min())
        if min_distance >= distance_floor:
            return True
    return False


def _build_template_from_mask(
    label: str,
    mask: np.ndarray,
    alpha_sum: np.ndarray,
    rgb_alpha_sum: np.ndarray,
    support_count: np.ndarray,
    member_indices: tuple[int, ...],
    family_count: int,
) -> ConsensusTemplate:
    safe_mask = mask & (alpha_sum > 1e-6)
    mean_alpha = np.zeros_like(alpha_sum, dtype=np.float32)
    mean_alpha[safe_mask] = alpha_sum[safe_mask] / np.maximum(support_count[safe_mask], 1.0)

    rgb = np.zeros_like(rgb_alpha_sum, dtype=np.float32)
    denom = np.maximum(alpha_sum[:, :, None], 1e-6)
    rgb[safe_mask] = rgb_alpha_sum[safe_mask] / denom[safe_mask]

    bgra = np.zeros((*mask.shape, 4), dtype=np.uint8)
    bgra[:, :, :3] = np.clip(np.round(rgb), 0.0, 255.0).astype(np.uint8)
    bgra[:, :, 3] = np.clip(np.round(mean_alpha * 255.0), 0.0, 255.0).astype(np.uint8)
    bgra[~safe_mask] = 0
    cropped = _crop_bgra_to_alpha(bgra)
    alpha_mass = float(cropped[:, :, 3].astype(np.float32).sum() / 255.0)
    return ConsensusTemplate(
        label=label,
        bgra=cropped,
        alpha_mass=alpha_mass,
        member_indices=member_indices,
        family_count=family_count,
    )


def build_cluster_outputs(
    graph: CandidateGraph,
    cluster: ClusterRecord,
    stingy_ratio: float = 0.67,
    disputed_ratio_floor: float = 0.01,
    disputed_component_floor: float = 0.04,
    disputed_distance_floor: float = 6.0,
) -> list[ConsensusTemplate]:
    canvas_shape = _cluster_canvas_shape(graph, cluster)
    support_count = np.zeros(canvas_shape, dtype=np.float32)
    alpha_sum = np.zeros(canvas_shape, dtype=np.float32)
    rgb_alpha_sum = np.zeros((*canvas_shape, 3), dtype=np.float32)

    for member_index in cluster.member_indices:
        aligned_bgra, aligned_alpha, aligned_support = _align_record_to_anchor(
            graph,
            cluster.anchor_index,
            member_index,
            canvas_shape,
        )
        alpha_sum += aligned_alpha.astype(np.float32)
        support_count += (aligned_support > 0).astype(np.float32)
        rgb_alpha_sum += aligned_bgra[:, :, :3].astype(np.float32) * aligned_alpha[:, :, None].astype(np.float32)

    member_count = max(len(cluster.member_indices), 1)
    stingy_votes = 1 if member_count == 1 else max(2, int(np.round(stingy_ratio * member_count)))
    stingy_mask = (support_count >= float(stingy_votes)) & (alpha_sum > 1e-6)
    generous_mask = (support_count >= 1.0) & (alpha_sum > 1e-6)

    if not np.any(stingy_mask):
        stingy_mask = generous_mask.copy()

    disputed_mask = generous_mask & ~stingy_mask
    disputed_mass = float(alpha_sum[disputed_mask].sum())
    core_mass = float(alpha_sum[stingy_mask].sum())
    has_separated_component = _has_separated_disputed_component(
        disputed_mask,
        stingy_mask,
        area_ratio_floor=disputed_component_floor,
        distance_floor=disputed_distance_floor,
    )

    if (
        disputed_mass > 0.0
        and disputed_mass / max(core_mass, 1e-6) >= disputed_ratio_floor
        and has_separated_component
    ):
        stingy = _build_template_from_mask(
            label="stingy",
            mask=stingy_mask,
            alpha_sum=alpha_sum,
            rgb_alpha_sum=rgb_alpha_sum,
            support_count=support_count,
            member_indices=cluster.member_indices,
            family_count=cluster.family_count,
        )
        generous = _build_template_from_mask(
            label="generous",
            mask=generous_mask,
            alpha_sum=alpha_sum,
            rgb_alpha_sum=rgb_alpha_sum,
            support_count=support_count,
            member_indices=cluster.member_indices,
            family_count=cluster.family_count,
        )
        return [stingy, generous]

    consensus_mask = stingy_mask if np.any(stingy_mask) else generous_mask
    return [
        _build_template_from_mask(
            label="consensus",
            mask=consensus_mask,
            alpha_sum=alpha_sum,
            rgb_alpha_sum=rgb_alpha_sum,
            support_count=support_count,
            member_indices=cluster.member_indices,
            family_count=cluster.family_count,
        )
    ]


def _record_to_template(record: CandidateRecord) -> ConsensusTemplate:
    cropped = _crop_bgra_to_alpha(record.bgra)
    alpha_mass = float(cropped[:, :, 3].astype(np.float32).sum() / 255.0)
    return ConsensusTemplate(
        label="source",
        bgra=cropped,
        alpha_mass=alpha_mass,
        member_indices=(record.metadata.candidate_index,),
        family_count=1,
    )


def build_final_templates(
    records: list[CandidateRecord] | tuple[CandidateRecord, ...],
    edge_threshold: float = 0.45,
) -> list[ConsensusTemplate]:
    if not records:
        return []

    graph = build_candidate_graph(records)
    clusters = extract_candidate_clusters(graph, edge_threshold=edge_threshold)
    if not clusters:
        templates = [_record_to_template(record) for record in records]
        templates.sort(key=lambda template: -template.alpha_mass)
        return templates

    outputs: list[ConsensusTemplate] = []
    for cluster in clusters:
        outputs.extend(build_cluster_outputs(graph, cluster))
    for record in records:
        outputs.append(_record_to_template(record))
    return outputs
