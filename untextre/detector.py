"""Text detector loaders and adapters for EAST, EasyOCR, and YOLO11x.

This module owns shared model instances, normalizes detector outputs into the
project's detection shape, and exposes single-detector entry points used by the
consensus layer.
"""


import cv2
import gc
import numpy as np
import torch
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .utils import ImageArray, BBox, setup_logger, CLI_DEFAULT_CONFIDENCE

import warnings
warnings.filterwarnings("ignore", message="defusedxml.cElementTree is deprecated")

logger = setup_logger(__name__)

# Type alias for detection results
Detection = Dict[str, Any]  # {'geometry': points, 'confidence': score}

# Module-level model instances for persistent loading
_easyocr_reader: Optional[object] = None
_east_net: Optional[Any] = None
_yolo11x_model: Optional[object] = None

EAST_MODEL_URL = "https://github.com/oyyd/frozen_east_text_detection.pb/raw/master/frozen_east_text_detection.pb"
EAST_MODEL_DOCS = "docs/detector-models.md"
EAST_DOWNLOAD_TIMEOUT_SECONDS = 60
EAST_MODEL_MIN_BYTES = 10 * 1024 * 1024

YOLO11X_MODEL_URL = "https://huggingface.co/spaces/fancyfeast/joycaption-watermark-detection/resolve/main/yolo11x-train28-best.pt"
YOLO11X_MODEL_DOCS = "docs/detector-models.md"
YOLO11X_DOWNLOAD_TIMEOUT_SECONDS = 120
YOLO11X_MODEL_MIN_BYTES = 50 * 1024 * 1024




def get_easyocr_reader() -> object:
    """Return the shared EasyOCR reader."""
    global _easyocr_reader

    if _easyocr_reader is None:
        logger.info("Initializing EasyOCR model...")
        import easyocr
        _easyocr_reader = easyocr.Reader(['en'], verbose=False)
        logger.info("EasyOCR model ready")

    return _easyocr_reader


def get_east_net() -> Any:
    """Return the shared EAST text detector network."""
    global _east_net

    if _east_net is None:
        logger.info("Initializing EAST text detector...")
        _east_net = _load_east_model()
        logger.info("EAST model ready")

    return _east_net

def get_yolo11x_model() -> object:
    """Return the shared YOLO11x watermark detector."""
    global _yolo11x_model

    if _yolo11x_model is None:
        logger.info("Initializing YOLO11x watermark detector...")
        _yolo11x_model = _load_yolo11x_model()
        logger.info("YOLO11x model ready")

    return _yolo11x_model


def cleanup_vram() -> None:
    """Force cleanup of GPU memory after detection operations.
    
    Call this periodically during long-running batch operations to prevent
    VRAM accumulation from intermediate tensors.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def detect_text_regions(
    image: ImageArray,
    method: str = "east",
    confidence_threshold: float = CLI_DEFAULT_CONFIDENCE,
) -> List[BBox]:
    """Detect text regions in an image and return bounding boxes.
    
    This is the main entry point for text detection. It applies preprocessing,
    runs the specified detection method, and returns bounding boxes for detected text.
    
    Args:
        image: Input image as H×W×3 BGR uint8 numpy array
        method: Detection method to use ("east", "easyocr", or "yolo11x")
        confidence_threshold: Minimum confidence threshold for detections (0.0-1.0, default: 0.3)
        
    Returns:
        List of bounding boxes as (x, y, width, height) tuples
        
    Raises:
        ValueError: If image is invalid or method is unsupported
        RuntimeError: If detection fails
    """
    if method == "east":
        net = get_east_net()
        detections = _detect_with_east(image, net, min_confidence=confidence_threshold)
    elif method == "easyocr":
        reader = get_easyocr_reader()
        detections = _detect_with_easyocr(image, reader, confidence_threshold=confidence_threshold)
    elif method == "yolo11x":
        model = get_yolo11x_model()
        detections = _detect_with_yolo11x(image, model, confidence_threshold=confidence_threshold)
    else:
        raise ValueError(f"Unsupported detection method: {method}")
    
    logger.info(f"Using {method.upper()} text detection")
    
    # Convert detections to bounding boxes
    bboxes = []
    for det in detections:
        bbox = _geometry_to_bbox(det['geometry'])
        bboxes.append(bbox)
    
    if bboxes:
        bbox_coords = [f"({bbox[0]},{bbox[1]})" for bbox in bboxes]
        logger.info(f"Detected {len(bboxes)} text regions")
        logger.info(f"Found bounding boxes at: {', '.join(bbox_coords)}")
        # Also log dimensions for context
        bbox_dims = [f"{bbox[2]}x{bbox[3]}" for bbox in bboxes]
        logger.info(f"Bounding box dimensions: {', '.join(bbox_dims)}")
    else:
        logger.info("No text regions detected")
    
    return bboxes

def _detect_with_easyocr(image: ImageArray, reader: Any, confidence_threshold: float = CLI_DEFAULT_CONFIDENCE) -> List[Detection]:
    """Detect text regions using EasyOCR with pre-initialized reader.
    
    Args:
        image: Input image as H×W×3 BGR uint8 numpy array
        reader: Pre-initialized EasyOCR reader instance
        confidence_threshold: Minimum confidence threshold for detections (0.0-1.0, default: 0.3)
        
    Returns:
        List of detection dictionaries in the shared geometry/confidence format
        
    Raises:
        RuntimeError: If detection fails
    """
    try:
        # Convert BGR to RGB for EasyOCR
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run detection
        results = reader.readtext(rgb_image)
        
        # Convert EasyOCR results to our format, filtering by confidence
        detections = []
        for bbox, text, confidence in results:
            # Skip detections below confidence threshold (0.3)
            if confidence < confidence_threshold:
                continue
                
            # EasyOCR returns bbox as [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            # Convert to numpy array for consistency
            geometry = np.array(bbox, dtype=np.float32)
            
            detection = {
                'geometry': geometry,
                'confidence': float(confidence)
            }
            detections.append(detection)
            
        logger.debug(f"EasyOCR found {len(detections)} text regions")
        return detections
        
    except Exception as e:
        logger.error(f"EasyOCR detection failed: {e}")
        raise RuntimeError("EasyOCR detection failed") from e

def _detect_with_yolo11x(
    image: ImageArray,
    model: Any,
    confidence_threshold: float = CLI_DEFAULT_CONFIDENCE,
) -> List[Detection]:
    """Detect watermark regions using a YOLO11x model with pre-loaded weights."""
    try:
        results = model.predict(image, conf=confidence_threshold, verbose=False)
        detections: List[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", [])
            if boxes is None:
                continue

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                points = np.array(
                    [
                        [x1, y1],
                        [x2, y1],
                        [x2, y2],
                        [x1, y2],
                    ],
                    dtype=np.float32,
                )
                detections.append(
                    {
                        "geometry": points,
                        "confidence": confidence,
                    }
                )
        logger.debug(f"YOLO11x found {len(detections)} watermark regions")
        return detections
    except Exception as e:
        logger.error(f"YOLO11x detection failed: {e}")
        raise RuntimeError("YOLO11x detection failed") from e

def _get_east_model_path() -> Path:
    """Return the persistent cache path for the EAST model."""
    model_dir = Path.home() / ".untextre" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir / "frozen_east_text_detection.pb"


def _east_manual_download_message(model_path: Path, reason: str) -> str:
    return (
        f"{reason}. Download the EAST model manually from {EAST_MODEL_URL} "
        f"and save it as {model_path}. See {EAST_MODEL_DOCS} for detector model sources."
    )


def _validate_east_model_file(model_path: Path) -> None:
    if model_path.stat().st_size < EAST_MODEL_MIN_BYTES:
        raise RuntimeError(
            _east_manual_download_message(
                model_path,
                f"EAST model file is too small: {model_path}",
            )
        )


def _download_east_model(
    model_path: Path,
    *,
    urlopen=urllib.request.urlopen,
) -> None:
    """Download the EAST model atomically and reject truncated/error responses."""
    tmp_path = Path(f"{model_path}.tmp")
    try:
        with urlopen(EAST_MODEL_URL, timeout=EAST_DOWNLOAD_TIMEOUT_SECONDS) as response:
            with tmp_path.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)

        _validate_east_model_file(tmp_path)
        tmp_path.replace(model_path)
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        if isinstance(exc, RuntimeError):
            raise RuntimeError(
                _east_manual_download_message(model_path, "EAST model download failed")
            ) from exc
        raise RuntimeError(
            _east_manual_download_message(model_path, "EAST model download failed")
        ) from exc


def _load_east_model() -> Any:
    """Load the EAST text detection model.
    
    This function attempts to download the EAST model if it doesn't exist locally.
    The EAST (Efficient and Accurate Scene Text) detector is a deep learning model
    designed for text detection in natural scene images.
    
    Returns:
        Loaded OpenCV DNN network
        
    Raises:
        RuntimeError: If model loading fails
    """
    try:
        model_path = _get_east_model_path()
        
        # Download model if it doesn't exist
        if not model_path.exists():
            logger.info("Downloading EAST text detection model...")
            _download_east_model(model_path)
            logger.info(f"EAST model downloaded to: {model_path}")

        _validate_east_model_file(model_path)
        
        # Load the network
        net = cv2.dnn.readNet(str(model_path))
        logger.debug(f"EAST model loaded from: {model_path}")
        return net
        
    except RuntimeError:
        raise
    except Exception as e:
        logger.error(f"Failed to load EAST model: {e}")
        model_path = _get_east_model_path()
        raise RuntimeError(
            _east_manual_download_message(model_path, "EAST model loading failed")
        ) from e


def _get_yolo11x_model_path() -> Path:
    """Return the persistent cache path for the YOLO11x model."""
    model_dir = Path.home() / ".untextre" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir / "yolo11x-train28-best.pt"


def _yolo11x_manual_download_message(model_path: Path, reason: str) -> str:
    return (
        f"{reason}. Download the YOLO11x model manually from {YOLO11X_MODEL_URL} "
        f"and save it as {model_path}. See {YOLO11X_MODEL_DOCS} for detector model sources."
    )


def _validate_yolo11x_model_file(model_path: Path) -> None:
    if model_path.stat().st_size < YOLO11X_MODEL_MIN_BYTES:
        raise RuntimeError(
            _yolo11x_manual_download_message(
                model_path,
                f"YOLO11x model file is too small: {model_path}",
            )
        )


def _download_yolo11x_model(
    model_path: Path,
    *,
    urlopen=urllib.request.urlopen,
) -> None:
    """Download the YOLO11x model atomically and reject truncated/error responses."""
    tmp_path = Path(f"{model_path}.tmp")
    try:
        with urlopen(YOLO11X_MODEL_URL, timeout=YOLO11X_DOWNLOAD_TIMEOUT_SECONDS) as response:
            with tmp_path.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)

        _validate_yolo11x_model_file(tmp_path)
        tmp_path.replace(model_path)
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        if isinstance(exc, RuntimeError):
            raise RuntimeError(
                _yolo11x_manual_download_message(model_path, "YOLO11x model download failed")
            ) from exc
        raise RuntimeError(
            _yolo11x_manual_download_message(model_path, "YOLO11x model download failed")
        ) from exc


def _load_yolo11x_model() -> object:
    """Load the YOLO11x watermark detection model.

    This function attempts to download the YOLO11x model if it doesn't exist
    locally. The model is an Ultralytics YOLO checkpoint fine-tuned for
    watermark detection.

    Returns:
        ultralytics.YOLO instance ready for prediction

    Raises:
        RuntimeError: If the model file is missing, truncated, or fails to load
    """
    try:
        model_path = _get_yolo11x_model_path()

        if not model_path.exists():
            logger.info("Downloading YOLO11x watermark detection model...")
            _download_yolo11x_model(model_path)
            logger.info(f"YOLO11x model downloaded to: {model_path}")

        _validate_yolo11x_model_file(model_path)

        from ultralytics import YOLO

        model = YOLO(str(model_path))
        logger.debug(f"YOLO11x model loaded from: {model_path}")
        return model

    except RuntimeError:
        raise
    except Exception as e:
        logger.error(f"Failed to load YOLO11x model: {e}")
        model_path = _get_yolo11x_model_path()
        raise RuntimeError(
            _yolo11x_manual_download_message(model_path, "YOLO11x model loading failed")
        ) from e


def _detect_with_east(image: ImageArray, net: Any,
                     min_confidence: float = 0.3, 
                     nms_threshold: float = 0.4,
                     width: int = 640, 
                     height: int = 640) -> List[Detection]:
    """Detect text regions using EAST text detector with OpenCV DNN.
    
    EAST (Efficient and Accurate Scene Text) is a deep learning model specifically
    designed for text detection in natural scenes. It can handle text at various
    orientations and scales.
    
    Args:
        image: Input image as H×W×3 BGR uint8 numpy array
        net: Pre-loaded EAST network
        min_confidence: Minimum confidence threshold for detections (0.0-1.0)
        nms_threshold: Non-maximum suppression threshold (0.0-1.0) 
        width: Network input width (must be multiple of 32)
        height: Network input height (must be multiple of 32)
        
    Returns:
        List of detection dictionaries in the shared geometry/confidence format
        
    Raises:
        RuntimeError: If detection fails
    """
    try:
        # Store original dimensions for coordinate scaling
        (orig_h, orig_w) = image.shape[:2]
        
        # Calculate scaling ratios
        r_w = orig_w / float(width)
        r_h = orig_h / float(height)
        
        # Resize image for EAST network (must be multiple of 32)
        resized = cv2.resize(image, (width, height))
        
        # Prepare blob for network input
        # EAST expects RGB input with specific mean subtraction
        blob = cv2.dnn.blobFromImage(resized, 1.0, (width, height),
                                    (123.68, 116.78, 103.94), swapRB=True, crop=False)
        
        # Set network input and run forward pass
        net.setInput(blob)
        
        # EAST has two output layers:
        # 1. Probability scores (whether region contains text)
        # 2. Geometry predictions (bounding box coordinates)
        layer_names = [
            "feature_fusion/Conv_7/Sigmoid",    # Scores
            "feature_fusion/concat_3"           # Geometry
        ]
        (scores, geometry) = net.forward(layer_names)
        
        # Decode predictions into bounding boxes and confidences
        rectangles, confidences = _decode_east_predictions(scores, geometry, min_confidence)
        
        if not rectangles:
            logger.debug("No text regions found by EAST")
            return []
        
        # Apply non-maximum suppression to remove overlapping detections.
        boxes = [[x, y, w, h] for (x, y, w, h) in rectangles]
        
        # Apply NMS; fall back to local implementation on any cv2.dnn binding failure.
        try:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, nms_threshold)
            if len(indices) > 0:
                # NMSBoxes returns a 2-D array in OpenCV <4.5, 1-D in >=4.5.
                if isinstance(indices, np.ndarray) and indices.ndim > 1:
                    indices = indices.flatten()
                selected_indices = indices
            else:
                selected_indices = []
        except Exception:
            logger.warning(
                "OpenCV NMS failed; using local NMS fallback "
                f"(score_threshold={min_confidence:.3f}, nms_threshold={nms_threshold:.3f})"
            )
            selected_indices = _non_max_suppression_indices(
                boxes, confidences, min_confidence, nms_threshold
            )
        
        # Convert selected rectangles to our detection format
        detections = []
        for i in selected_indices:
            (x, y, w, h) = rectangles[i]
            confidence = confidences[i]
            
            # Scale coordinates back to original image size
            x = int(x * r_w)
            y = int(y * r_h) 
            w = int(w * r_w)
            h = int(h * r_h)
            
            # Create 4-point polygon from rectangle (for consistency with other detectors)
            geometry = np.array([
                [x, y],         # Top-left
                [x + w, y],     # Top-right  
                [x + w, y + h], # Bottom-right
                [x, y + h]      # Bottom-left
            ], dtype=np.float32)
            
            detection = {
                'geometry': geometry,
                'confidence': float(confidence)
            }
            detections.append(detection)
        
        logger.debug(f"EAST found {len(detections)} text regions after NMS")
        return detections
        
    except Exception as e:
        logger.error(f"EAST detection failed: {e}")
        raise RuntimeError("EAST detection failed") from e


def _non_max_suppression_indices(
    boxes: List[List[int]],
    confidences: List[float],
    min_confidence: float,
    nms_threshold: float,
) -> List[int]:
    """Return kept indices for axis-aligned (x, y, w, h) boxes."""
    candidate_indices = [
        idx for idx, confidence in enumerate(confidences)
        if confidence >= min_confidence
    ]
    candidate_indices.sort(key=lambda idx: confidences[idx], reverse=True)

    selected: List[int] = []
    while candidate_indices:
        current = candidate_indices.pop(0)
        selected.append(current)
        candidate_indices = [
            idx for idx in candidate_indices
            if _calculate_xywh_iou(boxes[current], boxes[idx]) <= nms_threshold
        ]

    return selected


def _calculate_xywh_iou(box_a: List[int], box_b: List[int]) -> float:
    """Calculate IoU for axis-aligned boxes stored as (x, y, w, h)."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    a_x2 = ax + aw
    a_y2 = ay + ah
    b_x2 = bx + bw
    b_y2 = by + bh

    inter_w = max(0, min(a_x2, b_x2) - max(ax, bx))
    inter_h = max(0, min(a_y2, b_y2) - max(ay, by))
    intersection = inter_w * inter_h
    if intersection == 0:
        return 0.0

    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _decode_east_predictions(scores: np.ndarray, geometry: np.ndarray, 
                           min_confidence: float) -> Tuple[List[Tuple[int, int, int, int]], List[float]]:
    """Decode EAST network predictions into bounding boxes and confidences.
    
    Args:
        scores: Network output scores array (confidence predictions)
        geometry: Network output geometry array (bounding box predictions)  
        min_confidence: Minimum confidence threshold
        
    Returns:
        Tuple of (rectangles, confidences) where rectangles are (x,y,w,h) tuples
    """
    # Extract dimensions from score volume
    (num_rows, num_cols) = scores.shape[2:4]
    rectangles = []
    confidences = []
    
    # Loop over each row and column of the score map
    for y in range(0, num_rows):
        # Extract scores and geometry data for current row
        scores_data = scores[0, 0, y]
        x_data_0 = geometry[0, 0, y]  # Distance to top edge
        x_data_1 = geometry[0, 1, y]  # Distance to right edge
        x_data_2 = geometry[0, 2, y]  # Distance to bottom edge
        x_data_3 = geometry[0, 3, y]  # Distance to left edge
        angles_data = geometry[0, 4, y]  # Rotation angles
        
        for x in range(0, num_cols):
            # Skip if confidence is too low
            if scores_data[x] < min_confidence:
                continue
            
            # Calculate offset - EAST output is 4x smaller than input
            (offset_x, offset_y) = (x * 4.0, y * 4.0)
            
            # Extract rotation angle and calculate sin/cos
            angle = angles_data[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            
            # Calculate width and height of bounding box
            h = x_data_0[x] + x_data_2[x]
            w = x_data_1[x] + x_data_3[x]
            
            # Rotation-aware bbox: angle from EAST geometry channel 4 orients the box.
            end_x = int(offset_x + (cos * x_data_1[x]) + (sin * x_data_2[x]))
            end_y = int(offset_y - (sin * x_data_1[x]) + (cos * x_data_2[x]))
            start_x = int(end_x - w)
            start_y = int(end_y - h)
            
            # Store rectangle as (x, y, width, height)
            rectangles.append((start_x, start_y, int(w), int(h)))
            confidences.append(float(scores_data[x]))
    
    return rectangles, confidences

def _geometry_to_bbox(geometry: np.ndarray) -> BBox:
    """Convert geometry points to bounding box.
    
    Args:
        geometry: Array of polygon points
        
    Returns:
        Bounding box as (x, y, width, height)
    """
    x_coords = geometry[:, 0]
    y_coords = geometry[:, 1]
    
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    
    return (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))

