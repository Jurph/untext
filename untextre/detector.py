"""Text detection module using DocTR.

This module provides functionality to detect text regions in images using the DocTR library.
It includes preprocessing integration and returns bounding boxes for detected text regions.

WARNING: DO NOT ASSUME DocTR's OUTPUT FORMAT! 
The format is documented in their codebase at docs/doctr/doctr/io/elements.py.
"""


import cv2
import gc
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from doctr.models import detection
from doctr.models.detection.predictor import DetectionPredictor
from doctr.models.preprocessor.pytorch import PreProcessor

from .utils import ImageArray, BBox, setup_logger, CLI_DEFAULT_CONFIDENCE

import warnings
warnings.filterwarnings("ignore", message="defusedxml.cElementTree is deprecated")

logger = setup_logger(__name__)

# Type alias for detection results
Detection = Dict[str, np.ndarray]  # {'geometry': points, 'confidence': score}

# Module-level model instances for persistent loading
_doctr_detector: Optional['TextDetector'] = None
_easyocr_reader: Optional[object] = None
_east_net: Optional[cv2.dnn_Net] = None


def get_doctr_detector(
    confidence_threshold: float = CLI_DEFAULT_CONFIDENCE,
    min_text_size: int = 10,
) -> 'TextDetector':
    """Return the shared DocTR detector, upgrading to a more permissive config if needed."""
    global _doctr_detector

    needs_init = _doctr_detector is None
    if _doctr_detector is not None:
        needs_init = (
            _doctr_detector.confidence_threshold > confidence_threshold
            or _doctr_detector.min_text_size > min_text_size
        )

    if needs_init:
        logger.info("Initializing DocTR model...")
        _doctr_detector = TextDetector(
            confidence_threshold=confidence_threshold,
            min_text_size=min_text_size,
        )
        logger.info("DocTR model ready")

    return _doctr_detector


def get_easyocr_reader() -> object:
    """Return the shared EasyOCR reader."""
    global _easyocr_reader

    if _easyocr_reader is None:
        logger.info("Initializing EasyOCR model...")
        import easyocr
        _easyocr_reader = easyocr.Reader(['en'], verbose=False)
        logger.info("EasyOCR model ready")

    return _easyocr_reader


def get_east_net() -> cv2.dnn_Net:
    """Return the shared EAST text detector network."""
    global _east_net

    if _east_net is None:
        logger.info("Initializing EAST text detector...")
        _east_net = _load_east_model()
        logger.info("EAST model ready")

    return _east_net


def initialize_models(
    detector_methods: List[str],
    doctr_confidence_threshold: float = CLI_DEFAULT_CONFIDENCE,
    doctr_min_text_size: int = 10,
) -> None:
    """Initialize detection models once for reuse across multiple images.
    
    Args:
        detector_methods: List of methods to initialize ("doctr", "easyocr", "east", or combinations)
    """
    if "doctr" in detector_methods:
        get_doctr_detector(
            confidence_threshold=doctr_confidence_threshold,
            min_text_size=doctr_min_text_size,
        )

    if "easyocr" in detector_methods:
        get_easyocr_reader()

    if "east" in detector_methods:
        get_east_net()


def cleanup_vram() -> None:
    """Force cleanup of GPU memory after detection operations.
    
    Call this periodically during long-running batch operations to prevent
    VRAM accumulation from intermediate tensors.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def detect_text_regions(image: ImageArray, method: str = "doctr", confidence_threshold: float = 0.3) -> List[BBox]:
    """Detect text regions in an image and return bounding boxes.
    
    This is the main entry point for text detection. It applies preprocessing,
    runs the specified detection method, and returns bounding boxes for detected text.
    
    Args:
        image: Input image as H×W×3 BGR uint8 numpy array
        method: Detection method to use ("doctr", "easyocr", or "east")
        confidence_threshold: Minimum confidence threshold for detections (0.0-1.0, default: 0.3)
        
    Returns:
        List of bounding boxes as (x, y, width, height) tuples
        
    Raises:
        ValueError: If image is invalid or method is unsupported
        RuntimeError: If detection fails
    """
    if method == "doctr":
        detector = get_doctr_detector()
        # Reuse global detector - confidence threshold is applied as post-filter
        detections = detector.detect(image)
        # Filter by confidence threshold
        detections = [d for d in detections if d.get('confidence', 0) >= confidence_threshold]
    elif method == "easyocr":
        reader = get_easyocr_reader()
        detections = _detect_with_easyocr(image, reader, confidence_threshold=confidence_threshold)
    elif method == "east":
        net = get_east_net()
        detections = _detect_with_east(image, net, min_confidence=confidence_threshold)
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
        bbox_dims = [f"{bbox[2]}×{bbox[3]}" for bbox in bboxes]
        logger.info(f"Bounding box dimensions: {', '.join(bbox_dims)}")
    else:
        logger.info("No text regions detected")
    
    return bboxes

def _detect_with_easyocr(image: ImageArray, reader: object, confidence_threshold: float = CLI_DEFAULT_CONFIDENCE) -> List[Detection]:
    """Detect text regions using EasyOCR with pre-initialized reader.
    
    Args:
        image: Input image as H×W×3 BGR uint8 numpy array
        reader: Pre-initialized EasyOCR reader instance
        confidence_threshold: Minimum confidence threshold for detections (0.0-1.0, default: 0.3)
        
    Returns:
        List of detection dictionaries in the same format as DocTR
        
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

def _load_east_model() -> cv2.dnn_Net:
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
        import urllib.request
        from pathlib import Path
        
        # Define model path in user's home directory for persistence
        model_dir = Path.home() / ".untextre" / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "frozen_east_text_detection.pb"
        
        # Download model if it doesn't exist
        if not model_path.exists():
            logger.info("Downloading EAST text detection model...")
            model_url = "https://github.com/oyyd/frozen_east_text_detection.pb/raw/master/frozen_east_text_detection.pb"
            urllib.request.urlretrieve(model_url, model_path)
            logger.info(f"EAST model downloaded to: {model_path}")
        
        # Load the network
        net = cv2.dnn.readNet(str(model_path))
        logger.debug(f"EAST model loaded from: {model_path}")
        return net
        
    except Exception as e:
        logger.error(f"Failed to load EAST model: {e}")
        raise RuntimeError("EAST model loading failed") from e

def _detect_with_east(image: ImageArray, net: cv2.dnn_Net, 
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
        List of detection dictionaries in the same format as DocTR
        
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
        
        # Apply OpenCV's NMS if available, otherwise use simple overlap removal
        try:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, nms_threshold)
            if len(indices) > 0:
                # Flatten indices array if needed (OpenCV version differences)
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
            
            # Calculate bounding box coordinates
            # Note: This is a simplified version - full EAST can handle rotation
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

class TextDetector:
    """Text detector using DocTR with optimized preprocessing."""
    
    def __init__(
        self, 
        confidence_threshold: float = CLI_DEFAULT_CONFIDENCE, 
        min_text_size: int = 10
    ) -> None:
        """Initialize the TextDetector with a pre-trained model.
        
        Args:
            confidence_threshold: Minimum confidence score for detections (0-1)
            min_text_size: Minimum size of text regions to detect
        """
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if min_text_size <= 0:
            raise ValueError("min_text_size must be positive")
            
        self.confidence_threshold = confidence_threshold
        self.min_text_size = min_text_size
        
        try:
            # Initialize detection model
            self.model = detection.db_resnet50(pretrained=True)
            self.pre_processor = PreProcessor(
                output_size=(1024, 1024),
                batch_size=1,
                mean=(0.798, 0.785, 0.772),
                std=(0.264, 0.2749, 0.287)
            )
            self.predictor = DetectionPredictor(
                pre_processor=self.pre_processor,
                model=self.model
            )
            logger.info("Initialized TextDetector with DB ResNet50 model")
        except Exception as e:
            logger.error(f"Failed to initialize TextDetector: {e}")
            raise RuntimeError("Failed to initialize TextDetector") from e
    
    @torch.no_grad()
    def detect(self, image: ImageArray) -> List[Detection]:
        """Detect text regions in an image.
        
        Args:
            image: Input image as H×W×3 BGR uint8 numpy array
            
        Returns:
            List of detection dictionaries, each containing:
                - 'geometry': numpy array of polygon points
                - 'confidence': confidence score
        
        Raises:
            ValueError: If image is invalid
            RuntimeError: If detection fails
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be a numpy array")
        
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Image must be H×W×3 BGR")
        
        try:
            # Use the image as provided (preprocessing now handled uniformly by caller)
            logger.debug("Running DocTR detection on preprocessed image")
            
            # Run detection (torch.no_grad context prevents gradient accumulation)
            raw = self.predictor([image])
            if not isinstance(raw, list) or len(raw) == 0:
                logger.warning("No text detected")
                return []
            
            # Convert DocTR output to our format
            detections = self._parse_doctr_output(raw[0], image.shape[:2])
            
            # Filter detections
            detections = self._filter_detections(detections, image.shape[:2])
            
            logger.debug(f"Detected {len(detections)} text regions")
            return detections
            
        except Exception as e:
            logger.error(f"Text detection failed: {e}")
            raise RuntimeError("Text detection failed") from e

    def _parse_doctr_output(self, page: Dict, image_shape: Tuple[int, int]) -> List[Detection]:
        """Convert DocTR output to our detection format.
        
        Args:
            page: Dictionary from DocTR containing predictions
            image_shape: Shape of the image (height, width)
            
        Returns:
            List of detection dictionaries in our format
        """
        if not isinstance(page, dict):
            logger.warning("Invalid page format from DocTR")
            return []
            
        detections = []
        # DocTR returns predictions as numpy arrays with shape (*, 5) or (*, 6)
        # where each row is [x1, y1, x2, y2, confidence, ...]
        if 'words' in page and isinstance(page['words'], np.ndarray):
            for pred in page['words']:
                if len(pred) >= 5:  # Ensure we have at least x1,y1,x2,y2,conf
                    x1, y1, x2, y2, conf = pred[:5]
                    
                    # Skip low confidence detections
                    if conf < self.confidence_threshold:
                        continue
                    
                    # Convert normalized coordinates to pixel coordinates
                    h, w = image_shape
                    points = np.array([
                        [x1 * w, y1 * h],
                        [x2 * w, y1 * h],
                        [x2 * w, y2 * h],
                        [x1 * w, y2 * h]
                    ], dtype=np.float32)
                    
                    detections.append({
                        'geometry': points,
                        'confidence': float(conf)
                    })
                
        return detections

    def _filter_detections(self, detections: List[Dict], image_shape: Tuple[int, int]) -> List[Dict]:
        """Filter detections based on size and position.
        
        Args:
            detections: List of detection dictionaries
            image_shape: Shape of the image (height, width)
            
        Returns:
            Filtered list of detections
        """
        if not isinstance(image_shape, tuple) or len(image_shape) != 2:
            raise ValueError("image_shape must be a tuple of two integers")
            
        filtered = []
        for det in detections:
            if not isinstance(det, dict) or 'geometry' not in det:
                continue
                
            geometry = det['geometry']
            if not isinstance(geometry, np.ndarray) or geometry.shape[0] < 3:
                continue
                
            # Calculate width and height
            x_coords = geometry[:, 0]
            y_coords = geometry[:, 1]
            width = np.max(x_coords) - np.min(x_coords)
            height = np.max(y_coords) - np.min(y_coords)
            
            # Check size
            if width < self.min_text_size or height < self.min_text_size:
                continue
                
            # TODO: Add smarter position-based filtering (corner preference)
            
            filtered.append(det)
            
        return filtered 
