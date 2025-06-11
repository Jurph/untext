"""Text detection module using DocTR.

This module provides functionality to detect text regions in images using the DocTR library.
It includes preprocessing integration and returns bounding boxes for detected text regions.

WARNING: DO NOT ASSUME DocTR's OUTPUT FORMAT! 
The format is documented in their codebase at docs/doctr/doctr/io/elements.py.
"""


import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from doctr.models import detection
from doctr.models.detection.predictor import DetectionPredictor
from doctr.models.preprocessor.pytorch import PreProcessor
import easyocr

from .utils import ImageArray, BBox, setup_logger
from .preprocessor import preprocess_image

import warnings
warnings.filterwarnings("ignore", message="defusedxml.cElementTree is deprecated")

logger = setup_logger(__name__)

# Type alias for detection results
Detection = Dict[str, np.ndarray]  # {'geometry': points, 'confidence': score}

# Module-level model instances for persistent loading
_doctr_detector: Optional['TextDetector'] = None
_easyocr_reader: Optional[easyocr.Reader] = None
_east_net: Optional[cv2.dnn_Net] = None

def initialize_models(detector_methods: List[str]) -> None:
    """Initialize detection models once for reuse across multiple images.
    
    Args:
        detector_methods: List of methods to initialize ("doctr", "easyocr", "east", or combinations)
    """
    global _doctr_detector, _easyocr_reader, _east_net
    
    if "doctr" in detector_methods and _doctr_detector is None:
        logger.info("Initializing DocTR model...")
        _doctr_detector = TextDetector()
        logger.info("DocTR model ready")
    
    if "easyocr" in detector_methods and _easyocr_reader is None:
        logger.info("Initializing EasyOCR model...")
        _easyocr_reader = easyocr.Reader(['en'], verbose=False)
        logger.info("EasyOCR model ready")
    
    if "east" in detector_methods and _east_net is None:
        logger.info("Initializing EAST text detector...")
        _east_net = _load_east_model()
        logger.info("EAST model ready")

def detect_text_regions(image: ImageArray, method: str = "doctr") -> List[BBox]:
    """Detect text regions in an image and return bounding boxes.
    
    This is the main entry point for text detection. It applies preprocessing,
    runs the specified detection method, and returns bounding boxes for detected text.
    
    Args:
        image: Input image as H×W×3 BGR uint8 numpy array
        method: Detection method to use ("doctr", "easyocr", or "east")
        
    Returns:
        List of bounding boxes as (x, y, width, height) tuples
        
    Raises:
        ValueError: If image is invalid or method is unsupported
        RuntimeError: If detection fails
    """
    global _doctr_detector, _easyocr_reader, _east_net
    
    if method == "doctr":
        if _doctr_detector is None:
            initialize_models(["doctr"])
        detections = _doctr_detector.detect(image)
    elif method == "easyocr":
        if _easyocr_reader is None:
            initialize_models(["easyocr"])
        detections = _detect_with_easyocr(image, _easyocr_reader)
    elif method == "east":
        if _east_net is None:
            initialize_models(["east"])
        detections = _detect_with_east(image, _east_net)
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

def _detect_with_easyocr(image: ImageArray, reader: easyocr.Reader) -> List[Detection]:
    """Detect text regions using EasyOCR with pre-initialized reader.
    
    Args:
        image: Input image as H×W×3 BGR uint8 numpy array
        reader: Pre-initialized EasyOCR reader instance
        
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
        
        # Convert EasyOCR results to our format
        detections = []
        for bbox, text, confidence in results:
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
                     min_confidence: float = 0.5, 
                     nms_threshold: float = 0.4,
                     width: int = 320, 
                     height: int = 320) -> List[Detection]:
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
        
        # Apply non-maximum suppression to remove overlapping detections
        # Convert rectangles to (x, y, w, h) format for NMS
        boxes = []
        for (x, y, w, h) in rectangles:
            boxes.append([x, y, x + w, y + h])
        
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
        except:
            # Fallback to simple confidence-based selection if NMS fails
            logger.warning("OpenCV NMS failed, using confidence-based selection")
            conf_threshold = min_confidence + 0.1  # Slightly higher threshold
            selected_indices = [i for i, conf in enumerate(confidences) if conf >= conf_threshold]
        
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

def get_largest_text_region(image: ImageArray, method: str = "doctr") -> BBox:
    """Get the largest detected text region, merged with boxes at the same height.
    
    This function finds the largest bounding box, then merges it with any other
    bounding boxes whose vertical span includes the centroid Y-coordinate of
    the largest box. This groups text that appears on the same line.
    
    Args:
        image: Input image as H×W×3 BGR uint8 numpy array
        
    Returns:
        Merged bounding box as (x, y, width, height)
        
    Raises:
        ValueError: If no text is detected
    """
    bboxes = detect_text_regions(image, method)
    
    if not bboxes:
        # TODO: Add fallover to corner-based region selection
        raise ValueError("No text regions detected")
    
    # Find largest bbox by area
    largest_bbox = max(bboxes, key=lambda bbox: bbox[2] * bbox[3])
    area = largest_bbox[2] * largest_bbox[3]
    logger.info(f"Largest text region: ({largest_bbox[0]},{largest_bbox[1]}) "
                f"size {largest_bbox[2]}×{largest_bbox[3]} (area: {area} pixels)")
    
    # Calculate centroid Y-coordinate of the largest box
    largest_x, largest_y, largest_w, largest_h = largest_bbox
    centroid_y = largest_y + largest_h // 2
    logger.info(f"Largest box centroid Y-coordinate: {centroid_y}")
    
    # Find boxes whose vertical span includes the centroid Y-coordinate
    boxes_to_merge = [largest_bbox]
    for bbox in bboxes:
        if bbox == largest_bbox:
            continue
        
        x, y, w, h = bbox
        # Check if centroid_y falls within this box's vertical span
        if y <= centroid_y <= y + h:
            boxes_to_merge.append(bbox)
            logger.info(f"Merging box ({x},{y}) size {w}×{h} - overlaps at height {centroid_y}")
    
    # Merge all selected boxes into one
    merged_bbox = _merge_bboxes(boxes_to_merge)
    merged_area = merged_bbox[2] * merged_bbox[3]
    
    logger.info(f"Merged {len(boxes_to_merge)} boxes into final region: "
                f"({merged_bbox[0]},{merged_bbox[1]}) size {merged_bbox[2]}×{merged_bbox[3]} "
                f"(area: {merged_area} pixels)")
    
    return merged_bbox

def _merge_bboxes(bboxes: List[BBox]) -> BBox:
    """Merge multiple bounding boxes into one encompassing box.
    
    Args:
        bboxes: List of bounding boxes as (x, y, width, height) tuples
        
    Returns:
        Merged bounding box as (x, y, width, height)
    """
    if not bboxes:
        raise ValueError("Cannot merge empty list of bounding boxes")
    
    if len(bboxes) == 1:
        return bboxes[0]
    
    # Convert to (x1, y1, x2, y2) format for easier calculation
    x1_coords = []
    y1_coords = []
    x2_coords = []
    y2_coords = []
    
    for x, y, w, h in bboxes:
        x1_coords.append(x)
        y1_coords.append(y)
        x2_coords.append(x + w)
        y2_coords.append(y + h)
    
    # Find the encompassing bounds
    merged_x1 = min(x1_coords)
    merged_y1 = min(y1_coords)
    merged_x2 = max(x2_coords)
    merged_y2 = max(y2_coords)
    
    # Convert back to (x, y, width, height) format
    merged_w = merged_x2 - merged_x1
    merged_h = merged_y2 - merged_y1
    
    return (merged_x1, merged_y1, merged_w, merged_h)

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
        confidence_threshold: float = 0.3, 
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
            # Apply preprocessing
            logger.debug("Applying optimized preprocessing before text detection")
            processed_image = preprocess_image(image)
            if processed_image is None:
                logger.warning("Preprocessing failed, using original image")
                processed_image = image
            else:
                # Convert back to BGR if needed (preprocessor returns RGB)
                if processed_image.shape[2] == 3:
                    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
            
            # Run detection
            raw = self.predictor([processed_image])
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