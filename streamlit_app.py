#!/usr/bin/env python3
"""Streamlit web interface for untextre text watermark removal.

This provides a drag-and-drop web interface for removing text watermarks
from images using the untextre pipeline. All models are loaded at startup
for fast processing.

Usage:
    streamlit run streamlit_app.py
"""

# Workaround for PyTorch/Streamlit compatibility issue
import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

import streamlit as st
import cv2
import tempfile
import time
from pathlib import Path
from PIL import Image
import io
from streamlit_drawable_canvas import st_canvas
from streamlit_js_eval import streamlit_js_eval

# Import our untextre modules
from untextre.utils import (
    load_image,
    calculate_bbox_superset,
    WEB_DEFAULT_CONFIDENCE,
)
from untextre.preprocessor import preprocess_image
from untextre.orb_matcher import (
    load_watermark_templates,
    try_watermark_cascade,
)
from untextre.pipeline import (
    initialize_consensus_models,
    process_single_image,
)
import numpy as np
from untextre.consensus import find_consensus_boxes
from untextre.inpaint import initialize_lama_model, get_lama_status, reset_lama_model
import hashlib
from PIL import ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# UI / layout constants  (single source of truth for magic numbers)
# ---------------------------------------------------------------------------

# Detection mode radio options — used in st.radio and equality checks.
# Changing the label here automatically updates the widget and all comparisons.
MODE_AUTO_DETECT = "🤖 Auto-detect"
MODE_GRABCUT_EXPAND = "🔬 Color-guided expand"
MODE_DRAW_MANUALLY = "✏️ Draw manually"

# Color input method radio options
COLOR_INPUT_PICKER = "Color picker"
COLOR_INPUT_HEX = "Hex code"

# Layout
COLUMN_WIDTH_RATIO = 0.42          # fraction of window.innerWidth for each column
RESULT_COLUMN_SPACER_PX = 258      # vertical spacer so result image aligns with source

# File handling
MAX_DOWNLOAD_BYTES = int(14.5 * 1024 * 1024)   # Streamlit download button limit
JPEG_QUALITY_CASCADE = (95, 90, 85, 80, 75)     # try highest first, shrink to fit

# Detection defaults
BBOX_PAD_FRACTION = 0.1            # pad consensus boxes by this fraction of their size
CONSENSUS_OVERLAP_THRESHOLD = 0.1  # IoU threshold for merging consensus regions

# Inpainting defaults
DEFAULT_GRANULARITY = 4            # K-means cluster count for color analysis
DEFAULT_COLOR_SENSITIVITY = 16     # ± pixel-value tolerance around target color

# Watermark template library
WATERMARKS_DIR = Path(__file__).resolve().parent / "watermarks"

# ---------------------------------------------------------------------------
# Pure helpers (no Streamlit dependency — testable in isolation)
# ---------------------------------------------------------------------------


def make_image_state_id(image_name, image_bytes):
    """Return a stable per-image id for widget/session-state keys."""
    if image_bytes is None:
        return None
    image_name = image_name or "image"
    digest = hashlib.md5(image_bytes).hexdigest()[:12]
    return f"{Path(image_name).stem}_{digest}"


def resolve_active_image(ingested_bytes, ingested_name, uploaded_file):
    """Resolve the active image source, with ingested results taking priority."""
    if ingested_bytes is not None:
        return ingested_bytes, ingested_name
    if uploaded_file is not None:
        return uploaded_file.getvalue(), uploaded_file.name
    return None, None


def bbox_to_fabric_rect(bbox, scale_x, scale_y):
    """Convert an image-coordinate bbox to a Fabric.js ``initial_drawing`` dict.

    Args:
        bbox: ``(x, y, w, h)`` in original image pixel coordinates.
        scale_x: ``image_width / canvas_width`` (>1 when image is larger).
        scale_y: ``image_height / canvas_height``.

    Returns:
        A dict suitable for ``st_canvas(initial_drawing=...)``.
    """
    x, y, w, h = bbox
    # Version must match the Fabric.js version bundled by streamlit-drawable-canvas.
    return {
        "version": "4.4.0",
        "objects": [
            {
                "type": "rect",
                "version": "4.4.0",
                "originX": "left",
                "originY": "top",
                "left": x / scale_x,
                "top": y / scale_y,
                "width": w / scale_x,
                "height": h / scale_y,
                "fill": "rgba(0, 0, 0, 0)",
                "stroke": "rgb(0, 255, 0)",
                "strokeWidth": 3,
                "strokeDashArray": None,
                "strokeLineCap": "butt",
                "strokeDashOffset": 0,
                "strokeLineJoin": "miter",
                "strokeUniform": True,
                "strokeMiterLimit": 10,
                "scaleX": 1,
                "scaleY": 1,
                "angle": 0,
                "flipX": False,
                "flipY": False,
                "opacity": 1,
                "shadow": None,
                "visible": True,
                "backgroundColor": "",
                "fillRule": "nonzero",
                "paintFirst": "fill",
                "globalCompositeOperation": "source-over",
                "skewX": 0,
                "skewY": 0,
                "rx": 0,
                "ry": 0,
                "lockUniScaling": False,
            }
        ],
    }


def fabric_rect_to_bbox(rect, scale_x, scale_y, image_width, image_height):
    """Convert a Fabric.js rect object to an image-coordinate bbox.

    Handles Fabric.js ``scaleX/scaleY`` transforms, ensures proper corner
    ordering, scales back to original image coordinates, and clamps to the
    image bounds.

    Args:
        rect: A single Fabric.js object dict (``type == "rect"``).
        scale_x: ``image_width / canvas_width``.
        scale_y: ``image_height / canvas_height``.
        image_width: Original image width in pixels.
        image_height: Original image height in pixels.

    Returns:
        ``(x, y, w, h)`` clamped to ``[0, image_width) × [0, image_height)``,
        with ``w >= 1`` and ``h >= 1``.  Returns ``None`` if *rect* is not a
        valid rectangle.
    """
    if not rect or rect.get("type") != "rect":
        return None

    # Fabric.js stores base dimensions; actual size = base * scale
    canvas_x = rect.get("left", 0)
    canvas_y = rect.get("top", 0)
    canvas_w = rect.get("width", 0) * rect.get("scaleX", 1.0)
    canvas_h = rect.get("height", 0) * rect.get("scaleY", 1.0)

    # Corner ordering (handles negative-drag rectangles)
    x1c, x2c = sorted([canvas_x, canvas_x + canvas_w])
    y1c, y2c = sorted([canvas_y, canvas_y + canvas_h])

    # Scale to image coordinates
    x1 = int(round(x1c * scale_x))
    y1 = int(round(y1c * scale_y))
    x2 = int(round(x2c * scale_x))
    y2 = int(round(y2c * scale_y))

    # Clamp to image bounds
    x1 = max(0, min(x1, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))
    x2 = max(x1 + 1, min(x2, image_width))
    y2 = max(y1 + 1, min(y2, image_height))

    return (x1, y1, x2 - x1, y2 - y1)


def encode_result_for_download(result_pil, original_filename):
    """Encode a PIL image for the Streamlit download button.

    Picks the output format based on the original file's extension, attempts
    JPEG quality cascading to stay under ``MAX_DOWNLOAD_BYTES`` (best-effort;
    very large images may still exceed the limit), and converts BMP to PNG.

    Args:
        result_pil: A ``PIL.Image`` of the processed result.
        original_filename: Original filename (used only to determine format).

    Returns:
        ``(buf_bytes, download_name, mime_type)`` where *buf_bytes* is the
        encoded image as ``bytes``.
    """
    suffix = Path(original_filename).suffix.lower()
    stem = Path(original_filename).stem

    if suffix in (".jpg", ".jpeg"):
        mime_type = "image/jpeg"
        download_name = f"{stem}_clean.jpg"
        for quality in JPEG_QUALITY_CASCADE:
            buf = io.BytesIO()
            result_pil.save(buf, format="JPEG", quality=quality, optimize=True)
            if buf.tell() <= MAX_DOWNLOAD_BYTES:
                break
    elif suffix == ".bmp":
        # BMP is uncompressed and huge — convert to PNG
        buf = io.BytesIO()
        result_pil.save(buf, format="PNG", optimize=True)
        mime_type = "image/png"
        download_name = f"{stem}_clean.png"
    elif suffix in (".tif", ".tiff"):
        buf = io.BytesIO()
        result_pil.save(buf, format="TIFF", compression="tiff_deflate")
        mime_type = "image/tiff"
        download_name = f"{stem}_clean.tiff"
    else:
        # Default to PNG (lossless). WEBP input is intentionally saved as PNG.
        buf = io.BytesIO()
        result_pil.save(buf, format="PNG", optimize=True)
        mime_type = "image/png"
        download_name = f"{stem}_clean.png"

    return buf.getvalue(), download_name, mime_type


# Page configuration
st.set_page_config(
    page_title="UnTextre - Text Watermark Removal",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def initialize_models():
    """Preload detection models and try to initialize LaMa with progress feedback."""
    
    # Create placeholder for progress messages
    progress_placeholder = st.empty()
    
    try:
        progress_placeholder.info("🚀 Starting UnTextre... Initializing AI models (this may take 30-60 seconds on first run)")
        
        # Initialize consensus detection models
        with st.spinner("📥 Loading detection models (EAST, DocTR, EasyOCR)..."):
            initialize_consensus_models(device="cuda")
        
        progress_placeholder.success("✅ Detection models loaded successfully")
        
        # Initialize LaMa inpainting model
        with st.spinner("🎨 Loading LaMa inpainting model..."):
            lama_success = initialize_lama_model(device="cuda")
        
        if lama_success:
            progress_placeholder.success("✅ All models ready! You can now upload images.")
        else:
            progress_placeholder.warning("⚠️ Detection models ready. LaMa failed to load (will auto-retry if needed).")
        
        # Clear progress messages after a moment
        import time
        time.sleep(2)
        progress_placeholder.empty()
        
        return {"lama_initialized": lama_success}
        
    except Exception as e:
        progress_placeholder.error(f"❌ Model initialization failed: {e}")
        return {"lama_initialized": False}


def run_detections_cached(image_bytes, confidence_threshold=WEB_DEFAULT_CONFIDENCE):
    """Run consensus detection and cache results by image hash.
    
    Args:
        image_bytes: Image file bytes
        confidence_threshold: Detection confidence threshold (see utils.py for constant docs)
        
    Returns:
        List of consensus detection dictionaries with bbox, confidence, detectors
    """
    # Create cache key from image content
    image_hash = hashlib.md5(image_bytes).hexdigest()
    cache_key = f"detections_{image_hash}_{confidence_threshold}"
    
    # Check if already computed
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    
    # Load and preprocess image
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = Path(tmp.name)
    
    try:
        image = load_image(tmp_path)
        preprocessed = preprocess_image(image)
        
        if preprocessed is None:
            st.error("❌ Image preprocessing failed")
            return []
        
        # Run all detectors with progress feedback
        from untextre.consensus import detect_with_doctr, detect_with_easyocr, detect_with_east
        
        if len(preprocessed.shape) == 2:
            image_bgr = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR)
        else:
            image_bgr = preprocessed
        
        # Progress updates
        progress = st.empty()
        
        progress.text("🔍 Running EAST detector (1/3)...")
        east_dets = detect_with_east(image_bgr, confidence_threshold)
        
        progress.text(f"🔍 EAST: {len(east_dets)} | Running DocTR (2/3)...")
        doctr_dets = detect_with_doctr(image_bgr, confidence_threshold)
        
        progress.text(f"🔍 EAST: {len(east_dets)} | DocTR: {len(doctr_dets)} | Running EasyOCR (3/3)...")
        easyocr_dets = detect_with_easyocr(image_bgr, confidence_threshold)
        
        progress.text("✅ All detectors complete | Finding consensus regions...")
        
        # Get detailed detections
        detections = {
            'east': east_dets,
            'doctr': doctr_dets,
            'easyocr': easyocr_dets
        }
        
        # Find consensus with full metadata
        consensus_detailed = find_consensus_boxes(detections, overlap_threshold=CONSENSUS_OVERLAP_THRESHOLD)
        
        progress.success(f"✅ Found {len(consensus_detailed)} consensus regions where 2+ detectors agree!")
        time.sleep(1.5)  # Let user see the result
        progress.empty()  # Clear progress messages
        
        # Pad consensus boxes (same as CLI does)
        h, w = image.shape[:2]
        padded_consensus = []
        for cons in consensus_detailed:
            x, y, box_w, box_h = cons['bbox']
            
            # Pad by BBOX_PAD_FRACTION on each side (2 * fraction = total expansion)
            pad_w = int(box_w * BBOX_PAD_FRACTION)
            pad_h = int(box_h * BBOX_PAD_FRACTION)
            padded_x = max(0, x - pad_w)
            padded_y = max(0, y - pad_h)
            padded_w = min(w - padded_x, box_w + 2 * pad_w)
            padded_h = min(h - padded_y, box_h + 2 * pad_h)
            
            cons['bbox'] = (padded_x, padded_y, padded_w, padded_h)
            padded_consensus.append(cons)
        
        # Cache result
        st.session_state[cache_key] = padded_consensus
        return padded_consensus
        
    finally:
        tmp_path.unlink(missing_ok=True)


def draw_detection_overlays(image_pil, detections, selected_indices=None, superset_bbox=None):
    """Draw detection boxes on image with color coding.
    
    Args:
        image_pil: PIL Image
        detections: List of detection dictionaries
        selected_indices: Set or list of selected detection indices (highlighted in green)
        superset_bbox: Optional superset bounding box to draw
        
    Returns:
        PIL Image with detection boxes drawn
    """
    # Make a copy to draw on
    annotated = image_pil.copy()
    draw = ImageDraw.Draw(annotated)
    
    # Color scheme
    COLOR_2WAY = (255, 0, 255)      # Magenta for 2-detector consensus
    COLOR_3WAY = (0, 255, 255)      # Cyan for 3-detector consensus
    COLOR_SELECTED = (0, 255, 0)    # Green for selected
    COLOR_SUPERSET = (255, 165, 0)   # Orange for superset box
    
    if selected_indices is None:
        selected_indices = set()
    elif not isinstance(selected_indices, set):
        selected_indices = set(selected_indices)
    
    # Draw superset box first (if provided) so it appears behind other boxes
    if superset_bbox is not None:
        sx, sy, sw, sh = superset_bbox
        draw.rectangle([sx, sy, sx+sw, sy+sh], outline=COLOR_SUPERSET, width=4)
        # Draw superset label
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except (OSError, IOError):
            font = ImageFont.load_default()
        label = "Superset (All Selected)"
        bbox_text = draw.textbbox((sx, sy-25), label, font=font)
        text_width = bbox_text[2] - bbox_text[0]
        text_height = bbox_text[3] - bbox_text[1]
        draw.rectangle([sx, sy-text_height-30, sx+text_width+4, sy-25], fill=COLOR_SUPERSET)
        draw.text((sx+2, sy-text_height-28), label, fill=(255, 255, 255), font=font)
    
    # Draw individual detection boxes
    for i, det in enumerate(detections):
        x, y, w, h = det['bbox']
        detector_count = det.get('detector_count', len(det.get('detectors', [])))
        
        # Choose color
        if i in selected_indices:
            color = COLOR_SELECTED
            width = 5
        elif detector_count >= 3:
            color = COLOR_3WAY
            width = 3
        else:
            color = COLOR_2WAY
            width = 3
        
        # Draw rectangle
        draw.rectangle([x, y, x+w, y+h], outline=color, width=width)
        
        # Draw label with clearer format
        detectors_str = "+".join(sorted(det.get('detectors', [])))
        conf = det.get('confidence', 0)
        # More descriptive label
        label = f"Region {i+1}: {detectors_str} consensus ({conf:.0%} confidence)"
        
        # Draw label background
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except (OSError, IOError):
            font = ImageFont.load_default()
        
        # Get text size
        bbox_text = draw.textbbox((x, y-20), label, font=font)
        text_width = bbox_text[2] - bbox_text[0]
        text_height = bbox_text[3] - bbox_text[1]
        
        # Draw background rectangle
        draw.rectangle([x, y-text_height-25, x+text_width+4, y-20], fill=color)
        
        # Draw text
        draw.text((x+2, y-text_height-23), label, fill=(255, 255, 255), font=font)
    
    return annotated

def display_lama_status():
    """Display LaMa status indicator in the sidebar."""
    status = get_lama_status()
    
    st.subheader("🤖 LaMa Status")
    
    # Overall status indicator
    if status["available"] and status["initialized"] and status["healthy"]:
        st.success("✅ LaMa Ready")
        st.caption(f"Device: {status['device']}")
    elif status["available"] and status["initialized"] and not status["healthy"]:
        st.warning("⚠️ LaMa Unhealthy")
        st.caption("Model loaded but not responding correctly")
    elif status["available"] and not status["initialized"]:
        st.error("❌ LaMa Not Initialized")
        if status["init_failed"]:
            st.caption("Previous initialization failed")
        else:
            st.caption("Model not loaded")
    elif not status["available"]:
        st.error("❌ LaMa Not Available")
        st.caption("Installation issue - check simple-lama-inpainting")
    
    # Detailed status
    with st.expander("📊 Detailed Status"):
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.metric("Available", "✅ Yes" if status["available"] else "❌ No")
            st.metric("Initialized", "✅ Yes" if status["initialized"] else "❌ No")
        
        with col_b:
            st.metric("Healthy", "✅ Yes" if status["healthy"] else "❌ No")
            st.metric("Device", status["device"] or "Unknown")
    
    # Action buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Restart LaMa", help="Reinitialize LaMa model"):
            with st.spinner("Restarting LaMa..."):
                reset_lama_model()
                success = initialize_lama_model(device="cuda", force_reinit=True)
                if success:
                    st.success("LaMa restarted successfully!")
                else:
                    st.error("Failed to restart LaMa")
                st.rerun()
    
    with col2:
        if st.button("🩺 Health Check", help="Test LaMa responsiveness"):
            with st.spinner("Testing LaMa..."):
                new_status = get_lama_status()
                if new_status["healthy"]:
                    st.success("LaMa is healthy!")
                else:
                    st.error("LaMa health check failed")
                st.rerun()
    
    return status

def process_image_streamlit(
    image_bytes, confidence_threshold, granularity, method, keep_masks,
    target_color=None, color_sensitivity=3, forced_bbox=None,
    use_grabcut=False,
    use_grabcut_expand=False,
):
    """Process an uploaded image via consensus detection and return the result.

    Watermark template matching is handled at the UI layer (before this
    function is called).  This function only runs the consensus detection
    and inpainting path.
    """

    # Create temporary files
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as input_file:
        input_file.write(image_bytes)
        input_path = Path(input_file.name)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        
        try:
            # Check LaMa status before processing if method is lama
            if method == "lama":
                lama_status = get_lama_status()
                if not lama_status["available"]:
                    raise RuntimeError("LaMa inpainter is not available. Please install simple-lama-inpainting.")
                elif not lama_status["initialized"]:
                    st.warning("LaMa not initialized. Attempting to initialize...")
                    if not initialize_lama_model(device="cuda"):
                        raise RuntimeError("Failed to initialize LaMa model.")
                elif not lama_status["healthy"]:
                    st.warning("LaMa appears unhealthy. Attempting restart...")
                    reset_lama_model()
                    if not initialize_lama_model(device="cuda", force_reinit=True):
                        raise RuntimeError("Failed to restart LaMa model.")

            # ── Consensus detection path ──────────────────────────────
            # Convert hex color to BGR tuple if provided
            target_color_bgr = None
            if target_color and target_color.startswith('#') and len(target_color) == 7:
                try:
                    hex_color = target_color[1:]
                    r = int(hex_color[0:2], 16)
                    g = int(hex_color[2:4], 16) 
                    b = int(hex_color[4:6], 16)
                    target_color_bgr = (b, g, r)
                except ValueError:
                    st.warning(f"Invalid hex color format: {target_color}. Ignoring color enhancement.")
                    target_color_bgr = None
            
            # Web UI: user controls granularity directly, no auto-retry or bbox expansion
            timing_data = process_single_image(
                image_path=input_path,
                output_dir=output_dir,
                target_color=target_color_bgr,
                keep_masks=keep_masks,
                method=method,
                maskfile=None,
                confidence_threshold=confidence_threshold,
                granularity=granularity,
                forced_bbox=forced_bbox,
                color_sensitivity=color_sensitivity,
                expand_bboxes=False,
                auto_retry=False,
                use_grabcut=use_grabcut,
                use_grabcut_expand=use_grabcut_expand,
            )
            
            # Load the result
            result_path = output_dir / f"{input_path.stem}_clean{input_path.suffix}"
            if result_path.exists():
                result_image = load_image(result_path)
                result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                
                mask_image = None
                if keep_masks:
                    mask_path = output_dir / f"{input_path.stem}_mask.png"
                    if mask_path.exists():
                        mask_image = load_image(mask_path)
                
                return result_rgb, mask_image, timing_data
            else:
                st.error("Processing failed - no output image generated")
                return None, None, timing_data
                
        except Exception as e:
            error_msg = str(e)
            if "LaMa" in error_msg or "lama" in error_msg.lower():
                st.error(f"LaMa Error: {error_msg}")
                st.info("Try restarting LaMa using the button in the sidebar, or switch to TELEA inpainting.")
            else:
                st.error(f"Error processing image: {error_msg}")
            return None, None, None
        finally:
            input_path.unlink(missing_ok=True)

def main():
    """Main Streamlit application.

    Organized as a single top-to-bottom render pass (idiomatic Streamlit).
    Sections are marked with comment banners for navigation.  Pure
    computation is delegated to helper functions defined above; UI
    rendering stays inline per Streamlit convention.
    """

    # ── Bootstrap: measure layout & load models ──────────────────────────
    # This gives JS time to execute while the rest of the UI loads
    # We use a simple, reliable measurement: half the window width minus padding
    js_measured_width = streamlit_js_eval(
        js_expressions=f'Math.floor(window.innerWidth * {COLUMN_WIDTH_RATIO})',
        key="app_column_width_measurement"
    )
    
    # Block until we have the measurement - this is essential for correct canvas sizing
    if js_measured_width is None:
        st.info("⏳ Initializing display...")
        st.stop()
    
    # Store in session state for use later
    st.session_state["column_width"] = js_measured_width
    
    # Initialize models
    init_results = initialize_models()
    
    # Title and description
    st.title("🎨 UnTextre - Text Watermark Removal")
    st.markdown("""
    Upload an image with text watermarks and watch them disappear! This tool uses advanced AI models 
    to detect and remove text overlays while preserving the underlying image.
    """)

    # Widget state is available before widgets are declared during a rerun. Resolve
    # the active image here so sidebar controls never lag one rerun behind uploads.
    pending_uploaded_file = st.session_state.get("source_image_uploader")
    current_image_bytes, current_image_name = resolve_active_image(
        st.session_state.get("ingested_image_bytes"),
        st.session_state.get("ingested_image_name"),
        pending_uploaded_file,
    )
    current_image_id = make_image_state_id(current_image_name, current_image_bytes)
    st.session_state.current_image_bytes = current_image_bytes
    st.session_state.current_image_name = current_image_name
    st.session_state.current_image_id = current_image_id
    
    # ── Sidebar: processing options ─────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Processing Options")
        
        # Detection Mode Selector
        st.subheader("🎯 Detection Mode")
        
        # Check for mode override (when user draws on canvas in auto mode)
        _all_modes = [MODE_AUTO_DETECT, MODE_GRABCUT_EXPAND, MODE_DRAW_MANUALLY]
        if hasattr(st.session_state, 'detection_mode_override'):
            override_mode = st.session_state.detection_mode_override
            mode_index = _all_modes.index(override_mode) if override_mode in _all_modes else 0
            # Clear the override after using it
            del st.session_state.detection_mode_override
        else:
            mode_index = 0

        detection_mode = st.radio(
            "How to select region:",
            options=_all_modes,
            index=mode_index,
            help=(
                "Auto-detect: Let AI find text regions\n"
                "Color-guided expand: Auto-detect then extend the mask using the "
                "watermark's color cluster and GrabCut (good for partial detections)\n"
                "Draw manually: Specify exact coordinates"
            )
        )
        is_auto_mode = detection_mode != MODE_DRAW_MANUALLY
        use_grabcut_expand = detection_mode == MODE_GRABCUT_EXPAND
        
        st.divider()
        
        # Detection settings
        st.subheader("🔍 Detection")
        
        # Target Color options
        enable_color_enhancement = st.checkbox(
            "Enable target color",
            value=False,
            help="Enhance subtle watermarks by targeting specific colors"
        )
        
        # Color selection options (enabled only when image is loaded and enhancement is enabled)
        image_loaded = st.session_state.get('current_image_bytes') is not None
        color_controls_enabled = enable_color_enhancement and image_loaded
        
        if not image_loaded and enable_color_enhancement:
            st.info("💡 Upload an image to access target color options")
        
        color_input_method = st.radio(
            "Color input method:",
            [COLOR_INPUT_PICKER, COLOR_INPUT_HEX],
            disabled=not color_controls_enabled,
            help="Choose how to specify the target color"
        )
        
        target_color = None
        color_sensitivity = DEFAULT_COLOR_SENSITIVITY
        
        if color_controls_enabled:
            if color_input_method == COLOR_INPUT_PICKER:
                # Streamlit color picker returns hex format
                picked_color = st.color_picker(
                    "Target color",
                    value="#808080",
                    help="Pick the color you want to enhance"
                )
                target_color = picked_color
                
            else:  # Hex code
                hex_input = st.text_input(
                    "Hex color code",
                    value="#808080",
                    placeholder="#RRGGBB",
                    help="Enter color in hex format (e.g., #808080)"
                )
                # Validate hex format
                if hex_input.startswith('#') and len(hex_input) == 7:
                    try:
                        # Test if it's valid hex
                        int(hex_input[1:], 16)
                        target_color = hex_input
                    except ValueError:
                        st.error("Invalid hex format")
                        target_color = None
                elif hex_input:
                    st.error("Format: #RRGGBB (e.g., #808080)")
                    target_color = None
            
            # Color sensitivity slider
            color_sensitivity = st.slider(
                "Color sensitivity",
                min_value=1,
                max_value=32,
                value=DEFAULT_COLOR_SENSITIVITY,
                step=1,
                help="±N values around target color (higher = more tolerance)"
            )
            
            # Show current target if valid
            if target_color:
                st.success(f"🎨 Target: {target_color} ±{color_sensitivity}")

        use_grabcut = st.checkbox(
            "GrabCut mask refinement",
            value=False,
            help="Refine text masks with GrabCut for smoother, spatially coherent edges. "
                 "Adds ~50-200ms per region. Try this if masks look ragged or grab stray pixels."
        )

        st.divider()
        
        # Manual coordinate input (only used in manual mode) - reactive inputs, no button needed
        if detection_mode == MODE_DRAW_MANUALLY:
            # Check if image is loaded to show dimensions and enable controls
            current_bytes = st.session_state.get('current_image_bytes')
            current_id = st.session_state.get('current_image_id')
            image_loaded = current_bytes is not None
            
            if image_loaded:
                # Get image dimensions
                original_image = Image.open(io.BytesIO(current_bytes))
                img_width, img_height = original_image.size
                st.caption(f"📐 Image is {img_width}w × {img_height}h")
                
                # Get current pipeline bbox or use defaults
                global_bbox_key = f"pipeline_bbox_{current_id}"
                current_bbox = st.session_state.get(global_bbox_key)
                
                if current_bbox:
                    default_x, default_y, default_w, default_h = current_bbox
                    default_x = max(0, min(default_x, img_width - 1))
                    default_y = max(0, min(default_y, img_height - 1))
                    default_w = max(1, min(default_w, img_width - default_x))
                    default_h = max(1, min(default_h, img_height - default_y))
                else:
                    default_x, default_y, default_w, default_h = 0, 0, 100, 100
                    default_w = min(default_w, img_width)
                    default_h = min(default_h, img_height)
                
                # Manual coordinate inputs - update pipeline bbox immediately on change
                col_x, col_y = st.columns(2)
                with col_x:
                    manual_x = st.number_input(
                        "X (left)", 
                        min_value=0, 
                        max_value=img_width-1, 
                        value=default_x,
                        key=f"manual_x_{current_id}"
                    )
                with col_y:
                    manual_y = st.number_input(
                        "Y (top)", 
                        min_value=0, 
                        max_value=img_height-1, 
                        value=default_y,
                        key=f"manual_y_{current_id}"
                    )
                
                col_w, col_h = st.columns(2)
                with col_w:
                    manual_w = st.number_input(
                        "W (width)", 
                        min_value=1, 
                        max_value=img_width, 
                        value=default_w,
                        key=f"manual_w_{current_id}"
                    )
                with col_h:
                    manual_h = st.number_input(
                        "H (height)", 
                        min_value=1, 
                        max_value=img_height, 
                        value=default_h,
                        key=f"manual_h_{current_id}"
                    )
                
                # Update pipeline bbox immediately when inputs change (no button needed)
                new_x = int(manual_x)
                new_y = int(manual_y)
                new_w = min(int(manual_w), img_width - new_x)
                new_h = min(int(manual_h), img_height - new_y)
                new_coords = (new_x, new_y, new_w, new_h)
                if new_coords != current_bbox:
                    st.session_state[global_bbox_key] = new_coords
                
                # Clear manual selection button
                if st.button("🗑️ Clear Manual Selection", help="Remove manual bounding box selection"):
                    st.session_state[global_bbox_key] = None
                    st.success("Manual selection cleared!")
                    st.rerun()
            else:
                st.info("💡 Upload an image to set coordinates")
        
        st.divider()

        # ── Sidebar: known watermark templates ────────────────────────────
        st.subheader("📂 Known Watermarks")

        # Discover templates on disk
        available_templates = load_watermark_templates(WATERMARKS_DIR)

        use_watermark_templates = False
        selected_templates = []

        if available_templates:
            use_watermark_templates = st.checkbox(
                "Try known watermarks first",
                value=True,
                help="Try matching known watermark templates before running text detection. "
                     "Much faster when a template matches.",
            )
            if use_watermark_templates:
                template_names = [template.name for template in available_templates]
                selected_names = st.multiselect(
                    "Templates to try",
                    options=template_names,
                    default=template_names,
                    help="Select which templates to try (first match wins)",
                )
                selected_templates = [
                    template for template in available_templates
                    if template.name in selected_names
                ]
                if selected_templates:
                    st.caption(f"{len(selected_templates)} template(s) selected")

                # Quick-test button: run ORB cascade against current image
                test_image_bytes = st.session_state.get("current_image_bytes")
                if test_image_bytes and st.button(
                    "🔍 Test templates now",
                    help="Run ORB matching against the current image without inpainting.",
                ):
                    import cv2 as _cv2
                    import time as _time
                    arr = np.frombuffer(test_image_bytes, dtype=np.uint8)
                    test_img = _cv2.imdecode(arr, _cv2.IMREAD_COLOR)
                    if test_img is not None:
                        t0 = _time.perf_counter()
                        cascade_hit = try_watermark_cascade(test_img, selected_templates)
                        elapsed = _time.perf_counter() - t0
                        if cascade_hit is not None:
                            _, bbox, matched_name = cascade_hit
                            st.success(
                                f"Matched **{matched_name}** at "
                                f"({bbox[0]}, {bbox[1]}) {bbox[2]}x{bbox[3]}px "
                                f"in {elapsed:.2f}s"
                            )
                        else:
                            st.warning(f"No template matched ({elapsed:.2f}s)")
                    else:
                        st.error("Could not decode the current image")
        else:
            st.caption("No templates in watermarks/ folder")

        # Upload new template
        uploaded_template = st.file_uploader(
            "Add template (RGBA PNG)",
            type=["png"],
            help="Upload an RGBA PNG where the alpha channel marks watermark pixels. "
                 "Saved to the watermarks/ folder for future use.",
            key="watermark_uploader",
        )
        if uploaded_template is not None:
            import cv2 as _cv2
            uploaded_template_bytes = uploaded_template.getvalue()
            upload_fingerprint = (
                f"{uploaded_template.name}:"
                f"{hashlib.md5(uploaded_template_bytes).hexdigest()}"
            )
            file_bytes = np.frombuffer(uploaded_template_bytes, dtype=np.uint8)
            rgba = _cv2.imdecode(file_bytes, _cv2.IMREAD_UNCHANGED)
            if rgba is not None and rgba.ndim == 3 and rgba.shape[2] == 4:
                save_path = WATERMARKS_DIR / uploaded_template.name
                if st.session_state.get("last_saved_template_upload") != upload_fingerprint:
                    WATERMARKS_DIR.mkdir(parents=True, exist_ok=True)
                    _cv2.imwrite(str(save_path), rgba)
                    st.session_state.last_saved_template_upload = upload_fingerprint
                    st.success(f"Saved: {uploaded_template.name}")
                    st.rerun()
                else:
                    st.caption(f"Template already saved: {uploaded_template.name}")
            else:
                st.error("File must be an RGBA PNG (4 channels)")

        st.divider()
        
        # Inpainting settings
        st.subheader("🎨 Inpainting")
        granularity = st.slider(
            "Color Granularity",
            min_value=3,
            max_value=20,
            value=DEFAULT_GRANULARITY,
            step=1,
            help="Number of color clusters for text detection. Lower values (3-6) are more aggressive; higher values (12-20) are more selective. Default of 4 works well for most images."
        )
        
        method = st.selectbox(
            "Method",
            options=["lama", "telea"],
            index=0,
            help="LaMa: High quality but slower, TELEA: Fast but lower quality"
        )
        
        st.divider()
        
        # Status section
        st.subheader("📊 Status")
        
        # Ready indicator
        if init_results.get("lama_initialized", False):
            st.success("✅ Ready for processing!")
        else:
            st.warning("⚠️ Some models may not be ready")
        
        # Detector status and controls
        col_det1, col_det2 = st.columns(2)
        with col_det1:
            st.metric("Detectors", "✅ Loaded")
        with col_det2:
            if st.button("🔄 Reload", help="Reload detection models", key="reload_detectors"):
                st.cache_resource.clear()
                st.success("Models reloaded!")
                st.rerun()
        
        # LaMa status monitoring
        lama_status = display_lama_status()
        
        # Debug options
        with st.expander("🔧 Advanced / Debug"):
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.01,
                max_value=0.99,
                value=WEB_DEFAULT_CONFIDENCE,
                step=0.01,
                help="Lower values detect more text (may include false positives). "
                     "Because we require 2+ detectors to agree, values as low as "
                     "0.03 still work well. Rarely needs adjustment."
            )
            keep_masks = st.checkbox(
                "Show detection masks",
                value=False,
                help="Display the detected text regions"
            )
    
    # ── Main content area ──────────────────────────────────────────────
    final_target_color = target_color if enable_color_enhancement and target_color else None
    auto_detections = []
    sorted_detections = []
    selected_detection_indices = set()
    superset_bbox = None
    use_superset = False

    col1, col2 = st.columns(2)
    
    # ── Column 1: upload, detect, select regions ───────────────────────
    with col1:
        st.header("📤 Upload Image")
        
        # File uploader, Ingest, and Refresh side by side
        upload_col, ingest_col, refresh_col = st.columns([3, 1, 1])
        
        with upload_col:
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'],
                help="Drag and drop an image or click to browse",
                key="source_image_uploader",
            )
        
        with ingest_col:
            # Add some vertical spacing to align with file uploader
            st.write("")  # Spacer
            st.write("")  # Spacer
            
            # Check if we have a result to ingest
            has_result = hasattr(st.session_state, 'result_image') and st.session_state.result_image is not None
            
            if st.button("🔄 Ingest Result", disabled=not has_result, 
                        help="Load the processed result as the new input for another pass"):
                if has_result:
                    # Convert result numpy array to bytes
                    result_pil = Image.fromarray(st.session_state.result_image)
                    buf = io.BytesIO()
                    result_pil.save(buf, format='PNG')
                    buf.seek(0)
                    
                    # Store ingested image data in session state
                    st.session_state.ingested_image_bytes = buf.getvalue()
                    st.session_state.ingested_image_name = f"pass_{st.session_state.get('ingest_count', 0) + 1}_{st.session_state.get('original_filename', 'image.png')}"
                    st.session_state.ingest_count = st.session_state.get('ingest_count', 0) + 1
                    
                    # Clear the previous result so we can run again
                    st.session_state.result_image = None
                    st.session_state.mask_image = None
                    
                    st.rerun()
        
        with refresh_col:
            st.write("")  # Spacer
            st.write("")  # Spacer
            if st.button("🔃 Refresh Canvas",
                         help="Force the canvas to re-render (fixes occasional blackouts)"):
                # Bump a counter that is part of the canvas widget key,
                # which forces Streamlit to destroy and recreate it.
                st.session_state["canvas_refresh_counter"] = (
                    st.session_state.get("canvas_refresh_counter", 0) + 1
                )
                st.rerun()
        
        # Determine which image source to use: ingested image takes priority over uploaded file
        image_bytes, image_name = resolve_active_image(
            st.session_state.get("ingested_image_bytes"),
            st.session_state.get("ingested_image_name"),
            uploaded_file,
        )
        image_state_id = make_image_state_id(image_name, image_bytes)

        if st.session_state.get("active_image_id") != image_state_id:
            st.session_state["sorted_detections"] = []
            st.session_state.result_image = None
            st.session_state.mask_image = None
            st.session_state.timing_data = None
            st.session_state.active_image_id = image_state_id

        if 'ingested_image_bytes' in st.session_state and st.session_state.ingested_image_bytes is not None:
            st.info(f"🔄 Using ingested result: {image_name}")
            
            # Add button to clear ingested and go back to file uploader
            if st.button("❌ Clear ingested image"):
                st.session_state.ingested_image_bytes = None
                st.session_state.ingested_image_name = None
                st.rerun()
        
        # Store in session state for other parts of the app
        st.session_state.uploaded_file = uploaded_file
        st.session_state.current_image_bytes = image_bytes
        st.session_state.current_image_name = image_name
        st.session_state.current_image_id = image_state_id
        
        if image_bytes is not None:
            # Load original image from bytes
            original_image = Image.open(io.BytesIO(image_bytes))
            
            # Force full image load (PIL uses lazy loading which can cause race conditions)
            original_image.load()
            
            # Convert to RGB if needed (canvas may not handle all modes like 'P', 'LA', 'CMYK')
            if original_image.mode not in ('RGB', 'RGBA'):
                original_image = original_image.convert('RGB')
            
            img_width, img_height = original_image.size
            
            # Image info
            st.info(f"📊 Size: {img_width}×{img_height} pixels")
            
            # ── Watermark cascade (fast path) ────────────────────────────
            # Try known templates before heavy detection.  If a template
            # matches we store the result and skip detection entirely;
            # the single "Remove Watermark" button at the bottom handles
            # both the template-match and consensus-detection paths.
            watermark_handled = False
            wm_match = None          # (mask, bbox, name, img_bgr) on hit
            if use_watermark_templates and selected_templates:
                import cv2 as _cv2
                import time as _time
                arr = np.frombuffer(image_bytes, dtype=np.uint8)
                img_bgr = _cv2.imdecode(arr, _cv2.IMREAD_COLOR)
                if img_bgr is not None:
                    t0 = _time.perf_counter()
                    cascade_hit = try_watermark_cascade(img_bgr, selected_templates)
                    wm_cascade_elapsed = _time.perf_counter() - t0
                    if cascade_hit is not None:
                        wm_mask, wm_bbox, wm_tmpl_name = cascade_hit
                        wm_match = (wm_mask, wm_bbox, wm_tmpl_name, img_bgr)
                        st.success(
                            f"🎯 Matched watermark template **{wm_tmpl_name}** "
                            f"at ({wm_bbox[0]}, {wm_bbox[1]}) "
                            f"{wm_bbox[2]}x{wm_bbox[3]}px  ({wm_cascade_elapsed:.2f}s)"
                        )
                        # Draw bounding box overlay on the image
                        annotated = original_image.copy()
                        draw = ImageDraw.Draw(annotated)
                        x, y, w, h = wm_bbox
                        draw.rectangle(
                            [x, y, x + w, y + h],
                            outline=(0, 255, 0), width=3,
                        )
                        # Label with template filename (sans extension)
                        label = Path(wm_tmpl_name).stem
                        try:
                            font = ImageFont.truetype("arial.ttf", size=max(16, img_height // 40))
                        except OSError:
                            font = ImageFont.load_default()
                        draw.text(
                            (x, max(0, y - font.size - 4)),
                            label, fill=(0, 255, 0), font=font,
                        )
                        st.image(annotated, caption=f"Matched: {wm_tmpl_name}", width='stretch')
                        watermark_handled = True
                    else:
                        st.caption(
                            f"No watermark template matched ({wm_cascade_elapsed:.2f}s)"
                        )
            
            # ── Auto-detect or manual canvas ─────────────────────────────
            if not watermark_handled and is_auto_mode:
                # Run detections immediately
                with st.spinner("🔍 Running consensus detection..."):
                    auto_detections = run_detections_cached(
                        image_bytes, 
                        confidence_threshold
                    )
                
                if auto_detections:
                    st.success(f"✅ Found {len(auto_detections)} consensus regions")
                    
                    # Sort detections by consensus strength (# detectors + confidence)
                    # Higher is better: 3-way consensus ranks above 2-way, and within
                    # same consensus level, higher confidence wins
                    def detection_score(det):
                        num_detectors = len(det.get('detectors', []))
                        confidence = det.get('confidence', 0)
                        return num_detectors + confidence  # e.g., 3-way @ 80% = 3.8
                    
                    sorted_detections = sorted(auto_detections, key=detection_score, reverse=True)
                    
                    # Store sorted detections in session state for use in processing section
                    st.session_state['sorted_detections'] = sorted_detections
                    
                    # Detection selector with checkboxes
                    st.subheader("📦 Select Regions to Remove")
                    
                    # Initialize session state for selected boxes - auto-select first (best) region
                    selection_key = f"selected_detections_{image_state_id}"
                    if selection_key not in st.session_state:
                        # Auto-select the first (highest-scoring) detection
                        st.session_state[selection_key] = {0} if sorted_detections else set()
                    
                    # Checkboxes for each detection (now sorted by score)
                    for i, det in enumerate(sorted_detections):
                        x, y, w, h = det['bbox']
                        conf = det.get('confidence', 0)
                        detectors = "+".join(sorted(det.get('detectors', [])))
                        detector_count = len(det.get('detectors', []))
                        
                        # Create clearer label
                        consensus_type = f"{detector_count}-way" if detector_count >= 2 else "1-way"
                        label = f"Region {i+1}: {consensus_type} consensus ({detectors}) - {conf:.0%} confidence - {w}×{h}px at ({x},{y})"
                        
                        # Checkbox
                        is_selected = st.checkbox(
                            label,
                            value=(i in st.session_state[selection_key]),
                            key=f"detection_checkbox_{i}_{image_state_id}"
                        )
                        
                        # Update session state
                        if is_selected:
                            st.session_state[selection_key].add(i)
                        else:
                            st.session_state[selection_key].discard(i)
                    
                    # Get selected indices
                    selected_detection_indices = st.session_state[selection_key]
                    
                    # Superset box option (only if at least one box is selected)
                    superset_bbox = None
                    if selected_detection_indices:
                        selected_bboxes = [sorted_detections[i]['bbox'] for i in selected_detection_indices]
                        superset_bbox = calculate_bbox_superset(selected_bboxes, (img_height, img_width))
                        
                        if superset_bbox:
                            sx, sy, sw, sh = superset_bbox
                            superset_label = f"Superset Box: Contains all {len(selected_detection_indices)} selected region(s) - {sw}×{sh}px at ({sx},{sy})"
                            
                            # Default to False - let user explicitly opt into superset
                            use_superset = st.checkbox(
                                superset_label,
                                value=False,
                                key=f"superset_checkbox_{image_state_id}",
                                help="Use a single bounding box that contains all selected regions"
                            )
                    else:
                        st.info("💡 Select one or more regions above to enable the superset box option")
                    
                    # Show annotated image with overlays
                    annotated_image = draw_detection_overlays(
                        original_image, 
                        sorted_detections, 
                        selected_indices=selected_detection_indices,
                        superset_bbox=superset_bbox if use_superset else None
                    )
                    
                    caption = "Detected Regions"
                    if selected_detection_indices:
                        caption += f" (Green = {len(selected_detection_indices)} selected"
                        if use_superset:
                            caption += ", Orange = Superset"
                        caption += ")"
                    st.image(annotated_image, caption=caption, width='stretch')
                    
                else:
                    st.session_state['sorted_detections'] = []
                    st.warning("⚠️ No consensus regions detected")
                    st.info("💡 Try lowering confidence threshold or switch to manual mode")
                    st.image(original_image, caption="Original Image", width='stretch')
            
            elif not watermark_handled:
                # Manual mode - show interactive canvas overlay on image
                global_bbox_key = f"pipeline_bbox_{image_state_id}"
                
                # Initialize session state
                if global_bbox_key not in st.session_state:
                    st.session_state[global_bbox_key] = None
                
                # Get the measured column width (measured at app startup)
                # This ensures the canvas fills the column width exactly
                orig_width, orig_height = original_image.size
                orig_aspect = orig_width / orig_height
                
                # Use the column width measured at app startup
                container_width = st.session_state["column_width"]
                
                # Calculate display dimensions to fill container width while preserving aspect ratio
                display_width = int(container_width)
                display_height = int(round(display_width / orig_aspect))
                
                # Calculate scale factors for coordinate conversion
                scale_x = orig_width / display_width
                scale_y = orig_height / display_height
                
                # Build initial_drawing for the canvas (handles dragging, resizing, etc.)
                initial_drawing = None
                if st.session_state[global_bbox_key] is not None:
                    initial_drawing = bbox_to_fabric_rect(
                        st.session_state[global_bbox_key], scale_x, scale_y
                    )
                
                # Create canvas with background image
                # Canvas uses large pixel dimensions - will fill container via Streamlit's natural sizing
                # Coordinate mapping uses these pixel dimensions (canvas coordinate system)
                canvas_result = st_canvas(
                    fill_color="rgba(0, 0, 0, 0)",  # Transparent fill
                    stroke_width=3,
                    stroke_color="rgb(0, 255, 0)",  # Green stroke (matches initial_drawing)
                    background_image=original_image,  # Package will resize internally
                    update_streamlit=True,  # Critical: updates immediately on draw
                    width=display_width,    # Canvas pixel width (coordinate system uses this)
                    height=display_height,  # Canvas pixel height (coordinate system uses this)
                    drawing_mode="rect" if initial_drawing is None else "transform",  # Allow drawing if no existing rectangle
                    point_display_radius=0,
                    key=f"manual_rect_canvas_{image_state_id}_{st.session_state.get('canvas_refresh_counter', 0)}",
                    initial_drawing=initial_drawing  # This handles everything - dragging, resizing, out-of-frame
                )
                
                # Convert canvas result back to image coordinates
                if canvas_result.json_data is not None:
                    objects = canvas_result.json_data.get("objects", [])
                    if objects:
                        new_coords = fabric_rect_to_bbox(
                            objects[0], scale_x, scale_y, orig_width, orig_height
                        )
                        if new_coords is not None:
                            st.session_state[global_bbox_key] = new_coords
                            # Automatically switch to manual mode if not already
                            if detection_mode != MODE_DRAW_MANUALLY:
                                st.session_state.detection_mode_override = MODE_DRAW_MANUALLY
                                st.rerun()
                    elif not objects and st.session_state[global_bbox_key] is not None:
                        # User cleared the drawing - clear the bbox
                        st.session_state[global_bbox_key] = None
            
            # ── Resolve final bbox for processing ────────────────────────
            force_bbox_coords = None
            
            # Get sorted detections from session state (sorted by consensus score)
            sorted_detections = st.session_state.get('sorted_detections', [])
            
            if is_auto_mode and sorted_detections:
                if use_superset and superset_bbox:
                    # Use superset box
                    force_bbox_coords = superset_bbox
                    sx, sy, sw, sh = superset_bbox
                    st.success(f"🎯 Using superset box containing {len(selected_detection_indices)} selected region(s)")
                    st.caption(f"Superset: {sw}×{sh}px at ({sx},{sy})")
                elif selected_detection_indices:
                    # Use union of selected boxes (calculate superset but don't show as selected)
                    selected_bboxes = [sorted_detections[i]['bbox'] for i in selected_detection_indices]
                    force_bbox_coords = calculate_bbox_superset(selected_bboxes, (img_height, img_width))
                    st.success(f"🎯 Using union of {len(selected_detection_indices)} selected region(s)")
                    if force_bbox_coords:
                        sx, sy, sw, sh = force_bbox_coords
                        st.caption(f"Union box: {sw}×{sh}px at ({sx},{sy})")
                else:
                    # No selection - warn user
                    force_bbox_coords = None
                    st.warning("⚠️ Please select at least one region to process")
            
            elif not watermark_handled and detection_mode == MODE_DRAW_MANUALLY:
                # Manual mode - get bbox from pipeline state
                global_bbox_key = f"pipeline_bbox_{image_state_id}"
                if global_bbox_key in st.session_state and st.session_state[global_bbox_key] is not None:
                    force_bbox_coords = st.session_state[global_bbox_key]
                    x, y, w, h = force_bbox_coords
                    coverage = (w * h) / (img_width * img_height) * 100
                    st.success(f"🎯 Pipeline will use coordinates: {force_bbox_coords}")
                    st.caption(f"📊 Detection box covers {coverage:.1f}% of the image")
                else:
                    force_bbox_coords = None
                    st.info("💡 Drag a rectangle on the image above to set detection coordinates")
            
            # ── Process button (always in the same location) ─────────────
            # Warning for LaMa issues
            if method == "lama":
                lama_status = get_lama_status()
                if not lama_status["available"]:
                    st.error("⚠️ LaMa not available - please install simple-lama-inpainting or switch to TELEA")
                elif not lama_status["healthy"]:
                    st.warning("⚠️ LaMa may not be working correctly - consider restarting it")
            
            if st.button("🚀 Remove Watermark", type="primary"):

                # ── Watermark template path ───────────────────────────
                if wm_match is not None:
                    from untextre.inpaint import inpaint_image
                    wm_mask, wm_bbox, wm_tmpl_name, img_bgr = wm_match
                    with st.spinner(f"Removing watermark ({wm_tmpl_name})..."):
                        start_time = time.time()
                        result_bgr = inpaint_image(
                            img_bgr, wm_mask,
                            bbox=wm_bbox, method=method,
                        )
                        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
                        processing_time = time.time() - start_time
                    st.session_state.result_image = result_rgb
                    st.session_state.mask_image = wm_mask if keep_masks else None
                    st.session_state.timing_data = {
                        "matched_template": wm_tmpl_name,
                        "mask_found": True,
                        "total_time": processing_time,
                    }
                    st.session_state.processing_time = processing_time
                    st.session_state.original_filename = image_name
                    st.success(f"✅ Watermark removed in {processing_time:.1f}s!")
                    st.rerun()

                # ── Consensus detection path ──────────────────────────
                else:
                    if is_auto_mode:
                        if not selected_detection_indices:
                            st.error("❌ Please select at least one region to process")
                            st.stop()
                        if force_bbox_coords is None:
                            st.error("❌ No valid bounding box selected")
                            st.stop()
                    
                    start_time = time.time()
                    
                    with st.spinner("Processing image... This may take a few seconds."):
                        result_image, mask_image, timing_data = process_image_streamlit(
                            image_bytes, confidence_threshold, granularity, method, keep_masks,
                            target_color=final_target_color, color_sensitivity=color_sensitivity,
                            forced_bbox=force_bbox_coords,
                            use_grabcut=use_grabcut,
                            use_grabcut_expand=use_grabcut_expand,
                        )
                    
                    processing_time = time.time() - start_time
                    
                    if result_image is not None:
                        st.session_state.result_image = result_image
                        st.session_state.mask_image = mask_image
                        st.session_state.timing_data = timing_data
                        st.session_state.processing_time = processing_time
                        st.session_state.original_filename = image_name
                        
                        st.success(f"✅ Processing complete in {processing_time:.1f} seconds!")
                        st.rerun()
    
    # ── Column 2: result display, download, stats ──────────────────────
    with col2:
        st.header("📥 Result")
        
        # Spacer to align result image with the source image in col1
        # (col1 has file uploader, info boxes, and size display above its image)
        st.markdown(f"<div style='height: {RESULT_COLUMN_SPACER_PX}px'></div>", unsafe_allow_html=True)
        
        if hasattr(st.session_state, 'result_image') and st.session_state.result_image is not None:
            # Display result image
            st.image(st.session_state.result_image, caption="Processed Image", width='stretch')
            
            # Encode result for download (format chosen by original extension)
            result_pil = Image.fromarray(st.session_state.result_image)
            buf_bytes, download_name, mime_type = encode_result_for_download(
                result_pil, st.session_state.original_filename
            )
            file_size_mb = len(buf_bytes) / (1024 * 1024)
            
            st.download_button(
                label=f"💾 Download Result ({file_size_mb:.1f}MB)",
                data=buf_bytes,
                file_name=download_name,
                mime=mime_type
            )
            
            # Show processing stats
            if st.session_state.timing_data:
                timing = st.session_state.timing_data
                
                with st.expander("📊 Processing Details"):
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.metric("Total Time", f"{timing['total_time']:.1f}s")
                        
                        matched_tmpl = timing.get("matched_template")
                        if matched_tmpl:
                            # Watermark template path — show template name
                            st.metric("Method", "Template Match")
                            st.caption(f"Matched: {matched_tmpl}")
                        elif is_auto_mode and sorted_detections:
                            st.metric("Detection Time", "Cached ✨", help="Detection ran when image was uploaded")
                            st.metric("Consensus Boxes", timing.get('consensus_boxes_count', 0))
                        else:
                            det_time = timing.get('detection_time')
                            if det_time is not None:
                                st.metric("Detection Time", f"{det_time:.1f}s")
                            st.metric("Consensus Boxes", timing.get('consensus_boxes_count', 0))
                    
                    with col_b:
                        color_time = timing.get('color_time')
                        if color_time is not None:
                            st.metric("TF-IDF Time", f"{color_time:.1f}s")
                        inpaint_time = timing.get('inpaint_time')
                        if inpaint_time is not None:
                            st.metric("Inpainting Time", f"{inpaint_time:.1f}s")
                        
                        failover_type = timing.get('failover_type', 'none')
                        if failover_type != 'none':
                            st.metric("Failover Used", failover_type.title())
            
            # Show mask if available
            if keep_masks and hasattr(st.session_state, 'mask_image') and st.session_state.mask_image is not None:
                with st.expander("🎭 Detection Mask"):
                    st.image(st.session_state.mask_image, caption="Detected Text Regions", width='stretch')
        
        else:
            st.info("👆 Upload an image and click 'Remove Text Watermarks' to see results here")
    
    # ── Footer ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with ❤️ using Streamlit • Powered by LaMa, DocTR, EasyOCR, and EAST</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 
