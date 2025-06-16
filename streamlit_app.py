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
import numpy as np
import tempfile
import time
from pathlib import Path
from PIL import Image
import io
from streamlit_image_annotation import detection

# Import our untextre modules
from untextre.utils import load_image, save_image
from untextre.preprocessor import preprocess_image
from untextre.cli import initialize_consensus_models, run_consensus_detection, process_single_image
from untextre.inpaint import initialize_lama_model, get_lama_status, reset_lama_model

# Page configuration
st.set_page_config(
    page_title="UnTextre - Text Watermark Removal",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def initialize_models():
    """Initialize all models once at startup."""
    with st.spinner("Loading AI models... This may take a minute on first run."):
        # Initialize consensus detection models
        initialize_consensus_models(confidence_threshold=0.3, device="cuda")
        
        # Initialize LaMa inpainting model
        lama_success = initialize_lama_model(device="cuda")
        
    return {"lama_initialized": lama_success}

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

def process_image_streamlit(image_bytes, confidence_threshold, granularity, method, keep_masks, target_color=None, color_sensitivity=3, forced_bbox=None):
    """Process an uploaded image and return the result."""
    
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
            
            # Convert hex color to BGR tuple if provided
            target_color_bgr = None
            if target_color and target_color.startswith('#') and len(target_color) == 7:
                try:
                    # Parse hex color #RRGGBB to BGR tuple
                    hex_color = target_color[1:]  # Remove #
                    r = int(hex_color[0:2], 16)
                    g = int(hex_color[2:4], 16) 
                    b = int(hex_color[4:6], 16)
                    target_color_bgr = (b, g, r)  # OpenCV uses BGR order
                except ValueError:
                    st.warning(f"Invalid hex color format: {target_color}. Ignoring color enhancement.")
                    target_color_bgr = None
            
            # Debug logging for forced bbox
            if forced_bbox is not None:
                st.info(f"🔧 DEBUG: Passing forced_bbox to CLI: {forced_bbox}")
            else:
                st.info("🔧 DEBUG: No forced_bbox provided to CLI")
            
            # Process the image using our existing pipeline
            timing_data = process_single_image(
                image_path=input_path,
                output_dir=output_dir,
                target_color=target_color_bgr,  # Pass BGR tuple instead of hex string
                keep_masks=keep_masks,
                method=method,
                maskfile=None,
                confidence_threshold=confidence_threshold,
                granularity=granularity,
                forced_bbox=forced_bbox,
                color_sensitivity=color_sensitivity
            )
            
            # Load the result
            result_path = output_dir / f"{input_path.stem}_clean{input_path.suffix}"
            if result_path.exists():
                result_image = load_image(result_path)
                
                # Convert BGR to RGB for display
                result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                
                # Load mask if available
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
            # Check if it's a LaMa-related error and try to provide helpful feedback
            error_msg = str(e)
            if "LaMa" in error_msg or "lama" in error_msg.lower():
                st.error(f"LaMa Error: {error_msg}")
                st.info("Try restarting LaMa using the button in the sidebar, or switch to TELEA inpainting.")
            else:
                st.error(f"Error processing image: {error_msg}")
            return None, None, None
        finally:
            # Clean up input file
            input_path.unlink(missing_ok=True)

def main():
    """Main Streamlit application."""
    
    # Initialize models
    init_results = initialize_models()
    
    # Title and description
    st.title("🎨 UnTextre - Text Watermark Removal")
    st.markdown("""
    Upload an image with text watermarks and watch them disappear! This tool uses advanced AI models 
    to detect and remove text overlays while preserving the underlying image.
    """)
    
    # Sidebar for options - using session state to maintain values across reruns
    with st.sidebar:
        st.header("⚙️ Processing Options")
        
        # Detection settings
        st.subheader("Detection Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.01,
            max_value=0.99,
            value=0.025,
            step=0.01,
            help="Lower values detect more text (may include false positives)"
        )
        
        granularity = st.slider(
            "Color Granularity",
            min_value=4,
            max_value=48,
            value=24,
            step=1,
            help="Number of color clusters for text detection"
        )
        
        # Inpainting method
        st.subheader("Inpainting Method")
        method = st.selectbox(
            "Method",
            options=["lama", "telea"],
            index=0,
            help="LaMa: High quality but slower, TELEA: Fast but lower quality"
        )
        
        # Target Color options
        st.subheader("🎨 Target Color")
        enable_color_enhancement = st.checkbox(
            "Enable target color",
            value=False,
            help="Enhance subtle watermarks by targeting specific colors"
        )
        
        # Color selection options (enabled only when image is loaded and enhancement is enabled)
        image_loaded = 'uploaded_file' in st.session_state and st.session_state.uploaded_file is not None
        color_controls_enabled = enable_color_enhancement and image_loaded
        
        if not image_loaded and enable_color_enhancement:
            st.info("💡 Upload an image to access target color options")
        
        color_input_method = st.radio(
            "Color input method:",
            ["Color picker", "Hex code"],
            disabled=not color_controls_enabled,
            help="Choose how to specify the target color"
        )
        
        target_color = None
        color_sensitivity = 3  # Default value
        
        if color_controls_enabled:
            if color_input_method == "Color picker":
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
                value=3,
                step=1,
                help="±N values around target color (higher = more tolerance)"
            )
            
            # Show current target if valid
            if target_color:
                st.success(f"🎨 Target: {target_color} ±{color_sensitivity}")
        
        # Force Detection Box
        st.subheader("🔍 Force Detection Box")
        enable_force_bbox = st.checkbox(
            "Enable force detection box",
            value=False,
            help="Draw a bounding box to force detection in a specific region"
        )
        
        
        # Debug options
        st.subheader("Debug Options")
        keep_masks = st.checkbox(
            "Show detection masks",
            value=False,
            help="Display the detected text regions"
        )
        
        # LaMa status monitoring
        lama_status = display_lama_status()
        
        # General model status
        st.subheader("🔧 General Status")
        if init_results.get("lama_initialized", False):
            st.success("✅ Detection models loaded")
        else:
            st.warning("⚠️ Some models may not be ready")
            
        st.info("Ready for processing!")
    
    # Store sidebar values in variables accessible to main area
    final_target_color = target_color if enable_color_enhancement and target_color else None
    
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📤 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Drag and drop an image or click to browse"
        )
        
        # Store uploaded file in session state for color control logic
        st.session_state.uploaded_file = uploaded_file
        
        if uploaded_file is not None:
            # Display original image
            original_image = Image.open(uploaded_file)
            st.image(original_image, caption="Original Image", use_container_width=True)
            
            # Image info
            st.info(f"📊 Size: {original_image.size[0]}×{original_image.size[1]} pixels")
            
            # Force detection box drawing interface
            force_bbox_coords = None
            if enable_force_bbox:
                st.subheader("🖱️ Draw Detection Box")
                st.info("💡 Click and drag on the image to draw a rectangle around the area you want to force detection on")
                
                # Helpful UI guide
                with st.expander("ℹ️ How to use the drawing tools", expanded=False):
                    st.markdown("""
                    **Drawing Tools Guide:**
                    
                    🎯 **Mode Selection** (top-left buttons):
                    - **Add box**: Click this to draw new rectangles
                    - **Remove box**: Click this to delete existing rectangles
                    
                    🏷️ **Class Selection**: 
                    - Leave as "Detection Area" (ignore this dropdown)
                    
                    🖱️ **Drawing**:
                    1. Select **Add box** mode
                    2. Click and drag on the image to draw a rectangle  
                    3. Release to complete the rectangle
                    
                    🗑️ **Removing**:
                    1. Select **Remove box** mode
                    2. Click on any existing rectangle to delete it
                    
                    ⌨️ **Keyboard**: Press **Space** to finish/confirm your selection
                    """)
                
                st.caption("💡 **Tip**: Draw your rectangle, then click **Set Detection Box** below to save it")
                
                # Initialize session state for force bbox if not exists
                bbox_key = f"force_bbox_{uploaded_file.name}"
                if bbox_key not in st.session_state:
                    st.session_state[bbox_key] = None
                
                # Calculate display dimensions for annotation tool
                orig_width, orig_height = original_image.size
                display_height = min(500, orig_height)
                display_width = min(700, orig_width)
                
                # Save image temporarily for annotation - use a more persistent approach
                temp_dir = Path(tempfile.gettempdir()) / "streamlit_annotation"
                temp_dir.mkdir(exist_ok=True)
                temp_path = temp_dir / f"annotation_{hash(str(uploaded_file.name))}.png"
                
                try:
                    # Save image for annotation
                    original_image.save(temp_path)
                    
                    # Get existing bboxes from session state
                    existing_bboxes = []
                    existing_labels = []
                    if st.session_state[bbox_key] is not None:
                        # The component expects coordinates in original image space, same as what we store
                        # So we can pass them directly without conversion
                        orig_bbox = st.session_state[bbox_key]
                        existing_bboxes = [list(orig_bbox)]  # Convert tuple to list
                        existing_labels = [0]
                    
                    # Use streamlit-image-annotation for detection
                    bboxes = detection(
                        image_path=str(temp_path),
                        label_list=["Detection Area"],  # Simplified single label
                        bboxes=existing_bboxes,
                        labels=existing_labels,
                        height=display_height,
                        width=display_width,
                        use_space=True,
                        key=f"force_detection_annotation_{uploaded_file.name}"
                    )
                    
                    if bboxes and len(bboxes) > 0:
                        # Get the first/latest bounding box
                        bbox_data = bboxes[-1]  # Use the last drawn box
                        bbox = bbox_data["bbox"]  # [x, y, width, height] - ALREADY in original image coordinates!
                        
                        # The component returns coordinates in original image space, not display space
                        # So we can use them directly without scaling
                        orig_left, orig_top, orig_width_box, orig_height_box = bbox
                        
                        # Convert to integers and ensure coordinates are within image bounds
                        orig_left = max(0, min(int(orig_left), orig_width - 1))
                        orig_top = max(0, min(int(orig_top), orig_height - 1))
                        orig_width_box = max(1, min(int(orig_width_box), orig_width - orig_left))
                        orig_height_box = max(1, min(int(orig_height_box), orig_height - orig_top))
                        
                        force_bbox_coords = (orig_left, orig_top, orig_width_box, orig_height_box)
                        
                        # Store in session state to persist across renders
                        st.session_state[bbox_key] = force_bbox_coords
                        
                        # Show the detected bounding box
                        coverage = (orig_width_box * orig_height_box) / (orig_width * orig_height) * 100
                        st.success(f"🔍 Force detection box: ({orig_left}, {orig_top}, {orig_width_box}, {orig_height_box})")
                        st.info(f"📊 Detection box covers {coverage:.1f}% of the image")
                    else:
                        # Retrieve from session state if no new bbox drawn
                        if st.session_state[bbox_key] is not None:
                            force_bbox_coords = st.session_state[bbox_key]
                            st.success(f"🔍 Using saved detection box: {force_bbox_coords}")
                        else:
                            st.info("👆 Click and drag on the image to draw a detection rectangle")
                    
                    # User-friendly action buttons
                    col_btn1, col_btn2 = st.columns(2)
                    
                    with col_btn1:
                        if st.button("✅ Set Detection Box", type="primary", help="Save the current bounding box"):
                            if force_bbox_coords is not None:
                                st.session_state[bbox_key] = force_bbox_coords
                                st.success("Detection box saved!")
                            else:
                                st.warning("Please draw a bounding box first")
                    
                    with col_btn2:
                        if st.button("🗑️ Clear Detection Box", help="Remove the current bounding box"):
                            st.session_state[bbox_key] = None
                            st.success("Detection box cleared!")
                            st.rerun()
                        
                except Exception as e:
                    st.error(f"Error with bounding box annotation tool: {e}")
                    st.error(f"Error details: {type(e).__name__}: {str(e)}")
                    # Fallback: manual coordinate input
                    st.warning("Falling back to manual coordinate input")
                    
                    col_a, col_b, col_c, col_d = st.columns(4)
                    with col_a:
                        manual_x = st.number_input("X (left)", min_value=0, max_value=orig_width-1, value=0)
                    with col_b:
                        manual_y = st.number_input("Y (top)", min_value=0, max_value=orig_height-1, value=0)
                    with col_c:
                        manual_w = st.number_input("Width", min_value=1, max_value=orig_width, value=100)
                    with col_d:
                        manual_h = st.number_input("Height", min_value=1, max_value=orig_height, value=100)
                    
                    if st.button("Use Manual Coordinates"):
                        force_bbox_coords = (manual_x, manual_y, manual_w, manual_h)
                        st.session_state[bbox_key] = force_bbox_coords
                        st.success(f"🔍 Manual detection box: {force_bbox_coords}")
                
                # Clean up temp file when done
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except:
                        pass  # Ignore cleanup errors
                
                # If we still don't have coordinates from the current render, get from session state
                if force_bbox_coords is None and st.session_state[bbox_key] is not None:
                    force_bbox_coords = st.session_state[bbox_key]
            
            # Show current force bbox status
            if enable_force_bbox:
                if force_bbox_coords is not None:
                    st.info(f"🎯 Force detection will be applied to region: {force_bbox_coords}")
                else:
                    st.warning("⚠️ Force detection is enabled but no bounding box is set")
            
            # Warning for LaMa issues
            if method == "lama":
                lama_status = get_lama_status()
                if not lama_status["available"]:
                    st.error("⚠️ LaMa not available - please install simple-lama-inpainting or switch to TELEA")
                elif not lama_status["healthy"]:
                    st.warning("⚠️ LaMa may not be working correctly - consider restarting it")
            
            # Process button
            if st.button("🚀 Remove Text Watermarks", type="primary"):
                start_time = time.time()
                
                with st.spinner("Processing image... This may take a few seconds."):
                    # Get image bytes
                    image_bytes = uploaded_file.getvalue()
                    
                    # Process the image
                    result_image, mask_image, timing_data = process_image_streamlit(
                        image_bytes, confidence_threshold, granularity, method, keep_masks,
                        target_color=final_target_color, color_sensitivity=color_sensitivity,
                        forced_bbox=force_bbox_coords
                    )
                
                processing_time = time.time() - start_time
                
                if result_image is not None:
                    # Store results in session state
                    st.session_state.result_image = result_image
                    st.session_state.mask_image = mask_image
                    st.session_state.timing_data = timing_data
                    st.session_state.processing_time = processing_time
                    st.session_state.original_filename = uploaded_file.name
                    
                    st.success(f"✅ Processing complete in {processing_time:.1f} seconds!")
                    st.rerun()
    
    with col2:
        st.header("📥 Result")
        
        if hasattr(st.session_state, 'result_image') and st.session_state.result_image is not None:
            # Display result image
            st.image(st.session_state.result_image, caption="Processed Image", use_container_width=True)
            
            # Download button
            result_pil = Image.fromarray(st.session_state.result_image)
            buf = io.BytesIO()
            result_pil.save(buf, format='PNG')
            
            original_name = Path(st.session_state.original_filename)
            download_name = f"{original_name.stem}_clean{original_name.suffix}"
            
            st.download_button(
                label="💾 Download Result",
                data=buf.getvalue(),
                file_name=download_name,
                mime="image/png"
            )
            
            # Show processing stats
            if st.session_state.timing_data:
                timing = st.session_state.timing_data
                
                with st.expander("📊 Processing Details"):
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.metric("Total Time", f"{timing['total_time']:.1f}s")
                        st.metric("Detection Time", f"{timing['detection_time']:.1f}s")
                        st.metric("Consensus Boxes", timing['consensus_boxes_count'])
                    
                    with col_b:
                        if timing['color_time'] is not None:
                            st.metric("TF-IDF Time", f"{timing['color_time']:.1f}s")
                        if timing['inpaint_time'] is not None:
                            st.metric("Inpainting Time", f"{timing['inpaint_time']:.1f}s")
                        
                        failover_type = timing.get('failover_type', 'none')
                        if failover_type != 'none':
                            st.metric("Failover Used", failover_type.title())
            
            # Show mask if available
            if keep_masks and hasattr(st.session_state, 'mask_image') and st.session_state.mask_image is not None:
                with st.expander("🎭 Detection Mask"):
                    st.image(st.session_state.mask_image, caption="Detected Text Regions", use_container_width=True)
        
        else:
            st.info("👆 Upload an image and click 'Remove Text Watermarks' to see results here")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with ❤️ using Streamlit • Powered by LaMa, DocTR, EasyOCR, and EAST</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 