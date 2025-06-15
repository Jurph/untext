#!/usr/bin/env python3
"""Streamlit web interface for untextre text watermark removal.

This provides a drag-and-drop web interface for removing text watermarks
from images using the untextre pipeline. All models are loaded at startup
for fast processing.

Usage:
    streamlit run streamlit_app.py
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import time
from pathlib import Path
from PIL import Image
import io

# Import our untextre modules
from untextre.utils import load_image, save_image
from untextre.preprocessor import preprocess_image
from untextre.cli import initialize_consensus_models, run_consensus_detection, process_single_image
from untextre.inpaint import initialize_lama_model

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
        initialize_lama_model(device="cuda")
        
    return True

def process_image_streamlit(image_bytes, confidence_threshold, granularity, method, keep_masks):
    """Process an uploaded image and return the result."""
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as input_file:
        input_file.write(image_bytes)
        input_path = Path(input_file.name)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        
        try:
            # Process the image using our existing pipeline
            timing_data = process_single_image(
                image_path=input_path,
                output_dir=output_dir,
                target_color=None,  # Use spatial TF-IDF
                keep_masks=keep_masks,
                method=method,
                maskfile=None,
                confidence_threshold=confidence_threshold,
                granularity=granularity,
                forced_bbox=None
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
            st.error(f"Error processing image: {str(e)}")
            return None, None, None
        finally:
            # Clean up input file
            input_path.unlink(missing_ok=True)

def main():
    """Main Streamlit application."""
    
    # Initialize models
    models_loaded = initialize_models()
    
    # Title and description
    st.title("🎨 UnTextre - Text Watermark Removal")
    st.markdown("""
    Upload an image with text watermarks and watch them disappear! This tool uses advanced AI models 
    to detect and remove text overlays while preserving the underlying image.
    """)
    
    # Sidebar for options
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
        
        # Debug options
        st.subheader("Debug Options")
        keep_masks = st.checkbox(
            "Show detection masks",
            value=False,
            help="Display the detected text regions"
        )
        
        # Model status
        st.subheader("🤖 Model Status")
        if models_loaded:
            st.success("✅ All models loaded")
            st.info("Ready for processing!")
        else:
            st.error("❌ Models not loaded")
    
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📤 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Drag and drop an image or click to browse"
        )
        
        if uploaded_file is not None:
            # Display original image
            original_image = Image.open(uploaded_file)
            st.image(original_image, caption="Original Image", use_container_width=True)
            
            # Image info
            st.info(f"📊 Size: {original_image.size[0]}×{original_image.size[1]} pixels")
            
            # Process button
            if st.button("🚀 Remove Text Watermarks", type="primary"):
                start_time = time.time()
                
                with st.spinner("Processing image... This may take a few seconds."):
                    # Get image bytes
                    image_bytes = uploaded_file.getvalue()
                    
                    # Process the image
                    result_image, mask_image, timing_data = process_image_streamlit(
                        image_bytes, confidence_threshold, granularity, method, keep_masks
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