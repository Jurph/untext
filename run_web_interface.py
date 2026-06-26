#!/usr/bin/env python3
"""Startup script for the UnTextre web interface.

This script launches the Streamlit web interface with optimized settings
for the text watermark removal tool.
"""

import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path


def main():
    """Launch the Streamlit web interface."""

    # Check if streamlit is installed
    if find_spec("streamlit") is None:
        print("[ERROR] Streamlit not found. Please install the web dependencies:")
        print("   uv sync --extra web")
        sys.exit(1)

    # Get the path to the streamlit app
    app_path = Path(__file__).parent / "streamlit_app.py"

    if not app_path.exists():
        print(f"[ERROR] Streamlit app not found at {app_path}")
        sys.exit(1)

    print("=" * 70)
    print("[START] Starting UnTextre Web Interface...")
    print("=" * 70)
    print()
    print("[INFO] The app will open in your browser automatically at:")
    print("   http://localhost:8501")
    print()
    print("[INFO] First-time startup: ~30-60 seconds (downloading/loading AI models)")
    print("   - EAST text detection model (~80MB)")
    print("   - DocTR document OCR model (~100MB)")
    print("   - EasyOCR recognition model (~120MB)")
    print("   - LaMa inpainting model (~200MB)")
    print()
    print("[INFO] After initial load: Processing is fast (~2-5 seconds per image)")
    print()
    print("[STOP] To stop the server: Press Ctrl+C in this terminal")
    print("=" * 70)
    print()

    # Launch streamlit with optimized settings
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.maxUploadSize", "50",  # 50MB max upload
        "--server.maxMessageSize", "50",  # 50MB max message
        "--browser.gatherUsageStats", "false",  # Disable telemetry
        "--theme.base", "dark",  # Dark theme
    ]

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n[STOP] Shutting down UnTextre Web Interface")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Error starting Streamlit: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
