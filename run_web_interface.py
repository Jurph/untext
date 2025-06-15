#!/usr/bin/env python3
"""Startup script for the UnTextre web interface.

This script launches the Streamlit web interface with optimized settings
for the text watermark removal tool.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Launch the Streamlit web interface."""
    
    # Check if streamlit is installed
    try:
        import streamlit
    except ImportError:
        print("❌ Streamlit not found. Please install requirements:")
        print("   pip install -r requirements_streamlit.txt")
        sys.exit(1)
    
    # Get the path to the streamlit app
    app_path = Path(__file__).parent / "streamlit_app.py"
    
    if not app_path.exists():
        print(f"❌ Streamlit app not found at {app_path}")
        sys.exit(1)
    
    print("🚀 Starting UnTextre Web Interface...")
    print("📱 The web interface will open in your browser automatically")
    print("🛑 Press Ctrl+C to stop the server")
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
        print("\n👋 Shutting down UnTextre Web Interface")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 