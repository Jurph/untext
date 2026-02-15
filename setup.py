"""Setup configuration for untextre package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="untextre",
    version="0.1.0",
    author="Jurph",
    description="Remove text-based watermarks from images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jurph/untextre",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    # NOTE: torch and torchvision are intentionally excluded.
    # pip install will pull CPU-only builds from PyPI, overwriting any
    # CUDA-enabled installation.  See README.md for PyTorch setup.
    install_requires=[
        "python-doctr>=0.6.0",
        "Pillow>=9.5.0",
        "simple-lama-inpainting==0.1.2",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "scikit-image>=0.21.0",
        "easyocr>=1.7.0",
    ],
    extras_require={
        "web": [
            "streamlit>=1.28.0",
            "streamlit-drawable-canvas-fix>=0.9.8",
            "streamlit-image-annotation>=0.4.0",
            "streamlit-js-eval>=0.1.5",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "untextre=untextre.cli:main",
        ],
    },
)
