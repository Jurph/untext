"""Setup configuration for untextre package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="untextre",
    version="0.1.0",
    author="Jurph",
    author_email="jurph@example.com",  # TODO: Update with actual email
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
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "python-doctr>=0.6.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "Pillow>=9.5.0,<10.0",
        "scikit-image>=0.21.0",  # For image processing
        "tqdm>=4.65.0",  # For progress bars
        "matplotlib>=3.7.0",  # For visualization
        "kornia>=0.6.0",
        "albumentations>=1.3.0",
        "omegaconf>=2.3.0",
        "simple-lama-inpainting==0.1.2",
        "streamlit-image-annotation>=0.4.0",  # For force detection boxes
    ],
    extras_require={
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