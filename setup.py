"""
Setup configuration for Deepfake Detection System
CO3201 Final Year Project
"""

# setuptools: standard Python packaging library
from setuptools import setup, find_packages # find_packages: automatically finds all folders with __init__.py
from pathlib import Path # Path: modern cross platform file path handling

# Reads requirements from requirements.txt and converts to a list
# Path(__file__).parent gets the folder containing this setup.py file
requirements_path = Path(__file__).parent / "requirements.txt"
with open(requirements_path) as f:
    requirements = [
        line.strip().split('#')[0].strip()
        for line in f
        if line.strip()
        and not line.startswith('#')
        and not line.startswith('-')
        # Skips blank lines, comments, and pip directives like --extra-index-url
        # (setuptools install_requires only accepts package specs, not pip flags)
    ]

# setup() makes this project installable as a Python package via "pip install -e ."
setup(
    name="deepfake-detection",  # Package name used by pip
    version="0.1.0",
    description="Deep Learning System for Deepfake Video Detection",
    author="Raul Blanco Vazquez",

    # find_packages scans for folders containing __init__.py
    # 'src' is the src folder itself and 'src.*' is all the subfolders (src.models, src.preprocessing, src.training)
    packages=find_packages(include=['src', 'src.*']),

    python_requires='>=3.8',  # Minimum Python version (PyTorch 2.0 requires 3.8+)

    # Automatically installs everything from requirements.txt when someone runs pip install
    install_requires=requirements
)
