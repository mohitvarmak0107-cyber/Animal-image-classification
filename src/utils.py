"""
Utility helper functions for Animal Image Classification project.

Provides functions for dataset verification and system environment checking.
Designed to be easy to read and explain during college project vivas.
"""

import sys
import platform
from pathlib import Path
from typing import Dict, Any

from src.config import RAW_DATA_DIR, ANIMAL_CLASSES

# Supported image file extensions
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def check_system_environment() -> Dict[str, Any]:
    """
    Checks the local Python and Deep Learning environment.

    Returns:
        dict: Information about Python version, platform, and installed DL libraries.
    """
    info = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "tensorflow_version": None,
        "keras_version": None,
        "streamlit_version": None,
        "gpu_available": False,
    }

    try:
        import tensorflow as tf
        info["tensorflow_version"] = tf.__version__
        gpus = tf.config.list_physical_devices("GPU")
        info["gpu_available"] = len(gpus) > 0
    except ImportError:
        info["tensorflow_version"] = "Not installed"

    try:
        import keras
        info["keras_version"] = keras.__version__
    except ImportError:
        info["keras_version"] = "Not installed"

    try:
        import streamlit as st
        info["streamlit_version"] = st.__version__
    except ImportError:
        info["streamlit_version"] = "Not installed"

    return info


def inspect_raw_dataset(raw_dir: Path = RAW_DATA_DIR) -> Dict[str, int]:
    """
    Scans the data/raw/ directory to check how many images exist
    in each class folder.

    Args:
        raw_dir (Path): Path to the raw data directory.

    Returns:
        dict: Mapping from class name to number of valid images found.
    """
    summary = {}
    for class_name in ANIMAL_CLASSES:
        class_folder = raw_dir / class_name
        if not class_folder.exists():
            summary[class_name] = 0
            continue

        # Count files matching valid image extensions (case-insensitive)
        count = sum(
            1 for file in class_folder.iterdir()
            if file.is_file() and file.suffix.lower() in VALID_IMAGE_EXTENSIONS
        )
        summary[class_name] = count

    return summary
