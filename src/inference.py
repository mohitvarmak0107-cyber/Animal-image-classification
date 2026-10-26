"""
Inference and Preprocessing Utilities for Animal Image Classification.
Stage 4: Streamlit Web Application

Provides reusable, tested functions for:
1. Loading the trained CNN model and class label mapping.
2. Preprocessing uploaded images (RGB conversion, 128x128 resize, [0, 1] normalization).
3. Running model inference and formatting prediction confidence scores.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
from PIL import Image

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf

from src.config import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    IMAGE_SIZE,
    ANIMAL_CLASSES,
    MODEL_SAVE_PATH,
    CLASS_INDICES_PATH,
)


def load_model_and_labels(
    model_path: Path = MODEL_SAVE_PATH,
    indices_path: Path = CLASS_INDICES_PATH
) -> Tuple[Optional[tf.keras.Model], Optional[List[str]], Optional[str]]:
    """
    Loads the trained Keras model and class indices from disk.

    Returns:
        (model, class_names, error_message)
        If the model or labels file is missing, model will be None.
    """
    if not model_path.exists():
        return None, None, f"Model file not found at '{model_path}'. Please train the model in Stage 3 first."

    try:
        model = tf.keras.models.load_model(str(model_path))
    except Exception as e:
        return None, None, f"Failed to load model: {str(e)}"

    # Load class labels if available, otherwise default to config classes
    class_names = ANIMAL_CLASSES
    if indices_path.exists():
        try:
            with open(indices_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "classes" in data and isinstance(data["classes"], list):
                    class_names = data["classes"]
        except Exception:
            class_names = ANIMAL_CLASSES

    return model, class_names, None


def preprocess_image(
    image: Image.Image,
    target_size: Tuple[int, int] = IMAGE_SIZE
) -> np.ndarray:
    """
    Preprocesses a PIL Image to match the exact input requirements of the trained CNN:
    1. Converts color mode to 3-channel RGB (handles RGBA, grayscale, CMYK, etc.)
    2. Resizes image to target resolution (128x128) using high-quality bilinear interpolation.
    3. Converts image pixels to float32 NumPy array.
    4. Normalizes pixel values from [0, 255] to [0.0, 1.0].
    5. Adds batch dimension: shape (1, 128, 128, 3).

    Args:
        image: PIL.Image instance
        target_size: Tuple of (height, width) -> (128, 128)

    Returns:
        np.ndarray with shape (1, 128, 128, 3) and values in [0.0, 1.0].
    """
    # 1. Ensure 3-channel RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    # 2. Resize to 128x128
    resized_img = image.resize((target_size[1], target_size[0]), Image.Resampling.BILINEAR)

    # 3. Convert to float32 array
    img_array = np.array(resized_img, dtype=np.float32)

    # 4. Normalize to [0.0, 1.0]
    normalized_array = img_array / 255.0

    # 5. Expand dimensions for batch format: (1, 128, 128, 3)
    batch_tensor = np.expand_dims(normalized_array, axis=0)
    return batch_tensor


def predict_animal(
    model: tf.keras.Model,
    preprocessed_tensor: np.ndarray,
    class_names: List[str] = ANIMAL_CLASSES
) -> Dict[str, Any]:
    """
    Runs model inference on preprocessed image tensor and extracts prediction scores.

    Args:
        model: Loaded tf.keras.Model
        preprocessed_tensor: ndarray of shape (1, 128, 128, 3)
        class_names: List of 5 animal category names

    Returns:
        Dict containing predicted_class, confidence_percentage, and all class probabilities.
    """
    # Model returns shape (1, 5) with Softmax probabilities
    raw_predictions = model.predict(preprocessed_tensor, verbose=0)
    probabilities = raw_predictions[0]

    predicted_idx = int(np.argmax(probabilities))
    predicted_class = class_names[predicted_idx]
    confidence_score = float(probabilities[predicted_idx]) * 100.0

    # Format all class scores sorted by highest probability first
    all_scores = {}
    for idx, cls_name in enumerate(class_names):
        all_scores[cls_name] = float(probabilities[idx]) * 100.0

    sorted_scores = dict(sorted(all_scores.items(), key=lambda item: item[1], reverse=True))

    # Determine confidence reliability level
    if confidence_score >= 75.0:
        confidence_level = "High Confidence"
        is_low_confidence = False
    elif confidence_score >= 50.0:
        confidence_level = "Moderate Confidence"
        is_low_confidence = False
    else:
        confidence_level = "Low Confidence / Uncertain"
        is_low_confidence = True

    return {
        "predicted_class": predicted_class,
        "predicted_index": predicted_idx,
        "confidence": confidence_score,
        "confidence_level": confidence_level,
        "is_low_confidence": is_low_confidence,
        "all_scores": sorted_scores,
        "raw_probabilities": probabilities.tolist(),
    }
