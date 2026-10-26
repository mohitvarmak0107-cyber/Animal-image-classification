"""
Unit and Integration Tests for Stage 4 Inference & Preprocessing Pipeline.
Tests:
1. Preprocessing image conversion (RGBA, Grayscale, irregular resolutions -> (1, 128, 128, 3)).
2. Pixel normalization strictly in [0.0, 1.0].
3. Model prediction execution, probability formatting, and top class selection.
4. Graceful handling of missing model files.
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ANIMAL_CLASSES
from src.model import build_animal_cnn
from src.inference import (
    preprocess_image,
    predict_animal,
    load_model_and_labels,
)


def test_inference_module():
    print("--- Running Stage 4 Inference Unit Tests ---")

    # 1. Test Preprocessing with standard RGB
    rgb_img = Image.new("RGB", (300, 200), color=(120, 80, 200))
    tensor = preprocess_image(rgb_img, target_size=(128, 128))
    assert tensor.shape == (1, 128, 128, 3), f"Incorrect shape: {tensor.shape}"
    assert float(np.min(tensor)) >= 0.0 and float(np.max(tensor)) <= 1.0
    print("[TEST PASSED] Standard RGB image preprocessed to (1, 128, 128, 3) in [0, 1].")

    # 2. Test Preprocessing with RGBA (Transparency)
    rgba_img = Image.new("RGBA", (50, 50), color=(255, 0, 0, 128))
    tensor_rgba = preprocess_image(rgba_img, target_size=(128, 128))
    assert tensor_rgba.shape == (1, 128, 128, 3), "RGBA failed to convert to 3 channels"
    print("[TEST PASSED] RGBA image converted to 3-channel RGB.")

    # 3. Test Preprocessing with Grayscale
    gray_img = Image.new("L", (80, 80), color=128)
    tensor_gray = preprocess_image(gray_img, target_size=(128, 128))
    assert tensor_gray.shape == (1, 128, 128, 3), "Grayscale failed to convert to 3 channels"
    print("[TEST PASSED] Grayscale image converted to 3-channel RGB.")

    # 4. Test Missing Model Handling
    missing_path = PROJECT_ROOT / "models" / "non_existent_model.keras"
    model, labels, err = load_model_and_labels(missing_path)
    assert model is None
    assert err is not None
    print("[TEST PASSED] Missing model gracefully handled without exception.")

    # 5. Test Prediction with active CNN
    cnn = build_animal_cnn(input_shape=(128, 128, 3), num_classes=5)
    pred_result = predict_animal(cnn, tensor, ANIMAL_CLASSES)

    assert pred_result["predicted_class"] in ANIMAL_CLASSES
    assert 0.0 <= pred_result["confidence"] <= 100.0
    assert len(pred_result["all_scores"]) == 5
    total_percent = sum(pred_result["all_scores"].values())
    assert abs(total_percent - 100.0) < 0.1, f"Probabilities don't sum to ~100%: {total_percent}"
    print(f"[TEST PASSED] Prediction works: Predicted '{pred_result['predicted_class']}' with {pred_result['confidence']:.2f}% confidence.")

    # 6. Test Class Mapping Order and Trained Model on Elephant Test Image
    trained_model_path = PROJECT_ROOT / "models" / "animal_cnn_best.keras"
    indices_path = PROJECT_ROOT / "models" / "class_indices.json"
    if trained_model_path.exists():
        t_model, t_labels, t_err = load_model_and_labels(trained_model_path, indices_path)
        assert t_err is None
        assert t_labels == ["cats", "dogs", "elephants", "horses", "lions"]

        # Test on real holdout test image of elephant
        test_ele_path = PROJECT_ROOT / "data" / "processed" / "test" / "elephants" / "Elephant.jpg"
        if test_ele_path.exists():
            ele_img = Image.open(test_ele_path)
            ele_tensor = preprocess_image(ele_img, target_size=(128, 128))
            ele_pred = predict_animal(t_model, ele_tensor, t_labels)
            assert ele_pred["predicted_class"] == "elephants", f"Elephant predicted as {ele_pred['predicted_class']}"
            assert ele_pred["confidence"] >= 80.0, f"Confidence too low: {ele_pred['confidence']}"
            print(f"[TEST PASSED] Elephant test image verified: '{ele_pred['predicted_class']}' ({ele_pred['confidence']:.2f}% confidence).")

    print("--- All Stage 4 Inference Unit Tests Passed Successfully! ---")


if __name__ == "__main__":
    test_inference_module()
