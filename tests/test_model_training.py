"""
Unit and Integration Test for Stage 3 (Model Training Pipeline).
Tests:
1. Model construction and compilation.
2. Model fit execution for 2 epochs on mock images.
3. Callback execution and model artifact generation (.keras).
4. Training history plot creation.
5. Confusion matrix and classification report calculation.
"""

import sys
import shutil
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ANIMAL_CLASSES
from src.model import build_animal_cnn
from src.train import (
    save_class_indices,
    train_animal_model,
)


def test_training_pipeline_end_to_end():
    print("--- Running Model Training Pipeline Test in Temporary Sandbox ---")
    temp_dir = Path(tempfile.mkdtemp(prefix="test_animal_train_"))

    try:
        raw_mock_dir = temp_dir / "raw"
        models_mock_dir = temp_dir / "models"
        figures_mock_dir = temp_dir / "figures"
        results_mock_dir = temp_dir / "results"

        models_mock_dir.mkdir(parents=True, exist_ok=True)
        figures_mock_dir.mkdir(parents=True, exist_ok=True)
        results_mock_dir.mkdir(parents=True, exist_ok=True)

        # Create 5 class subfolders with 8 sample images each (40 total)
        for cls in ANIMAL_CLASSES:
            cls_folder = raw_mock_dir / cls
            cls_folder.mkdir(parents=True, exist_ok=True)
            for i in range(8):
                arr = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
                img = Image.fromarray(arr)
                img.save(cls_folder / f"{cls}_{i}.jpg")

        # 1. Test Model Architecture
        model = build_animal_cnn(input_shape=(128, 128, 3), num_classes=5)
        assert model.input_shape == (None, 128, 128, 3)
        assert model.output_shape == (None, 5)
        print("[TEST PASSED] CNN architecture created with input (None, 128, 128, 3) and output (None, 5).")

        # 2. Test Class Indices
        class_idx_path = models_mock_dir / "class_indices.json"
        save_class_indices(ANIMAL_CLASSES, class_idx_path)
        assert class_idx_path.exists()
        print("[TEST PASSED] Class index mapping JSON saved successfully.")

        # 3. Test Full Training with 2 Epochs
        model_save_path = models_mock_dir / "test_cnn.keras"
        history_plot_path = figures_mock_dir / "test_history.png"
        cm_plot_path = figures_mock_dir / "test_cm.png"
        metrics_json_path = results_mock_dir / "test_metrics.json"

        # Monkey-patch config paths for isolated test
        import src.train as train_mod
        orig_model_path = train_mod.MODEL_SAVE_PATH
        orig_indices_path = train_mod.CLASS_INDICES_PATH
        orig_history_path = train_mod.TRAINING_HISTORY_PATH
        orig_cm_path = train_mod.CONFUSION_MATRIX_PATH
        orig_metrics_path = train_mod.TRAINING_METRICS_PATH

        train_mod.MODEL_SAVE_PATH = model_save_path
        train_mod.CLASS_INDICES_PATH = class_idx_path
        train_mod.TRAINING_HISTORY_PATH = history_plot_path
        train_mod.CONFUSION_MATRIX_PATH = cm_plot_path
        train_mod.TRAINING_METRICS_PATH = metrics_json_path

        try:
            results = train_animal_model(
                raw_dir=raw_mock_dir,
                processed_dir=temp_dir / "processed",
                epochs=2,
                batch_size=8,
                image_size=(128, 128),
            )
            assert results["status"] == "success", f"Training failed: {results}"
            assert model_save_path.exists(), "Trained model was not saved"
            assert history_plot_path.exists(), "Training history plot was not saved"
            assert cm_plot_path.exists(), "Confusion matrix plot was not saved"
            assert metrics_json_path.exists(), "Metrics JSON was not saved"
            print(f"[TEST PASSED] Model trained for {results['epochs_trained']} epochs.")
            print(f"[TEST PASSED] Saved model artifact: {model_save_path.stat().st_size} bytes.")
            print(f"[TEST PASSED] Test accuracy: {results['test_accuracy']:.2f}")
            print(f"[TEST PASSED] Plots and metrics generated.")
        finally:
            # Restore paths
            train_mod.MODEL_SAVE_PATH = orig_model_path
            train_mod.CLASS_INDICES_PATH = orig_indices_path
            train_mod.TRAINING_HISTORY_PATH = orig_history_path
            train_mod.CONFUSION_MATRIX_PATH = orig_cm_path
            train_mod.TRAINING_METRICS_PATH = orig_metrics_path

        print("--- All Stage 3 Model Training Tests Passed Successfully! ---")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_training_pipeline_end_to_end()
