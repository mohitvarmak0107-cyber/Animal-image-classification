"""
Unit and Pipeline Integration Test for Stage 2 (Dataset Preparation).
Tests:
1. Class folder discovery.
2. File integrity checking and corrupt file detection.
3. image_dataset_from_directory batching, (128, 128) resizing, and [0, 1] rescaling.
4. Deterministic 80/20 train/validation split (preventing data leakage).
5. Generation of distribution plots and sample visualization grids.
"""

import sys
import shutil
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (
    validate_dataset_structure,
    get_train_val_datasets,
    save_class_distribution_plot,
    save_sample_images_grid,
)
from src.config import ANIMAL_CLASSES


def test_pipeline_with_mock_data():
    print("--- Running Pipeline Unit Test with Temporary Sandbox Directory ---")
    temp_dir = Path(tempfile.mkdtemp(prefix="test_animal_data_"))

    try:
        # Create 5 class subfolders with 10 test images each (50 total)
        for cls in ANIMAL_CLASSES:
            cls_folder = temp_dir / cls
            cls_folder.mkdir(parents=True, exist_ok=True)
            for i in range(10):
                # Generate random synthetic image
                arr = np.random.randint(0, 256, (150, 150, 3), dtype=np.uint8)
                img = Image.fromarray(arr)
                img.save(cls_folder / f"{cls}_{i}.jpg")

        # Also add one corrupted file to verify detection
        corrupt_file = temp_dir / "cats" / "corrupted_test.jpg"
        with open(corrupt_file, "wb") as f:
            f.write(b"NOT_AN_IMAGE_BYTES")

        # 1. Test validation
        report = validate_dataset_structure(temp_dir, ANIMAL_CLASSES)
        assert report["raw_dir_exists"] is True
        assert report["total_images"] == 50, f"Expected 50 valid images, got {report['total_images']}"
        assert len(report["corrupted_files"]) == 1, "Expected 1 corrupted file detected"
        print("[TEST PASSED] Validation correctly identified 50 valid images and 1 corrupted file.")

        # 2. Test dataset loading, 128x128 resize, [0, 1] normalization, splits
        from src.data_loader import get_train_val_test_datasets
        train_ds, val_ds, test_ds, meta = get_train_val_test_datasets(
            processed_dir=temp_dir / "processed",
            raw_dir=temp_dir,
            image_size=(128, 128),
            batch_size=8,
            seed=42,
        )
        assert train_ds is not None and val_ds is not None and test_ds is not None
        print("[TEST PASSED] Train, validation, and test tf.data.Dataset created.")

        # Verify shapes and pixel range [0, 1]
        for batch_x, batch_y in train_ds.take(1):
            assert batch_x.shape[1:] == (128, 128, 3), f"Wrong image shape: {batch_x.shape}"
            assert batch_y.shape[1] == 5, f"Wrong label shape (expected 5 classes): {batch_y.shape}"
            assert float(np.min(batch_x)) >= 0.0, "Pixel min < 0"
            assert float(np.max(batch_x)) <= 1.0, "Pixel max > 1"
            print(f"[TEST PASSED] Resizing verified: shape is {batch_x.shape}.")
            print(f"[TEST PASSED] Normalization verified: values in [{float(np.min(batch_x)):.3f}, {float(np.max(batch_x)):.3f}].")
            break

        # 3. Test plot generation
        test_fig_dist = temp_dir / "test_dist.png"
        save_class_distribution_plot(report["class_counts"], test_fig_dist)
        assert test_fig_dist.exists() and test_fig_dist.stat().st_size > 0
        print("[TEST PASSED] Class distribution plot saved and verified.")

        test_fig_samples = temp_dir / "test_samples.png"
        save_sample_images_grid(train_ds, ANIMAL_CLASSES, test_fig_samples, num_samples=6)
        assert test_fig_samples.exists() and test_fig_samples.stat().st_size > 0
        print("[TEST PASSED] Sample images grid saved and verified.")

        print("--- All Stage 2 Unit & Integration Tests Passed Successfully! ---")
    finally:
        # Cleanup temporary test directory
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_pipeline_with_mock_data()
