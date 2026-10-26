"""
Stage 3: Deep Learning CNN Model Training Runner
BTech Mini Project: Animal Image Classification

Run this script to:
1. Verify the prepared dataset and train/val/test splits.
2. Build and compile the Convolutional Neural Network (CNN).
3. Train the model using training data and validate with validation data.
4. Apply EarlyStopping, ModelCheckpoint, and learning rate scheduling.
5. Save the best model to models/animal_cnn_best.keras and class labels.
6. Evaluate on the independent TEST set.
7. Save loss/accuracy curves, confusion matrix, and classification report in results/.
"""

import sys
from pathlib import Path

# Safe Windows console encoding
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    ANIMAL_CLASSES,
    IMAGE_SIZE,
    BATCH_SIZE,
    DEFAULT_EPOCHS,
    MODEL_SAVE_PATH,
    CLASS_INDICES_PATH,
    TRAINING_HISTORY_PATH,
    CONFUSION_MATRIX_PATH,
    TRAINING_METRICS_PATH,
)
from src.data_loader import validate_dataset_structure
from src.model import build_animal_cnn
from src.train import train_animal_model


def run_stage_3():
    print("=" * 72)
    print("  STAGE 3: DEEP LEARNING MODEL TRAINING (CNN)")
    print("  Project: Animal Image Classification (5 Classes)")
    print("=" * 72)

    # Step 1: Verify Dataset Readiness
    print("\n[Step 1] Auditing Dataset Availability...")
    report = validate_dataset_structure(RAW_DATA_DIR, ANIMAL_CLASSES)

    print(f"  * Total Valid Images Detected in data/raw: {report['total_images']}")
    for cls in ANIMAL_CLASSES:
        count = report["class_counts"].get(cls, 0)
        status_tag = "[OK]  " if count >= 10 else "[LOW] "
        print(f"    {status_tag} {cls:<12} : {count} images")

    # Step 2: Handle Missing Dataset
    if not report["is_ready_for_training"]:
        print("\n" + "=" * 72)
        print("  [ERROR] CANNOT START TRAINING: DATASET HAS INSUFFICIENT IMAGES")
        print("=" * 72)
        print("Please run `py -3.12 prepare_dataset.py` first to automatically download")
        print("and prepare the dataset splits, or place at least 15 images in each folder:")
        for cls in ANIMAL_CLASSES:
            print(f"  * {cls:<12} -> {RAW_DATA_DIR / cls}")

        print("\n[Architecture Verification] CNN Model Architecture Preview:")
        print("-" * 72)
        cnn_model = build_animal_cnn(
            input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
            num_classes=len(ANIMAL_CLASSES)
        )
        cnn_model.summary()
        print("-" * 72)
        return

    # Step 3: Run Full Model Training
    print("\n[Step 2] Dataset verified. Launching CNN Training Pipeline...")
    print(f"  * Input Dimension : {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} RGB")
    print(f"  * Batch Size      : {BATCH_SIZE}")
    print(f"  * Target Epochs   : {DEFAULT_EPOCHS} (with EarlyStopping)")
    print(f"  * Target Classes  : {', '.join(ANIMAL_CLASSES)}")

    results = train_animal_model(
        raw_dir=RAW_DATA_DIR,
        processed_dir=PROCESSED_DATA_DIR,
        epochs=DEFAULT_EPOCHS,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
    )

    if results.get("status") != "success":
        print(f"\n[ERROR] Training failed: {results.get('message')}")
        return

    # Step 4: Display Results & Viva Metrics
    print("\n" + "=" * 72)
    print("  STAGE 3: MODEL TRAINING & TEST EVALUATION COMPLETE!")
    print("=" * 72)
    print(f"  * Epochs Trained           : {results['epochs_trained']}")
    print(f"  * Final Training Loss      : {results['final_training_loss']:.4f}")
    print(f"  * Final Training Accuracy  : {results['final_training_accuracy'] * 100:.2f}%")
    print(f"  * Best Validation Loss     : {results['best_validation_loss']:.4f}")
    print(f"  * Best Validation Accuracy : {results['best_validation_accuracy'] * 100:.2f}%")
    print(f"  * Holdout TEST Loss        : {results['test_loss']:.4f}")
    print(f"  * Holdout TEST Accuracy    : {results['test_accuracy'] * 100:.2f}%")
    print("\n  Generated Artifacts:")
    print(f"  [OK] Best Model Saved    : {results['model_path']}")
    print(f"  [OK] Class Indices Saved : {results['class_indices_path']}")
    print(f"  [OK] Loss/Accuracy Curves: {results['history_plot']}")
    print(f"  [OK] Confusion Matrix    : {results['confusion_matrix_plot']}")
    print(f"  [OK] Full Metrics JSON   : {TRAINING_METRICS_PATH}")

    # Confirm model file exists on disk
    if Path(results['model_path']).exists():
        size_mb = Path(results['model_path']).stat().st_size / (1024 * 1024)
        print(f"\n[CONFIRMED] Saved model file verified on disk ({size_mb:.2f} MB).")
        print("[READY] You can now launch the web app with `py -3.12 -m streamlit run app.py`.")


if __name__ == "__main__":
    run_stage_3()
