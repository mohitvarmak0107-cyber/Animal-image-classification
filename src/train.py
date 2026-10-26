"""
Training Pipeline for Animal Image Classification.
Stage 3 & Stage 5: Deep Learning Model Training & Test Set Evaluation

Handles:
1. Model training with Adam optimizer and Categorical Cross-Entropy loss.
2. Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau.
3. Training and validation loss/accuracy curve plotting.
4. Independent evaluation on the holdout TEST set.
5. Multi-class confusion matrix and classification report on test data.
6. Saving the best trained model to models/animal_cnn_best.keras and class index mapping.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from src.config import (
    ANIMAL_CLASSES,
    IMAGE_SIZE,
    BATCH_SIZE,
    RANDOM_SEED,
    DEFAULT_EPOCHS,
    PATIENCE_EARLY_STOPPING,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODEL_SAVE_PATH,
    CLASS_INDICES_PATH,
    TRAINING_HISTORY_PATH,
    CONFUSION_MATRIX_PATH,
    TRAINING_METRICS_PATH,
)
from src.data_loader import get_train_val_test_datasets, validate_dataset_structure
from src.model import build_animal_cnn, build_animal_model


# =====================================================================
# 1. Class Indices Persistence
# =====================================================================

def save_class_indices(
    classes: list = ANIMAL_CLASSES,
    output_path: Path = CLASS_INDICES_PATH
) -> Path:
    """
    Saves a JSON mapping of class names and index mappings.
    Ensures that prediction probabilities during inference match the exact
    label ordering from the dataset loader.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mapping = {
        "classes": classes,
        "class_to_idx": {cls: idx for idx, cls in enumerate(classes)},
        "idx_to_class": {str(idx): cls for idx, cls in enumerate(classes)},
        "num_classes": len(classes),
        "image_size": [IMAGE_SIZE[0], IMAGE_SIZE[1], 3],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=4)
    return output_path


# =====================================================================
# 2. Callbacks Setup
# =====================================================================

def get_training_callbacks(
    model_save_path: Path = MODEL_SAVE_PATH,
    patience: int = PATIENCE_EARLY_STOPPING
) -> list:
    """
    Constructs Keras callbacks for training monitoring.

    Callbacks Explanation:
    - EarlyStopping: Monitors validation loss and stops training if no improvement
      is seen for `patience` consecutive epochs, preventing overfitting.
    - ModelCheckpoint: Saves the best model weights based on lowest validation loss.
    - ReduceLROnPlateau: Lowers the learning rate if validation loss plateaus.
    """
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(model_save_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]
    return callbacks


# =====================================================================
# 3. Training & Validation Visualizations
# =====================================================================

def plot_training_history(
    history: tf.keras.callbacks.History,
    output_path: Path = TRAINING_HISTORY_PATH
) -> Path:
    """
    Plots training vs. validation loss and accuracy curves side by side.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs_range = range(1, len(history.history["loss"]) + 1)
    train_loss = history.history["loss"]
    val_loss = history.history.get("val_loss", [])
    train_acc = history.history["accuracy"]
    val_acc = history.history.get("val_accuracy", [])

    plt.figure(figsize=(14, 5))

    # Subplot 1: Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_loss, label="Training Loss", color="#D0021B", lw=2)
    if val_loss:
        plt.plot(epochs_range, val_loss, label="Validation Loss", color="#4A90E2", lw=2, linestyle="--")
    plt.title("Model Loss Across Epochs", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Loss (Categorical Cross-Entropy)", fontsize=10)
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=":", alpha=0.6)

    # Subplot 2: Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_acc, label="Training Accuracy", color="#50E3C2", lw=2)
    if val_acc:
        plt.plot(epochs_range, val_acc, label="Validation Accuracy", color="#9013FE", lw=2, linestyle="--")
    plt.title("Model Accuracy Across Epochs", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Accuracy", fontsize=10)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list = ANIMAL_CLASSES,
    output_path: Path = CONFUSION_MATRIX_PATH,
    title: str = "Test Set Confusion Matrix",
) -> Path:
    """
    Calculates and plots a multi-class confusion matrix with integer count annotations.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))

    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    class_labels = [c.capitalize() for c in classes]
    plt.xticks(tick_marks, class_labels, rotation=45, fontsize=10)
    plt.yticks(tick_marks, class_labels, fontsize=10)

    # Annotate counts inside cells
    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontweight="bold"
            )

    plt.ylabel("True Class", fontsize=11)
    plt.xlabel("Predicted Class", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


# =====================================================================
# 4. Model Training & Evaluation Orchestrator
# =====================================================================

def train_animal_model(
    raw_dir: Path = RAW_DATA_DIR,
    processed_dir: Path = PROCESSED_DATA_DIR,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = BATCH_SIZE,
    image_size: Tuple[int, int] = IMAGE_SIZE,
    seed: int = RANDOM_SEED,
    model_type: str = "transfer",
) -> Dict[str, Any]:
    """
    Executes the full end-to-end model training workflow:
    1. Loads Train, Validation, and Test datasets.
    2. Compiles Deep Learning model (Transfer Learning via MobileNetV2 or Custom CNN).
    3. Fits model with EarlyStopping and ModelCheckpoint.
    4. Evaluates on holdout Test dataset.
    5. Saves confusion matrix, training curves, metrics, and models/animal_cnn_best.keras.
    """
    # 1. Dataset Integrity Check
    report = validate_dataset_structure(raw_dir)
    if not report["is_ready_for_training"]:
        return {
            "status": "error",
            "message": "Dataset is not ready. All 5 animal class folders must contain valid images.",
            "report": report,
        }

    # 2. Load Train, Val, Test Datasets
    train_ds, val_ds, test_ds, meta = get_train_val_test_datasets(
        processed_dir=processed_dir,
        raw_dir=raw_dir,
        image_size=image_size,
        batch_size=batch_size,
        seed=seed,
    )

    if train_ds is None or val_ds is None or test_ds is None:
        return {
            "status": "error",
            "message": f"Failed to load dataset: {meta.get('message')}",
        }

    # 3. Build & Compile Model
    model = build_animal_model(
        model_type=model_type,
        input_shape=(image_size[0], image_size[1], 3),
        num_classes=len(ANIMAL_CLASSES),
    )

    # 4. Save Class Indices
    class_idx_path = save_class_indices(ANIMAL_CLASSES, CLASS_INDICES_PATH)

    # 5. Callbacks
    callbacks = get_training_callbacks(MODEL_SAVE_PATH, patience=PATIENCE_EARLY_STOPPING)

    # 6. Fit Model
    print(f"\n[INFO] Starting CNN training for up to {epochs} epochs (with EarlyStopping)...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Ensure best model file exists and is confirmed saved
    if not MODEL_SAVE_PATH.exists():
        model.save(str(MODEL_SAVE_PATH))

    # 7. Generate Training Curves
    history_plot = plot_training_history(history, TRAINING_HISTORY_PATH)

    # 8. Evaluate on Independent Test Set
    print("\n[INFO] Evaluating best model on holdout TEST dataset...")
    # Load the best saved model weights for evaluation
    best_model = tf.keras.models.load_model(str(MODEL_SAVE_PATH))
    test_loss, test_acc = best_model.evaluate(test_ds, verbose=0)
    val_loss, val_acc = best_model.evaluate(val_ds, verbose=0)

    # Collect predictions and ground truth labels for test set confusion matrix
    y_true_list = []
    y_pred_list = []
    for images, labels in test_ds:
        preds = best_model.predict(images, verbose=0)
        y_true_list.extend(np.argmax(labels.numpy(), axis=1))
        y_pred_list.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    cm_plot = plot_confusion_matrix(y_true, y_pred, ANIMAL_CLASSES, CONFUSION_MATRIX_PATH)
    cls_report = classification_report(
        y_true, y_pred,
        target_names=ANIMAL_CLASSES,
        output_dict=True,
        zero_division=0,
    )

    # 9. Save Real Metrics Summary
    metrics_summary = {
        "status": "success",
        "epochs_trained": len(history.history["loss"]),
        "final_training_loss": float(history.history["loss"][-1]),
        "final_training_accuracy": float(history.history["accuracy"][-1]),
        "best_validation_loss": float(val_loss),
        "best_validation_accuracy": float(val_acc),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "model_path": str(MODEL_SAVE_PATH),
        "class_indices_path": str(class_idx_path),
        "history_plot": str(history_plot),
        "confusion_matrix_plot": str(cm_plot),
        "classification_report": cls_report,
    }

    with open(TRAINING_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=4)

    return metrics_summary
