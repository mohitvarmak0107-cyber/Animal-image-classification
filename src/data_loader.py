"""
Data Preparation, Splitting, and Loading Pipeline for Animal Image Classification.
Stage 2 & Stage 5: Leak-Free Dataset Management

This module handles:
1. Verification of class folders and image integrity (Pillow header & RGB validation).
2. Detection and reporting of corrupted, 0-byte, and duplicate image files.
3. Reproducible partitioning into Train (70%), Validation (15%), and Test (15%) splits
   saved in data/processed/ to strictly prevent data leakage.
4. Loading batched tf.data.Dataset pipelines for train, validation, and testing.
5. Pixel resizing to 128x128 and normalization to [0.0, 1.0].
6. Generating distribution plots, sample image grids, and JSON summaries.
"""

import os
import json
import shutil
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import tensorflow as tf

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
    ANIMAL_CLASSES,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    IMAGE_SIZE,
    BATCH_SIZE,
    TRAIN_SPLIT,
    VAL_SPLIT,
    TEST_SPLIT,
    RANDOM_SEED,
    DATASET_SUMMARY_PATH,
    SAMPLE_IMAGES_PATH,
    CLASS_DISTRIBUTION_PATH,
)

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# =====================================================================
# 1. Dataset Integrity & File Validation
# =====================================================================

def calculate_file_hash(filepath: Path) -> str:
    """Calculates SHA-256 hash of a file to detect duplicates."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_dataset_structure(
    raw_dir: Path = RAW_DATA_DIR,
    classes: List[str] = ANIMAL_CLASSES,
    min_samples_per_class: int = 5
) -> Dict[str, Any]:
    """
    Scans the raw data directory to verify class folders, validate image readability,
    flag corrupted/unsupported files, and detect duplicates.

    Returns:
        Structured audit report.
    """
    report = {
        "raw_dir_exists": raw_dir.exists(),
        "classes": classes,
        "folder_status": {},
        "class_counts": {cls: 0 for cls in classes},
        "valid_files": {cls: [] for cls in classes},
        "total_images": 0,
        "corrupted_files": [],
        "invalid_format_files": [],
        "duplicate_files": [],
        "is_ready_for_training": False,
    }

    if not raw_dir.exists():
        return report

    seen_hashes = {}

    for cls in classes:
        folder = raw_dir / cls
        folder_exists = folder.is_dir()
        report["folder_status"][cls] = folder_exists

        if not folder_exists:
            continue

        for file in sorted(folder.iterdir()):
            if file.name.startswith(".") or file.name == ".gitkeep":
                continue

            if not file.is_file():
                continue

            # Check extension
            ext = file.suffix.lower()
            if ext not in VALID_IMAGE_EXTENSIONS:
                report["invalid_format_files"].append(str(file))
                continue

            # Verify integrity with Pillow
            try:
                if file.stat().st_size == 0:
                    report["corrupted_files"].append(f"{file} (0 bytes)")
                    continue

                with Image.open(file) as img:
                    img.verify()

                # Re-open to verify raster decoding and color mode
                with Image.open(file) as img:
                    img.convert("RGB")

                # Check for duplicates
                f_hash = calculate_file_hash(file)
                if f_hash in seen_hashes:
                    report["duplicate_files"].append((str(file), seen_hashes[f_hash]))
                else:
                    seen_hashes[f_hash] = str(file)

                report["valid_files"][cls].append(file)

            except Exception as err:
                report["corrupted_files"].append(f"{file} ({str(err)})")

        report["class_counts"][cls] = len(report["valid_files"][cls])

    report["total_images"] = sum(report["class_counts"].values())
    report["is_ready_for_training"] = (
        report["total_images"] > 0 and
        all(report["class_counts"][c] >= min_samples_per_class for c in classes)
    )
    return report


# =====================================================================
# 2. Leak-Free Train / Val / Test Partitioning
# =====================================================================

def prepare_dataset_splits(
    raw_dir: Path = RAW_DATA_DIR,
    processed_dir: Path = PROCESSED_DATA_DIR,
    train_ratio: float = TRAIN_SPLIT,
    val_ratio: float = VAL_SPLIT,
    test_ratio: float = TEST_SPLIT,
    seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """
    Partitions valid images from data/raw/ into:
      - data/processed/train/<class>/
      - data/processed/val/<class>/
      - data/processed/test/<class>/

    Data Leakage Prevention:
    - Fixed deterministic pseudo-random shuffling per class (seed=42).
    - Every image is copied to exactly ONE split directory.
    - Zero overlap across train, validation, and test sets.
    """
    report = validate_dataset_structure(raw_dir)
    if not report["is_ready_for_training"]:
        raise ValueError(
            "Cannot prepare dataset splits: one or more classes have insufficient valid images.\n"
            f"Class counts: {report['class_counts']}"
        )

    # Clean existing processed folders to ensure no stale data
    train_dir = processed_dir / "train"
    val_dir = processed_dir / "val"
    test_dir = processed_dir / "test"

    for split_dir in [train_dir, val_dir, test_dir]:
        if split_dir.exists():
            shutil.rmtree(split_dir)
        for cls in ANIMAL_CLASSES:
            (split_dir / cls).mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(seed)
    split_stats = {
        "train": {cls: 0 for cls in ANIMAL_CLASSES},
        "val": {cls: 0 for cls in ANIMAL_CLASSES},
        "test": {cls: 0 for cls in ANIMAL_CLASSES},
    }

    manifest = []

    for cls in ANIMAL_CLASSES:
        files = list(report["valid_files"][cls])
        # Sort first for deterministic ordering before random shuffle
        files.sort(key=lambda p: p.name)
        rng.shuffle(files)

        total_cls = len(files)
        n_train = int(total_cls * train_ratio)
        n_val = int(total_cls * val_ratio)
        # Ensure test gets the remaining files (at least 1 if total >= 5)
        n_test = total_cls - n_train - n_val

        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]

        for f in train_files:
            dest = train_dir / cls / f.name
            shutil.copy2(f, dest)
            manifest.append({"file": f.name, "class": cls, "split": "train"})
            split_stats["train"][cls] += 1

        for f in val_files:
            dest = val_dir / cls / f.name
            shutil.copy2(f, dest)
            manifest.append({"file": f.name, "class": cls, "split": "val"})
            split_stats["val"][cls] += 1

        for f in test_files:
            dest = test_dir / cls / f.name
            shutil.copy2(f, dest)
            manifest.append({"file": f.name, "class": cls, "split": "test"})
            split_stats["test"][cls] += 1

    manifest_path = processed_dir / "split_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "seed": seed,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "counts": split_stats,
            "manifest_size": len(manifest),
        }, f, indent=4)

    return {
        "status": "success",
        "split_stats": split_stats,
        "manifest_path": str(manifest_path),
        "total_split_images": len(manifest),
    }


# =====================================================================
# 3. Keras Dataset Loading (Train, Val, Test)
# =====================================================================

def get_train_val_test_datasets(
    processed_dir: Path = PROCESSED_DATA_DIR,
    raw_dir: Path = RAW_DATA_DIR,
    image_size: Tuple[int, int] = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
    seed: int = RANDOM_SEED,
) -> Tuple[Optional[tf.data.Dataset], Optional[tf.data.Dataset], Optional[tf.data.Dataset], Dict[str, Any]]:
    """
    Loads normalized tf.data.Dataset pipelines for Train, Validation, and Test sets.
    If data/processed/ is missing or empty, automatically partitions data/raw/.
    """
    train_dir = processed_dir / "train"
    val_dir = processed_dir / "val"
    test_dir = processed_dir / "test"

    # Check if processed splits exist and have images
    train_has_images = train_dir.exists() and any(
        any((train_dir / cls).iterdir()) for cls in ANIMAL_CLASSES if (train_dir / cls).exists()
    )

    if not train_has_images:
        print("[INFO] Processed dataset splits not found. Preparing splits from data/raw/...")
        try:
            prepare_dataset_splits(raw_dir=raw_dir, processed_dir=processed_dir, seed=seed)
        except Exception as e:
            return None, None, None, {
                "status": "error",
                "message": f"Failed to prepare dataset splits: {str(e)}",
            }

    # Load from processed split folders
    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory=str(train_dir),
        labels="inferred",
        label_mode="categorical",
        class_names=ANIMAL_CLASSES,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=seed,
        interpolation="bilinear",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        directory=str(val_dir),
        labels="inferred",
        label_mode="categorical",
        class_names=ANIMAL_CLASSES,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False,
        interpolation="bilinear",
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        directory=str(test_dir),
        labels="inferred",
        label_mode="categorical",
        class_names=ANIMAL_CLASSES,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False,
        interpolation="bilinear",
    )

    # Pixel Normalization [0.0, 1.0]
    rescaling_layer = tf.keras.layers.Rescaling(1.0 / 255.0)
    train_ds = train_ds.map(lambda x, y: (rescaling_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (rescaling_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(lambda x, y: (rescaling_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    # In-memory caching and prefetching
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    meta = {
        "status": "success",
        "classes": ANIMAL_CLASSES,
        "image_size": [image_size[0], image_size[1], 3],
        "batch_size": batch_size,
        "normalization": "Rescaling [0.0, 1.0]",
    }

    return train_ds, val_ds, test_ds, meta


# Compatibility alias for Stage 2
def get_train_val_datasets(
    raw_dir: Path = RAW_DATA_DIR,
    image_size: Tuple[int, int] = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
    val_split: float = VAL_SPLIT,
    seed: int = RANDOM_SEED,
) -> Tuple[Optional[tf.data.Dataset], Optional[tf.data.Dataset], Dict[str, Any]]:
    train_ds, val_ds, _, meta = get_train_val_test_datasets(
        raw_dir=raw_dir,
        image_size=image_size,
        batch_size=batch_size,
        seed=seed,
    )
    return train_ds, val_ds, meta


# =====================================================================
# 4. Visualizations & Summary Storage
# =====================================================================

def save_class_distribution_plot(
    class_counts: Dict[str, int],
    output_path: Path = CLASS_DISTRIBUTION_PATH,
    split_counts: Optional[Dict[str, Dict[str, int]]] = None,
) -> Path:
    """
    Creates and saves a bar chart showing class counts and split breakdown.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    classes = list(class_counts.keys())

    if split_counts and "train" in split_counts:
        # Stacked bar chart showing Train, Val, Test per class
        plt.figure(figsize=(9, 5))
        x = np.arange(len(classes))
        width = 0.55

        train_vals = [split_counts["train"].get(c, 0) for c in classes]
        val_vals = [split_counts["val"].get(c, 0) for c in classes]
        test_vals = [split_counts["test"].get(c, 0) for c in classes]

        p1 = plt.bar(x, train_vals, width, label="Train (70%)", color="#4A90E2")
        p2 = plt.bar(x, val_vals, width, bottom=train_vals, label="Validation (15%)", color="#50E3C2")
        p3 = plt.bar(
            x, test_vals, width,
            bottom=[train_vals[i] + val_vals[i] for i in range(len(classes))],
            label="Test (15%)", color="#F5A623"
        )

        plt.title("Animal Dataset - Class & Split Distribution", fontsize=13, fontweight="bold")
        plt.xlabel("Animal Class", fontsize=11)
        plt.ylabel("Number of Images", fontsize=11)
        plt.xticks(x, [c.capitalize() for c in classes], fontsize=10)
        plt.legend(loc="upper right")
        plt.grid(axis="y", linestyle="--", alpha=0.6)

        for i in range(len(classes)):
            tot = train_vals[i] + val_vals[i] + test_vals[i]
            plt.text(i, tot + 1, str(tot), ha="center", va="bottom", fontsize=10, fontweight="bold")
    else:
        plt.figure(figsize=(8, 5))
        counts = [class_counts[c] for c in classes]
        bars = plt.bar([c.capitalize() for c in classes], counts, color=["#4A90E2", "#50E3C2", "#F5A623", "#9013FE", "#D0021B"])
        plt.title("Animal Dataset - Class Distribution", fontsize=13, fontweight="bold")
        plt.xlabel("Animal Class", fontsize=11)
        plt.ylabel("Number of Images", fontsize=11)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        for bar in bars:
            h = bar.get_height()
            plt.annotate(f"{int(h)}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3),
                         textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def save_sample_images_grid(
    dataset: tf.data.Dataset,
    classes: List[str] = ANIMAL_CLASSES,
    output_path: Path = SAMPLE_IMAGES_PATH,
    num_samples: int = 10
) -> Optional[Path]:
    """
    Extracts a batch of normalized images and saves a sample visualization grid with labels.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for images, labels in dataset.take(1):
        images_np = images.numpy()
        labels_np = labels.numpy()
        break
    else:
        return None

    actual_samples = min(num_samples, len(images_np))
    if actual_samples == 0:
        return None

    rows = 2
    cols = (actual_samples + 1) // 2

    plt.figure(figsize=(15, 6))
    for i in range(actual_samples):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images_np[i])
        class_idx = int(np.argmax(labels_np[i]))
        class_name = classes[class_idx]
        plt.title(f"{class_name.capitalize()}", fontsize=11, fontweight="bold")
        plt.axis("off")

    plt.suptitle("Animal Dataset - Sample Images (128x128, Normalized [0, 1])", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def save_dataset_summary(
    summary_data: Dict[str, Any],
    output_path: Path = DATASET_SUMMARY_PATH
) -> Path:
    """Saves structured dataset summary to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)
    return output_path
