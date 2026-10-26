"""
Stage 2: Dataset Preparation, Validation, and Splitting Runner
BTech Deep Learning Mini Project - Animal Image Classification

Run this script to:
1. Verify the existence and structure of class folders.
2. Automatically download real animal images if folders are empty.
3. Scan for corrupted, invalid format, or duplicate files.
4. Partition images into Train (70%), Validation (15%), and Test (15%) splits
   in data/processed/ (strictly preventing data leakage).
5. Generate class distribution plots, sample image grids, and JSON summaries.
"""

import sys
import argparse
from pathlib import Path

# Fix Windows console encoding
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
    DATASET_NAME,
    DATASET_LICENSE,
)
from src.dataset_downloader import download_and_extract_dataset
from src.data_loader import (
    validate_dataset_structure,
    prepare_dataset_splits,
    get_train_val_test_datasets,
    save_class_distribution_plot,
    save_sample_images_grid,
    save_dataset_summary,
)


def run_stage_2(auto_download: bool = True, force_download: bool = False):
    print("=" * 72)
    print("  STAGE 2: DATASET PREPARATION, VALIDATION & SPLITTING")
    print("  Project: Animal Image Classification (5 Classes)")
    print("=" * 72)

    # Step 1: Verify folder paths
    print(f"\n[Step 1] Checking Raw Dataset Directory: {RAW_DATA_DIR}")
    if not RAW_DATA_DIR.exists():
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("[OK] Raw dataset directory exists.")
    print(f"[OK] Target Animal Classes: {', '.join(ANIMAL_CLASSES)}")

    # Step 2: Validate existing images
    print("\n[Step 2] Scanning Class Folders for Images & Validating File Integrity...")
    report = validate_dataset_structure(RAW_DATA_DIR, ANIMAL_CLASSES)

    # If insufficient images and auto_download is enabled, download public dataset
    if (not report["is_ready_for_training"] and auto_download) or force_download:
        print("\n[INFO] Class folders contain insufficient images for multi-split training.")
        print("[INFO] Initiating automated download of real animal images...")
        dl_res = download_and_extract_dataset(RAW_DATA_DIR, force_download=force_download)
        if dl_res["status"] in ("success", "already_present"):
            # Re-scan after download
            report = validate_dataset_structure(RAW_DATA_DIR, ANIMAL_CLASSES)

    # Print Class Audit Table
    print("\n  Class Folder Audit:")
    print("  " + "-" * 55)
    for cls in ANIMAL_CLASSES:
        exists = report["folder_status"].get(cls, False)
        count = report["class_counts"].get(cls, 0)
        status_tag = "[OK]  " if count >= 10 else "[LOW] "
        print(f"  {status_tag} class: {cls:<12} | Folder Exists: {str(exists):<5} | Valid Images: {count}")
    print("  " + "-" * 55)
    print(f"  Total Valid Images Found: {report['total_images']}")

    # Report unreadable or invalid format files
    if report["corrupted_files"]:
        print(f"\n[WARNING] Flagged {len(report['corrupted_files'])} unreadable/corrupted files:")
        for bad_file in report["corrupted_files"][:5]:
            print(f"   * {bad_file}")
        if len(report["corrupted_files"]) > 5:
            print(f"   ... and {len(report['corrupted_files']) - 5} more.")
    else:
        print("[OK] No corrupted image files detected.")

    if report["duplicate_files"]:
        print(f"\n[INFO] Detected {len(report['duplicate_files'])} duplicate images (unique files retained).")

    if not report["is_ready_for_training"]:
        print("\n" + "=" * 72)
        print("  [ERROR] DATASET PREPARATION INCOMPLETE")
        print("=" * 72)
        print("One or more animal classes have insufficient images for training.")
        print("Please ensure image files are placed in:")
        for cls in ANIMAL_CLASSES:
            print(f"  * {cls:<12} -> {RAW_DATA_DIR / cls}")
        sys.exit(1)

    # Step 3: Prepare Leak-Free Train / Val / Test Splits
    print("\n[Step 3] Partitioning Data into Train (70%), Validation (15%), and Test (15%) Splits...")
    split_res = prepare_dataset_splits(
        raw_dir=RAW_DATA_DIR,
        processed_dir=PROCESSED_DATA_DIR,
        train_ratio=TRAIN_SPLIT,
        val_ratio=VAL_SPLIT,
        test_ratio=TEST_SPLIT,
        seed=RANDOM_SEED,
    )
    print(f"[OK] Partitions created in: {PROCESSED_DATA_DIR}")
    for split_name in ["train", "val", "test"]:
        sub_counts = split_res["split_stats"][split_name]
        total_sub = sum(sub_counts.values())
        print(f"  * {split_name.capitalize():<5} Split: {total_sub} images {sub_counts}")

    # Step 4: Verify Keras Loading & Preprocessing
    print("\n[Step 4] Verifying Keras tf.data.Dataset Loading & [0, 1] Normalization...")
    train_ds, val_ds, test_ds, meta = get_train_val_test_datasets(
        processed_dir=PROCESSED_DATA_DIR,
        raw_dir=RAW_DATA_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        seed=RANDOM_SEED,
    )

    if train_ds is None:
        print(f"[ERROR] Loading datasets: {meta.get('message')}")
        sys.exit(1)
    print(f"[OK] Datasets ready. Target input size: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}x3 (RGB).")

    # Step 5: Save Visualizations & Summaries
    print("\n[Step 5] Generating Visualizations and Dataset Summary Report...")
    dist_path = save_class_distribution_plot(
        report["class_counts"],
        CLASS_DISTRIBUTION_PATH,
        split_counts=split_res["split_stats"],
    )
    print(f"[OK] Class distribution chart saved to: {dist_path}")

    sample_path = save_sample_images_grid(train_ds, ANIMAL_CLASSES, SAMPLE_IMAGES_PATH)
    if sample_path:
        print(f"[OK] Sample images grid saved to: {sample_path}")

    summary_payload = {
        "stage": "Stage 2: Dataset Preparation",
        "status": "ready",
        "dataset_name": DATASET_NAME,
        "dataset_license": DATASET_LICENSE,
        "target_classes": ANIMAL_CLASSES,
        "target_image_size": [IMAGE_HEIGHT, IMAGE_WIDTH, 3],
        "normalization_range": [0.0, 1.0],
        "split_ratio": {"training": TRAIN_SPLIT, "validation": VAL_SPLIT, "testing": TEST_SPLIT},
        "random_seed": RANDOM_SEED,
        "class_counts_raw": report["class_counts"],
        "total_raw_valid_images": report["total_images"],
        "split_counts": split_res["split_stats"],
        "total_split_images": split_res["total_split_images"],
        "corrupted_files_count": len(report["corrupted_files"]),
        "duplicate_files_count": len(report["duplicate_files"]),
        "data_leakage_prevention": [
            "Physical directory separation into data/processed/train, val, test",
            "Deterministic stratified pseudorandom shuffling per class (seed=42)",
            "Disjoint non-overlapping file assignments",
            "Fixed analytical 1/255.0 rescaling without sharing dataset statistics"
        ]
    }
    summary_file = save_dataset_summary(summary_payload, DATASET_SUMMARY_PATH)
    print(f"[OK] Comprehensive dataset summary saved to: {summary_file}")

    print("\n" + "=" * 72)
    print("  STAGE 2: DATASET PREPARATION COMPLETE & VERIFIED!")
    print("=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare and split animal image classification dataset.")
    parser.add_argument("--download", action="store_true", help="Force download public dataset archive.")
    parser.add_argument("--no-download", action="store_true", help="Do not attempt to download images.")
    args = parser.parse_args()

    run_stage_2(auto_download=not args.no_download, force_download=args.download)
