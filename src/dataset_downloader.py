"""
Automated Dataset Downloader and Extractor.
Animal Image Classification Project

Downloads a legitimate, publicly accessible, Apache-2.0 licensed animal image
dataset (featuring Cats, Dogs, Elephants, Horses, and Lions) from Hugging Face
and organizes the images cleanly into data/raw/<class_name>/.
"""

import sys
import io
import zipfile
import urllib.request
from pathlib import Path
from typing import Dict, Any
from PIL import Image

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    RAW_DATA_DIR,
    ANIMAL_CLASSES,
    DATASET_SOURCE_URL,
    DATASET_NAME,
    DATASET_LICENSE,
)

# Mapping from archive folder names to project class folders
ARCHIVE_FOLDER_MAPPING = {
    "cat": "cats",
    "dog": "dogs",
    "elephant": "elephants",
    "horse": "horses",
    "lion": "lions",
}

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def download_and_extract_dataset(
    target_raw_dir: Path = RAW_DATA_DIR,
    source_url: str = DATASET_SOURCE_URL,
    min_images_per_class: int = 15,
    force_download: bool = False,
) -> Dict[str, Any]:
    """
    Downloads and extracts real animal images into data/raw/<class_name>/
    if the directory is empty or contains insufficient images.

    Args:
        target_raw_dir: Directory pointing to data/raw/
        source_url: Direct download URL of dataset archive (.zip)
        min_images_per_class: Minimum images required before triggering auto-download
        force_download: If True, downloads regardless of existing counts

    Returns:
        Dict reporting download status, extracted counts, and errors if any.
    """
    target_raw_dir.mkdir(parents=True, exist_ok=True)
    for cls in ANIMAL_CLASSES:
        (target_raw_dir / cls).mkdir(parents=True, exist_ok=True)

    # 1. Check existing images
    existing_counts = {}
    for cls in ANIMAL_CLASSES:
        cls_dir = target_raw_dir / cls
        count = sum(1 for f in cls_dir.iterdir() if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS)
        existing_counts[cls] = count

    has_enough_images = all(c >= min_images_per_class for c in existing_counts.values())
    if has_enough_images and not force_download:
        return {
            "status": "already_present",
            "message": "Dataset already contains sufficient images in all 5 class folders.",
            "counts": existing_counts,
            "source": "Local data/raw/",
        }

    print("=" * 72)
    print("  DOWNLOADING LEGITIMATE PUBLIC ANIMAL DATASET")
    print("=" * 72)
    print(f"  Dataset Name   : {DATASET_NAME}")
    print(f"  License        : {DATASET_LICENSE}")
    print(f"  Source URL     : {source_url}")
    print(f"  Target Classes : {', '.join(ANIMAL_CLASSES)}")
    print("  Please wait while the dataset archive is downloaded...")

    try:
        req = urllib.request.Request(
            source_url,
            headers={"User-Agent": "AnimalClassifierBot/1.0 (academic student mini project)"}
        )
        with urllib.request.urlopen(req, timeout=120) as response:
            archive_bytes = response.read()

        print(f"[OK] Downloaded archive successfully ({len(archive_bytes) / (1024 * 1024):.2f} MB).")
        print("[INFO] Extracting and validating images into class directories...")

        extracted_counts = {cls: 0 for cls in ANIMAL_CLASSES}
        corrupted_in_archive = 0

        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zf:
            for item in zf.infolist():
                if item.is_dir():
                    continue

                parts = item.filename.split("/")
                if len(parts) < 3 or parts[0] != "dataset":
                    continue

                source_class_raw = parts[1].lower()
                if source_class_raw not in ARCHIVE_FOLDER_MAPPING:
                    continue

                target_class = ARCHIVE_FOLDER_MAPPING[source_class_raw]
                filename = parts[-1]
                ext = Path(filename).suffix.lower()

                if ext not in VALID_EXTENSIONS:
                    continue

                # Read image bytes and verify integrity
                img_data = zf.read(item.filename)
                if not img_data or len(img_data) == 0:
                    corrupted_in_archive += 1
                    continue

                try:
                    with Image.open(io.BytesIO(img_data)) as test_img:
                        test_img.verify()
                except Exception:
                    corrupted_in_archive += 1
                    continue

                # Write verified image to data/raw/<class>/
                dest_file = target_raw_dir / target_class / filename
                if not dest_file.exists():
                    with open(dest_file, "wb") as f_out:
                        f_out.write(img_data)
                    extracted_counts[target_class] += 1
                else:
                    extracted_counts[target_class] += 1

        print("\n  Extraction Summary:")
        print("  " + "-" * 50)
        for cls, count in extracted_counts.items():
            print(f"  * {cls:<12} : {count} valid images extracted")
        print("  " + "-" * 50)
        if corrupted_in_archive > 0:
            print(f"  * Flagged & skipped {corrupted_in_archive} unreadable files in archive.")

        return {
            "status": "success",
            "message": "Dataset successfully downloaded, verified, and extracted.",
            "extracted_counts": extracted_counts,
            "license": DATASET_LICENSE,
            "source_url": source_url,
        }

    except Exception as e:
        err_msg = f"Failed to download or extract dataset: {str(e)}"
        print(f"[ERROR] {err_msg}")
        return {
            "status": "error",
            "message": err_msg,
        }


if __name__ == "__main__":
    download_and_extract_dataset()
