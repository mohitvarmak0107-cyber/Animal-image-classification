"""
Master Test Suite Runner for Animal Image Classification Project.
BTech Deep Learning Mini Project - Stage 5 Verification

Executes all automated unit and integration tests across:
1. Dataset preparation and validation pipeline (tests/test_data_loader.py)
2. CNN training and callback verification (tests/test_model_training.py)
3. Preprocessing and inference pipeline (tests/test_inference.py)
"""

import sys
from pathlib import Path

# Fix Windows console encoding
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.test_data_loader import test_pipeline_with_mock_data
from tests.test_model_training import test_training_pipeline_end_to_end
from tests.test_inference import test_inference_module


def run_master_test_suite():
    print("=" * 72)
    print("  ANIMAL IMAGE CLASSIFICATION - MASTER TEST SUITE")
    print("  Stage 5: Final Testing & Verification")
    print("=" * 72)

    tests = [
        ("Dataset Preparation & Leak-Free Splitting", test_pipeline_with_mock_data),
        ("CNN Training, Callbacks & Artifact Generation", test_training_pipeline_end_to_end),
        ("Image Preprocessing (RGB/RGBA/Gray) & Inference", test_inference_module),
    ]

    all_passed = True
    for idx, (name, test_func) in enumerate(tests, 1):
        print(f"\n[{idx}/3] Testing: {name}...")
        try:
            test_func()
            print(f"[PASSED] {name}")
        except Exception as e:
            all_passed = False
            print(f"[FAILED] {name}: {str(e)}")

    print("\n" + "=" * 72)
    if all_passed:
        print("  ALL TESTS PASSED SUCCESSFULLY! (100% HEALTHY)")
        print("  Project is ready for college demonstration and viva defense.")
    else:
        print("  SOME TESTS ENCOUNTERED ERRORS. Please review log output above.")
    print("=" * 72)


if __name__ == "__main__":
    run_master_test_suite()
