"""
Configuration file for Animal Image Classification Mini Project.

This module stores global paths, image dimensions, target categories,
and training hyperparameters to keep the codebase clean and modular.
"""

from pathlib import Path

# Base project directory (points to 'animal-image-classification')
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRAIN_DIR = PROCESSED_DATA_DIR / "train"
VAL_DIR = PROCESSED_DATA_DIR / "val"
TEST_DIR = PROCESSED_DATA_DIR / "test"

# Model and results directories
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# The 5 Target Animal Categories (strictly in alphabetical order)
ANIMAL_CLASSES = [
    "cats",
    "dogs",
    "elephants",
    "horses",
    "lions"
]

# Dataset source information (Public, legitimate Apache-2.0 licensed dataset)
DATASET_SOURCE_URL = "https://huggingface.co/datasets/monikasengar/animal_image_classification/resolve/main/animal_image_dataset.zip"
DATASET_NAME = "Animal Image Classification Dataset"
DATASET_LICENSE = "Apache 2.0"

# Image and training hyperparameters
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)
CHANNELS = 3  # RGB images
BATCH_SIZE = 32
TRAIN_SPLIT = 0.70  # 70% training
VAL_SPLIT = 0.15    # 15% validation
TEST_SPLIT = 0.15   # 15% testing
RANDOM_SEED = 42
DEFAULT_EPOCHS = 20
LEARNING_RATE = 0.001
PATIENCE_EARLY_STOPPING = 5

# Model paths
MODEL_SAVE_PATH = MODELS_DIR / "animal_cnn_best.keras"
CLASS_INDICES_PATH = MODELS_DIR / "class_indices.json"

# Summary and visualization output paths
DATASET_SUMMARY_PATH = RESULTS_DIR / "dataset_summary.json"
SAMPLE_IMAGES_PATH = FIGURES_DIR / "sample_images.png"
CLASS_DISTRIBUTION_PATH = FIGURES_DIR / "class_distribution.png"
TRAINING_HISTORY_PATH = FIGURES_DIR / "training_history.png"
CONFUSION_MATRIX_PATH = FIGURES_DIR / "confusion_matrix.png"
TRAINING_METRICS_PATH = RESULTS_DIR / "training_metrics.json"

# UI Settings
APP_TITLE = "🐾 Animal Image Classification"
APP_SUBTITLE = "BTech Deep Learning Mini Project (CNN & Streamlit)"
