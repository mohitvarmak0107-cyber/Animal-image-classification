"""
Deep Learning Model Architecture for Animal Image Classification.
Stage 3 & Stage 5: CNN & Transfer Learning Architectures

Provides:
1. build_transfer_model: MobileNetV2 backbone pretrained on ImageNet + Data Augmentation
   for high accuracy (>90%) and robust generalization on small/modest datasets.
2. build_custom_cnn: 3-block Convolutional Neural Network built from scratch for viva comparison.
3. build_animal_model: Factory function selecting between 'transfer' (default) and 'custom_cnn'.
4. build_animal_cnn: Backwards-compatible alias returning the primary model.
"""

import sys
from pathlib import Path
from typing import Tuple

# Ensure project root is in sys.path when module is run directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

from src.config import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    CHANNELS,
    ANIMAL_CLASSES,
    LEARNING_RATE,
)


def build_transfer_model(
    input_shape: Tuple[int, int, int] = (IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS),
    num_classes: int = len(ANIMAL_CLASSES),
    learning_rate: float = LEARNING_RATE,
) -> tf.keras.Model:
    """
    Constructs a Transfer Learning model using the MobileNetV2 backbone pretrained on ImageNet.

    Why Transfer Learning Solves the Small Dataset Problem (Viva Explanation):
    1. Training a deep network from scratch with random weights on modest datasets (~600 images)
       leads to overfitting and weak feature extraction (e.g. confusing elephants with dogs).
    2. MobileNetV2 was pretrained on 1.4 million ImageNet images across 1,000 categories,
       giving it rich prior knowledge of animal anatomy, textures, ears, eyes, and shapes.
    3. We freeze the MobileNetV2 feature extractor and train only the custom classification head
       with Data Augmentation, ensuring fast convergence and high generalization accuracy.

    Architecture Breakdown:
    1. Input Layer: (128, 128, 3) in [0.0, 1.0].
    2. Data Augmentation (RandomFlip, RandomRotation, RandomZoom): Active during training only.
    3. Internal Rescaling: Maps [0.0, 1.0] to [-1.0, 1.0] expected by MobileNetV2.
    4. MobileNetV2 Backbone (154 layers, frozen weights): Extracts visual feature maps.
    5. GlobalAveragePooling2D: Replaces heavy Flatten layers, summarizing features into 1280 values.
    6. Dropout (0.3): Regularization against overfitting.
    7. Dense Layer (128 units, ReLU): Learns animal-specific combinations.
    8. Dropout (0.3): Secondary regularization.
    9. Dense Output (5 units, Softmax): Outputs probability distribution across the 5 classes.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # Data Augmentation pipeline (active only during training)
    x = layers.RandomFlip("horizontal", name="aug_random_flip")(inputs)
    x = layers.RandomRotation(0.1, name="aug_random_rotation")(x)
    x = layers.RandomZoom(0.1, name="aug_random_zoom")(x)

    # Scale [0.0, 1.0] -> [-1.0, 1.0] for MobileNetV2
    x = layers.Rescaling(scale=2.0, offset=-1.0, name="mobilenet_rescaling")(x)

    # Pretrained MobileNetV2 feature extractor (weights frozen)
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False
    x = base_model(x, training=False)

    # Classification Head
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = layers.Dropout(0.3, name="dropout_head_1")(x)
    x = layers.Dense(128, activation="relu", name="dense_features")(x)
    x = layers.Dropout(0.3, name="dropout_head_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output_probabilities")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="Animal_MobileNetV2_Transfer")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.CategoricalCrossentropy()
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return model


def build_custom_cnn(
    input_shape: Tuple[int, int, int] = (IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS),
    num_classes: int = len(ANIMAL_CLASSES),
    learning_rate: float = LEARNING_RATE,
) -> models.Sequential:
    """
    Constructs a beginner-friendly 3-block Convolutional Neural Network (CNN) from scratch.
    Useful for college viva comparison to illustrate how CNNs work at a basic layer level.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape, name="input_image"),

        layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu", name="conv1"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool1"),

        layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu", name="conv2"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool2"),

        layers.Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu", name="conv3"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool3"),

        layers.GlobalAveragePooling2D(name="global_pool"),
        layers.Dense(128, activation="relu", name="dense_representation"),
        layers.Dropout(0.4, name="dropout_regularization"),
        layers.Dense(num_classes, activation="softmax", name="output_probabilities"),
    ], name="Animal_Custom_CNN")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.CategoricalCrossentropy()
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return model


def build_animal_model(
    model_type: str = "transfer",
    input_shape: Tuple[int, int, int] = (IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS),
    num_classes: int = len(ANIMAL_CLASSES),
    learning_rate: float = LEARNING_RATE,
) -> tf.keras.Model:
    """
    Factory function to instantiate either the high-accuracy Transfer Learning model
    or the educational Custom CNN.
    """
    if model_type == "custom_cnn":
        return build_custom_cnn(input_shape=input_shape, num_classes=num_classes, learning_rate=learning_rate)
    return build_transfer_model(input_shape=input_shape, num_classes=num_classes, learning_rate=learning_rate)


def build_animal_cnn(
    input_shape: Tuple[int, int, int] = (IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS),
    num_classes: int = len(ANIMAL_CLASSES),
    learning_rate: float = LEARNING_RATE,
) -> tf.keras.Model:
    """
    Backwards-compatible interface. Returns the primary high-accuracy model.
    """
    return build_animal_model(
        model_type="transfer",
        input_shape=input_shape,
        num_classes=num_classes,
        learning_rate=learning_rate,
    )


if __name__ == "__main__":
    model = build_animal_model()
    model.summary()
