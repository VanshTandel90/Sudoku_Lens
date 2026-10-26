"""
Train the CNN that recognizes printed digits 1-9.

Run once from the project folder, after generate_dataset.py:
    python training/train_model.py

Output: models/digit_model.keras and models/training_history.json
"""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "generated" / "digits.npz"
MODEL_DIR = ROOT / "models"
EPOCHS = 10


def build_model():
    """
    A small CNN:
      2 convolution + pooling blocks learn stroke and shape patterns,
      then dense layers turn them into one of 9 classes (digits 1-9).
    """
    return tf.keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(9, activation="softmax"),
    ])


def main():
    data = np.load(DATA_FILE)
    images = data["images"].astype("float32")[..., None] / 255.0   # scale pixels to 0-1
    labels = data["labels"] - 1                                    # digits 1-9 -> classes 0-8

    # Shuffle and keep 15% aside to check accuracy on images the model never trained on
    rng = np.random.default_rng(0)
    order = rng.permutation(len(images))
    split = int(len(images) * 0.85)
    train, val = order[:split], order[split:]

    model = build_model()
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    history = model.fit(images[train], labels[train], epochs=EPOCHS, batch_size=128,
                        validation_data=(images[val], labels[val]), verbose=2)

    MODEL_DIR.mkdir(exist_ok=True)
    model.save(MODEL_DIR / "digit_model.keras")
    with open(MODEL_DIR / "training_history.json", "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)
    print(f"Validation accuracy: {history.history['val_accuracy'][-1]:.4f}")


if __name__ == "__main__":
    main()
