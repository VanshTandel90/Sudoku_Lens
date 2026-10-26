"""
Step 6 of the pipeline: read each extracted digit with the trained CNN.
"""

from pathlib import Path

import numpy as np

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "digit_model.keras"

_model = None


def load_model():
    """Load the trained model once and reuse it."""
    global _model
    if _model is None:
        import tensorflow as tf     # imported here because it is slow to load
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


def recognize(cells):
    """
    Turn the 9x9 grid of digit images into numbers.

    Returns (board, confidence): board[r][c] is 1-9 or 0 for an empty cell,
    confidence[r][c] is how sure the model is (0-1).
    """
    board = np.zeros((9, 9), int)
    confidence = np.zeros((9, 9))
    positions = [(r, c) for r in range(9) for c in range(9) if cells[r][c] is not None]
    if not positions:
        return board, confidence

    batch = np.array([cells[r][c] for r, c in positions], "float32")[..., None] / 255.0
    probabilities = load_model().predict(batch, verbose=0)

    for (r, c), probs in zip(positions, probabilities):
        board[r, c] = probs.argmax() + 1       # classes 0-8 -> digits 1-9
        confidence[r, c] = probs.max()
    return board, confidence
