"""
Step 4-5 of the pipeline: split the straightened grid into 81 cells and
pull out the digit (if any) from each cell.

    straightened grid -> threshold -> remove grid lines
                      -> for each cell: find the main blob near the centre
                      -> crop it and centre it in a 28x28 image for the model
"""

import cv2
import numpy as np

from src.grid import CELL, find_line_masks

DIGIT_SIZE = 28             # input size of the digit model
MIN_CONTRAST = 50           # how much darker than the paper a digit must be


def clean_grid(warped):
    """Threshold the straightened grid and erase the grid lines, leaving only digits."""
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 7
    )
    horizontal, vertical = find_line_masks(thresh)
    lines = cv2.dilate(cv2.bitwise_or(horizontal, vertical), np.ones((3, 3), np.uint8))
    digits_only = cv2.bitwise_and(thresh, cv2.bitwise_not(lines))
    return gray, thresh, digits_only


def center_digit(mask):
    """Scale a digit crop to fit 20x20 and centre it in a 28x28 image (like MNIST)."""
    h, w = mask.shape
    factor = 20 / max(h, w)
    resized = cv2.resize(mask, (max(1, round(w * factor)), max(1, round(h * factor))),
                         interpolation=cv2.INTER_AREA)
    canvas = np.zeros((DIGIT_SIZE, DIGIT_SIZE), np.uint8)
    rh, rw = resized.shape
    top, left = (DIGIT_SIZE - rh) // 2, (DIGIT_SIZE - rw) // 2
    canvas[top:top + rh, left:left + rw] = resized
    return canvas


def extract_digit(cell_mask, cell_gray):
    """
    Find the digit in one cell. Returns (28x28 digit image, bounding box) or None if empty.

    A blob counts as a digit when it is tall enough, near the centre of the
    cell, and clearly darker than the paper (so faint pencil marks are ignored).
    """
    border = 4
    mask = cell_mask.copy()
    mask[:border, :] = mask[-border:, :] = 0
    mask[:, :border] = mask[:, -border:] = 0

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    centre = np.array([CELL / 2, CELL / 2])
    paper = np.median(cell_gray)

    best, best_dist = None, None
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if h < 0.3 * CELL or h > 0.95 * CELL or w > 0.9 * CELL or area < 25:
            continue
        dist = np.linalg.norm(centroids[i] - centre)
        if dist > 0.3 * CELL:
            continue
        ink = cell_gray[labels == i].mean()
        if paper - ink < MIN_CONTRAST:
            continue
        if best is None or dist < best_dist:
            best, best_dist = i, dist

    if best is None:
        return None
    x, y, w, h, _ = stats[best]
    digit_mask = np.where(labels[y:y + h, x:x + w] == best, 255, 0).astype(np.uint8)
    return center_digit(digit_mask), (x, y, w, h)


def extract_cells(warped):
    """
    Split the straightened grid into 81 cells.

    Returns a 9x9 list where each entry is a 28x28 digit image or None (empty cell),
    plus intermediate images for display.
    """
    gray, thresh, digits_only = clean_grid(warped)
    cells = [[None] * 9 for _ in range(9)]
    boxes = warped.copy()

    for r in range(9):
        for c in range(9):
            y, x = r * CELL, c * CELL
            found = extract_digit(digits_only[y:y + CELL, x:x + CELL], gray[y:y + CELL, x:x + CELL])
            if found is None:
                continue
            cells[r][c], (bx, by, bw, bh) = found
            cv2.rectangle(boxes, (x + bx, y + by), (x + bx + bw, y + by + bh), (0, 0, 255), 2)

    return cells, {"cell_thresh": thresh, "digits_only": digits_only, "digit_boxes": boxes}
