"""
Step 1-3 of the pipeline: find the Sudoku grid in a photo and straighten it.

    photo -> grayscale -> blur -> adaptive threshold -> find contours
          -> pick the 4-cornered shape that looks most like a 9x9 grid
          -> perspective transform into a flat square
"""

import cv2
import numpy as np

PROCESS_SIZE = 1000         # every photo is resized so its longer side is this many pixels
WARP_SIZE = 450             # size of the straightened grid (9 cells x 50 px)
CELL = WARP_SIZE // 9
MIN_GRID_SCORE = 0.35       # below this, we say "no grid found"


def resize_for_processing(image):
    """Resize so the longer side is PROCESS_SIZE. Returns the image and the scale used."""
    h, w = image.shape[:2]
    scale = PROCESS_SIZE / max(h, w)
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=interpolation), scale


def preprocess(image):
    """Grayscale -> blur -> adaptive threshold (white lines/digits on black)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    # Adaptive threshold compares each pixel with its neighbourhood,
    # so shadows and uneven lighting don't break it
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    return gray, blurred, thresh


def order_corners(pts):
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)             # x + y: smallest at top-left, largest at bottom-right
    d = np.diff(pts, axis=1).ravel()  # y - x: smallest at top-right, largest at bottom-left
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]])


def warp(image, corners, size=WARP_SIZE):
    """Perspective transform: map the 4 corners onto a flat size x size square."""
    target = np.float32([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]])
    matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), target)
    return cv2.warpPerspective(image, matrix, (size, size))


def find_line_masks(thresh):
    """Keep only long horizontal and vertical strokes (grid lines), removing digits."""
    horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (CELL, 1)))
    vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, CELL)))
    return horizontal, vertical


def grid_score(warped_gray):
    """
    How much does a straightened image look like a 9x9 grid? (0 to 1)

    A real grid has 10 horizontal and 10 vertical lines at equal spacing.
    We measure how strong the lines are exactly at those 10 positions.
    """
    thresh = cv2.adaptiveThreshold(
        warped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    horizontal, vertical = find_line_masks(thresh)
    row_profile = horizontal.mean(axis=1) / 255     # how "full" each row is
    col_profile = vertical.mean(axis=0) / 255

    def peaks(profile):
        window = 6
        values = []
        for i in range(10):
            p = min(i * CELL, WARP_SIZE - 1)
            values.append(profile[max(0, p - window): p + window + 1].max())
        return np.mean(values)

    return (peaks(row_profile) + peaks(col_profile)) / 2


def contour_to_corners(contour):
    """Turn a contour into 4 corner points (simplifying its outline if needed)."""
    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)
    for eps in (0.02, 0.03, 0.05, 0.08):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            return order_corners(approx)
    # Fallback: the extreme points of the shape
    return order_corners(np.array([
        hull[hull[:, :, 0].argmin()][0], hull[hull[:, :, 1].argmin()][0],
        hull[hull[:, :, 0].argmax()][0], hull[hull[:, :, 1].argmax()][0],
    ]))


def is_reasonable_quad(corners, image_area):
    """Reject shapes that are too small or far too stretched to be a Sudoku grid."""
    sides = [np.linalg.norm(corners[i] - corners[(i + 1) % 4]) for i in range(4)]
    area = cv2.contourArea(corners.astype(np.float32))
    return area > 0.04 * image_area and min(sides) / max(sides) > 0.5


def find_grid(image):
    """
    Find the Sudoku grid in a BGR photo.

    Returns a dict with the corners (in original image coordinates),
    the straightened grid, the score and the intermediate images for display.
    Returns corners=None if no grid was found.
    """
    small, scale = resize_for_processing(image)
    gray, blurred, thresh = preprocess(small)

    # Thicken lines slightly so small gaps don't split the grid outline
    closed = cv2.dilate(thresh, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:25]

    # Try the biggest shapes and keep the one that looks most like a grid
    best_score, best_corners = 0.0, None
    image_area = small.shape[0] * small.shape[1]
    for contour in contours:
        corners = contour_to_corners(contour)
        if not is_reasonable_quad(corners, image_area):
            continue
        score = grid_score(warp(gray, corners))
        if score > best_score:
            best_score, best_corners = score, corners

    result = {
        "score": best_score,
        "corners": None,
        "warped": None,
        "steps": {"gray": gray, "blurred": blurred, "thresh": thresh},
    }
    if best_corners is None or best_score < MIN_GRID_SCORE:
        return result

    corners = best_corners / scale      # back to original photo coordinates
    result["corners"] = corners
    result["warped"] = warp(image, corners)   # warp the full-quality photo

    outline = small.copy()
    cv2.polylines(outline, [best_corners.astype(np.int32)], True, (0, 255, 0), 4)
    result["steps"]["outline"] = outline
    return result
