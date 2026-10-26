"""
Step 8 of the pipeline: write the solution back onto the original photo.

We draw the missing digits on a flat square, then use the INVERSE
perspective transform to bend that square back onto the tilted grid
in the photo.
"""

import cv2
import numpy as np

DRAW_SIZE = 900             # draw on a big square so digits stay sharp after warping
SOLUTION_COLOR = (40, 160, 40)   # green (BGR)


def draw_solution(photo, corners, given, solved):
    """
    photo:   original BGR image
    corners: 4 grid corners in the photo (top-left, top-right, bottom-right, bottom-left)
    given:   9x9 board read from the photo (0 = empty)
    solved:  9x9 solved board
    Returns a copy of the photo with the missing digits filled in.
    """
    cell = DRAW_SIZE / 9
    layer = np.zeros((DRAW_SIZE, DRAW_SIZE, 3), np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = cell / 40, max(2, int(cell / 18))

    for r in range(9):
        for c in range(9):
            if given[r][c] != 0:
                continue
            text = str(solved[r][c])
            (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
            x = int(c * cell + (cell - tw) / 2)
            y = int(r * cell + (cell + th) / 2)
            cv2.putText(layer, text, (x, y), font, scale, SOLUTION_COLOR, thickness, cv2.LINE_AA)

    # Inverse perspective: flat square -> tilted grid in the photo
    square = np.float32([[0, 0], [DRAW_SIZE - 1, 0], [DRAW_SIZE - 1, DRAW_SIZE - 1], [0, DRAW_SIZE - 1]])
    matrix = cv2.getPerspectiveTransform(square, np.float32(corners))
    h, w = photo.shape[:2]
    warped = cv2.warpPerspective(layer, matrix, (w, h))

    # Copy only the pixels where a digit was drawn
    mask = warped.max(axis=2) > 0
    result = photo.copy()
    result[mask] = warped[mask]
    return result


def draw_board(board, given=None, conflicts=(), size=450):
    """
    Draw a clean 9x9 board image.
    Digits from the photo (given) are black, solved digits are green,
    and cells in `conflicts` get a red background.
    """
    cell = size // 9
    image = np.full((size + 1, size + 1, 3), 255, np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = cell / 40, max(1, cell // 20)

    for r, c in conflicts:
        cv2.rectangle(image, (c * cell, r * cell), ((c + 1) * cell, (r + 1) * cell), (180, 180, 255), -1)

    for i in range(10):
        width = 3 if i % 3 == 0 else 1
        cv2.line(image, (0, i * cell), (size, i * cell), (0, 0, 0), width)
        cv2.line(image, (i * cell, 0), (i * cell, size), (0, 0, 0), width)

    for r in range(9):
        for c in range(9):
            digit = board[r][c]
            if digit == 0:
                continue
            is_given = given is None or given[r][c] != 0
            color = (0, 0, 0) if is_given else SOLUTION_COLOR
            text = str(digit)
            (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
            x = int(c * cell + (cell - tw) / 2)
            y = int(r * cell + (cell + th) / 2)
            cv2.putText(image, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
    return image
