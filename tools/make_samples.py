"""
Generate sample Sudoku images for testing and the "try a sample" buttons.

Each sample is a valid, unsolved puzzle drawn on a white grid, then made
to look like a real photo using different effects:
    - clean        : a flat, screenshot-like image
    - tilted       : grid printed on paper, photographed at an angle
    - shadow       : uneven lighting across the page
    - noisy        : slight rotation, camera noise and blur

Run once from the project folder:
    python tools/make_samples.py
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "samples"

# Well-known valid puzzles (0 = empty cell)
PUZZLES = {
    "easy": [
        "530070000",
        "600195000",
        "098000060",
        "800060003",
        "400803001",
        "700020006",
        "060000280",
        "000419005",
        "000080079",
    ],
    "medium": [
        "003020600",
        "900305001",
        "001806400",
        "008102900",
        "700000008",
        "006708200",
        "002609500",
        "800203009",
        "005010300",
    ],
    "hard": [
        "200080300",
        "060070084",
        "030500209",
        "000105408",
        "000000000",
        "402706000",
        "301007040",
        "720040060",
        "004010003",
    ],
}

GRID_SIZE = 540             # size of the drawn grid in pixels
CELL = GRID_SIZE // 9       # size of one cell
MARGIN = 40                 # white border around the grid


def load_font(size):
    """Load a common system font, falling back to Pillow's built-in font."""
    for name in ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def draw_puzzle(rows):
    """Draw a clean Sudoku grid with its digits and return it as a BGR image."""
    size = GRID_SIZE + 2 * MARGIN
    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    font = load_font(int(CELL * 0.6))

    # Grid lines: thick every 3 cells, thin otherwise
    for i in range(10):
        width = 5 if i % 3 == 0 else 2
        pos = MARGIN + i * CELL
        draw.line([(MARGIN, pos), (MARGIN + GRID_SIZE, pos)], fill="black", width=width)
        draw.line([(pos, MARGIN), (pos, MARGIN + GRID_SIZE)], fill="black", width=width)

    # Digits, centred in their cells
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == "0":
                continue
            cx = MARGIN + c * CELL + CELL / 2
            cy = MARGIN + r * CELL + CELL / 2
            draw.text((cx, cy), ch, fill="black", font=font, anchor="mm")

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def place_on_background(page, tilt=True):
    """Put the page on a table-like background, optionally at an angle."""
    h, w = page.shape[:2]
    bg_size = int(w * 1.6)
    rng = np.random.default_rng(7)

    # Brownish "table" with some texture
    background = np.full((bg_size, bg_size, 3), (60, 95, 140), np.uint8)
    texture = rng.normal(0, 12, (bg_size, bg_size, 1)).astype(np.int16)
    background = np.clip(background.astype(np.int16) + texture, 0, 255).astype(np.uint8)

    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    off = (bg_size - w) / 2
    if tilt:
        # Move the corners so the page looks photographed from an angle
        dst = np.float32([
            [off + 60, off + 20],
            [off + w - 10, off + 70],
            [off + w + 40, off + h - 20],
            [off - 30, off + h + 10],
        ])
    else:
        dst = src + off

    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(page, matrix, (bg_size, bg_size))
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), matrix, (bg_size, bg_size))

    result = background.copy()
    result[mask > 0] = warped[mask > 0]
    return result


def add_shadow(img):
    """Darken the image gradually from one corner to simulate uneven lighting."""
    h, w = img.shape[:2]
    x = np.linspace(0, 1, w)
    y = np.linspace(0, 1, h)
    gradient = 0.45 + 0.55 * np.outer(y, x)          # dark top-left, bright bottom-right
    return np.clip(img * gradient[..., None], 0, 255).astype(np.uint8)


def add_noise_and_blur(img):
    """Slight rotation, camera noise and blur."""
    # Pad first so the rotated grid's corners stay inside the image
    img = cv2.copyMakeBorder(img, 60, 60, 60, 60, cv2.BORDER_CONSTANT, value=(200, 200, 200))
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), 6, 1.0)
    img = cv2.warpAffine(img, matrix, (w, h), borderValue=(200, 200, 200))
    rng = np.random.default_rng(3)
    noise = rng.normal(0, 14, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(img, (3, 3), 0)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    samples = {
        "generated_easy_clean.jpg": draw_puzzle(PUZZLES["easy"]),
        "generated_medium_tilted.jpg": place_on_background(draw_puzzle(PUZZLES["medium"])),
        "generated_hard_shadow.jpg": add_shadow(place_on_background(draw_puzzle(PUZZLES["hard"]), tilt=False)),
        "generated_easy_noisy.jpg": add_noise_and_blur(draw_puzzle(PUZZLES["easy"])),
    }

    for name, img in samples.items():
        cv2.imwrite(str(OUTPUT_DIR / name), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print("saved", name)


if __name__ == "__main__":
    main()
