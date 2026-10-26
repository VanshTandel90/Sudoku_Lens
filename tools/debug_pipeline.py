"""
Developer tool: run the grid + cell pipeline on images and save a picture of
each stage, so we can see where it works and where it fails.

Usage:
    python tools/debug_pipeline.py OUTPUT_DIR IMAGE [IMAGE ...]
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cells import extract_cells
from src.grid import find_grid


def fit(img, height=450):
    """Resize to a fixed height and convert to 3 channels so images can sit side by side."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    return cv2.resize(img, (round(w * height / h), height))


def main():
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in sys.argv[2:]:
        image = cv2.imread(path)
        name = Path(path).stem
        result = find_grid(image)
        steps = result["steps"]

        if result["corners"] is None:
            print(f"{name:28s} NO GRID   (score {result['score']:.2f})")
            panel = np.hstack([fit(image), fit(steps["thresh"])])
        else:
            cells, cell_steps = extract_cells(result["warped"])
            filled = sum(cell is not None for row in cells for cell in row)
            print(f"{name:28s} grid found (score {result['score']:.2f}), {filled} digits")
            panel = np.hstack([fit(steps["outline"]), fit(cell_steps["digits_only"]),
                               fit(cell_steps["digit_boxes"])])
        cv2.imwrite(str(out_dir / f"{name}_debug.jpg"), panel)


if __name__ == "__main__":
    main()
