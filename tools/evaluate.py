"""
Measure how accurately the full pipeline reads the sample images.

Compares the board read from each image with the correct answer in
samples/labels.json and prints the accuracy and every mistake.

Usage (from the project folder):
    python tools/evaluate.py
"""

import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.cells import extract_cells
from src.grid import find_grid
from src.recognizer import recognize

SAMPLES_DIR = ROOT / "samples"


def main():
    labels = json.loads((SAMPLES_DIR / "labels.json").read_text())
    total_cells = total_correct = 0

    for name, rows in labels.items():
        if name.startswith("_"):
            continue
        expected = [[int(ch) for ch in row] for row in rows]
        result = find_grid(cv2.imread(str(SAMPLES_DIR / name)))
        if result["corners"] is None:
            print(f"{name:30s} NO GRID FOUND")
            total_cells += 81
            continue

        board, confidence = recognize(extract_cells(result["warped"])[0])
        mistakes = [
            f"r{r + 1}c{c + 1}: read {board[r][c] or '-'} expected {expected[r][c] or '-'} "
            f"(conf {confidence[r][c]:.2f})"
            for r in range(9) for c in range(9) if board[r][c] != expected[r][c]
        ]
        correct = 81 - len(mistakes)
        total_cells += 81
        total_correct += correct
        print(f"{name:30s} {correct}/81 cells correct")
        for m in mistakes:
            print("    " + m)

    print(f"\nOverall: {total_correct}/{total_cells} cells correct "
          f"({100 * total_correct / total_cells:.1f}%)")


if __name__ == "__main__":
    main()
