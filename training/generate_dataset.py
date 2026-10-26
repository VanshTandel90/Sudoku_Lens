"""
Generate the training data for the digit recognition model.

Sudoku puzzles use PRINTED digits, so instead of the handwritten MNIST
dataset we render digits 1-9 in many fonts. Each digit is drawn inside a
fake Sudoku cell with random size, position, rotation, lighting, blur and
noise, and then passed through the SAME extract_digit() code the app uses
on real photos. That way the model trains on exactly what it will see.

Run once from the project folder (takes about a minute):
    python training/generate_dataset.py

Output: data/generated/digits.npz (+ a preview image)
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.cells import clean_grid, extract_digit
from src.grid import CELL

OUTPUT_DIR = ROOT / "data" / "generated"
FONT_DIRS = [Path("C:/Windows/Fonts"), Path("/usr/share/fonts"), Path("/Library/Fonts")]
SAMPLES_PER_FONT_PER_DIGIT = 20

# Plain printed fonts that look like newspaper / puzzle-book digits
FONT_FILES = [
    "arial.ttf", "arialbd.ttf", "ARIALN.TTF", "ARIALNB.TTF", "ariblk.ttf", "ARLRDBD.TTF",
    "bahnschrift.ttf", "BASKVILL.TTF", "BELL.TTF", "BELLB.TTF", "BKANT.TTF", "BOD_R.TTF",
    "BOD_B.TTF", "BOOKOS.TTF", "BOOKOSB.TTF", "calibri.ttf", "calibrib.ttf", "calibril.ttf",
    "CALIFR.TTF", "CALIFB.TTF", "CALIST.TTF", "CALISTB.TTF", "cambriab.ttf", "Candara.ttf",
    "Candarab.ttf", "Candaral.ttf", "CENSCBK.TTF", "CENTURY.TTF", "comic.ttf", "comicbd.ttf",
    "consola.ttf", "consolab.ttf", "constan.ttf", "constanb.ttf", "corbel.ttf", "corbelb.ttf",
    "cour.ttf", "courbd.ttf", "DUBAI-REGULAR.TTF", "DUBAI-BOLD.TTF", "ebrima.ttf", "ERASMD.TTF",
    "ERASDEMI.TTF", "FRABK.TTF", "FRADM.TTF", "FRAMDCN.TTF", "framd.ttf", "gadugi.ttf",
    "gadugib.ttf", "GARA.TTF", "GARABD.TTF", "georgia.ttf", "georgiab.ttf", "GIL_____.TTF",
    "GILB____.TTF", "GOTHIC.TTF", "GOTHICB.TTF", "GOUDOS.TTF", "GOUDOSB.TTF", "LEELAWAD.TTF",
    "LeelawUI.ttf", "lucon.ttf", "LSANS.TTF", "LSANSD.TTF", "LTYPE.TTF", "micross.ttf",
    "Nirmala.ttf", "NirmalaB.ttf", "pala.ttf", "palab.ttf", "PER_____.TTF", "PERB____.TTF",
    "ROCK.TTF", "ROCKB.TTF", "SCHLBKB.TTF", "segoeui.ttf", "segoeuib.ttf", "seguisb.ttf",
    "tahoma.ttf", "tahomabd.ttf", "TCM_____.TTF", "TCB_____.TTF", "times.ttf", "timesbd.ttf",
    "trebuc.ttf", "trebucbd.ttf", "verdana.ttf", "verdanab.ttf", "malgun.ttf", "sylfaen.ttf",
    "REFSAN.TTF", "DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Regular.ttf",
    "LiberationSerif-Regular.ttf",
]


def find_fonts():
    """Return the paths of the fonts from FONT_FILES that exist on this computer."""
    found = {}
    for folder in FONT_DIRS:
        if not folder.exists():
            continue
        for path in folder.rglob("*"):
            if path.name.lower() in {f.lower() for f in FONT_FILES}:
                found.setdefault(path.name.lower(), path)
    return list(found.values())


def render_cell(digit, font_path, rng):
    """Draw one digit inside a fake Sudoku cell, with random variations."""
    size = CELL * 2                                  # draw at 2x, then shrink (smoother edges)
    paper = int(rng.integers(150, 256))
    ink = int(rng.integers(0, min(90, paper - 60)))

    img = Image.new("L", (size, size), paper)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(str(font_path), int(size * rng.uniform(0.5, 0.85)))
    cx = size / 2 + rng.uniform(-0.08, 0.08) * size
    cy = size / 2 + rng.uniform(-0.08, 0.08) * size
    stroke = int(rng.choice([0, 0, 0, 1, 2]))
    draw.text((cx, cy), str(digit), fill=ink, font=font, anchor="mm",
              stroke_width=stroke, stroke_fill=ink)

    # Grid lines around the cell, sometimes shifted inwards like an imperfect warp
    if rng.random() < 0.7:
        width = int(rng.integers(2, 8))
        for _ in range(4):
            offset = int(rng.integers(0, 7))
            side = rng.integers(4)
            if side == 0:
                draw.line([(0, offset), (size, offset)], fill=ink, width=width)
            elif side == 1:
                draw.line([(0, size - offset), (size, size - offset)], fill=ink, width=width)
            elif side == 2:
                draw.line([(offset, 0), (offset, size)], fill=ink, width=width)
            else:
                draw.line([(size - offset, 0), (size - offset, size)], fill=ink, width=width)

    img = img.rotate(rng.uniform(-8, 8), resample=Image.BILINEAR, fillcolor=paper)
    cell = cv2.resize(np.array(img), (CELL, CELL), interpolation=cv2.INTER_AREA)

    if rng.random() < 0.25:                          # simulate a low-resolution photo
        factor = rng.uniform(0.4, 0.7)
        small = cv2.resize(cell, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        cell = cv2.resize(small, (CELL, CELL), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.4:
        cell = cv2.GaussianBlur(cell, (3, 3), 0)
    noise = rng.normal(0, rng.uniform(0, 10), cell.shape)
    return np.clip(cell + noise, 0, 255).astype(np.uint8)


def main():
    fonts = find_fonts()
    if not fonts:
        sys.exit("No usable fonts found.")
    print(f"Using {len(fonts)} fonts")

    rng = np.random.default_rng(42)
    images, labels = [], []
    for font_path in fonts:
        for digit in range(1, 10):
            for _ in range(SAMPLES_PER_FONT_PER_DIGIT):
                cell = render_cell(digit, font_path, rng)
                # Same processing as the real app
                _, _, digits_only = clean_grid(cv2.cvtColor(cell, cv2.COLOR_GRAY2BGR))
                found = extract_digit(digits_only, cell)
                if found is None:
                    continue
                images.append(found[0])
                labels.append(digit)

    images, labels = np.array(images), np.array(labels)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT_DIR / "digits.npz", images=images, labels=labels)
    print(f"Saved {len(images)} digit images")

    # Preview: 9 rows (digits 1-9) x 30 random examples
    rows = []
    for digit in range(1, 10):
        picks = rng.choice(np.flatnonzero(labels == digit), 30, replace=False)
        rows.append(np.hstack(images[picks]))
    cv2.imwrite(str(OUTPUT_DIR / "preview.png"), np.vstack(rows))


if __name__ == "__main__":
    main()
