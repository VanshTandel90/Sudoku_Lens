"""
SudokuLens - Image-Based Sudoku Grid Detection and Solving using Computer Vision.
Streamlit web app.

Run from the project folder:
    streamlit run app.py
"""

import hashlib
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from src.cells import extract_cells
from src.grid import find_grid
from src.overlay import draw_board, draw_solution
from src.recognizer import load_model, recognize
from src.solver import find_conflicts, solve

ROOT = Path(__file__).resolve().parent
SAMPLES_DIR = ROOT / "samples"
LOW_CONFIDENCE = 0.90       # digits the model is less sure about than this get flagged
MIN_CLUES = 17              # no valid Sudoku has fewer starting digits than this
STEP_WIDTH = 320            # width of each image in the Step-by-step tab

# Editable grid layout: 9 cells per row with a narrow gap column between 3x3 blocks
GRID_COLUMNS = [1, 1, 1, 0.3, 1, 1, 1, 0.3, 1, 1, 1]
CELL_POSITIONS = [0, 1, 2, 4, 5, 6, 8, 9, 10]
SHADE_COLOR = "rgba(128, 128, 128, 0.18)"
UNSURE_COLOR = "rgba(255, 165, 0, 0.40)"
CONFLICT_COLOR = "rgba(255, 60, 60, 0.45)"
GRID_CSS = """
<style>
.st-key-sudoku_grid { gap: 0.3rem; }
.st-key-sudoku_grid [data-testid="stHorizontalBlock"] { gap: 0.3rem; }
[class*="st-key-cell_"] input {
    text-align: center;
    font-size: 1.3rem;
    font-weight: 600;
    padding: 0.35rem 0;
}
</style>
"""

st.set_page_config(page_title="SudokuLens", page_icon="🧩", layout="wide")


# ---------- Cached helpers ----------

@st.cache_resource
def get_model():
    """Load the CNN once for the whole app."""
    return load_model()


@st.cache_data
def process_image(image_bytes):
    """Run the vision pipeline on an image: find grid -> extract cells -> read digits."""
    photo = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if photo is None:
        return {"error": "This file could not be read as an image."}

    grid = find_grid(photo)
    result = {"photo": photo, "grid": grid}
    if grid["corners"] is None:
        return result

    get_model()
    cells, cell_steps = extract_cells(grid["warped"])
    board, confidence = recognize(cells)
    result.update(cells=cells, cell_steps=cell_steps, board=board, confidence=confidence)
    return result


@st.cache_data
def thumbnail(path, size=200):
    """Small square preview of a sample image (padded, not cropped)."""
    img = cv2.imread(str(path))
    h, w = img.shape[:2]
    scale = size / max(h, w)
    img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    top, left = (size - img.shape[0]) // 2, (size - img.shape[1]) // 2
    return cv2.copyMakeBorder(img, top, size - img.shape[0] - top, left, size - img.shape[1] - left,
                              cv2.BORDER_CONSTANT, value=(240, 240, 240))


def digit_tiles(cells, board, confidence):
    """Montage of the 28x28 images fed to the CNN, with the predicted digit and confidence."""
    tiles = []
    for r in range(9):
        for c in range(9):
            if cells[r][c] is None:
                continue
            tile = cv2.cvtColor(cv2.resize(cells[r][c], (84, 84), interpolation=cv2.INTER_NEAREST),
                                cv2.COLOR_GRAY2BGR)
            label = np.full((26, 84, 3), 255, np.uint8)
            color = (0, 0, 0) if confidence[r][c] >= LOW_CONFIDENCE else (0, 0, 220)
            cv2.putText(label, f"{board[r][c]}  {confidence[r][c]:.0%}", (6, 19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            tiles.append(cv2.copyMakeBorder(np.vstack([tile, label]), 3, 3, 3, 3,
                                            cv2.BORDER_CONSTANT, value=(255, 255, 255)))
    if not tiles:
        return None
    per_row = 10
    blank = np.full_like(tiles[0], 255)
    tiles += [blank] * (-len(tiles) % per_row)
    return np.vstack([np.hstack(tiles[i:i + per_row]) for i in range(0, len(tiles), per_row)])


def nice_name(path):
    return path.stem.replace("_", " ").capitalize()


def png_bytes(image):
    """Encode a BGR image as PNG bytes for the download buttons."""
    return cv2.imencode(".png", image)[1].tobytes()


def cell_key(r, c):
    return f"cell_{image_id}_{r}_{c}"


def color_cells(cells, color):
    """Give the listed grid cells a background colour."""
    rules = " ".join(f".st-key-{cell_key(r, c)} input {{ background-color: {color}; }}" for r, c in cells)
    if rules:
        st.html(f"<style>{rules}</style>")


def board_input(board):
    """
    Editable 9x9 Sudoku grid: one small box per cell, with gaps and shading
    to show the 3x3 blocks. Returns the current board and any invalid cells.
    """
    st.html(GRID_CSS)
    color_cells([(r, c) for r in range(9) for c in range(9) if (r // 3 + c // 3) % 2 == 1], SHADE_COLOR)

    current = np.zeros((9, 9), int)
    invalid = []
    with st.container(key="sudoku_grid"):
        for r in range(9):
            if r in (3, 6):
                st.html('<div style="height: 0.6rem"></div>')     # gap between block rows
            columns = st.columns(GRID_COLUMNS, gap="small")
            for c, position in enumerate(CELL_POSITIONS):
                with columns[position]:
                    value = st.text_input(
                        f"Row {r + 1}, column {c + 1}",
                        value=str(board[r][c]) if board[r][c] else "",
                        max_chars=1, key=cell_key(r, c), label_visibility="collapsed",
                    ).strip()
                if value and value in "123456789":
                    current[r, c] = int(value)
                elif value:
                    invalid.append((r, c))
    return current, invalid


# ---------- Choosing an image ----------

st.title("🧩 SudokuLens")
st.caption("Upload a photo or screenshot of an unsolved Sudoku. The app finds the grid, "
           "reads the digits with a CNN, and solves it.")

if "source" not in st.session_state:
    st.session_state.source = None


def use_upload():
    st.session_state.source = "upload"


uploaded = st.file_uploader("Upload a Sudoku image", type=["png", "jpg", "jpeg"], on_change=use_upload)

st.write("**Or try a sample:**")
samples = sorted(p for p in SAMPLES_DIR.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
per_row = 5
for start in range(0, len(samples), per_row):
    columns = st.columns(per_row)
    for column, path in zip(columns, samples[start:start + per_row]):
        with column:
            st.image(thumbnail(path), channels="BGR", width="stretch")
            if st.button(nice_name(path), key=f"sample_{path.name}", width="stretch"):
                st.session_state.source = str(path)

if st.button("✏️ Enter a puzzle by hand"):
    st.session_state.source = "manual"

source = st.session_state.source
manual = source == "manual"
if manual:
    image_id, result = "manual", None
else:
    if source == "upload" and uploaded is not None:
        image_bytes, image_name = uploaded.getvalue(), uploaded.name
    elif source not in (None, "upload"):
        image_bytes, image_name = Path(source).read_bytes(), Path(source).name
    else:
        st.info("Upload an image, click a sample, or enter a puzzle by hand to begin.")
        st.stop()

    image_id = hashlib.md5(image_bytes).hexdigest()[:12]
    with st.spinner("Reading the puzzle..."):
        result = process_image(image_bytes)

st.divider()
st.subheader("Manual entry" if manual else f"Image: {image_name}")

if not manual and "error" in result:
    st.error(result["error"])
    st.stop()

grid_found = manual or result["grid"]["corners"] is not None
solve_tab, steps_tab = st.tabs(["Solve", "Step-by-step"])


# ---------- Solve tab ----------

with solve_tab:
    if not grid_found:
        st.error("No Sudoku grid was found in this image. Tips: make sure all 4 corners of the "
                 "grid are visible, the photo is not too blurry, and the lighting is even. "
                 "The Step-by-step tab shows what the app saw.")
        st.image(result["photo"], channels="BGR", width=500)
    else:
        if manual:
            board, confidence = np.zeros((9, 9), int), np.zeros((9, 9))
            grid_col, side_col = st.columns(2)
            with side_col:
                st.markdown("**How to enter a puzzle**")
                st.caption("Type one digit (1–9) in each given cell and leave the empty cells blank. "
                           "Press Tab to move to the next cell. Cells that break the Sudoku rules "
                           "turn red.")
        else:
            board, confidence = result["board"], result["confidence"]
            photo_col, grid_col = st.columns(2)
            with photo_col:
                st.markdown("**Original photo**")
                st.image(result["photo"], channels="BGR", width="stretch")

        with grid_col:
            if manual:
                st.markdown("**Type the puzzle's digits** — leave empty cells blank")
            else:
                st.markdown("**Digits read from the photo** — click a cell to fix it, or clear it if empty")
            current, invalid = board_input(board)

            unsure = [(r, c) for r in range(9) for c in range(9)
                      if board[r][c] and confidence[r][c] < LOW_CONFIDENCE]
            color_cells(unsure, UNSURE_COLOR)
            color_cells(find_conflicts(current) | set(invalid), CONFLICT_COLOR)

            if unsure:
                st.warning("Orange cells: please double-check these, the model was less sure: " + ", ".join(
                    f"row {r + 1} col {c + 1} → {board[r][c]} ({confidence[r][c]:.0%})" for r, c in unsure))
            if invalid:
                st.warning("Only digits 1–9 are allowed. These cells are ignored: " + ", ".join(
                    f"row {r + 1} col {c + 1}" for r, c in invalid))

            clues = int((current > 0).sum())
            st.caption(f"{clues} digits on the board.")
            solve_clicked = st.button("Solve ▶", type="primary")

        if solve_clicked:
            conflicts = find_conflicts(current)
            if clues < MIN_CLUES:
                st.session_state.solution = {"id": image_id, "board": current.tolist(), "too_few": True}
            elif conflicts:
                st.session_state.solution = {"id": image_id, "board": current.tolist(), "conflicts": sorted(conflicts)}
            else:
                with st.spinner("Solving..."):
                    solved = solve(current)
                st.session_state.solution = {
                    "id": image_id, "board": current.tolist(),
                    "solved": None if solved is None else solved.tolist(),
                }

        solution = st.session_state.get("solution")
        if solution and solution["id"] == image_id:
            if solution["board"] != current.tolist():
                st.info("The grid was changed after solving. Press Solve again.")
            elif solution.get("too_few"):
                st.error(f"Only {clues} digits on the board. A proper Sudoku has at least {MIN_CLUES} "
                         "starting digits — add the missing ones and solve again.")
            elif solution.get("conflicts"):
                cells = ", ".join(f"row {r + 1} col {c + 1}" for r, c in solution["conflicts"])
                st.error(f"Some digits break the Sudoku rules (same digit twice in a row, column or box): "
                         f"{cells}. Fix the red cells in the grid and solve again.")
            elif solution["solved"] is None:
                st.error("This puzzle has no solution. Some digits may be wrong — check the grid.")
            else:
                st.success("Solved!")
                solved_board = draw_board(solution["solved"], given=current)
                if manual:
                    board_col = st.container()
                else:
                    overlay_col, board_col = st.columns(2)
                    with overlay_col:
                        st.markdown("**Solution on the photo**")
                        solved_photo = draw_solution(result["photo"], result["grid"]["corners"], current,
                                                     solution["solved"])
                        st.image(solved_photo, channels="BGR", width="stretch")
                        st.download_button("⬇️ Download solved photo", png_bytes(solved_photo),
                                           file_name=f"solved_{Path(image_name).stem}.png",
                                           mime="image/png", on_click="ignore")
                with board_col:
                    st.markdown("**Solved board** (green = filled in by the solver)")
                    st.image(solved_board, channels="BGR", width=450 if manual else "stretch")
                    st.download_button("⬇️ Download solved board", png_bytes(solved_board),
                                       file_name="solved_board.png", mime="image/png", on_click="ignore")


# ---------- Step-by-step tab ----------

def render_steps():
    steps = result["grid"]["steps"]

    items = [
        ("1. Original photo", result["photo"], "The image as uploaded."),
        ("2. Grayscale", steps["gray"], "Colour is removed; only brightness matters for finding lines and digits."),
        ("3. Gaussian blur", steps["blurred"], "Smooths out paper texture and camera noise."),
        ("4. Adaptive threshold", steps["thresh"],
         "Each pixel is compared with its neighbourhood, so lines and digits turn white even under shadows."),
    ]
    if grid_found:
        cell_steps = result["cell_steps"]
        items += [
            ("5. Grid detected", steps["outline"],
             "The biggest 4-cornered shapes are tested; the one with 10 evenly spaced lines "
             f"in each direction wins (score {result['grid']['score']:.2f})."),
            ("6. Perspective transform", result["grid"]["warped"],
             "The 4 corners are mapped onto a flat 450×450 square, so every cell is 50×50 px."),
            ("7. Grid lines removed", cell_steps["digits_only"],
             "Long horizontal and vertical strokes are detected with morphology and erased, "
             "leaving only the digits."),
            ("8. Digits found", cell_steps["digit_boxes"],
             "In each cell, a blob that is centred, tall enough and dark enough counts as a digit."),
        ]

    # Three steps per row, each image at a fixed size
    for start in range(0, len(items), 3):
        for column, (title, image, text) in zip(st.columns(3), items[start:start + 3]):
            with column:
                st.markdown(f"**{title}**")
                st.image(image, channels="BGR" if image.ndim == 3 else "GRAY", width=STEP_WIDTH)
                st.caption(text)

    if not grid_found:
        st.error(f"No grid found (best grid score {result['grid']['score']:.2f}, needs "
                 "at least 0.35). The steps stop here.")
    else:
        st.markdown("**9. CNN input and predictions**")
        tiles = digit_tiles(result["cells"], result["board"], result["confidence"])
        if tiles is not None:
            st.image(tiles, channels="BGR")
        st.caption("Each digit is scaled into a 28×28 image and classified by the CNN. "
                   "The number under each is the predicted digit and how sure the model is "
                   f"(red if below {LOW_CONFIDENCE:.0%}).")

        st.markdown("**10. Solve and overlay**")
        st.caption("Backtracking fills the empty cells, then the inverse perspective transform "
                   "draws the answers back onto the original photo. See the Solve tab.")


with steps_tab:
    if manual:
        st.info("The step-by-step view shows how an image is processed. "
                "Upload an image or click a sample to see it.")
    else:
        render_steps()
