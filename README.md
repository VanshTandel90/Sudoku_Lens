# 🧩 SudokuLens

**Image-Based Sudoku Grid Detection and Solving using Computer Vision**

A Computer Vision project that reads an unsolved Sudoku from a photo or screenshot, recognizes the printed digits with a CNN, solves the puzzle, and draws the answers back onto the original image.

Everything runs offline on a normal laptop. No GPU or camera is needed.

## Features

- **Upload** a photo or screenshot of a Sudoku (PNG/JPG), or **try one of 10 sample images** with one click
- **Automatic grid detection**, even on tilted photos, shadows, and newspaper pages with other content around the puzzle
- **Digit recognition** with a small CNN trained on printed digits
- **Editable grid** to fix any misread digit before solving
  - Orange cells: digits the model was less sure about
  - Red cells: digits that break the Sudoku rules
- **Solution drawn on the original photo**, plus a clean solved board
- **Download** the solved photo and the solved board as PNG
- **Enter a puzzle by hand** without an image
- **Step-by-step view** showing every stage of the image processing

## How it works

| Step | What happens | Technique |
|---|---|---|
| 1 | Clean up the image | Grayscale, Gaussian blur, adaptive threshold |
| 2 | Find the Sudoku grid | Contours, convex hull, polygon approximation, grid-likeness score |
| 3 | Straighten the grid | Perspective transform to a flat 450×450 square |
| 4 | Find the digits | Morphological line removal, connected components |
| 5 | Read the digits | CNN classifies each 28×28 digit image as 1–9 |
| 6 | Solve | Rule check, then backtracking (most-constrained cell first) |
| 7 | Show the answer | Inverse perspective transform onto the original photo |

## Setup

These steps are for Windows. You need **Python 3.12**, because TensorFlow does not support newer versions yet. Python 3.12 can be installed alongside any other Python version without conflicts.

### 1. Install Python 3.12

Download it from [python.org](https://www.python.org/downloads/release/python-31210/), or run:

```
winget install Python.Python.3.12
```

Check that it's installed:

```
py --list
```

You should see `3.12` in the list.

### 2. Get the project

```
git clone https://github.com/VanshTandel90/Sudoku_Lens.git
cd Sudoku_Lens
```

### 3. Create the virtual environment and install the libraries

```
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

After `activate`, your prompt starts with `(.venv)`. The install downloads about 500 MB (mostly TensorFlow), so it can take a few minutes.

### 4. Run the app

```
streamlit run app.py
```

The app opens in your browser at http://localhost:8501. The first time, Streamlit asks for an email address. Just press Enter to skip.

To stop the app, press **Ctrl + C** in the terminal.

### Running it again later

```
cd Sudoku_Lens
.venv\Scripts\activate
streamlit run app.py
```

### Troubleshooting

| Problem | Fix |
|---|---|
| `streamlit` is not recognized | The virtual environment isn't active. Run `.venv\Scripts\activate` first |
| `running scripts is disabled on this system` | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate again |
| `No matching distribution found for tensorflow` | The venv was made with the wrong Python. Delete `.venv` and redo step 3 with `py -3.12` |
| `Port 8501 is already in use` | The app is already running in another terminal |
| TensorFlow warnings in the terminal (oneDNN, GPU) | Normal. They can be ignored |

On macOS or Linux, use `python3.12 -m venv .venv` and `source .venv/bin/activate` instead.

## Using the app

1. Upload a Sudoku image, click a sample, or click **✏️ Enter a puzzle by hand**.
2. Check the digits in the grid. Click a cell to fix a wrong digit or clear a false one.
3. Press **Solve ▶**.
4. See the solution on the photo, and download it if you like.
5. Open the **Step-by-step** tab to see how the image was processed.

**Tips for photos:** keep all four corners of the grid visible, avoid strong glare, and use a reasonably sharp image. Unsolved puzzles work best. Handwritten digits may be read as well.

## Project structure

```
Sudoku_Lens/
├── app.py                    # Streamlit web app
├── requirements.txt          # Python libraries with exact versions
├── src/                      # The vision and solving pipeline
│   ├── grid.py               #   Steps 1-3: preprocessing, grid detection, perspective transform
│   ├── cells.py              #   Step 4: line removal and digit extraction
│   ├── recognizer.py         #   Step 5: digit recognition with the CNN
│   ├── solver.py             #   Step 6: rule check and backtracking solver
│   └── overlay.py            #   Step 7: drawing the solution
├── models/
│   ├── digit_model.keras     # The trained CNN
│   └── training_history.json # Accuracy and loss per training epoch
├── training/
│   ├── generate_dataset.py   # Creates printed-digit training images from fonts
│   └── train_model.py        # Trains the CNN
├── samples/                  # Sample images + labels.json (correct digits for each)
└── tools/
    ├── make_samples.py       # Creates the generated sample images
    ├── debug_pipeline.py     # Saves a picture of each pipeline stage for given images
    └── evaluate.py           # Measures digit-reading accuracy on the samples
```

## The digit model

Sudoku puzzles use printed digits, so the model is not trained on the handwritten MNIST dataset. Instead, `training/generate_dataset.py` renders the digits 1–9 in about 90 fonts with random size, position, rotation, lighting, blur and noise, then passes each one through the same digit-extraction code the app uses.

- **Model:** 2 convolution + pooling layers, then 2 dense layers (about 225,000 parameters)
- **Data:** about 16,000 generated digit images
- **Training time:** under a minute on a normal CPU
- **Validation accuracy:** 99.5%
- **On the 10 sample images:** 810 / 810 cells read correctly

The trained model is included in `models/`, so **you do not need to train anything to run the app**.

### Retraining (optional)

```
python training/generate_dataset.py
python training/train_model.py
python tools/evaluate.py
```

The dataset script uses the fonts installed on the computer. Its font list is based on standard Windows fonts.

## Limitations

- Very curved pages, strong glare, or heavily blurred photos can make grid detection fail.
- Handwritten digits (for example, answers someone already filled in) may be read as given digits. Clear them in the grid before solving.
- Only standard 9×9 Sudoku is supported.

## Built with

Python · OpenCV · TensorFlow / Keras · NumPy · Pillow · Streamlit
