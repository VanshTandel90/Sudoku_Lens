"""
Step 7 of the pipeline: check the grid for mistakes and solve it.

Solving uses backtracking:
    1. pick an empty cell
    2. try a digit that doesn't break the rules
    3. move on to the next empty cell
    4. if a cell has no valid digit left, go back and change the previous choice
We always pick the empty cell with the FEWEST possible digits first,
which makes the search much faster.
"""

import numpy as np

MAX_STEPS = 500_000         # give up after this many tries (protects against bad input)


def find_conflicts(board):
    """Return the set of (row, col) cells whose digit appears twice in a row, column or box."""
    conflicts = set()
    groups = []
    for i in range(9):
        groups.append([(i, c) for c in range(9)])                     # row i
        groups.append([(r, i) for r in range(9)])                     # column i
        br, bc = 3 * (i // 3), 3 * (i % 3)
        groups.append([(br + r, bc + c) for r in range(3) for c in range(3)])  # box i

    for group in groups:
        seen = {}
        for r, c in group:
            digit = board[r][c]
            if digit == 0:
                continue
            if digit in seen:
                conflicts.update({(r, c), seen[digit]})
            else:
                seen[digit] = (r, c)
    return conflicts


def candidates(board, r, c):
    """Digits that can legally go in cell (r, c)."""
    used = set(board[r, :]) | set(board[:, c])
    br, bc = 3 * (r // 3), 3 * (c // 3)
    used |= set(board[br:br + 3, bc:bc + 3].ravel())
    return [d for d in range(1, 10) if d not in used]


def solve(board):
    """
    Solve a Sudoku. board is a 9x9 array with 0 for empty cells.
    Returns the solved 9x9 array, or None if it can't be solved.
    """
    board = np.array(board, int)
    if find_conflicts(board):
        return None
    steps = 0

    def backtrack():
        nonlocal steps
        steps += 1
        if steps > MAX_STEPS:
            return False

        # Find the empty cell with the fewest options
        best, best_options = None, None
        for r in range(9):
            for c in range(9):
                if board[r, c] == 0:
                    options = candidates(board, r, c)
                    if best is None or len(options) < len(best_options):
                        best, best_options = (r, c), options
                        if len(options) <= 1:
                            break
            if best_options is not None and len(best_options) <= 1:
                break

        if best is None:
            return True                 # no empty cells left: solved
        r, c = best
        for digit in best_options:
            board[r, c] = digit
            if backtrack():
                return True
        board[r, c] = 0                 # undo and go back
        return False

    return board if backtrack() else None
