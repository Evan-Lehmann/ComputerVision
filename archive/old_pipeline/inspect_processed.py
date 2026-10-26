
from pathlib import Path
import cv2
import numpy as np

INPUT = Path("output/processed_v2")
OUTPUT = Path("output/processed_contact_sheet.png")

files = sorted(
    p for p in INPUT.rglob("*.png")
    if p.is_file()
)

CELL_W = 160
CELL_H = 110
COLS = 6

if not files:
    raise FileNotFoundError("No processed images found.")

ROWS = (len(files) + COLS - 1) // COLS

sheet = np.full(
    (ROWS * CELL_H, COLS * CELL_W, 3),
    35,
    dtype=np.uint8
)

for index, path in enumerate(files):
    img = cv2.imread(str(path))

    if img is None:
        continue

    row, col = divmod(index, COLS)

    x = col * CELL_W
    y = row * CELL_H

    # Place rank image
    sheet[y + 5:y + 69, x + 48:x + 112] = img

    # Label with deck and rank
    deck = path.parent.parent.name
    rank = path.parent.name

    cv2.putText(
        sheet,
        f"{deck}: {rank}",
        (x + 8, y + 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (255, 255, 255),
        1
    )

    cv2.putText(
        sheet,
        path.stem[:22],
        (x + 8, y + 103),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.29,
        (180, 180, 180),
        1
    )

if not cv2.imwrite(str(OUTPUT), sheet):
    raise OSError("Failed to save contact sheet")

print(f"Created contact sheet with {len(files)} images.")
print(f"Saved to: {OUTPUT}")
