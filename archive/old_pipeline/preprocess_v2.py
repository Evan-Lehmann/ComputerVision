
from pathlib import Path
import csv
import re

import cv2
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = Path("samples/cards")
OUTPUT_DIR = Path("output/processed_v2")
METADATA_FILE = OUTPUT_DIR / "metadata.csv"

CANVAS_SIZE = 64
PADDING = 8

REFERENCE_WIDTH = 200
REFERENCE_HEIGHT = 290

RANK_NAMES = {
    "a": "ace", "ace": "ace",
    "2": "two", "two": "two",
    "3": "three", "three": "three",
    "4": "four", "four": "four",
    "5": "five", "five": "five",
    "6": "six", "six": "six",
    "7": "seven", "seven": "seven",
    "8": "eight", "eight": "eight",
    "9": "nine", "nine": "nine",
    "10": "ten", "ten": "ten",
    "j": "jack", "jack": "jack",
    "q": "queen", "queen": "queen",
    "k": "king", "king": "king",
}


# ============================================================
# FILENAME LABELING
# ============================================================

def extract_rank(filename):
    """
    Supports filenames such as:
      ace_spades.png
      ace of spades.png
      ace_of_spades2.png
      10_of_hearts.png
    """

    name = Path(filename).stem.lower()
    name = re.sub(r"[\s_-]+", " ", name)

    first_word = name.split()[0] if name.split() else ""

    return RANK_NAMES.get(first_word)


# ============================================================
# SYMBOL NORMALIZATION
# ============================================================

def normalize_symbol(binary):
    """
    Remove tiny noise components, crop to the symbol,
    and center it on a 64x64 black canvas.
    """

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    cleaned = np.zeros_like(binary)

    for i in range(1, count):
        area = stats[i, cv2.CC_STAT_AREA]

        if area >= 3:
            cleaned[labels == i] = 255

    ys, xs = np.where(cleaned > 0)

    if len(xs) == 0:
        raise ValueError("No symbol pixels found")

    symbol = cleaned[
        ys.min():ys.max() + 1,
        xs.min():xs.max() + 1
    ]

    sh, sw = symbol.shape

    scale = min(
        (CANVAS_SIZE - 2 * PADDING) / sw,
        (CANVAS_SIZE - 2 * PADDING) / sh
    )

    new_w = max(1, round(sw * scale))
    new_h = max(1, round(sh * scale))

    resized = cv2.resize(
        symbol,
        (new_w, new_h),
        interpolation=cv2.INTER_NEAREST
    )

    canvas = np.zeros(
        (CANVAS_SIZE, CANVAS_SIZE),
        dtype=np.uint8
    )

    x = (CANVAS_SIZE - new_w) // 2
    y = (CANVAS_SIZE - new_h) // 2

    canvas[
        y:y + new_h,
        x:x + new_w
    ] = resized

    return canvas


# ============================================================
# DECK-SPECIFIC PREPROCESSING
# ============================================================


def preprocess_card(image, source):
    """
    Extract a card's upper-left rank symbol using
    source-specific cropping boundaries.
    """

    if source == "original":
        # Small screenshots: crop before resizing
        h, w = image.shape[:2]

        roi = image[
            int(h * 0.05):int(h * 0.38),
            int(w * 0.05):int(w * 0.32)
        ]

        if roi.size == 0:
            raise ValueError("Empty original rank crop")

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        _, binary = cv2.threshold(
            gray, 210, 255, cv2.THRESH_BINARY_INV
        )

    else:
        # Standardize downloaded card dimensions
        resized = cv2.resize(
            image,
            (REFERENCE_WIDTH, REFERENCE_HEIGHT),
            interpolation=cv2.INTER_AREA
        )

        h, w = resized.shape[:2]

        if source == "deck2":
            # Wider crop for Deck 2's larger typography
            x1, x2 = 0.035, 0.17
            y1, y2 = 0.025, 0.135
        else:
            # Preserve existing Deck 1 settings
            x1, x2 = 0.035, 0.14
            y1, y2 = 0.025, 0.135

        roi = resized[
            int(h * y1):int(h * y2),
            int(w * x1):int(w * x2)
        ]

        if roi.size == 0:
            raise ValueError("Empty downloaded rank crop")

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Detect dark symbols
        dark_mask = cv2.inRange(
            hsv,
            np.array([0, 0, 0]),
            np.array([179, 255, 140])
        )

        # Detect red symbols
        red_mask1 = cv2.inRange(
            hsv,
            np.array([0, 75, 55]),
            np.array([15, 255, 255])
        )

        red_mask2 = cv2.inRange(
            hsv,
            np.array([165, 75, 55]),
            np.array([179, 255, 255])
        )

        binary = cv2.bitwise_or(
            dark_mask,
            cv2.bitwise_or(red_mask1, red_mask2)
        )

    return normalize_symbol(binary)

# ============================================================
# MAIN PROCESSING PIPELINE
# ============================================================

def main():

    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Input folder does not exist: {INPUT_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    extensions = {".png", ".jpg", ".jpeg"}

    files = sorted(
        path
        for path in INPUT_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in extensions
    )

    rows = []
    successful = 0
    skipped = 0

    print(f"Found {len(files)} image files.\n")

    for path in files:

        rank = extract_rank(path.name)

        if rank is None:
            print(f"Skipped unsupported file: {path.name}")
            skipped += 1
            continue

        image = cv2.imread(str(path))

        if image is None:
            print(f"Unreadable image: {path.name}")
            skipped += 1
            continue

        source = path.parent.name

        try:
            processed = preprocess_card(
                image,
                source
            )

            destination_dir = (
                OUTPUT_DIR / source / rank
            )

            destination_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            destination = (
                destination_dir / path.name
            )

            saved = cv2.imwrite(
                str(destination),
                processed
            )

            if not saved:
                raise OSError("Failed to save processed image")

            rows.append({
                "source_deck": source,
                "rank": rank,
                "original_file": str(path),
                "processed_file": str(destination)
            })

            successful += 1

            print(
                f"Processed: {source}/{path.name}"
            )

        except (ValueError, OSError) as error:
            skipped += 1

            print(
                f"Failed: {source}/{path.name} - {error}"
            )

    # Save metadata CSV
    with METADATA_FILE.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "source_deck",
                "rank",
                "original_file",
                "processed_file"
            ]
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\n--- PREPROCESSING COMPLETE ---")
    print(f"Total files: {len(files)}")
    print(f"Successful: {successful}")
    print(f"Skipped: {skipped}")
    print(f"Metadata: {METADATA_FILE.resolve()}")


if __name__ == "__main__":
    main()


