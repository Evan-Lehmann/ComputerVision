
from pathlib import Path
import csv
import re

import cv2
import numpy as np

# Directories
INPUT_DIR = Path("samples/cards")
OUTPUT_DIR = Path("output/processed_v2")
METADATA_FILE = OUTPUT_DIR / "metadata.csv"

# Standardized rank dimensions
CANVAS_SIZE = 64
PADDING = 8

# Normalize card images before extracting ranks
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


def extract_rank(filename):
    # Supports:
    # ace_spades.png
    # ace of spades.png
    # ace_of_spades2.png
    # 10_of_hearts.png

    name = Path(filename).stem.lower()
    name = re.sub(r"[\s_-]+", " ", name)

    first_word = name.split()[0]

    return RANK_NAMES.get(first_word)


def normalize_symbol(binary):
    # Remove small isolated components
    count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
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

    canvas[y:y + new_h, x:x + new_w] = resized

    return canvas


def preprocess_card(image):
    # Resize cards to consistent dimensions
    resized = cv2.resize(
        image,
        (REFERENCE_WIDTH, REFERENCE_HEIGHT),
        interpolation=cv2.INTER_AREA
    )

    # Extract the top-left rank region
    # These percentages are initial estimates.
    h, w = resized.shape[:2]

    roi = resized[
        int(h * 0.025):int(h * 0.14),
        int(w * 0.035):int(w * 0.19)
    ]

    if roi.size == 0:
        raise ValueError("Empty rank crop")

    # Convert to HSV and identify dark or red symbols
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Capture black/dark rank symbols
    dark_mask = cv2.inRange(
        hsv,
        np.array([0, 0, 0]),
        np.array([179, 255, 140])
    )

    # Capture saturated red symbols
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


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    successful = 0
    skipped = 0

    extensions = {".png", ".jpg", ".jpeg"}

    files = sorted(
        path for path in INPUT_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in extensions
    )

    for path in files:
        rank = extract_rank(path.name)

        if rank is None:
            print(f"Skipped unsupported file: {path.name}")
            skipped += 1
            continue

        image = cv2.imread(str(path))

        if image is None:
            print(f"Unreadable: {path.name}")
            skipped += 1
            continue

        source = path.parent.name

        try:
            processed = preprocess_card(image)

            destination_dir = OUTPUT_DIR / source / rank
            destination_dir.mkdir(
                parents=True, exist_ok=True
            )

            destination = destination_dir / path.name

            if not cv2.imwrite(
                str(destination), processed
            ):
                raise OSError("Failed to save image")

            rows.append({
                "source_deck": source,
                "rank": rank,
                "original_file": str(path),
                "processed_file": str(destination)
            })

            successful += 1
            print(f"Processed: {source}/{path.name}")

        except (ValueError, OSError) as error:
            skipped += 1
            print(f"Failed: {path.name}: {error}")

    with METADATA_FILE.open("w", newline="") as file:
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
    print(f"Successful: {successful}")
    print(f"Skipped: {skipped}")
    print(f"Metadata: {METADATA_FILE}")


if __name__ == "__main__":
    main()
