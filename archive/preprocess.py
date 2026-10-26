
from pathlib import Path
import cv2
import numpy as np

# File locations
INPUT_DIR = Path("samples/cards")
OUTPUT_DIR = Path("output/processed_ranks")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CANVAS_SIZE = 64
PADDING = 8


def preprocess_rank(image):
    h, w = image.shape[:2]

    # Crop the upper-left portion containing the rank
    roi = image[
        int(h * 0.05):int(h * 0.38),
        int(w * 0.05):int(w * 0.32)
    ]

    if roi.size == 0:
        raise ValueError("Empty rank crop")

    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Identify dark or red symbols on white backgrounds
    _, binary = cv2.threshold(
        gray, 210, 255, cv2.THRESH_BINARY_INV
    )

    # Remove isolated noise
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    cleaned = np.zeros_like(binary)

    for i in range(1, count):
        area = stats[i, cv2.CC_STAT_AREA]

        if area >= 3:
            cleaned[labels == i] = 255

    # Find the symbol boundaries
    ys, xs = np.where(cleaned > 0)

    if len(xs) == 0:
        raise ValueError("No rank symbol detected")

    symbol = cleaned[
        ys.min():ys.max() + 1,
        xs.min():xs.max() + 1
    ]

    sh, sw = symbol.shape

    # Resize while preserving proportions
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

    # Center the rank on a standard 64x64 canvas
    canvas = np.zeros(
        (CANVAS_SIZE, CANVAS_SIZE),
        dtype=np.uint8
    )

    x = (CANVAS_SIZE - new_w) // 2
    y = (CANVAS_SIZE - new_h) // 2

    canvas[y:y + new_h, x:x + new_w] = resized

    return canvas


def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Input directory not found: {INPUT_DIR}"
        )

    paths = sorted(INPUT_DIR.rglob("*.png"))

    if not paths:
        print("No PNG images found.")
        return

    successful = 0
    failed = 0

    for path in paths:
        image = cv2.imread(str(path))

        if image is None:
            print(f"Skipped unreadable file: {path}")
            failed += 1
            continue

        try:
            result = preprocess_rank(image)

            # Get rank from parent folder name
            # Example: cards/jack/jack_spades.png -> jack
            rank = path.parent.name

            # Preserve rank folders in output
            destination_dir = OUTPUT_DIR / rank
            destination_dir.mkdir(
                parents=True, exist_ok=True
            )

            destination = destination_dir / path.name

            if not cv2.imwrite(str(destination), result):
                raise OSError("Could not save processed image")

            print(f"Processed: {rank}/{path.name}")
            successful += 1

        except (ValueError, OSError) as error:
            print(f"Failed: {path.name} - {error}")
            failed += 1

    print("\n--- PREPROCESSING COMPLETE ---")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
