from pathlib import Path
import csv
import re

import cv2
import numpy as np


CAPTURES_DIR = Path("captures")
OUTPUT_CSV = Path("state_change_scores.csv")

# Compare each frame with one ~2 seconds earlier.
LAG_FRAMES = 4

# This crop is ONLY for offline dataset preparation.
# It has nothing to do with the eventual runtime detector.
X1_FRAC = 0.15
X2_FRAC = 0.85
Y1_FRAC = 0.10
Y2_FRAC = 0.78

WIDTH = 160
HEIGHT = 110


def natural_sort_key(path):
    parts = re.split(r"(\d+)", path.name)

    return [
        int(part) if part.isdigit() else part.lower()
        for part in parts
    ]


def get_paths():
    paths = []

    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        paths.extend(CAPTURES_DIR.glob(pattern))

    return sorted(paths, key=natural_sort_key)


def descriptor(frame):
    h, w = frame.shape[:2]

    x1 = int(w * X1_FRAC)
    x2 = int(w * X2_FRAC)

    y1 = int(h * Y1_FRAC)
    y2 = int(h * Y2_FRAC)

    crop = frame[y1:y2, x1:x2]

    crop = cv2.resize(
        crop,
        (WIDTH, HEIGHT),
        interpolation=cv2.INTER_AREA,
    )

    # --------------------------------------------------------
    # Grayscale appearance
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2GRAY,
    )

    gray = cv2.GaussianBlur(
        gray,
        (7, 7),
        0,
    )

    gray = (
        gray.astype(np.float32)
        / 255.0
    )

    # --------------------------------------------------------
    # Edge structure
    # --------------------------------------------------------

    edges = cv2.Canny(
        (gray * 255).astype(np.uint8),
        45,
        120,
    )

    edges = (
        edges.astype(np.float32)
        / 255.0
    )

    # --------------------------------------------------------
    # Bright-surface structure
    #
    # Useful because playing cards introduce substantial
    # light regions even when rank glyphs themselves are tiny.
    # --------------------------------------------------------

    hsv = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2HSV,
    )

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    bright = (
        (value >= 145)
        & (saturation <= 115)
    ).astype(np.float32)

    return gray, edges, bright


def distance(a, b):
    gray_a, edges_a, bright_a = a
    gray_b, edges_b, bright_b = b

    gray_diff = np.mean(
        np.abs(gray_a - gray_b)
    )

    edge_diff = np.mean(
        np.abs(edges_a - edges_b)
    )

    bright_diff = np.mean(
        np.abs(bright_a - bright_b)
    )

    # We are measuring, not classifying.
    # Keep the three signals separately as well as combined.
    combined = (
        0.45 * gray_diff
        + 0.25 * edge_diff
        + 0.30 * bright_diff
    )

    return (
        float(combined),
        float(gray_diff),
        float(edge_diff),
        float(bright_diff),
    )


def main():
    paths = get_paths()

    if not paths:
        print("No captures found.")
        return

    print(f"Frames found: {len(paths)}")
    print("Building descriptors...")

    descriptors = []

    for i, path in enumerate(paths, start=1):
        frame = cv2.imread(str(path))

        if frame is None:
            descriptors.append(None)
            continue

        descriptors.append(
            descriptor(frame)
        )

        if (
            i % 50 == 0
            or i == len(paths)
        ):
            print(
                f"Processed {i}/{len(paths)}"
            )

    rows = []
    combined_scores = []

    for i in range(
        LAG_FRAMES,
        len(paths)
    ):
        current = descriptors[i]
        previous = descriptors[
            i - LAG_FRAMES
        ]

        if current is None or previous is None:
            continue

        (
            combined,
            gray_diff,
            edge_diff,
            bright_diff,
        ) = distance(
            previous,
            current,
        )

        combined_scores.append(
            combined
        )

        rows.append(
            {
                "frame_number": i + 1,
                "filename": paths[i].name,
                "compared_to_frame":
                    i - LAG_FRAMES + 1,
                "combined_change":
                    f"{combined:.8f}",
                "gray_change":
                    f"{gray_diff:.8f}",
                "edge_change":
                    f"{edge_diff:.8f}",
                "bright_change":
                    f"{bright_diff:.8f}",
            }
        )

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_number",
                "filename",
                "compared_to_frame",
                "combined_change",
                "gray_change",
                "edge_change",
                "bright_change",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    scores = np.array(
        combined_scores,
        dtype=np.float32,
    )

    print()
    print(
        "--- STATE CHANGE ANALYSIS ---"
    )

    print(
        f"Comparisons: {len(scores)}"
    )

    for percentile in (
        0,
        10,
        25,
        50,
        75,
        90,
        95,
        97,
        99,
        100,
    ):
        value = np.percentile(
            scores,
            percentile,
        )

        print(
            f"P{percentile:>3}: "
            f"{value:.6f}"
        )

    print()
    print("Largest 20 changes:")

    top_indices = np.argsort(
        scores
    )[-20:][::-1]

    for rank, score_index in enumerate(
        top_indices,
        start=1,
    ):
        row = rows[
            int(score_index)
        ]

        print(
            f"{rank:02d}. "
            f"frame "
            f"{int(row['frame_number']):03d} "
            f"change="
            f"{float(row['combined_change']):.6f}"
        )

    print()
    print(
        f"Saved: {OUTPUT_CSV.resolve()}"
    )


if __name__ == "__main__":
    main()