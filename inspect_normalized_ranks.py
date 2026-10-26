from pathlib import Path
import csv

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent

SOURCES = [
    (
        ROOT / "rank_value_dataset",
        ROOT / "rank_value_dataset" / "labels.csv",
    ),
    (
        ROOT / "rank_value_candidates",
        ROOT / "rank_value_candidates" / "candidate_labels.csv",
    ),
]

TARGET_RANKS = [
    "4",
    "6",
    "7",
    "10",
    "J",
]

OUTPUT_DIR = (
    ROOT
    / "rank_classifier_v2"
    / "normalized_inspection"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SIZE = 64


def normalize_rank(image):
    """
    Normalize an already-detected rank crop.

    Candidate crops were saved with ~20% padding around
    the detector box. Rather than trying to rediscover
    the glyph with connected components, trim that
    padding back off and preserve the actual pixels.
    """

    if image is None or image.size == 0:
        return np.zeros(
            (SIZE, SIZE),
            dtype=np.uint8,
        )

    h, w = image.shape[:2]

    # A 20% pad on each side means the original detector
    # box is roughly the central 71% of the saved crop.
    #
    # Trim slightly less than the theoretical 14.3%
    # so we don't accidentally cut the glyph.
    trim_x = int(
        w * 0.11
    )

    trim_y = int(
        h * 0.11
    )

    if (
        w - 2 * trim_x > 2
        and h - 2 * trim_y > 2
    ):
        image = image[
            trim_y:h - trim_y,
            trim_x:w - trim_x,
        ]

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    # Mild contrast normalization only.
    gray = cv2.normalize(
        gray,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    )

    h, w = gray.shape[:2]

    available = SIZE - 8

    scale = min(
        available / max(w, 1),
        available / max(h, 1),
    )

    new_w = max(
        1,
        round(w * scale),
    )

    new_h = max(
        1,
        round(h * scale),
    )

    resized = cv2.resize(
        gray,
        (
            new_w,
            new_h,
        ),
        interpolation=cv2.INTER_CUBIC,
    )

    canvas = np.full(
        (SIZE, SIZE),
        255,
        dtype=np.uint8,
    )

    x = (
        SIZE - new_w
    ) // 2

    y = (
        SIZE - new_h
    ) // 2

    canvas[
        y:y + new_h,
        x:x + new_w
    ] = resized

    return canvas

def load_examples():
    examples = {
        rank: []
        for rank in TARGET_RANKS
    }

    for image_dir, csv_path in SOURCES:

        if not csv_path.exists():
            continue

        with csv_path.open(
            newline=""
        ) as f:

            reader = csv.DictReader(f)

            for row in reader:
                rank = row[
                    "rank"
                ].strip()

                if rank not in examples:
                    continue

                path = (
                    image_dir
                    / row["crop"]
                )

                if path.exists():
                    examples[
                        rank
                    ].append(path)

    return examples


def make_tile(
    path,
    rank,
):
    image = cv2.imread(
        str(path)
    )

    if image is None:
        return None

    normalized = normalize_rank(
        image
    )

    enlarged = cv2.resize(
        normalized,
        (180, 180),
        interpolation=cv2.INTER_NEAREST,
    )

    enlarged = cv2.cvtColor(
        enlarged,
        cv2.COLOR_GRAY2BGR,
    )

    canvas = np.full(
        (
            225,
            200,
            3,
        ),
        25,
        dtype=np.uint8,
    )

    canvas[
        35:215,
        10:190
    ] = enlarged

    cv2.putText(
        canvas,
        f"Rank {rank}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )

    return canvas


def main():
    examples = load_examples()

    for rank in TARGET_RANKS:

        tiles = []

        for path in examples[
            rank
        ]:
            tile = make_tile(
                path,
                rank,
            )

            if tile is not None:
                tiles.append(tile)

        if not tiles:
            continue

        columns = 5

        blank = np.full(
            (
                225,
                200,
                3,
            ),
            25,
            dtype=np.uint8,
        )

        while (
            len(tiles)
            % columns
        ):
            tiles.append(
                blank.copy()
            )

        rows = []

        for i in range(
            0,
            len(tiles),
            columns,
        ):
            rows.append(
                np.hstack(
                    tiles[
                        i:i + columns
                    ]
                )
            )

        sheet = np.vstack(
            rows
        )

        output = (
            OUTPUT_DIR
            / f"rank_{rank}.jpg"
        )

        cv2.imwrite(
            str(output),
            sheet,
        )

        print(
            f"{rank}: {output}"
        )


if __name__ == "__main__":
    main()