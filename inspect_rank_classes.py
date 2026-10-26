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
    / "class_inspection"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


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

                rank = (
                    row["rank"]
                    .strip()
                )

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

    # Fixed visual tile so every crop is easy
    # to compare.
    image = cv2.resize(
        image,
        (160, 160),
        interpolation=cv2.INTER_NEAREST,
    )

    canvas = np.full(
        (
            220,
            220,
            3,
        ),
        30,
        dtype=np.uint8,
    )

    canvas[
        45:205,
        30:190
    ] = image

    cv2.putText(
        canvas,
        f"Rank: {rank}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    short_name = path.name

    if len(short_name) > 28:
        short_name = (
            short_name[:25]
            + "..."
        )

    cv2.putText(
        canvas,
        short_name,
        (8, 215),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.33,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    return canvas


def make_sheet(
    rank,
    paths,
):
    tiles = []

    for path in paths:
        tile = make_tile(
            path,
            rank,
        )

        if tile is not None:
            tiles.append(tile)

    if not tiles:
        return None

    columns = 5

    rows = (
        len(tiles)
        + columns
        - 1
    ) // columns

    blank = np.full(
        (
            220,
            220,
            3,
        ),
        30,
        dtype=np.uint8,
    )

    while (
        len(tiles)
        < rows * columns
    ):
        tiles.append(
            blank.copy()
        )

    row_images = []

    for i in range(
        0,
        len(tiles),
        columns,
    ):
        row_images.append(
            np.hstack(
                tiles[
                    i:i + columns
                ]
            )
        )

    return np.vstack(
        row_images
    )


def main():
    examples = load_examples()

    for rank in TARGET_RANKS:

        paths = examples[
            rank
        ]

        print(
            f"{rank}: "
            f"{len(paths)} examples"
        )

        sheet = make_sheet(
            rank,
            paths,
        )

        if sheet is None:
            continue

        safe_rank = (
            "10"
            if rank == "10"
            else rank
        )

        output = (
            OUTPUT_DIR
            / f"rank_{safe_rank}.jpg"
        )

        cv2.imwrite(
            str(output),
            sheet,
        )

        print(
            f"  saved: {output}"
        )


if __name__ == "__main__":
    main()