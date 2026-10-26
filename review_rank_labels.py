from pathlib import Path
import csv

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent

SOURCES = [
    {
        "name": "original",
        "image_dir": ROOT / "rank_value_dataset",
        "csv_path": ROOT / "rank_value_dataset" / "labels.csv",
    },
    {
        "name": "candidate",
        "image_dir": ROOT / "rank_value_candidates",
        "csv_path": ROOT / "rank_value_candidates" / "candidate_labels.csv",
    },
]


def load_rows(source):
    rows = []

    with source["csv_path"].open(
        newline=""
    ) as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append(row)

    return rows


def save_rows(source, rows):
    with source["csv_path"].open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "crop",
                "rank",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)


def make_display(
    image,
    label,
    source_name,
    filename,
    index,
    total,
):
    h, w = image.shape[:2]

    scale = min(
        8.0,
        600 / max(h, w),
    )

    enlarged = cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_NEAREST,
    )

    canvas = cv2.copyMakeBorder(
        enlarged,
        120,
        40,
        30,
        30,
        cv2.BORDER_CONSTANT,
        value=(25, 25, 25),
    )

    cv2.putText(
        canvas,
        f"{index}/{total}   Label: {label}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        f"Source: {source_name}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    short_name = filename

    if len(short_name) > 55:
        short_name = (
            short_name[:52]
            + "..."
        )

    cv2.putText(
        canvas,
        short_name,
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        "K = keep   U = UNKNOWN   B = back   X = save + quit",
        (20, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return canvas


def main():
    entries = []

    loaded = {}

    for source in SOURCES:
        rows = load_rows(
            source
        )

        loaded[
            source["name"]
        ] = rows

        for row_index, row in enumerate(
            rows
        ):
            if (
                row["rank"]
                == "UNKNOWN"
            ):
                continue

            image_path = (
                source["image_dir"]
                / row["crop"]
            )

            if not image_path.exists():
                continue

            entries.append(
                {
                    "source":
                        source,
                    "row_index":
                        row_index,
                    "path":
                        image_path,
                }
            )

    print()
    print(
        f"Usable labels to review: "
        f"{len(entries)}"
    )

    print()
    print("Controls:")
    print("K = keep current label")
    print("U = mark UNKNOWN")
    print("B = go back one image")
    print("X = save and quit")
    print()

    index = 0

    while index < len(
        entries
    ):
        entry = entries[
            index
        ]

        source = entry[
            "source"
        ]

        rows = loaded[
            source["name"]
        ]

        row = rows[
            entry["row_index"]
        ]

        image = cv2.imread(
            str(
                entry["path"]
            )
        )

        if image is None:
            index += 1
            continue

        print(
            f"{index + 1}/{len(entries)} | "
            f"CURRENT LABEL = {row['rank']} | "
            f"{entry['path'].name}"
        )

        display = make_display(
            image,
            row["rank"],
            source["name"],
            entry["path"].name,
            index + 1,
            len(entries),
        )

        cv2.imshow(
            "Rank Label Review",
            display,
        )

        key = (
            cv2.waitKey(0)
            & 0xFF
        )

        if key == ord("k"):
            index += 1

        elif key == ord("u"):
            print(
                f"UNKNOWN: "
                f"{entry['path'].name} "
                f"(was {row['rank']})"
            )

            row["rank"] = (
                "UNKNOWN"
            )

            index += 1

        elif key == ord("b"):
            index = max(
                0,
                index - 1,
            )

        elif key == ord("x"):
            break

    cv2.destroyAllWindows()

    for source in SOURCES:
        save_rows(
            source,
            loaded[
                source["name"]
            ],
        )

    print()
    print(
        "Review saved."
    )

    # Updated counts
    counts = {}

    usable = 0

    for source in SOURCES:
        for row in loaded[
            source["name"]
        ]:
            rank = row["rank"]

            if rank == "UNKNOWN":
                continue

            counts[rank] = (
                counts.get(
                    rank,
                    0,
                )
                + 1
            )

            usable += 1

    order = [
        "A", "2", "3", "4", "5",
        "6", "7", "8", "9", "10",
        "J", "Q", "K",
    ]

    print()
    print(
        "Updated usable counts:"
    )

    for rank in order:
        print(
            f"{rank:>2}: "
            f"{counts.get(rank, 0)}"
        )

    print()
    print(
        f"Usable total: {usable}"
    )


if __name__ == "__main__":
    main()