from pathlib import Path
import csv

import cv2


ROOT = Path(__file__).resolve().parent

DATASET = ROOT / "detection_dataset"

OUTPUT_DIR = ROOT / "rank_value_dataset"
OUTPUT_DIR.mkdir(exist_ok=True)

CSV_PATH = OUTPUT_DIR / "labels.csv"


VALID_KEYS = {
    ord("a"): "A",
    ord("2"): "2",
    ord("3"): "3",
    ord("4"): "4",
    ord("5"): "5",
    ord("6"): "6",
    ord("7"): "7",
    ord("8"): "8",
    ord("9"): "9",
    ord("0"): "10",
    ord("j"): "J",
    ord("q"): "Q",
    ord("k"): "K",
}


def load_yolo_boxes(label_path, width, height):
    boxes = []

    if not label_path.exists():
        return boxes

    for line in label_path.read_text().splitlines():
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:
            continue

        _, cx, cy, bw, bh = map(
            float,
            parts,
        )

        x1 = int(
            (cx - bw / 2)
            * width
        )

        y1 = int(
            (cy - bh / 2)
            * height
        )

        x2 = int(
            (cx + bw / 2)
            * width
        )

        y2 = int(
            (cy + bh / 2)
            * height
        )

        boxes.append(
            (
                x1,
                y1,
                x2,
                y2,
            )
        )

    return boxes


def load_existing_labels():
    if not CSV_PATH.exists():
        return {}

    labels = {}

    with CSV_PATH.open(
        newline=""
    ) as file:

        reader = csv.DictReader(
            file
        )

        for row in reader:
            labels[
                row["crop"]
            ] = row["rank"]

    return labels


def save_labels(labels):
    with CSV_PATH.open(
        "w",
        newline="",
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "crop",
                "rank",
            ]
        )

        for crop_name in sorted(
            labels.keys()
        ):
            writer.writerow(
                [
                    crop_name,
                    labels[crop_name],
                ]
            )


def collect_crops():
    crops = []

    for split in (
        "train",
        "val",
    ):

        image_dir = (
            DATASET
            / split
            / "images"
        )

        label_dir = (
            DATASET
            / split
            / "labels"
        )

        for image_path in sorted(
            image_dir.glob("*.png")
        ):

            image = cv2.imread(
                str(image_path)
            )

            if image is None:
                continue

            height, width = (
                image.shape[:2]
            )

            boxes = load_yolo_boxes(
                label_dir
                / f"{image_path.stem}.txt",
                width,
                height,
            )

            for index, box in enumerate(
                boxes
            ):

                x1, y1, x2, y2 = box

                box_width = (
                    x2 - x1
                )

                box_height = (
                    y2 - y1
                )

                pad_x = max(
                    2,
                    int(
                        box_width
                        * 0.20
                    ),
                )

                pad_y = max(
                    2,
                    int(
                        box_height
                        * 0.20
                    ),
                )

                px1 = max(
                    0,
                    x1 - pad_x,
                )

                py1 = max(
                    0,
                    y1 - pad_y,
                )

                px2 = min(
                    width,
                    x2 + pad_x,
                )

                py2 = min(
                    height,
                    y2 + pad_y,
                )

                crop = image[
                    py1:py2,
                    px1:px2
                ]

                if crop.size == 0:
                    continue

                crop_name = (
                    f"{split}_"
                    f"{image_path.stem}_"
                    f"{index:02d}.png"
                )

                crop_path = (
                    OUTPUT_DIR
                    / crop_name
                )

                cv2.imwrite(
                    str(crop_path),
                    crop,
                )

                crops.append(
                    (
                        crop_name,
                        crop,
                    )
                )

    return crops


def make_display(crop, index, total):
    scale = 10

    enlarged = cv2.resize(
        crop,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_NEAREST,
    )

    top_margin = 55

    display = cv2.copyMakeBorder(
        enlarged,
        top_margin,
        20,
        20,
        20,
        cv2.BORDER_CONSTANT,
        value=(
            30,
            30,
            30,
        ),
    )

    cv2.putText(
        display,
        f"{index}/{total}",
        (
            20,
            30,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (
            0,
            255,
            0,
        ),
        2,
        cv2.LINE_AA,
    )

    return display


def main():
    labels = load_existing_labels()

    crops = collect_crops()

    print()
    print(
        f"Found {len(crops)} rank crops."
    )

    print(
        f"Already labeled: {len(labels)}"
    )

    print()
    print("Controls:")
    print("A = Ace")
    print("2-9 = number")
    print("0 = 10")
    print("J = Jack")
    print("Q = Queen")
    print("K = King")
    print("S = skip / ambiguous")
    print("X = quit and save")
    print()

    for index, (
        crop_name,
        crop,
    ) in enumerate(
        crops,
        start=1,
    ):

        if crop_name in labels:
            continue

        display = make_display(
            crop,
            index,
            len(crops),
        )

        cv2.imshow(
            "Rank Value Labeler",
            display,
        )

        while True:

            key = (
                cv2.waitKey(0)
                & 0xFF
            )

            if key in VALID_KEYS:

                rank = (
                    VALID_KEYS[key]
                )

                labels[
                    crop_name
                ] = rank

                save_labels(
                    labels
                )

                print(
                    f"{crop_name} "
                    f"-> {rank}"
                )

                break

            if key == ord("s"):

                labels[
                    crop_name
                ] = "UNKNOWN"

                save_labels(
                    labels
                )

                print(
                    f"{crop_name} "
                    f"-> UNKNOWN"
                )

                break

            if key == ord("x"):

                save_labels(
                    labels
                )

                cv2.destroyAllWindows()

                print()
                print(
                    "Saved progress."
                )

                return

    cv2.destroyAllWindows()

    print()
    print(
        "--- RANK LABELING COMPLETE ---"
    )

    print(
        f"Labeled crops: "
        f"{len(labels)}"
    )

    print(
        f"Saved to: "
        f"{CSV_PATH}"
    )


if __name__ == "__main__":
    main()