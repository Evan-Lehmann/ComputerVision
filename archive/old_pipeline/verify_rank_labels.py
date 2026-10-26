from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "detection_dataset_tiled"
OUTPUT = ROOT / "tiled_label_verification"


def render_split(split):
    image_dir = DATASET / split / "images"
    label_dir = DATASET / split / "labels"

    output_dir = OUTPUT / split
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        list(image_dir.glob("*.png"))
        + list(image_dir.glob("*.jpg"))
        + list(image_dir.glob("*.jpeg"))
    )

    total_boxes = 0

    for image_path in image_files:
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Could not read {image_path}")
            continue

        height, width = image.shape[:2]

        label_path = (
            label_dir
            / f"{image_path.stem}.txt"
        )

        boxes = []

        if label_path.exists():
            lines = label_path.read_text().splitlines()

            for line in lines:
                line = line.strip()

                if not line:
                    continue

                parts = line.split()

                if len(parts) != 5:
                    print(
                        f"BAD LABEL: {label_path.name}: "
                        f"{line}"
                    )
                    continue

                class_id, cx, cy, bw, bh = map(
                    float,
                    parts
                )

                x1 = int(
                    (cx - bw / 2) * width
                )
                y1 = int(
                    (cy - bh / 2) * height
                )
                x2 = int(
                    (cx + bw / 2) * width
                )
                y2 = int(
                    (cy + bh / 2) * height
                )

                boxes.append(
                    (x1, y1, x2, y2)
                )

        total_boxes += len(boxes)

        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

        text = (
            f"{split} | "
            f"{image_path.name} | "
            f"{len(boxes)} boxes"
        )

        cv2.putText(
            image,
            text,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        output_path = (
            output_dir
            / image_path.name
        )

        cv2.imwrite(
            str(output_path),
            image
        )

    print(
        f"{split}: "
        f"{len(image_files)} images, "
        f"{total_boxes} boxes"
    )


def main():
    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    render_split("train")
    render_split("val")

    print()
    print(
        "Verification images saved to:"
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()