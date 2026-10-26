from pathlib import Path
import shutil
import cv2


ROOT = Path(__file__).resolve().parent

SOURCE = ROOT / "detection_dataset"
OUTPUT = ROOT / "detection_dataset_tiled"

# Each tile is substantially smaller than the full screenshot,
# so the rank glyphs become much larger when YOLO resizes to 640.
TILE_W_FRAC = 0.40
TILE_H_FRAC = 0.50

# Overlap prevents ranks from constantly being cut by tile boundaries.
STRIDE_X_FRAC = 0.30
STRIDE_Y_FRAC = 0.35

# If a rank is cut too much by a tile boundary,
# discard that tile rather than train on a bad partial label.
MIN_VISIBLE = 0.95


def load_labels(label_path, img_w, img_h):
    boxes = []

    if not label_path.exists():
        return boxes

    for line in label_path.read_text().splitlines():
        line = line.strip()

        if not line:
            continue

        cls, cx, cy, bw, bh = map(float, line.split())

        x1 = (cx - bw / 2) * img_w
        y1 = (cy - bh / 2) * img_h
        x2 = (cx + bw / 2) * img_w
        y2 = (cy + bh / 2) * img_h

        boxes.append(
            {
                "class": int(cls),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )

    return boxes


def positions(length, tile_size, stride):
    values = list(range(0, max(1, length - tile_size + 1), stride))

    final = max(0, length - tile_size)

    if not values or values[-1] != final:
        values.append(final)

    return sorted(set(values))


def intersection(box, tx1, ty1, tx2, ty2):
    ix1 = max(box["x1"], tx1)
    iy1 = max(box["y1"], ty1)
    ix2 = min(box["x2"], tx2)
    iy2 = min(box["y2"], ty2)

    if ix2 <= ix1 or iy2 <= iy1:
        return None

    return ix1, iy1, ix2, iy2


def process_split(split):
    src_images = SOURCE / split / "images"
    src_labels = SOURCE / split / "labels"

    out_images = OUTPUT / split / "images"
    out_labels = OUTPUT / split / "labels"

    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    images = sorted(src_images.glob("*.png"))

    tiles_written = 0
    positive_tiles = 0
    negative_tiles = 0
    total_boxes = 0

    for image_path in images:
        image = cv2.imread(str(image_path))

        if image is None:
            continue

        h, w = image.shape[:2]

        boxes = load_labels(
            src_labels / f"{image_path.stem}.txt",
            w,
            h,
        )

        tile_w = int(w * TILE_W_FRAC)
        tile_h = int(h * TILE_H_FRAC)

        stride_x = max(1, int(w * STRIDE_X_FRAC))
        stride_y = max(1, int(h * STRIDE_Y_FRAC))

        xs = positions(w, tile_w, stride_x)
        ys = positions(h, tile_h, stride_y)

        tile_number = 0

        for y1 in ys:
            for x1 in xs:
                x2 = x1 + tile_w
                y2 = y1 + tile_h

                converted = []
                bad_partial = False

                for box in boxes:
                    hit = intersection(
                        box,
                        x1,
                        y1,
                        x2,
                        y2,
                    )

                    if hit is None:
                        continue

                    ix1, iy1, ix2, iy2 = hit

                    original_area = (
                        (box["x2"] - box["x1"])
                        * (box["y2"] - box["y1"])
                    )

                    intersection_area = (
                        (ix2 - ix1)
                        * (iy2 - iy1)
                    )

                    visible = (
                        intersection_area / original_area
                        if original_area > 0
                        else 0
                    )

                    # Don't create a training tile containing
                    # a badly chopped rank.
                    if visible < MIN_VISIBLE:
                        bad_partial = True
                        break

                    local_x1 = box["x1"] - x1
                    local_y1 = box["y1"] - y1
                    local_x2 = box["x2"] - x1
                    local_y2 = box["y2"] - y1

                    cx = (
                        (local_x1 + local_x2) / 2
                    ) / tile_w

                    cy = (
                        (local_y1 + local_y2) / 2
                    ) / tile_h

                    bw = (
                        local_x2 - local_x1
                    ) / tile_w

                    bh = (
                        local_y2 - local_y1
                    ) / tile_h

                    converted.append(
                        (
                            box["class"],
                            cx,
                            cy,
                            bw,
                            bh,
                        )
                    )

                if bad_partial:
                    continue

                crop = image[
                    y1:y2,
                    x1:x2
                ]

                tile_number += 1

                name = (
                    f"{image_path.stem}"
                    f"_tile_{tile_number:02d}"
                )

                cv2.imwrite(
                    str(out_images / f"{name}.png"),
                    crop,
                )

                label_path = (
                    out_labels / f"{name}.txt"
                )

                with label_path.open("w") as f:
                    for cls, cx, cy, bw, bh in converted:
                        f.write(
                            f"{cls} "
                            f"{cx:.6f} "
                            f"{cy:.6f} "
                            f"{bw:.6f} "
                            f"{bh:.6f}\n"
                        )

                tiles_written += 1
                total_boxes += len(converted)

                if converted:
                    positive_tiles += 1
                else:
                    negative_tiles += 1

    print(
        f"{split}: "
        f"{tiles_written} tiles | "
        f"{positive_tiles} positive | "
        f"{negative_tiles} negative | "
        f"{total_boxes} boxes"
    )


def main():
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)

    process_split("train")
    process_split("val")

    yaml_text = f"""path: {OUTPUT}
train: train/images
val: val/images

names:
  0: rank
"""

    (OUTPUT / "dataset.yaml").write_text(
        yaml_text
    )

    print()
    print("Created:")
    print(OUTPUT)
    print()
    print("Your original labeled dataset was NOT modified.")


if __name__ == "__main__":
    main()