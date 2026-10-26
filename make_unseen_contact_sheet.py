from pathlib import Path
import cv2
import math


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "unseen_detector_test"
OUTPUT = ROOT / "unseen_detector_contact_sheet.jpg"

THUMB_W = 480
THUMB_H = 270
COLS = 4


def main():
    files = sorted(INPUT.glob("*.png"))

    if not files:
        files = sorted(INPUT.glob("*.jpg"))

    if not files:
        raise RuntimeError(
            f"No images found in {INPUT}"
        )

    thumbs = []

    for path in files:
        img = cv2.imread(str(path))

        if img is None:
            continue

        thumb = cv2.resize(
            img,
            (THUMB_W, THUMB_H)
        )

        cv2.putText(
            thumb,
            path.stem[-6:],
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        thumbs.append(thumb)

    rows = math.ceil(len(thumbs) / COLS)

    sheet = 255 * (
        __import__("numpy")
        .ones(
            (
                rows * THUMB_H,
                COLS * THUMB_W,
                3
            ),
            dtype="uint8"
        )
    )

    for i, thumb in enumerate(thumbs):
        row = i // COLS
        col = i % COLS

        y1 = row * THUMB_H
        x1 = col * THUMB_W

        sheet[
            y1:y1 + THUMB_H,
            x1:x1 + THUMB_W
        ] = thumb

    cv2.imwrite(
        str(OUTPUT),
        sheet
    )

    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()