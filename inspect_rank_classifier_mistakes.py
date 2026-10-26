from pathlib import Path
import csv
import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent

CSV_PATH = (
    ROOT
    / "rank_classifier_v2"
    / "holdout_predictions.csv"
)

OUTPUT = (
    ROOT
    / "rank_classifier_v2"
    / "mistakes_montage.jpg"
)


def main():
    mistakes = []

    with CSV_PATH.open(newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row["correct"] == "1":
                continue

            path = Path(row["path"])

            image = cv2.imread(
                str(path)
            )

            if image is None:
                continue

            enlarged = cv2.resize(
                image,
                None,
                fx=8,
                fy=8,
                interpolation=cv2.INTER_NEAREST,
            )

            canvas = cv2.copyMakeBorder(
                enlarged,
                80,
                20,
                20,
                20,
                cv2.BORDER_CONSTANT,
                value=(30, 30, 30),
            )

            text1 = (
                f"Actual: {row['actual']}   "
                f"Pred: {row['predicted']}"
            )

            text2 = (
                f"Confidence: "
                f"{float(row['confidence']):.3f}"
            )

            cv2.putText(
                canvas,
                text1,
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                canvas,
                text2,
                (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

            mistakes.append(canvas)

    if not mistakes:
        print("No mistakes.")
        return

    width = max(
        img.shape[1]
        for img in mistakes
    )

    padded = []

    for img in mistakes:
        right = width - img.shape[1]

        padded.append(
            cv2.copyMakeBorder(
                img,
                0,
                0,
                0,
                right,
                cv2.BORDER_CONSTANT,
                value=(30, 30, 30),
            )
        )

    montage = np.vstack(padded)

    cv2.imwrite(
        str(OUTPUT),
        montage,
    )

    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()