from pathlib import Path
import csv

import cv2


ROOT = Path(__file__).resolve().parent

CANDIDATE_DIR = (
    ROOT
    / "rank_value_candidates"
)

OUTPUT_CSV = (
    CANDIDATE_DIR
    / "candidate_labels.csv"
)


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


def load_existing():
    labels = {}

    if not OUTPUT_CSV.exists():
        return labels

    with OUTPUT_CSV.open(
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            labels[
                row["crop"]
            ] = row["rank"]

    return labels


def save_labels(labels):
    with OUTPUT_CSV.open(
        "w",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "crop",
                "rank",
            ]
        )

        for name in sorted(labels):
            writer.writerow(
                [
                    name,
                    labels[name],
                ]
            )


def make_display(
    crop,
    index,
    total,
):
    # Enlarge tiny rank crops without blurring
    # them too heavily.
    scale = 10

    enlarged = cv2.resize(
        crop,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_NEAREST,
    )

    display = cv2.copyMakeBorder(
        enlarged,
        65,
        25,
        25,
        25,
        cv2.BORDER_CONSTANT,
        value=(30, 30, 30),
    )

    cv2.putText(
        display,
        f"{index}/{total}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return display


def main():
    candidates = sorted(
        CANDIDATE_DIR.glob(
            "candidate_*.png"
        )
    )

    labels = load_existing()

    print()
    print(
        f"Candidates: {len(candidates)}"
    )

    print(
        f"Already labeled: {len(labels)}"
    )

    print()
    print("Controls:")
    print("A     = Ace")
    print("2-9   = number")
    print("0     = 10")
    print("J     = Jack")
    print("Q     = Queen")
    print("K     = King")
    print("S     = unreadable / ambiguous")
    print("X     = save and quit")
    print()

    for index, path in enumerate(
        candidates,
        start=1,
    ):
        if path.name in labels:
            continue

        crop = cv2.imread(
            str(path)
        )

        if crop is None:
            continue

        display = make_display(
            crop,
            index,
            len(candidates),
        )

        cv2.imshow(
            "Candidate Rank Labeler",
            display,
        )

        while True:
            key = (
                cv2.waitKey(0)
                & 0xFF
            )

            if key in VALID_KEYS:
                rank = VALID_KEYS[key]

                labels[path.name] = rank

                save_labels(labels)

                print(
                    f"{path.name} -> {rank}"
                )

                break

            if key == ord("s"):
                labels[
                    path.name
                ] = "UNKNOWN"

                save_labels(labels)

                print(
                    f"{path.name} -> UNKNOWN"
                )

                break

            if key == ord("x"):
                save_labels(labels)

                cv2.destroyAllWindows()

                print()
                print("Progress saved.")

                return

    cv2.destroyAllWindows()

    print()
    print(
        "--- CANDIDATE LABELING COMPLETE ---"
    )

    print(
        f"Labels: {len(labels)}"
    )

    print(
        f"Saved to: {OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()