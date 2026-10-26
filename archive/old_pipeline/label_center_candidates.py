from pathlib import Path
import csv
import shutil

import cv2


# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = Path(
    "center_match_diagnostics/raw_candidates"
)

OUTPUT_ROOT = Path(
    "center_training_data"
)

LABELED_DIR = OUTPUT_ROOT / "labeled"
CSV_PATH = OUTPUT_ROOT / "labels.csv"

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

LABELED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LABEL MAP
# ============================================================

# Single-key controls.
#
# A = Ace
# 2-9 = numeric ranks
# 0 = 10
# J/Q/K = face cards
#
# S = suit
# N = noise/junk
# X = skip
# ESC = save + quit

KEY_TO_LABEL = {
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
    ord("s"): "SUIT",
    ord("n"): "JUNK",
    ord("x"): "SKIP",
}


# ============================================================
# EXISTING LABELS
# ============================================================

def load_existing_labels():
    """
    Read labels.csv so we can resume an interrupted session.
    """

    labels = {}

    if not CSV_PATH.exists():
        return labels

    with open(
        CSV_PATH,
        "r",
        newline=""
    ) as file:
        reader = csv.DictReader(
            file
        )

        for row in reader:
            labels[
                row["filename"]
            ] = row["label"]

    return labels


# ============================================================
# SAVE CSV
# ============================================================

def save_labels(labels):
    """
    Rewrite labels.csv from current label dictionary.
    """

    rows = []

    for filename in sorted(
        labels.keys()
    ):
        rows.append(
            {
                "filename": filename,
                "label": labels[filename],
            }
        )

    with open(
        CSV_PATH,
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "filename",
                "label",
            ]
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


# ============================================================
# COPY INTO CLASS DIRECTORY
# ============================================================

def copy_labeled_image(
    source_path,
    label
):
    """
    Copy each labeled image into:

    center_training_data/labeled/<LABEL>/
    """

    if label == "SKIP":
        return

    destination_dir = (
        LABELED_DIR
        / label
    )

    destination_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    destination = (
        destination_dir
        / source_path.name
    )

    shutil.copy2(
        source_path,
        destination
    )


# ============================================================
# DISPLAY
# ============================================================

def build_display(
    image,
    filename,
    index,
    total
):
    """
    Enlarge tiny candidate crops and add instructions.
    """

    scale = 10

    enlarged = cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_NEAREST
    )

    h, w = enlarged.shape[:2]

    top_space = 105

    display = cv2.copyMakeBorder(
        enlarged,
        top_space,
        25,
        25,
        25,
        cv2.BORDER_CONSTANT,
        value=(30, 30, 30)
    )

    cv2.putText(
        display,
        f"{index}/{total}   {filename}",
        (20, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        display,
        "A 2 3 4 5 6 7 8 9 0(10) J Q K",
        (20, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        display,
        "S=suit   N=junk   X=skip   ESC=quit",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1,
        cv2.LINE_AA
    )

    return display


# ============================================================
# SUMMARY
# ============================================================

def print_summary(labels):
    counts = {}

    for label in labels.values():
        counts[label] = (
            counts.get(
                label,
                0
            )
            + 1
        )

    print()
    print(
        "--- LABELING SUMMARY ---"
    )

    print(
        f"Total labeled/skipped: "
        f"{len(labels)}"
    )

    for label in sorted(
        counts.keys()
    ):
        print(
            f"{label}: "
            f"{counts[label]}"
        )

    print()

    print(
        f"CSV: "
        f"{CSV_PATH.resolve()}"
    )

    print(
        f"Dataset: "
        f"{LABELED_DIR.resolve()}"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    image_paths = sorted(
        INPUT_DIR.glob(
            "*.png"
        )
    )

    if not image_paths:
        print(
            "No center candidate images found."
        )

        print(
            f"Expected images in: "
            f"{INPUT_DIR.resolve()}"
        )

        return

    labels = (
        load_existing_labels()
    )

    remaining = [
        path
        for path in image_paths
        if path.name not in labels
    ]

    print(
        f"Candidate images: "
        f"{len(image_paths)}"
    )

    print(
        f"Already labeled: "
        f"{len(labels)}"
    )

    print(
        f"Remaining: "
        f"{len(remaining)}"
    )

    print()
    print(
        "Controls:"
    )

    print(
        "A,2-9,0,J,Q,K = rank"
    )

    print(
        "S = suit"
    )

    print(
        "N = junk/noise"
    )

    print(
        "X = skip"
    )

    print(
        "ESC = save and quit"
    )

    print()

    if not remaining:
        print(
            "Everything is already labeled."
        )

        print_summary(
            labels
        )

        return

    cv2.namedWindow(
        "Center Candidate Labeler",
        cv2.WINDOW_NORMAL
    )

    for position, path in enumerate(
        remaining,
        start=1
    ):
        image = cv2.imread(
            str(path)
        )

        if image is None:
            labels[
                path.name
            ] = "SKIP"

            continue

        display = build_display(
            image,
            path.name,
            position,
            len(remaining)
        )

        while True:
            cv2.imshow(
                "Center Candidate Labeler",
                display
            )

            key = (
                cv2.waitKey(0)
                & 0xFF
            )

            # ESC
            if key == 27:
                save_labels(
                    labels
                )

                cv2.destroyAllWindows()

                print(
                    "Stopped early. "
                    "Progress saved."
                )

                print_summary(
                    labels
                )

                return

            # Allow uppercase keys too.
            if (
                ord("A")
                <= key
                <= ord("Z")
            ):
                key = ord(
                    chr(key).lower()
                )

            if key not in KEY_TO_LABEL:
                continue

            label = KEY_TO_LABEL[
                key
            ]

            labels[
                path.name
            ] = label

            copy_labeled_image(
                path,
                label
            )

            print(
                f"{path.name} -> "
                f"{label}"
            )

            # Save after every image.
            save_labels(
                labels
            )

            break

    cv2.destroyAllWindows()

    print()
    print(
        "All candidates processed."
    )

    print_summary(
        labels
    )


if __name__ == "__main__":
    main()