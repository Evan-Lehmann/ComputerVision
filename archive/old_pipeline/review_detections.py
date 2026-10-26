from pathlib import Path
import csv
import random
import re

import cv2


# ============================================================
# PATHS
# ============================================================

DEBUG_DIR = Path("card_detection/debug")
FRAMES_CSV = Path("card_detection/frames.csv")

OUTPUT_CSV = Path("card_detection/review.csv")


# ============================================================
# CONFIG
# ============================================================

# Number of examples to inspect for each detected-card count.
SAMPLES_PER_COUNT = 5

# Keep deterministic so rerunning gives the same sample.
RANDOM_SEED = 42


# ============================================================
# LOAD FRAME DATA
# ============================================================

def load_frame_counts():
    rows = []

    with open(FRAMES_CSV, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append(
                {
                    "frame_number": int(row["frame_number"]),
                    "filename": row["filename"],
                    "detections": int(row["detections"]),
                    "full_cards": int(row["full_cards"]),
                    "anchors": int(row["anchors"]),
                }
            )

    return rows


# ============================================================
# DEBUG IMAGE LOOKUP
# ============================================================

def get_debug_images():
    images = {}

    pattern = re.compile(
        r"frame_(\d+)_count_(\d+)\.png"
    )

    for path in DEBUG_DIR.glob("*.png"):
        match = pattern.fullmatch(path.name)

        if not match:
            continue

        frame_number = int(
            match.group(1)
        )

        count = int(
            match.group(2)
        )

        images[frame_number] = {
            "path": path,
            "count": count,
        }

    return images


# ============================================================
# SAMPLE FRAMES
# ============================================================

def build_review_sample(
    frame_rows,
    debug_images,
):
    random.seed(
        RANDOM_SEED
    )

    by_count = {}

    for row in frame_rows:
        frame_number = row[
            "frame_number"
        ]

        if frame_number not in debug_images:
            continue

        count = row[
            "detections"
        ]

        by_count.setdefault(
            count,
            []
        ).append(
            row
        )

    samples = []

    for count in sorted(
        by_count.keys()
    ):
        candidates = (
            by_count[count]
        )

        amount = min(
            SAMPLES_PER_COUNT,
            len(candidates),
        )

        selected = random.sample(
            candidates,
            amount,
        )

        samples.extend(
            selected
        )

    samples.sort(
        key=lambda row:
            row["frame_number"]
    )

    return samples


# ============================================================
# DISPLAY
# ============================================================

def add_instructions(
    image,
    frame_number,
    detected_count,
    progress,
):
    output = image.copy()

    lines = [
        (
            f"Frame {frame_number} | "
            f"Detector count: {detected_count}"
        ),
        (
            f"Review {progress}"
        ),
        (
            "C correct | M missed | "
            "F false | B both"
        ),
        (
            "N actually no cards | "
            "U unsure | Q quit"
        ),
    ]

    y = 30

    for line in lines:
        cv2.putText(
            output,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        y += 30

    return output


# ============================================================
# REVIEW
# ============================================================

def review(samples, debug_images):
    results = []

    key_map = {
        ord("c"): "CORRECT",
        ord("m"): "MISSED",
        ord("f"): "FALSE",
        ord("b"): "BOTH",
        ord("n"): "NO_CARDS",
        ord("u"): "UNSURE",
    }

    total = len(samples)

    for index, row in enumerate(
        samples,
        start=1,
    ):
        frame_number = row[
            "frame_number"
        ]

        info = debug_images[
            frame_number
        ]

        image = cv2.imread(
            str(info["path"])
        )

        if image is None:
            continue

        while True:
            display = add_instructions(
                image,
                frame_number,
                row["detections"],
                f"{index}/{total}",
            )

            cv2.imshow(
                "Card Detector Review",
                display,
            )

            key = (
                cv2.waitKey(0)
                & 0xFF
            )

            # Q
            if key == ord("q"):
                cv2.destroyAllWindows()

                return results

            # Allow uppercase too
            if 65 <= key <= 90:
                key += 32

            if key not in key_map:
                continue

            label = key_map[key]

            results.append(
                {
                    "frame_number":
                        frame_number,

                    "source_filename":
                        row["filename"],

                    "detected_count":
                        row["detections"],

                    "full_cards":
                        row["full_cards"],

                    "anchors":
                        row["anchors"],

                    "review":
                        label,
                }
            )

            print(
                f"{index:02d}/{total} "
                f"frame={frame_number:06d} "
                f"count={row['detections']} "
                f"-> {label}"
            )

            break

    cv2.destroyAllWindows()

    return results


# ============================================================
# SAVE
# ============================================================

def save_results(results):
    if not results:
        return

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_number",
                "source_filename",
                "detected_count",
                "full_cards",
                "anchors",
                "review",
            ],
        )

        writer.writeheader()
        writer.writerows(
            results
        )


# ============================================================
# SUMMARY
# ============================================================

def print_summary(results):
    if not results:
        print(
            "\nNo frames reviewed."
        )
        return

    counts = {}

    for row in results:
        label = row[
            "review"
        ]

        counts[label] = (
            counts.get(
                label,
                0
            )
            + 1
        )

    total = len(results)

    print()
    print(
        "--- DETECTOR REVIEW COMPLETE ---"
    )

    print(
        f"Frames reviewed: {total}"
    )

    for label in [
        "CORRECT",
        "MISSED",
        "FALSE",
        "BOTH",
        "NO_CARDS",
        "UNSURE",
    ]:
        count = counts.get(
            label,
            0
        )

        percentage = (
            100.0
            * count
            / total
        )

        print(
            f"{label:10s}: "
            f"{count:2d} "
            f"({percentage:5.1f}%)"
        )

    print()
    print(
        f"Saved review: "
        f"{OUTPUT_CSV.resolve()}"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    if not FRAMES_CSV.exists():
        print(
            "Missing card_detection/frames.csv"
        )
        return

    if not DEBUG_DIR.exists():
        print(
            "Missing card_detection/debug/"
        )
        return

    frame_rows = (
        load_frame_counts()
    )

    debug_images = (
        get_debug_images()
    )

    samples = build_review_sample(
        frame_rows,
        debug_images,
    )

    print(
        f"Reviewing {len(samples)} "
        f"representative frames."
    )

    print(
        "Use C/M/F/B/N/U. "
        "Press Q to stop."
    )

    results = review(
        samples,
        debug_images,
    )

    save_results(
        results
    )

    print_summary(
        results
    )


if __name__ == "__main__":
    main()