from pathlib import Path
import csv
import math
import re
import shutil

import cv2
import numpy as np


# ============================================================
# PATHS
# ============================================================

CAPTURES_DIR = Path("captures")
OUTPUT_DIR = Path("detection_dataset")

TRAIN_DIR = OUTPUT_DIR / "train" / "images"
VAL_DIR = OUTPUT_DIR / "val" / "images"

MANIFEST_PATH = OUTPUT_DIR / "manifest.csv"
EVENTS_PATH = OUTPUT_DIR / "event_bursts.csv"

TRAIN_SHEET = OUTPUT_DIR / "train_contact_sheet.jpg"
VAL_SHEET = OUTPUT_DIR / "val_contact_sheet.jpg"


# ============================================================
# CONFIG
# ============================================================

# Compare with frame ~2 seconds earlier.
LAG_FRAMES = 4

# Adaptive threshold. No fixed change-score threshold.
EVENT_PERCENTILE = 90

# High-change frames this close together belong to one event.
EVENT_MERGE_GAP = 8

# After an event finishes, look forward for the most stable
# representative frame.
STABLE_SEARCH_START = 2
STABLE_SEARCH_END = 12

# Descriptor area used ONLY for offline dataset sampling.
X1_FRAC = 0.15
X2_FRAC = 0.85
Y1_FRAC = 0.10
Y2_FRAC = 0.78

DESCRIPTOR_WIDTH = 160
DESCRIPTOR_HEIGHT = 110

# Last ~25% of selected chronological states becomes validation.
VAL_FRACTION = 0.25

# Leave one selected state unused between train and validation.
USE_GUARD_STATE = True

THUMB_WIDTH = 300
CONTACT_COLUMNS = 5


# ============================================================
# FILES
# ============================================================

def natural_sort_key(path):
    parts = re.split(r"(\d+)", path.name)

    return [
        int(part) if part.isdigit() else part.lower()
        for part in parts
    ]


def get_capture_paths():
    paths = []

    for pattern in (
        "*.png",
        "*.jpg",
        "*.jpeg",
    ):
        paths.extend(
            CAPTURES_DIR.glob(pattern)
        )

    return sorted(
        paths,
        key=natural_sort_key,
    )


# ============================================================
# DESCRIPTOR
# ============================================================

def create_descriptor(frame):
    h, w = frame.shape[:2]

    x1 = int(w * X1_FRAC)
    x2 = int(w * X2_FRAC)

    y1 = int(h * Y1_FRAC)
    y2 = int(h * Y2_FRAC)

    crop = frame[
        y1:y2,
        x1:x2
    ]

    crop = cv2.resize(
        crop,
        (
            DESCRIPTOR_WIDTH,
            DESCRIPTOR_HEIGHT,
        ),
        interpolation=cv2.INTER_AREA,
    )

    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2GRAY,
    )

    gray = cv2.GaussianBlur(
        gray,
        (7, 7),
        0,
    )

    gray_float = (
        gray.astype(np.float32)
        / 255.0
    )

    edges = cv2.Canny(
        gray,
        45,
        120,
    ).astype(np.float32) / 255.0

    hsv = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2HSV,
    )

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    bright = (
        (value >= 145)
        & (saturation <= 115)
    ).astype(np.float32)

    return (
        gray_float,
        edges,
        bright,
    )


def descriptor_distance(a, b):
    gray_a, edges_a, bright_a = a
    gray_b, edges_b, bright_b = b

    gray_change = np.mean(
        np.abs(
            gray_a - gray_b
        )
    )

    edge_change = np.mean(
        np.abs(
            edges_a - edges_b
        )
    )

    bright_change = np.mean(
        np.abs(
            bright_a - bright_b
        )
    )

    return float(
        0.45 * gray_change
        + 0.25 * edge_change
        + 0.30 * bright_change
    )


# ============================================================
# LOAD RECORDING
# ============================================================

def load_recording(paths):
    frames = []

    print(
        f"Reading {len(paths)} captures..."
    )

    for index, path in enumerate(
        paths
    ):
        image = cv2.imread(
            str(path)
        )

        if image is None:
            continue

        frames.append(
            {
                "index": index,
                "path": path,
                "descriptor":
                    create_descriptor(image),
            }
        )

        if (
            (index + 1) % 50 == 0
            or index + 1 == len(paths)
        ):
            print(
                f"  processed "
                f"{index + 1}/{len(paths)}"
            )

    return frames


# ============================================================
# CHANGE SCORES
# ============================================================

def calculate_change_scores(frames):
    scores = np.zeros(
        len(frames),
        dtype=np.float32,
    )

    for i in range(
        LAG_FRAMES,
        len(frames)
    ):
        scores[i] = (
            descriptor_distance(
                frames[i]["descriptor"],
                frames[
                    i - LAG_FRAMES
                ]["descriptor"],
            )
        )

    return scores


# ============================================================
# EVENT BURSTS
# ============================================================

def find_event_bursts(
    frames,
    scores,
):
    usable_scores = scores[
        LAG_FRAMES:
    ]

    threshold = float(
        np.percentile(
            usable_scores,
            EVENT_PERCENTILE,
        )
    )

    high_change_indices = [
        index
        for index in range(
            LAG_FRAMES,
            len(frames)
        )
        if scores[index]
        >= threshold
    ]

    if not high_change_indices:
        return [], threshold

    bursts = []

    current = [
        high_change_indices[0]
    ]

    for index in high_change_indices[1:]:

        if (
            index - current[-1]
            <= EVENT_MERGE_GAP
        ):
            current.append(index)

        else:
            bursts.append(current)

            current = [index]

    bursts.append(current)

    return bursts, threshold


# ============================================================
# STABLE REPRESENTATIVE
# ============================================================

def one_frame_motion(
    frames,
    index,
):
    if index <= 0:
        return 0.0

    return descriptor_distance(
        frames[index]["descriptor"],
        frames[index - 1]["descriptor"],
    )


def choose_stable_after_event(
    frames,
    burst,
):
    event_end = max(burst)

    start = min(
        len(frames) - 1,
        event_end
        + STABLE_SEARCH_START,
    )

    end = min(
        len(frames) - 1,
        event_end
        + STABLE_SEARCH_END,
    )

    if start > end:
        return frames[event_end]

    candidates = range(
        start,
        end + 1
    )

    best_index = min(
        candidates,
        key=lambda i:
            one_frame_motion(
                frames,
                i,
            ),
    )

    return frames[
        best_index
    ]


def choose_initial_state(
    frames,
    first_burst,
):
    """
    Find a calm frame before the first major event.
    """

    end = max(
        1,
        min(first_burst) - 2,
    )

    candidates = range(
        1,
        end + 1,
    )

    best_index = min(
        candidates,
        key=lambda i:
            one_frame_motion(
                frames,
                i,
            ),
    )

    return frames[
        best_index
    ]


# ============================================================
# REPRESENTATIVE STATES
# ============================================================

def build_representatives(
    frames,
    bursts,
):
    if not bursts:
        return []

    representatives = [
        choose_initial_state(
            frames,
            bursts[0],
        )
    ]

    for burst in bursts:

        representative = (
            choose_stable_after_event(
                frames,
                burst,
            )
        )

        # Avoid exact duplicate source frames.
        if (
            representative["index"]
            != representatives[-1]["index"]
        ):
            representatives.append(
                representative
            )

    return representatives


# ============================================================
# CHRONOLOGICAL TRAIN / VAL SPLIT
# ============================================================

def split_representatives(
    representatives
):
    n = len(
        representatives
    )

    if n < 3:
        return (
            representatives,
            [],
            None,
        )

    val_count = max(
        2,
        int(
            round(
                n * VAL_FRACTION
            )
        ),
    )

    val_count = min(
        val_count,
        n - 1,
    )

    val_start = (
        n - val_count
    )

    guard = None

    if (
        USE_GUARD_STATE
        and val_start >= 2
    ):
        guard = representatives[
            val_start - 1
        ]

        train = representatives[
            :val_start - 1
        ]

    else:
        train = representatives[
            :val_start
        ]

    val = representatives[
        val_start:
    ]

    return (
        train,
        val,
        guard,
    )


# ============================================================
# COPY DATASET
# ============================================================

def copy_split(
    selected,
    destination,
    split_name,
):
    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []

    for number, item in enumerate(
        selected,
        start=1
    ):
        suffix = (
            item["path"]
            .suffix
            .lower()
        )

        filename = (
            f"{split_name}_"
            f"{number:03d}"
            f"_frame_"
            f"{item['index'] + 1:06d}"
            f"{suffix}"
        )

        shutil.copy2(
            item["path"],
            destination / filename,
        )

        records.append(
            {
                "split":
                    split_name,

                "dataset_filename":
                    filename,

                "source_filename":
                    item["path"].name,

                "source_frame_number":
                    item["index"] + 1,

                "source_capture_index":
                    item["index"],
            }
        )

    return records


# ============================================================
# CONTACT SHEETS
# ============================================================

def create_contact_sheet(
    records,
    image_dir,
    output_path,
):
    if not records:
        return

    thumbnails = []

    for record in records:

        path = (
            image_dir
            / record[
                "dataset_filename"
            ]
        )

        image = cv2.imread(
            str(path)
        )

        if image is None:
            continue

        h, w = image.shape[:2]

        scale = (
            THUMB_WIDTH
            / float(w)
        )

        thumb_h = int(
            h * scale
        )

        thumb = cv2.resize(
            image,
            (
                THUMB_WIDTH,
                thumb_h,
            ),
            interpolation=cv2.INTER_AREA,
        )

        label_height = 36

        canvas = np.zeros(
            (
                thumb_h
                + label_height,
                THUMB_WIDTH,
                3,
            ),
            dtype=np.uint8,
        )

        canvas[
            label_height:
        ] = thumb

        cv2.putText(
            canvas,
            (
                f"{record['split']} "
                f"source "
                f"{record['source_frame_number']}"
            ),
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        thumbnails.append(
            canvas
        )

    if not thumbnails:
        return

    max_height = max(
        thumb.shape[0]
        for thumb in thumbnails
    )

    rows = math.ceil(
        len(thumbnails)
        / CONTACT_COLUMNS
    )

    sheet = np.zeros(
        (
            rows * max_height,
            CONTACT_COLUMNS
            * THUMB_WIDTH,
            3,
        ),
        dtype=np.uint8,
    )

    for i, thumb in enumerate(
        thumbnails
    ):
        row = (
            i
            // CONTACT_COLUMNS
        )

        col = (
            i
            % CONTACT_COLUMNS
        )

        y = (
            row
            * max_height
        )

        x = (
            col
            * THUMB_WIDTH
        )

        sheet[
            y:y + thumb.shape[0],
            x:x + thumb.shape[1],
        ] = thumb

    cv2.imwrite(
        str(output_path),
        sheet,
    )


# ============================================================
# CSV OUTPUT
# ============================================================

def save_manifest(records):
    with open(
        MANIFEST_PATH,
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "split",
                "dataset_filename",
                "source_filename",
                "source_frame_number",
                "source_capture_index",
            ],
        )

        writer.writeheader()
        writer.writerows(
            records
        )


def save_event_bursts(
    frames,
    bursts,
    scores,
):
    with open(
        EVENTS_PATH,
        "w",
        newline="",
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "event",
                "start_frame",
                "end_frame",
                "peak_frame",
                "peak_change",
            ]
        )

        for number, burst in enumerate(
            bursts,
            start=1,
        ):
            peak = max(
                burst,
                key=lambda i:
                    scores[i],
            )

            writer.writerow(
                [
                    number,
                    min(burst) + 1,
                    max(burst) + 1,
                    peak + 1,
                    f"{scores[peak]:.6f}",
                ]
            )


# ============================================================
# MAIN
# ============================================================

def main():

    paths = get_capture_paths()

    if not paths:
        print(
            "No captures found."
        )
        return

    frames = load_recording(
        paths
    )

    scores = calculate_change_scores(
        frames
    )

    bursts, threshold = (
        find_event_bursts(
            frames,
            scores,
        )
    )

    representatives = (
        build_representatives(
            frames,
            bursts,
        )
    )

    (
        train,
        val,
        guard,
    ) = split_representatives(
        representatives
    )

    if OUTPUT_DIR.exists():
        shutil.rmtree(
            OUTPUT_DIR
        )

    train_records = copy_split(
        train,
        TRAIN_DIR,
        "train",
    )

    val_records = copy_split(
        val,
        VAL_DIR,
        "val",
    )

    all_records = (
        train_records
        + val_records
    )

    save_manifest(
        all_records
    )

    save_event_bursts(
        frames,
        bursts,
        scores,
    )

    create_contact_sheet(
        train_records,
        TRAIN_DIR,
        TRAIN_SHEET,
    )

    create_contact_sheet(
        val_records,
        VAL_DIR,
        VAL_SHEET,
    )

    print()
    print(
        "--- EVENT-BASED DATASET COMPLETE ---"
    )

    print(
        f"Original captures: "
        f"{len(frames)}"
    )

    print(
        f"Adaptive event threshold "
        f"(P{EVENT_PERCENTILE}): "
        f"{threshold:.6f}"
    )

    print(
        f"Event bursts found: "
        f"{len(bursts)}"
    )

    print(
        f"Stable representative states: "
        f"{len(representatives)}"
    )

    print(
        f"Training images: "
        f"{len(train_records)}"
    )

    print(
        f"Validation images: "
        f"{len(val_records)}"
    )

    if guard is not None:
        print(
            f"Guard state excluded: "
            f"source frame "
            f"{guard['index'] + 1}"
        )

    print()
    print(
        "Selected source frames:"
    )

    print(
        [
            item["index"] + 1
            for item in representatives
        ]
    )

    print()
    print(
        f"Train sheet: "
        f"{TRAIN_SHEET.resolve()}"
    )

    print(
        f"Val sheet: "
        f"{VAL_SHEET.resolve()}"
    )

    print()
    print(
        "Do NOT label yet."
    )


if __name__ == "__main__":
    main()