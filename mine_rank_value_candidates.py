from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

from test_rank_detector_unseen import detect_full_frame


ROOT = Path(__file__).resolve().parent

CAPTURES = ROOT / "captures"

MODEL_PATH = (
    ROOT
    / "rank_detector_runs"
    / "rank_detector_v2"
    / "weights"
    / "best.pt"
)

OUTPUT = ROOT / "rank_value_candidates"

EVERY_N_FRAMES = 5

# Prevent thousands of almost identical crops.
HASH_DISTANCE_THRESHOLD = 8

# Padding around detected rank.
PAD_FRAC = 0.20


def dhash(image, hash_size=8):
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    resized = cv2.resize(
        gray,
        (hash_size + 1, hash_size),
    )

    diff = (
        resized[:, 1:]
        >
        resized[:, :-1]
    )

    value = 0

    for bit in diff.flatten():
        value = (
            value << 1
        ) | int(bit)

    return value


def hamming(a, b):
    return bin(
        a ^ b
    ).count("1")


def normalize_crop(crop):
    if crop.size == 0:
        return None

    return cv2.resize(
        crop,
        (64, 64),
        interpolation=cv2.INTER_CUBIC,
    )


def main():
    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Clear old candidate images only.
    for path in OUTPUT.glob("*.png"):
        path.unlink()

    model = YOLO(
        str(MODEL_PATH)
    )

    frames = sorted(
        CAPTURES.glob("*.png")
    )

    selected = frames[
        ::EVERY_N_FRAMES
    ]

    print(
        f"Frames available: {len(frames)}"
    )

    print(
        f"Frames sampled: {len(selected)}"
    )

    saved_hashes = []

    candidate_number = 0

    for frame_index, frame_path in enumerate(
        selected,
        start=1,
    ):
        image = cv2.imread(
            str(frame_path)
        )

        if image is None:
            continue

        (
            _raw,
            _stage1,
            _clustered,
            detections,
            _duplicate_groups,
            _same_tile_removed,
            _final_overlap_removed,
        ) = detect_full_frame(
            model,
            image,
        )

        height, width = (
            image.shape[:2]
        )

        for det_index, det in enumerate(
            detections,
            start=1,
        ):
            x1 = int(det["x1"])
            y1 = int(det["y1"])
            x2 = int(det["x2"])
            y2 = int(det["y2"])

            box_w = x2 - x1
            box_h = y2 - y1

            pad_x = max(
                2,
                int(box_w * PAD_FRAC),
            )

            pad_y = max(
                2,
                int(box_h * PAD_FRAC),
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

            normalized = (
                normalize_crop(crop)
            )

            if normalized is None:
                continue

            crop_hash = dhash(
                normalized
            )

            # Skip crops extremely similar
            # to something already saved.
            duplicate = any(
                hamming(
                    crop_hash,
                    previous_hash,
                )
                <= HASH_DISTANCE_THRESHOLD
                for previous_hash
                in saved_hashes
            )

            if duplicate:
                continue

            saved_hashes.append(
                crop_hash
            )

            candidate_number += 1

            output_name = (
                f"candidate_"
                f"{candidate_number:04d}_"
                f"{frame_path.stem}_"
                f"{det_index:02d}.png"
            )

            cv2.imwrite(
                str(
                    OUTPUT
                    / output_name
                ),
                crop,
            )

        print(
            f"{frame_index:03d}/"
            f"{len(selected):03d} "
            f"unique candidates: "
            f"{candidate_number}",
            end="\r",
        )

    print()
    print()
    print(
        "--- MINING COMPLETE ---"
    )

    print(
        f"Unique rank candidates: "
        f"{candidate_number}"
    )

    print(
        f"Saved to: {OUTPUT}"
    )


if __name__ == "__main__":
    main()