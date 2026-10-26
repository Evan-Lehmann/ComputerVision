from pathlib import Path

import cv2
import numpy as np


# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = Path("captures")

HARVEST_ROOT = Path("harvested")
CARDS_DIR = HARVEST_ROOT / "cards"
DEBUG_DIR = HARVEST_ROOT / "debug"

CARDS_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

NORMALIZED_SIZE = (80, 116)

# Face-up cards in your screenshots are mostly white.
# Card backs are noticeably less white.
MIN_WHITE_RATIO = 0.60

# Correlation threshold for duplicate detection.
# Higher = stricter match requirement.
DUPLICATE_SIMILARITY = 0.955


# ============================================================
# TABLE REGION
# ============================================================

def crop_table_region(image):
    """
    Restrict detection to the blackjack table area.

    This avoids browser chrome, Dock icons, sidebars,
    and unrelated UI elements.
    """

    h, w = image.shape[:2]

    x1 = int(w * 0.07)
    x2 = int(w * 0.92)

    y1 = int(h * 0.24)
    y2 = int(h * 0.92)

    roi = image[y1:y2, x1:x2]

    return roi, (x1, y1)


# ============================================================
# FACE-UP CARD FILTER
# ============================================================

def white_ratio(card):
    """
    Return the fraction of pixels that are bright/white.
    """

    hsv = cv2.cvtColor(
        card,
        cv2.COLOR_BGR2HSV
    )

    white_mask = cv2.inRange(
        hsv,
        np.array([0, 0, 170]),
        np.array([179, 100, 255])
    )

    white_pixels = np.count_nonzero(
        white_mask
    )

    total_pixels = white_mask.size

    if total_pixels == 0:
        return 0.0

    return white_pixels / total_pixels


def looks_face_up(card):
    """
    Reject card backs and dark UI candidates.
    """

    return white_ratio(card) >= MIN_WHITE_RATIO


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_card(card):
    """
    Convert a card crop into a standardized grayscale image.

    This makes duplicate comparison less sensitive to
    tiny changes in position or scale.
    """

    gray = cv2.cvtColor(
        card,
        cv2.COLOR_BGR2GRAY
    )

    resized = cv2.resize(
        gray,
        NORMALIZED_SIZE,
        interpolation=cv2.INTER_AREA
    )

    resized = cv2.GaussianBlur(
        resized,
        (3, 3),
        0
    )

    return resized


# ============================================================
# DUPLICATE DETECTION
# ============================================================

def similarity_score(a, b):
    """
    Compute normalized correlation.

    ~1.0 = extremely similar
    ~0.0 = unrelated
    """

    a = a.astype(np.float32)
    b = b.astype(np.float32)

    a_std = a.std()
    b_std = b.std()

    if a_std < 1e-6 or b_std < 1e-6:
        return 0.0

    a = (
        a - a.mean()
    ) / a_std

    b = (
        b - b.mean()
    ) / b_std

    return float(
        np.mean(a * b)
    )


def is_duplicate(card, saved_cards):
    """
    Compare a candidate against all previously saved cards.
    """

    normalized = normalize_card(
        card
    )

    best_score = None

    for previous in saved_cards:
        score = similarity_score(
            normalized,
            previous
        )

        if (
            best_score is None
            or score > best_score
        ):
            best_score = score

        if score >= DUPLICATE_SIMILARITY:
            return True, score

    return False, best_score


# ============================================================
# CARD DETECTION
# ============================================================

def detect_cards(table_image):
    """
    Detect large upright card-shaped white regions.
    """

    h, w = table_image.shape[:2]
    image_area = h * w

    hsv = cv2.cvtColor(
        table_image,
        cv2.COLOR_BGR2HSV
    )

    # Detect bright card surfaces.
    mask = cv2.inRange(
        hsv,
        np.array([0, 0, 175]),
        np.array([179, 95, 255])
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    detections = []

    for contour in contours:
        x, y, box_w, box_h = cv2.boundingRect(
            contour
        )

        if box_h == 0:
            continue

        area = box_w * box_h

        relative_area = (
            area / image_area
        )

        aspect_ratio = (
            box_w / box_h
        )

        contour_area = cv2.contourArea(
            contour
        )

        rectangularity = (
            contour_area
            / max(area, 1)
        )

        # Typical portrait-oriented playing card.
        if not 0.40 <= aspect_ratio <= 0.90:
            continue

        # Reject tiny interface elements and large panels.
        if not 0.00030 <= relative_area <= 0.025:
            continue

        if rectangularity < 0.68:
            continue

        detections.append(
            (x, y, box_w, box_h)
        )

    return detections


# ============================================================
# CROPPING
# ============================================================

def padded_crop(
    image,
    x,
    y,
    w,
    h,
    padding=3
):
    """
    Add a small margin around detected cards.
    """

    x1 = max(
        0,
        x - padding
    )

    y1 = max(
        0,
        y - padding
    )

    x2 = min(
        image.shape[1],
        x + w + padding
    )

    y2 = min(
        image.shape[0],
        y + h + padding
    )

    return image[
        y1:y2,
        x1:x2
    ]


# ============================================================
# MAIN
# ============================================================

def main():
    image_paths = sorted(
        INPUT_DIR.glob("*.png")
    )

    if not image_paths:
        print(
            f"No screenshots found in: "
            f"{INPUT_DIR.resolve()}"
        )
        return

    print(
        f"Found {len(image_paths)} captured frames."
    )

    saved_cards = []

    candidate_count = 0
    saved_count = 0
    duplicate_count = 0
    rejected_count = 0

    for frame_number, path in enumerate(
        image_paths,
        start=1
    ):
        image = cv2.imread(
            str(path)
        )

        if image is None:
            continue

        table, offset = crop_table_region(
            image
        )

        offset_x, offset_y = offset

        detections = detect_cards(
            table
        )

        annotated = image.copy()

        for x, y, w, h in detections:
            candidate_count += 1

            crop = padded_crop(
                table,
                x,
                y,
                w,
                h
            )

            if crop.size == 0:
                continue

            full_x = (
                x + offset_x
            )

            full_y = (
                y + offset_y
            )

            # --------------------------------------------
            # FACE-DOWN / UI REJECTION
            # --------------------------------------------

            if not looks_face_up(
                crop
            ):
                rejected_count += 1

                cv2.rectangle(
                    annotated,
                    (
                        full_x,
                        full_y
                    ),
                    (
                        full_x + w,
                        full_y + h
                    ),
                    (255, 0, 255),
                    2
                )

                cv2.putText(
                    annotated,
                    "REJECT",
                    (
                        full_x,
                        max(
                            20,
                            full_y - 8
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 0, 255),
                    2
                )

                continue

            # --------------------------------------------
            # DUPLICATE CHECK
            # --------------------------------------------

            duplicate, score = is_duplicate(
                crop,
                saved_cards
            )

            if duplicate:
                duplicate_count += 1

                cv2.rectangle(
                    annotated,
                    (
                        full_x,
                        full_y
                    ),
                    (
                        full_x + w,
                        full_y + h
                    ),
                    (0, 165, 255),
                    2
                )

                cv2.putText(
                    annotated,
                    "DUP",
                    (
                        full_x,
                        max(
                            20,
                            full_y - 8
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 165, 255),
                    2
                )

                continue

            # --------------------------------------------
            # SAVE NEW UNIQUE FACE-UP CARD
            # --------------------------------------------

            normalized = normalize_card(
                crop
            )

            saved_cards.append(
                normalized
            )

            saved_count += 1

            filename = (
                CARDS_DIR
                / f"card_{saved_count:04d}.png"
            )

            cv2.imwrite(
                str(filename),
                crop
            )

            cv2.rectangle(
                annotated,
                (
                    full_x,
                    full_y
                ),
                (
                    full_x + w,
                    full_y + h
                ),
                (0, 255, 0),
                2
            )

            cv2.putText(
                annotated,
                f"NEW {saved_count}",
                (
                    full_x,
                    max(
                        20,
                        full_y - 8
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        # Save debug frames only when something was detected.
        if detections:
            debug_path = (
                DEBUG_DIR
                / f"frame_{frame_number:04d}.png"
            )

            cv2.imwrite(
                str(debug_path),
                annotated
            )

    print()
    print("--- HARVEST COMPLETE ---")

    print(
        f"Frames scanned: "
        f"{len(image_paths)}"
    )

    print(
        f"Candidate detections: "
        f"{candidate_count}"
    )

    print(
        f"Rejected card backs/UI: "
        f"{rejected_count}"
    )

    print(
        f"Duplicates rejected: "
        f"{duplicate_count}"
    )

    print(
        f"Unique face-up cards saved: "
        f"{saved_count}"
    )

    print(
        f"Cards: "
        f"{CARDS_DIR.resolve()}"
    )

    print(
        f"Debug images: "
        f"{DEBUG_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()