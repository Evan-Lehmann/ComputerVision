from pathlib import Path

import cv2
import numpy as np


# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = Path("captures")

OUTPUT_ROOT = Path("harvested_center")
RANKS_DIR = OUTPUT_ROOT / "ranks"
DEBUG_DIR = OUTPUT_ROOT / "debug"

RANKS_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

# Normalized size for comparing rank crops
NORMALIZED_SIZE = (40, 50)

# Correlation threshold for duplicate rank crops
DUPLICATE_SIMILARITY = 0.94


# ============================================================
# CENTER REGION
# ============================================================

def crop_center_region(image):
    """
    Restrict processing tightly to the center flipped-card pile.

    Excludes:
    - action buttons above
    - betting chip below
    - side bets / unrelated UI
    """

    h, w = image.shape[:2]

    # Much tighter horizontal range
    x1 = int(w * 0.43)
    x2 = int(w * 0.57)

    # Center-card pile sits roughly here
    y1 = int(h * 0.67)
    y2 = int(h * 0.75)

    roi = image[y1:y2, x1:x2]

    return roi, (x1, y1)


# ============================================================
# LOCATE CARD PILE
# ============================================================

def find_card_pile(center_roi):
    """
    Locate the bright overlapping-card pile inside
    the center ROI.
    """

    hsv = cv2.cvtColor(
        center_roi,
        cv2.COLOR_BGR2HSV
    )

    white_mask = cv2.inRange(
        hsv,
        np.array([0, 0, 170]),
        np.array([179, 120, 255])
    )

    # Join nearby white fragments from overlapping cards.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (7, 3)
    )

    merged = cv2.morphologyEx(
        white_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    merged = cv2.dilate(
        merged,
        kernel,
        iterations=1
    )

    contours, _ = cv2.findContours(
        merged,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        area = w * h

        # Center piles are small but wider than a single glyph.
        if area < 150:
            continue

        if w < 20 or h < 10:
            continue

        candidates.append(
            (area, x, y, w, h)
        )

    if not candidates:
        return None

    # The largest suitable white object should be the pile.
    candidates.sort(reverse=True)

    _, x, y, w, h = candidates[0]

    # Slight padding
    pad_x = 4
    pad_y = 3

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)

    x2 = min(
        center_roi.shape[1],
        x + w + pad_x
    )

    y2 = min(
        center_roi.shape[0],
        y + h + pad_y
    )

    return x1, y1, x2, y2


# ============================================================
# RANK MASK
# ============================================================

def create_rank_mask(top_band):
    """
    Detect both black and red rank characters.
    """

    hsv = cv2.cvtColor(
        top_band,
        cv2.COLOR_BGR2HSV
    )

    gray = cv2.cvtColor(
        top_band,
        cv2.COLOR_BGR2GRAY
    )

    # Black/dark ranks
    dark_mask = cv2.inRange(
        gray,
        0,
        155
    )

    # Red ranks: low hue range
    red1 = cv2.inRange(
        hsv,
        np.array([0, 70, 50]),
        np.array([15, 255, 255])
    )

    # Red ranks: high hue range
    red2 = cv2.inRange(
        hsv,
        np.array([165, 70, 50]),
        np.array([179, 255, 255])
    )

    red_mask = cv2.bitwise_or(
        red1,
        red2
    )

    mask = cv2.bitwise_or(
        dark_mask,
        red_mask
    )

    # Remove isolated single pixels
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (2, 2)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    return mask


# ============================================================
# CHARACTER DETECTION
# ============================================================

def detect_rank_components(top_band):
    """
    Locate individual rank-character components.

    For most ranks this is one component.
    '10' may initially appear as two adjacent components.
    """

    mask = create_rank_mask(
        top_band
    )

    count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    boxes = []

    for i in range(1, count):
        x = stats[
            i,
            cv2.CC_STAT_LEFT
        ]

        y = stats[
            i,
            cv2.CC_STAT_TOP
        ]

        w = stats[
            i,
            cv2.CC_STAT_WIDTH
        ]

        h = stats[
            i,
            cv2.CC_STAT_HEIGHT
        ]

        area = stats[
            i,
            cv2.CC_STAT_AREA
        ]

        # Rank glyphs are tiny in these screenshots.
        if area < 3:
            continue

        if h < 4:
            continue

        if h > top_band.shape[0]:
            continue

        if w > 25:
            continue

        boxes.append(
            [x, y, w, h]
        )

    boxes.sort(
        key=lambda b: b[0]
    )

    return boxes, mask


# ============================================================
# GROUP "10"
# ============================================================

def group_adjacent_components(boxes):
    """
    Combine very close neighboring components.

    This primarily handles the two-character rank '10'.
    """

    if not boxes:
        return []

    grouped = []

    current = boxes[0]

    for box in boxes[1:]:
        x, y, w, h = box

        cx, cy, cw, ch = current

        current_right = (
            cx + cw
        )

        gap = x - current_right

        vertical_overlap = (
            min(cy + ch, y + h)
            - max(cy, y)
        )

        min_height = min(
            ch,
            h
        )

        overlap_ratio = (
            vertical_overlap
            / max(min_height, 1)
        )

        # Two digits belonging to the same "10"
        # should be extremely close horizontally.
        if (
            0 <= gap <= 2
            and overlap_ratio >= 0.5
        ):
            new_x = min(
                cx,
                x
            )

            new_y = min(
                cy,
                y
            )

            new_right = max(
                cx + cw,
                x + w
            )

            new_bottom = max(
                cy + ch,
                y + h
            )

            current = [
                new_x,
                new_y,
                new_right - new_x,
                new_bottom - new_y
            ]

        else:
            grouped.append(
                current
            )

            current = box

    grouped.append(
        current
    )

    return grouped


# ============================================================
# NORMALIZATION / DEDUPLICATION
# ============================================================

def normalize_rank(crop):
    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2GRAY
    )

    resized = cv2.resize(
        gray,
        NORMALIZED_SIZE,
        interpolation=cv2.INTER_AREA
    )

    return resized


def similarity_score(a, b):
    a = a.astype(
        np.float32
    )

    b = b.astype(
        np.float32
    )

    a_std = a.std()
    b_std = b.std()

    if (
        a_std < 1e-6
        or b_std < 1e-6
    ):
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


def is_duplicate(
    crop,
    saved_ranks
):
    normalized = normalize_rank(
        crop
    )

    best_score = None

    for previous in saved_ranks:
        score = similarity_score(
            normalized,
            previous
        )

        if (
            best_score is None
            or score > best_score
        ):
            best_score = score

        if (
            score
            >= DUPLICATE_SIMILARITY
        ):
            return True, score

    return False, best_score


# ============================================================
# SAFE CROP
# ============================================================

def padded_crop(
    image,
    x,
    y,
    w,
    h,
    padding=3
):
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
        INPUT_DIR.glob(
            "*.png"
        )
    )

    if not image_paths:
        print(
            "No capture screenshots found."
        )
        return

    print(
        f"Found {len(image_paths)} captured frames."
    )

    saved_ranks = []

    piles_found = 0
    candidates_found = 0
    duplicates = 0
    saved_count = 0

    for frame_number, path in enumerate(
        image_paths,
        start=1
    ):
        image = cv2.imread(
            str(path)
        )

        if image is None:
            continue

        center, offset = (
            crop_center_region(
                image
            )
        )

        offset_x, offset_y = offset

        pile_box = find_card_pile(
            center
        )

        if pile_box is None:
            continue

        piles_found += 1

        px1, py1, px2, py2 = (
            pile_box
        )

        pile = center[
            py1:py2,
            px1:px2
        ]

        if pile.size == 0:
            continue

        # Rank symbols are concentrated in
        # the upper portion of the fanned pile.
        pile_h = pile.shape[0]

        top_band_height = max(
            1,
            int(
                pile_h * 0.55
            )
        )

        top_band = pile[
            :top_band_height,
            :
        ]

        boxes, rank_mask = (
            detect_rank_components(
                top_band
            )
        )

        boxes = (
            group_adjacent_components(
                boxes
            )
        )

        annotated = image.copy()

        # Show pile in blue
        full_px1 = (
            offset_x + px1
        )

        full_py1 = (
            offset_y + py1
        )

        cv2.rectangle(
            annotated,
            (
                full_px1,
                full_py1
            ),
            (
                offset_x + px2,
                offset_y + py2
            ),
            (255, 100, 0),
            2
        )

        for x, y, w, h in boxes:
            candidates_found += 1

            crop = padded_crop(
                top_band,
                x,
                y,
                w,
                h,
                padding=2
            )

            if crop.size == 0:
                continue

            full_x = (
                offset_x
                + px1
                + x
            )

            full_y = (
                offset_y
                + py1
                + y
            )

            duplicate, score = (
                is_duplicate(
                    crop,
                    saved_ranks
                )
            )

            if duplicate:
                duplicates += 1

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
                    1
                )

                continue

            normalized = (
                normalize_rank(
                    crop
                )
            )

            saved_ranks.append(
                normalized
            )

            saved_count += 1

            destination = (
                RANKS_DIR
                / f"rank_{saved_count:04d}.png"
            )

            cv2.imwrite(
                str(destination),
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
                1
            )

            cv2.putText(
                annotated,
                f"R{saved_count}",
                (
                    full_x,
                    max(
                        15,
                        full_y - 4
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 255, 0),
                1
            )

        debug_path = (
            DEBUG_DIR
            / f"frame_{frame_number:04d}.png"
        )

        cv2.imwrite(
            str(debug_path),
            annotated
        )

    print()
    print(
        "--- CENTER HARVEST COMPLETE ---"
    )

    print(
        f"Frames scanned: "
        f"{len(image_paths)}"
    )

    print(
        f"Center piles found: "
        f"{piles_found}"
    )

    print(
        f"Rank candidates found: "
        f"{candidates_found}"
    )

    print(
        f"Duplicates rejected: "
        f"{duplicates}"
    )

    print(
        f"Unique rank crops saved: "
        f"{saved_count}"
    )

    print(
        f"Ranks: "
        f"{RANKS_DIR.resolve()}"
    )

    print(
        f"Debug: "
        f"{DEBUG_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()