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

NORMALIZED_SIZE = (48, 48)

# Very conservative duplicate removal.
# Better to save an extra copy than collapse different ranks.
DUPLICATE_SIMILARITY = 0.985


# ============================================================
# CENTER ROI
# ============================================================

def crop_center_region(image):
    """
    Tight ROI around the center discard-card area.
    """

    h, w = image.shape[:2]

    x1 = int(w * 0.43)
    x2 = int(w * 0.57)

    y1 = int(h * 0.67)
    y2 = int(h * 0.75)

    roi = image[y1:y2, x1:x2]

    return roi, (x1, y1)


# ============================================================
# MASK HELPERS
# ============================================================

def create_white_mask(image):
    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    return cv2.inRange(
        hsv,
        np.array([0, 0, 165]),
        np.array([179, 125, 255])
    )


def create_rank_ink_mask(image):
    """
    Detect black/red printing.
    """

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    dark = cv2.inRange(
        gray,
        0,
        150
    )

    red_low = cv2.inRange(
        hsv,
        np.array([0, 70, 45]),
        np.array([18, 255, 255])
    )

    red_high = cv2.inRange(
        hsv,
        np.array([165, 70, 45]),
        np.array([179, 255, 255])
    )

    red = cv2.bitwise_or(
        red_low,
        red_high
    )

    return cv2.bitwise_or(
        dark,
        red
    )


# ============================================================
# CARD PILE TEST
# ============================================================

def evaluate_card_candidate(
    white_mask,
    contour
):
    """
    Score a bright object as a card pile.

    V5 adds stronger chip rejection:
    - cards should be wider than tall
    - cards should have strong top-edge coverage
    - cards should have substantial white in both top corners
    - cards should not look circular
    """

    x, y, w, h = cv2.boundingRect(
        contour
    )

    area = w * h

    if area < 120:
        return None

    if w < 22 or h < 8:
        return None

    aspect = (
        w / max(h, 1)
    )

    # Stronger than v4.
    # Circular chip tends to be around ~1:1.
    if aspect < 1.55:
        return None

    if aspect > 5.0:
        return None

    contour_area = cv2.contourArea(
        contour
    )

    rectangularity = (
        contour_area
        / max(area, 1)
    )

    # A ring/chip usually fills its bounding box poorly.
    if rectangularity < 0.38:
        return None

    crop = white_mask[
        y:y + h,
        x:x + w
    ]

    if crop.size == 0:
        return None

    binary = (
        crop > 0
    )

    white_ratio = float(
        np.mean(binary)
    )

    if white_ratio < 0.38:
        return None

    # --------------------------------------------------------
    # TOP EDGE
    # --------------------------------------------------------

    top_h = max(
        2,
        int(h * 0.25)
    )

    top = binary[
        :top_h,
        :
    ]

    top_column_coverage = float(
        np.mean(
            np.any(
                top,
                axis=0
            )
        )
    )

    if top_column_coverage < 0.70:
        return None

    # --------------------------------------------------------
    # TOP CORNERS
    # --------------------------------------------------------

    corner_w = max(
        2,
        int(w * 0.20)
    )

    corner_h = max(
        2,
        int(h * 0.28)
    )

    upper_left = float(
        np.mean(
            binary[
                :corner_h,
                :corner_w
            ]
        )
    )

    upper_right = float(
        np.mean(
            binary[
                :corner_h,
                w - corner_w:
            ]
        )
    )

    # Circular chips tend to fail one or both of these.
    if upper_left < 0.25:
        return None

    if upper_right < 0.25:
        return None

    # --------------------------------------------------------
    # CENTER DENSITY
    # --------------------------------------------------------

    center_x1 = int(
        w * 0.25
    )

    center_x2 = int(
        w * 0.75
    )

    center_region = binary[
        :,
        center_x1:center_x2
    ]

    center_fill = float(
        np.mean(
            center_region
        )
    )

    if center_fill < 0.30:
        return None

    score = (
        white_ratio
        + top_column_coverage
        + upper_left
        + upper_right
        + center_fill
    )

    score *= np.sqrt(
        area
    )

    return (
        score,
        x,
        y,
        w,
        h
    )


def find_card_pile(center_roi):
    """
    Find the center overlapping-card pile while rejecting
    the betting chip.
    """

    white = create_white_mask(
        center_roi
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (5, 2)
    )

    merged = cv2.morphologyEx(
        white,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        merged,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in contours:
        result = evaluate_card_candidate(
            white,
            contour
        )

        if result is not None:
            candidates.append(
                result
            )

    if not candidates:
        return None

    candidates.sort(
        reverse=True,
        key=lambda item: item[0]
    )

    _, x, y, w, h = candidates[0]

    pad_x = 3
    pad_y = 2

    x1 = max(
        0,
        x - pad_x
    )

    y1 = max(
        0,
        y - pad_y
    )

    x2 = min(
        center_roi.shape[1],
        x + w + pad_x
    )

    y2 = min(
        center_roi.shape[0],
        y + h + pad_y
    )

    return (
        x1,
        y1,
        x2,
        y2
    )


# ============================================================
# FIND RANK COMPONENTS
# ============================================================

def find_rank_components(pile):
    """
    V5 uses an even shallower search strip.

    Goal:
    include the rank row
    exclude the suit row below it
    """

    h, w = pile.shape[:2]

    # V4 used 30%.
    # V5 uses only 22%.
    strip_h = max(
        5,
        int(h * 0.22)
    )

    strip = pile[
        :strip_h,
        :
    ]

    mask = create_rank_ink_mask(
        strip
    )

    count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    components = []

    for i in range(
        1,
        count
    ):
        x = int(
            stats[
                i,
                cv2.CC_STAT_LEFT
            ]
        )

        y = int(
            stats[
                i,
                cv2.CC_STAT_TOP
            ]
        )

        cw = int(
            stats[
                i,
                cv2.CC_STAT_WIDTH
            ]
        )

        ch = int(
            stats[
                i,
                cv2.CC_STAT_HEIGHT
            ]
        )

        area = int(
            stats[
                i,
                cv2.CC_STAT_AREA
            ]
        )

        if area < 2:
            continue

        if area > 80:
            continue

        if ch < 3:
            continue

        if cw < 1:
            continue

        if cw > 15:
            continue

        aspect = (
            cw / max(ch, 1)
        )

        fill_ratio = (
            area
            / max(
                cw * ch,
                1
            )
        )

        # ----------------------------------------------------
        # SUIT-LIKE REJECTION
        # ----------------------------------------------------

        # Clubs/spades/diamonds/hearts tend to be squatter
        # and denser than rank characters in this interface.
        if aspect > 1.05:
            continue

        if fill_ratio > 0.82:
            continue

        # Very short squat components are usually suits/noise.
        if (
            ch <= 4
            and aspect > 0.75
        ):
            continue

        components.append(
            [
                x,
                y,
                cw,
                ch,
                area
            ]
        )

    components.sort(
        key=lambda item: item[0]
    )

    return (
        components,
        strip,
        mask
    )


# ============================================================
# SPECIAL "10" GROUPING
# ============================================================

def group_probable_tens(components):
    """
    Merge adjacent 1 + 0 components into a single 10.
    """

    if not components:
        return []

    output = []

    i = 0

    while i < len(components):
        first = components[i]

        if i + 1 < len(components):
            second = components[
                i + 1
            ]

            x1, y1, w1, h1, a1 = first
            x2, y2, w2, h2, a2 = second

            gap = (
                x2
                - (x1 + w1)
            )

            vertical_overlap = (
                min(
                    y1 + h1,
                    y2 + h2
                )
                - max(
                    y1,
                    y2
                )
            )

            overlap_ratio = (
                vertical_overlap
                / max(
                    min(
                        h1,
                        h2
                    ),
                    1
                )
            )

            first_is_narrow = (
                w1
                <= max(
                    3,
                    int(
                        h1 * 0.55
                    )
                )
            )

            if (
                0 <= gap <= 2
                and overlap_ratio >= 0.60
                and first_is_narrow
            ):
                nx1 = min(
                    x1,
                    x2
                )

                ny1 = min(
                    y1,
                    y2
                )

                nx2 = max(
                    x1 + w1,
                    x2 + w2
                )

                ny2 = max(
                    y1 + h1,
                    y2 + h2
                )

                output.append(
                    [
                        nx1,
                        ny1,
                        nx2 - nx1,
                        ny2 - ny1,
                        a1 + a2
                    ]
                )

                i += 2
                continue

        output.append(
            first
        )

        i += 1

    return output


# ============================================================
# CROP
# ============================================================

def padded_crop(
    image,
    x,
    y,
    w,
    h,
    padding=2
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
# NORMALIZATION
# ============================================================

def normalize_rank(crop):
    mask = create_rank_ink_mask(
        crop
    )

    points = cv2.findNonZero(
        mask
    )

    if points is None:
        return None

    x, y, w, h = cv2.boundingRect(
        points
    )

    glyph = mask[
        y:y + h,
        x:x + w
    ]

    canvas = np.zeros(
        NORMALIZED_SIZE,
        dtype=np.uint8
    )

    target_h = 36
    target_w = 36

    scale = min(
        target_w / max(w, 1),
        target_h / max(h, 1)
    )

    new_w = max(
        1,
        int(
            round(
                w * scale
            )
        )
    )

    new_h = max(
        1,
        int(
            round(
                h * scale
            )
        )
    )

    resized = cv2.resize(
        glyph,
        (
            new_w,
            new_h
        ),
        interpolation=cv2.INTER_NEAREST
    )

    x_offset = (
        NORMALIZED_SIZE[1]
        - new_w
    ) // 2

    y_offset = (
        NORMALIZED_SIZE[0]
        - new_h
    ) // 2

    canvas[
        y_offset:
        y_offset + new_h,
        x_offset:
        x_offset + new_w
    ] = resized

    return canvas


# ============================================================
# DUPLICATE DETECTION
# ============================================================

def similarity_score(a, b):
    if a is None or b is None:
        return 0.0

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
        np.mean(
            a * b
        )
    )


def is_duplicate(
    crop,
    saved_normalized
):
    normalized = normalize_rank(
        crop
    )

    if normalized is None:
        return (
            False,
            None,
            None
        )

    best = None

    for previous in saved_normalized:
        score = similarity_score(
            normalized,
            previous
        )

        if (
            best is None
            or score > best
        ):
            best = score

        if (
            score
            >= DUPLICATE_SIMILARITY
        ):
            return (
                True,
                score,
                normalized
            )

    return (
        False,
        best,
        normalized
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
            "No screenshots found."
        )
        return

    print(
        f"Found {len(image_paths)} captured frames."
    )

    saved_normalized = []

    piles_found = 0
    no_pile_frames = 0

    components_found = 0
    rejected_components = 0

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

        annotated = image.copy()

        center, offset = (
            crop_center_region(
                image
            )
        )

        offset_x, offset_y = (
            offset
        )

        pile_box = find_card_pile(
            center
        )

        # ====================================================
        # NO PILE
        # ====================================================

        if pile_box is None:
            no_pile_frames += 1

            cv2.putText(
                annotated,
                "NO CENTER PILE",
                (
                    offset_x,
                    offset_y - 5
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (150, 150, 150),
                1
            )

        # ====================================================
        # PILE FOUND
        # ====================================================

        else:
            piles_found += 1

            px1, py1, px2, py2 = (
                pile_box
            )

            pile = center[
                py1:py2,
                px1:px2
            ]

            full_x1 = (
                offset_x + px1
            )

            full_y1 = (
                offset_y + py1
            )

            full_x2 = (
                offset_x + px2
            )

            full_y2 = (
                offset_y + py2
            )

            cv2.rectangle(
                annotated,
                (
                    full_x1,
                    full_y1
                ),
                (
                    full_x2,
                    full_y2
                ),
                (255, 100, 0),
                2
            )

            (
                components,
                strip,
                mask
            ) = find_rank_components(
                pile
            )

            components = (
                group_probable_tens(
                    components
                )
            )

            for (
                x,
                y,
                w,
                h,
                area
            ) in components:

                components_found += 1

                crop = padded_crop(
                    strip,
                    x,
                    y,
                    w,
                    h,
                    padding=2
                )

                if crop.size == 0:
                    continue

                (
                    duplicate,
                    score,
                    normalized
                ) = is_duplicate(
                    crop,
                    saved_normalized
                )

                fx1 = (
                    full_x1 + x
                )

                fy1 = (
                    full_y1 + y
                )

                fx2 = (
                    fx1 + w
                )

                fy2 = (
                    fy1 + h
                )

                if duplicate:
                    duplicates += 1

                    cv2.rectangle(
                        annotated,
                        (
                            fx1,
                            fy1
                        ),
                        (
                            fx2,
                            fy2
                        ),
                        (0, 165, 255),
                        1
                    )

                    continue

                if normalized is None:
                    rejected_components += 1
                    continue

                saved_normalized.append(
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
                        fx1,
                        fy1
                    ),
                    (
                        fx2,
                        fy2
                    ),
                    (0, 255, 0),
                    1
                )

                cv2.putText(
                    annotated,
                    f"R{saved_count}",
                    (
                        fx1,
                        max(
                            15,
                            fy1 - 3
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.32,
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
        "--- CENTER HARVEST V5 COMPLETE ---"
    )

    print(
        f"Frames scanned: "
        f"{len(image_paths)}"
    )

    print(
        f"Card piles found: "
        f"{piles_found}"
    )

    print(
        f"No-pile frames: "
        f"{no_pile_frames}"
    )

    print(
        f"Rank components found: "
        f"{components_found}"
    )

    print(
        f"Rejected components: "
        f"{rejected_components}"
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