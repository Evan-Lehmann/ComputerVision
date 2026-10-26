from pathlib import Path
import csv

import cv2
import numpy as np


# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = Path("captures")
TEMPLATE_ROOT = Path("templates/ranks")

OUTPUT_ROOT = Path("center_match_diagnostics")
RAW_DIR = OUTPUT_ROOT / "raw_candidates"
NORMALIZED_DIR = OUTPUT_ROOT / "normalized_candidates"
DEBUG_DIR = OUTPUT_ROOT / "debug"

RAW_DIR.mkdir(parents=True, exist_ok=True)
NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT_ROOT / "matches.csv"

NORMALIZED_SIZE = (64, 64)


# ============================================================
# TEMPLATE LOADING
# ============================================================

def load_templates():
    templates = {}

    if not TEMPLATE_ROOT.exists():
        return templates

    for rank_dir in sorted(TEMPLATE_ROOT.iterdir()):
        if not rank_dir.is_dir():
            continue

        rank = rank_dir.name

        items = []

        for path in sorted(rank_dir.glob("*.png")):
            image = cv2.imread(
                str(path),
                cv2.IMREAD_GRAYSCALE
            )

            if image is not None:
                items.append(image)

        if items:
            templates[rank] = items

    return templates


# ============================================================
# CENTER REGION
# ============================================================

def crop_center_region(image):
    h, w = image.shape[:2]

    x1 = int(w * 0.43)
    x2 = int(w * 0.57)

    y1 = int(h * 0.67)
    y2 = int(h * 0.75)

    return image[y1:y2, x1:x2], (x1, y1)


# ============================================================
# MASKS
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


def create_rank_mask(image):
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
# CARD PILE DETECTION
# ============================================================

def evaluate_card_candidate(
    white_mask,
    contour
):
    x, y, w, h = cv2.boundingRect(
        contour
    )

    area = w * h

    if area < 120:
        return None

    if w < 22 or h < 8:
        return None

    aspect = w / max(h, 1)

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

    if rectangularity < 0.38:
        return None

    crop = white_mask[
        y:y + h,
        x:x + w
    ]

    if crop.size == 0:
        return None

    binary = crop > 0

    white_ratio = float(
        np.mean(binary)
    )

    if white_ratio < 0.38:
        return None

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

    if upper_left < 0.25:
        return None

    if upper_right < 0.25:
        return None

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
    ) * np.sqrt(area)

    return (
        score,
        x,
        y,
        w,
        h
    )


def find_card_pile(center_roi):
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
# CENTER COMPONENT DETECTION
# ============================================================

def find_rank_candidates(pile):
    """
    Deliberately permissive.

    We want to save the candidates first, then study
    what the matching scores actually look like.
    """

    h, w = pile.shape[:2]

    strip_h = max(
        6,
        int(h * 0.30)
    )

    strip = pile[
        :strip_h,
        :
    ]

    mask = create_rank_mask(
        strip
    )

    count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    components = []

    for i in range(1, count):
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

        if area > 120:
            continue

        if ch < 3:
            continue

        if cw < 1:
            continue

        if cw > 18:
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
        strip
    )


# ============================================================
# "10" GROUPING
# ============================================================

def group_probable_tens(components):
    if not components:
        return []

    output = []
    i = 0

    while i < len(components):
        first = components[i]

        if i + 1 < len(components):
            second = components[i + 1]

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
                    min(h1, h2),
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
# NORMALIZE
# ============================================================

def normalize_glyph(crop):
    mask = create_rank_mask(
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

    glyph_mask = mask[
        y:y + h,
        x:x + w
    ]

    canvas = np.zeros(
        NORMALIZED_SIZE,
        dtype=np.uint8
    )

    target_w = 46
    target_h = 46

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
        glyph_mask,
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
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = resized

    return canvas


# ============================================================
# SIMILARITY
# ============================================================

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
        np.mean(
            a * b
        )
    )


def score_against_templates(
    normalized,
    templates
):
    """
    Return:
    - best rank
    - best score
    - second-best rank
    - second-best score
    - margin
    """

    rank_scores = []

    for rank, rank_templates in templates.items():
        best_for_rank = -1.0

        for template in rank_templates:
            score = similarity_score(
                normalized,
                template
            )

            best_for_rank = max(
                best_for_rank,
                score
            )

        rank_scores.append(
            (
                rank,
                best_for_rank
            )
        )

    rank_scores.sort(
        key=lambda item: item[1],
        reverse=True
    )

    best_rank, best_score = rank_scores[0]

    if len(rank_scores) >= 2:
        second_rank, second_score = (
            rank_scores[1]
        )
    else:
        second_rank = ""
        second_score = -1.0

    margin = (
        best_score
        - second_score
    )

    return (
        best_rank,
        best_score,
        second_rank,
        second_score,
        margin
    )


# ============================================================
# MAIN
# ============================================================

def main():
    templates = load_templates()

    if not templates:
        print(
            "No templates found."
        )

        return

    print(
        "Loaded templates:"
    )

    for rank, items in templates.items():
        print(
            f"  {rank}: {len(items)}"
        )

    image_paths = sorted(
        INPUT_DIR.glob(
            "*.png"
        )
    )

    if not image_paths:
        print(
            "No captures found."
        )
        return

    rows = []

    candidate_number = 0
    piles_found = 0
    no_pile = 0

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

        center, offset = crop_center_region(
            image
        )

        offset_x, offset_y = offset

        pile_box = find_card_pile(
            center
        )

        if pile_box is None:
            no_pile += 1

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

            cv2.rectangle(
                annotated,
                (
                    full_x1,
                    full_y1
                ),
                (
                    offset_x + px2,
                    offset_y + py2
                ),
                (255, 100, 0),
                2
            )

            (
                components,
                strip
            ) = find_rank_candidates(
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

                normalized = normalize_glyph(
                    crop
                )

                if normalized is None:
                    continue

                candidate_number += 1

                (
                    best_rank,
                    best_score,
                    second_rank,
                    second_score,
                    margin
                ) = score_against_templates(
                    normalized,
                    templates
                )

                raw_name = (
                    f"candidate_{candidate_number:04d}.png"
                )

                normalized_name = (
                    f"candidate_{candidate_number:04d}_norm.png"
                )

                cv2.imwrite(
                    str(
                        RAW_DIR
                        / raw_name
                    ),
                    crop
                )

                cv2.imwrite(
                    str(
                        NORMALIZED_DIR
                        / normalized_name
                    ),
                    normalized
                )

                rows.append(
                    {
                        "candidate": candidate_number,
                        "frame": frame_number,
                        "source_file": path.name,
                        "x": x,
                        "y": y,
                        "width": w,
                        "height": h,
                        "area": area,
                        "best_rank": best_rank,
                        "best_score": best_score,
                        "second_rank": second_rank,
                        "second_score": second_score,
                        "margin": margin,
                        "raw_file": raw_name,
                        "normalized_file": normalized_name,
                    }
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

                label = (
                    f"{candidate_number}:"
                    f"{best_rank} "
                    f"{best_score:.2f}"
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
                    (0, 165, 255),
                    1
                )

                cv2.putText(
                    annotated,
                    label,
                    (
                        fx1,
                        max(
                            15,
                            fy1 - 3
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.30,
                    (0, 165, 255),
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

    # ========================================================
    # CSV
    # ========================================================

    fieldnames = [
        "candidate",
        "frame",
        "source_file",
        "x",
        "y",
        "width",
        "height",
        "area",
        "best_rank",
        "best_score",
        "second_rank",
        "second_score",
        "margin",
        "raw_file",
        "normalized_file",
    ]

    with open(
        CSV_PATH,
        "w",
        newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print(
        "--- CENTER MATCH DIAGNOSTIC COMPLETE ---"
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
        f"{no_pile}"
    )

    print(
        f"Candidates saved: "
        f"{candidate_number}"
    )

    if rows:
        scores = np.array(
            [
                row["best_score"]
                for row in rows
            ],
            dtype=np.float32
        )

        margins = np.array(
            [
                row["margin"]
                for row in rows
            ],
            dtype=np.float32
        )

        print(
            f"Best-score min: "
            f"{scores.min():.3f}"
        )

        print(
            f"Best-score mean: "
            f"{scores.mean():.3f}"
        )

        print(
            f"Best-score max: "
            f"{scores.max():.3f}"
        )

        print(
            f"Mean best-vs-second margin: "
            f"{margins.mean():.3f}"
        )

    print(
        f"CSV: "
        f"{CSV_PATH.resolve()}"
    )

    print(
        f"Raw candidates: "
        f"{RAW_DIR.resolve()}"
    )

    print(
        f"Normalized candidates: "
        f"{NORMALIZED_DIR.resolve()}"
    )

    print(
        f"Debug: "
        f"{DEBUG_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()