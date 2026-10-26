from pathlib import Path

import cv2
import numpy as np


# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = Path("harvested/cards")

OUTPUT_ROOT = Path("templates/ranks")
PREVIEW_DIR = Path("templates/previews")

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

VALID_RANKS = {
    "A",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "J",
    "Q",
    "K",
}

NORMALIZED_SIZE = (64, 64)


# ============================================================
# RANK CROP
# ============================================================

def crop_rank_region(card):
    """
    Extract the upper-left rank area from a clean large-card crop.
    """

    h, w = card.shape[:2]

    x1 = int(w * 0.02)
    x2 = int(w * 0.34)

    y1 = int(h * 0.02)
    y2 = int(h * 0.34)

    return card[
        y1:y2,
        x1:x2
    ]


# ============================================================
# RANK MASK
# ============================================================

def create_rank_mask(crop):
    """
    Detect both black and red printed rank glyphs.
    """

    hsv = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2HSV
    )

    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2GRAY
    )

    dark = cv2.inRange(
        gray,
        0,
        155
    )

    red_low = cv2.inRange(
        hsv,
        np.array([0, 70, 40]),
        np.array([18, 255, 255])
    )

    red_high = cv2.inRange(
        hsv,
        np.array([165, 70, 40]),
        np.array([179, 255, 255])
    )

    red = cv2.bitwise_or(
        red_low,
        red_high
    )

    mask = cv2.bitwise_or(
        dark,
        red
    )

    return mask


# ============================================================
# ISOLATE TOP RANK
# ============================================================

def isolate_rank_glyph(rank_crop):
    """
    Find the top-most meaningful ink component.

    This helps separate the rank from the suit symbol below it.
    """

    mask = create_rank_mask(
        rank_crop
    )

    count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    candidates = []

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

        w = int(
            stats[
                i,
                cv2.CC_STAT_WIDTH
            ]
        )

        h = int(
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

        if area < 3:
            continue

        if h < 4:
            continue

        if w < 1:
            continue

        candidates.append(
            (
                y,
                -area,
                x,
                y,
                w,
                h
            )
        )

    if not candidates:
        return None

    # Top-most component first.
    # Larger area breaks close ties.
    candidates.sort()

    _, _, x, y, w, h = candidates[0]

    pad = 2

    x1 = max(
        0,
        x - pad
    )

    y1 = max(
        0,
        y - pad
    )

    x2 = min(
        rank_crop.shape[1],
        x + w + pad
    )

    y2 = min(
        rank_crop.shape[0],
        y + h + pad
    )

    return rank_crop[
        y1:y2,
        x1:x2
    ]


# ============================================================
# SPECIAL HANDLING FOR 10
# ============================================================

def isolate_ten_glyph(rank_crop):
    """
    For rank 10, isolate the top two nearby glyph components:
    '1' and '0'.
    """

    mask = create_rank_mask(
        rank_crop
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

        w = int(
            stats[
                i,
                cv2.CC_STAT_WIDTH
            ]
        )

        h = int(
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

        if h < 4:
            continue

        components.append(
            (
                y,
                x,
                w,
                h,
                area
            )
        )

    if len(components) < 2:
        return None

    # Focus on the upper-most components.
    components.sort(
        key=lambda item: (
            item[0],
            item[1]
        )
    )

    top_candidates = components[:4]

    # Then choose the two left-most candidates.
    top_candidates.sort(
        key=lambda item: item[1]
    )

    first = top_candidates[0]
    second = top_candidates[1]

    y1a, x1a, w1, h1, _ = first
    y2a, x2a, w2, h2, _ = second

    x1 = min(
        x1a,
        x2a
    )

    y1 = min(
        y1a,
        y2a
    )

    x2 = max(
        x1a + w1,
        x2a + w2
    )

    y2 = max(
        y1a + h1,
        y2a + h2
    )

    pad = 2

    x1 = max(
        0,
        x1 - pad
    )

    y1 = max(
        0,
        y1 - pad
    )

    x2 = min(
        rank_crop.shape[1],
        x2 + pad
    )

    y2 = min(
        rank_crop.shape[0],
        y2 + pad
    )

    return rank_crop[
        y1:y2,
        x1:x2
    ]


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_glyph(glyph):
    """
    Convert the rank glyph to a centered white-on-black
    64x64 template.
    """

    mask = create_rank_mask(
        glyph
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
# SAVE TEMPLATE
# ============================================================

def save_template(
    rank,
    normalized
):
    rank_dir = (
        OUTPUT_ROOT
        / rank
    )

    rank_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    existing = sorted(
        rank_dir.glob(
            "*.png"
        )
    )

    index = len(existing) + 1

    destination = (
        rank_dir
        / f"{rank}_{index:03d}.png"
    )

    cv2.imwrite(
        str(destination),
        normalized
    )

    print(
        f"Saved {rank} template -> "
        f"{destination}"
    )


# ============================================================
# PREVIEW
# ============================================================

def save_preview(
    card,
    source_name
):
    preview = card.copy()

    h, w = card.shape[:2]

    x1 = int(w * 0.02)
    x2 = int(w * 0.34)

    y1 = int(h * 0.02)
    y2 = int(h * 0.34)

    cv2.rectangle(
        preview,
        (
            x1,
            y1
        ),
        (
            x2,
            y2
        ),
        (0, 255, 0),
        1
    )

    destination = (
        PREVIEW_DIR
        / f"{source_name}_preview.png"
    )

    cv2.imwrite(
        str(destination),
        preview
    )


# ============================================================
# USER LABEL
# ============================================================

def ask_rank(filename):
    while True:
        value = input(
            f"{filename} rank? "
            "[A,2-10,J,Q,K / s=skip / x=quit]: "
        ).strip().upper()

        if value == "S":
            return "SKIP"

        if value == "X":
            return "QUIT"

        if value in VALID_RANKS:
            return value

        print(
            "Invalid rank."
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
            "No harvested large-card crops found."
        )

        print(
            f"Expected files in: "
            f"{INPUT_DIR.resolve()}"
        )

        return

    print(
        f"Found {len(image_paths)} card crops."
    )

    print()
    print(
        "Label each card once."
    )

    print(
        "Use s to skip or x to stop."
    )

    print()

    saved = 0
    skipped = 0

    for path in image_paths:
        card = cv2.imread(
            str(path)
        )

        if card is None:
            continue

        rank = ask_rank(
            path.name
        )

        if rank == "QUIT":
            break

        if rank == "SKIP":
            skipped += 1
            continue

        rank_crop = crop_rank_region(
            card
        )

        if rank == "10":
            glyph = isolate_ten_glyph(
                rank_crop
            )
        else:
            glyph = isolate_rank_glyph(
                rank_crop
            )

        if glyph is None:
            print(
                f"Could not isolate rank "
                f"for {path.name}; skipping."
            )

            skipped += 1
            continue

        normalized = normalize_glyph(
            glyph
        )

        if normalized is None:
            print(
                f"Could not normalize rank "
                f"for {path.name}; skipping."
            )

            skipped += 1
            continue

        save_template(
            rank,
            normalized
        )

        save_preview(
            card,
            path.stem
        )

        saved += 1

    print()

    print(
        "--- TEMPLATE BUILD COMPLETE ---"
    )

    print(
        f"Saved templates: "
        f"{saved}"
    )

    print(
        f"Skipped: "
        f"{skipped}"
    )

    print(
        f"Templates: "
        f"{OUTPUT_ROOT.resolve()}"
    )

    print(
        f"Previews: "
        f"{PREVIEW_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()