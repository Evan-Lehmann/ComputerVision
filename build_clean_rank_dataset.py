from pathlib import Path
import csv
from collections import defaultdict

import cv2
import numpy as np


# =========================================================
# CONFIG
# =========================================================

ROOT = Path(__file__).resolve().parent

SOURCES = [
    (
        ROOT / "rank_value_dataset",
        ROOT / "rank_value_dataset" / "labels.csv",
    ),
    (
        ROOT / "rank_value_candidates",
        ROOT / "rank_value_candidates" / "candidate_labels.csv",
    ),
]

OUTPUT_DIR = ROOT / "rank_value_clean"
IMAGE_DIR = OUTPUT_DIR / "images"
REVIEW_DIR = OUTPUT_DIR / "review"

OUTPUT_CSV = OUTPUT_DIR / "labels.csv"

IMAGE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REVIEW_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RANKS = [
    "A", "2", "3", "4", "5", "6",
    "7", "8", "9", "10", "J", "Q", "K"
]

OUTPUT_SIZE = 64


# =========================================================
# FOREGROUND EXTRACTION
# =========================================================

def make_foreground_mask(image):
    """
    Detect black or red rank glyph pixels while trying
    to ignore the card background.

    Red rank:
        captured using HSV saturation / hue.

    Black rank:
        captured using adaptive thresholding.
    """

    # Work larger because the original crops are tiny.
    enlarged = cv2.resize(
        image,
        None,
        fx=4,
        fy=4,
        interpolation=cv2.INTER_CUBIC,
    )

    gray = cv2.cvtColor(
        enlarged,
        cv2.COLOR_BGR2GRAY,
    )

    hsv = cv2.cvtColor(
        enlarged,
        cv2.COLOR_BGR2HSV,
    )

    # -----------------------------------------------------
    # BLACK / DARK GLYPHS
    # -----------------------------------------------------

    dark_mask = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        9,
    )

    # -----------------------------------------------------
    # RED GLYPHS
    # -----------------------------------------------------

    h, s, v = cv2.split(hsv)

    red_low = (
        (h <= 12)
        & (s >= 65)
        & (v >= 60)
    )

    red_high = (
        (h >= 165)
        & (s >= 65)
        & (v >= 60)
    )

    red_mask = (
        red_low
        | red_high
    ).astype(
        np.uint8
    ) * 255

    # Red pixels should definitely count as foreground.
    mask = cv2.bitwise_or(
        dark_mask,
        red_mask,
    )

    # Remove isolated noise, but don't aggressively
    # distort the rank.
    kernel = np.ones(
        (2, 2),
        dtype=np.uint8,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
    )

    return enlarged, mask


# =========================================================
# COMPONENT SELECTION
# =========================================================

def extract_rank_components(mask):
    """
    Remove card borders and choose the central glyph.

    Key idea:
      - card borders tend to touch the crop boundary
      - the rank glyph is usually central
      - '10' requires two neighboring components
    """

    h, w = mask.shape

    (
        num_labels,
        labels,
        stats,
        centroids,
    ) = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    candidates = []

    image_area = (
        h * w
    )

    border_margin = max(
        2,
        int(
            min(h, w)
            * 0.025
        ),
    )

    for component in range(
        1,
        num_labels,
    ):
        x = stats[
            component,
            cv2.CC_STAT_LEFT,
        ]

        y = stats[
            component,
            cv2.CC_STAT_TOP,
        ]

        cw = stats[
            component,
            cv2.CC_STAT_WIDTH,
        ]

        ch = stats[
            component,
            cv2.CC_STAT_HEIGHT,
        ]

        area = stats[
            component,
            cv2.CC_STAT_AREA,
        ]

        cx, cy = centroids[
            component
        ]

        # Tiny noise.
        if area < (
            image_area
            * 0.0015
        ):
            continue

        if ch < (
            h * 0.10
        ):
            continue

        # -------------------------------------------------
        # BORDER REJECTION
        # -------------------------------------------------

        touches_left = (
            x <= border_margin
        )

        touches_right = (
            x + cw
            >= w - border_margin
        )

        touches_top = (
            y <= border_margin
        )

        touches_bottom = (
            y + ch
            >= h - border_margin
        )

        touches_edge = (
            touches_left
            or touches_right
            or touches_top
            or touches_bottom
        )

        # Card borders tend to be long and edge-attached.
        border_like = (
            (
                ch > 0.45 * h
                and cw < 0.20 * w
            )
            or
            (
                cw > 0.45 * w
                and ch < 0.20 * h
            )
        )

        if (
            touches_edge
            and border_like
        ):
            continue

        # Huge edge-connected blobs are also usually
        # pieces of card/UI chrome.
        if (
            touches_edge
            and area
            > image_area * 0.08
        ):
            continue

        # -------------------------------------------------
        # CENTRALITY SCORE
        # -------------------------------------------------

        dx = (
            cx - w / 2
        ) / w

        dy = (
            cy - h / 2
        ) / h

        distance = (
            dx * dx
            + dy * dy
        ) ** 0.5

        score = (
            area
            / (
                1.0
                + 5.0 * distance
            )
        )

        candidates.append(
            {
                "component":
                    component,

                "x":
                    x,

                "y":
                    y,

                "w":
                    cw,

                "h":
                    ch,

                "area":
                    area,

                "cx":
                    cx,

                "cy":
                    cy,

                "score":
                    score,
            }
        )

    if not candidates:
        return np.zeros_like(
            mask
        )

    candidates.sort(
        key=lambda item:
            item["score"],
        reverse=True,
    )

    main = candidates[0]

    selected = [
        main
    ]

    # -----------------------------------------------------
    # ADD SECOND GLYPH IF APPROPRIATE
    #
    # Needed mainly for "10".
    # Also allows a disconnected part of a glyph.
    # -----------------------------------------------------

    for candidate in candidates[1:]:

        vertical_center_difference = abs(
            candidate["cy"]
            - main["cy"]
        )

        horizontal_distance = abs(
            candidate["cx"]
            - main["cx"]
        )

        height_ratio = (
            candidate["h"]
            / max(
                main["h"],
                1,
            )
        )

        area_ratio = (
            candidate["area"]
            / max(
                main["area"],
                1,
            )
        )

        same_text_line = (
            vertical_center_difference
            < 0.30 * h
        )

        close_enough = (
            horizontal_distance
            < 0.45 * w
        )

        reasonable_height = (
            0.35
            <= height_ratio
            <= 2.5
        )

        substantial = (
            area_ratio
            >= 0.12
        )

        if (
            same_text_line
            and close_enough
            and reasonable_height
            and substantial
        ):
            selected.append(
                candidate
            )

        if len(selected) >= 2:
            break

    clean = np.zeros_like(
        mask
    )

    for item in selected:
        clean[
            labels
            == item["component"]
        ] = 255

    return clean


# =========================================================
# NORMALIZATION
# =========================================================

def normalize_rank(image):
    enlarged, mask = (
        make_foreground_mask(
            image
        )
    )

    clean = extract_rank_components(
        mask
    )

    points = cv2.findNonZero(
        clean
    )

    if points is None:
        return np.zeros(
            (
                OUTPUT_SIZE,
                OUTPUT_SIZE,
            ),
            dtype=np.uint8,
        )

    x, y, w, h = (
        cv2.boundingRect(
            points
        )
    )

    glyph = clean[
        y:y + h,
        x:x + w
    ]

    canvas = np.zeros(
        (
            OUTPUT_SIZE,
            OUTPUT_SIZE,
        ),
        dtype=np.uint8,
    )

    available = (
        OUTPUT_SIZE - 12
    )

    scale = min(
        available
        / max(w, 1),

        available
        / max(h, 1),
    )

    new_w = max(
        1,
        round(
            w * scale
        ),
    )

    new_h = max(
        1,
        round(
            h * scale
        ),
    )

    glyph = cv2.resize(
        glyph,
        (
            new_w,
            new_h,
        ),
        interpolation=cv2.INTER_NEAREST,
    )

    ox = (
        OUTPUT_SIZE - new_w
    ) // 2

    oy = (
        OUTPUT_SIZE - new_h
    ) // 2

    canvas[
        oy:oy + new_h,
        ox:ox + new_w
    ] = glyph

    return canvas


# =========================================================
# REVIEW TILE
# =========================================================

def make_review_tile(
    original,
    clean,
    rank,
):
    original = cv2.resize(
        original,
        (130, 130),
        interpolation=cv2.INTER_NEAREST,
    )

    clean = cv2.resize(
        clean,
        (130, 130),
        interpolation=cv2.INTER_NEAREST,
    )

    clean = cv2.cvtColor(
        clean,
        cv2.COLOR_GRAY2BGR,
    )

    tile = np.full(
        (
            180,
            280,
            3,
        ),
        25,
        dtype=np.uint8,
    )

    tile[
        40:170,
        5:135
    ] = original

    tile[
        40:170,
        145:275
    ] = clean

    cv2.putText(
        tile,
        f"Rank {rank}",
        (8, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        tile,
        "raw",
        (45, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (220, 220, 220),
        1,
    )

    cv2.putText(
        tile,
        "clean",
        (185, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (220, 220, 220),
        1,
    )

    return tile


# =========================================================
# MAIN
# =========================================================

def main():
    output_rows = []

    review_tiles = defaultdict(
        list
    )

    usable = 0

    for (
        image_dir,
        csv_path,
    ) in SOURCES:

        if not csv_path.exists():
            continue

        with csv_path.open(
            newline=""
        ) as f:

            reader = csv.DictReader(
                f
            )

            for row in reader:

                rank = (
                    row["rank"]
                    .strip()
                )

                if rank == "UNKNOWN":
                    continue

                if rank not in RANKS:
                    continue

                source_path = (
                    image_dir
                    / row["crop"]
                )

                if not source_path.exists():
                    continue

                image = cv2.imread(
                    str(
                        source_path
                    )
                )

                if image is None:
                    continue

                clean = normalize_rank(
                    image
                )

                output_name = (
                    f"{usable:04d}_"
                    f"{rank}_"
                    f"{source_path.name}"
                )

                output_path = (
                    IMAGE_DIR
                    / output_name
                )

                cv2.imwrite(
                    str(
                        output_path
                    ),
                    clean,
                )

                output_rows.append(
                    {
                        "crop":
                            output_name,

                        "rank":
                            rank,

                        "source":
                            source_path.name,
                    }
                )

                review_tiles[
                    rank
                ].append(
                    make_review_tile(
                        image,
                        clean,
                        rank,
                    )
                )

                usable += 1

    # -----------------------------------------------------
    # SAVE LABEL CSV
    # -----------------------------------------------------

    with OUTPUT_CSV.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "crop",
                "rank",
                "source",
            ],
        )

        writer.writeheader()

        writer.writerows(
            output_rows
        )

    # -----------------------------------------------------
    # SAVE REVIEW SHEETS
    # -----------------------------------------------------

    for rank in RANKS:

        tiles = review_tiles.get(
            rank,
            []
        )

        if not tiles:
            continue

        columns = 4

        blank = np.full(
            (
                180,
                280,
                3,
            ),
            25,
            dtype=np.uint8,
        )

        while (
            len(tiles)
            % columns
        ):
            tiles.append(
                blank.copy()
            )

        rows = []

        for i in range(
            0,
            len(tiles),
            columns,
        ):
            rows.append(
                np.hstack(
                    tiles[
                        i:i + columns
                    ]
                )
            )

        sheet = np.vstack(
            rows
        )

        path = (
            REVIEW_DIR
            / f"rank_{rank}.jpg"
        )

        cv2.imwrite(
            str(path),
            sheet,
        )

    print()
    print(
        "--- CLEAN DATASET COMPLETE ---"
    )

    print(
        f"Clean crops: {usable}"
    )

    print(
        f"Images: {IMAGE_DIR}"
    )

    print(
        f"Labels: {OUTPUT_CSV}"
    )

    print(
        f"Review sheets: {REVIEW_DIR}"
    )


if __name__ == "__main__":
    main()