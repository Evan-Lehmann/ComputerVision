from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
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

OUTPUT_DIR = Path("card_detection")
DEBUG_DIR = OUTPUT_DIR / "debug"
CROPS_DIR = OUTPUT_DIR / "crops"

DETECTIONS_CSV = OUTPUT_DIR / "detections.csv"
FRAMES_CSV = OUTPUT_DIR / "frames.csv"


# ============================================================
# CONFIG
#
# IMPORTANT:
# These are expressed mostly as fractions of the screenshot
# dimensions rather than absolute pixels.
#
# That means moving the browser window or changing monitor
# resolution should not inherently break detection.
# ============================================================

# Bright / neutral card surface
MIN_CARD_VALUE = 145
MAX_CARD_SATURATION = 105

# Full-card geometry
MIN_CARD_HEIGHT_FRAC = 0.025
MAX_CARD_HEIGHT_FRAC = 0.32

MIN_CARD_WIDTH_FRAC = 0.010
MAX_CARD_WIDTH_FRAC = 0.22

MIN_CARD_ASPECT = 0.34
MAX_CARD_ASPECT = 0.98

IDEAL_CARD_ASPECT = 0.70

MIN_WHITE_RATIO = 0.48
MIN_RECTANGULARITY = 0.42

# There should normally be some printed rank/suit ink
# toward the upper-left of a face-up card.
MIN_CORNER_INK_RATIO = 0.004


# Exposed corner / rank-anchor fallback
MIN_ANCHOR_HEIGHT_FRAC = 0.006
MAX_ANCHOR_HEIGHT_FRAC = 0.095

MIN_ANCHOR_WIDTH_FRAC = 0.002
MAX_ANCHOR_WIDTH_FRAC = 0.065

MIN_ANCHOR_WHITE_CONTEXT = 0.28


# Candidate filtering
MIN_FULL_CARD_SCORE = 0.48
MIN_ANCHOR_SCORE = 0.48

FULL_CARD_NMS_IOU = 0.72
ANCHOR_NMS_IOU = 0.45


# Debug output
DEBUG_EVERY_N_FRAMES = 5


# ============================================================
# DATA MODEL
# ============================================================

@dataclass
class Detection:
    kind: str

    x: int
    y: int
    w: int
    h: int

    score: float

    white_ratio: float = 0.0
    rectangularity: float = 0.0
    ink_ratio: float = 0.0

    def box(self) -> Tuple[int, int, int, int]:
        return (
            self.x,
            self.y,
            self.w,
            self.h,
        )

    def center(self) -> Tuple[float, float]:
        return (
            self.x + self.w / 2.0,
            self.y + self.h / 2.0,
        )


# ============================================================
# FILE ORDERING
# ============================================================

def natural_sort_key(path: Path):
    parts = re.split(
        r"(\d+)",
        path.name,
    )

    return [
        int(part)
        if part.isdigit()
        else part.lower()
        for part in parts
    ]


def get_capture_paths() -> List[Path]:
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
# GEOMETRY
# ============================================================

def box_iou(
    a: Detection,
    b: Detection,
) -> float:

    ax1 = a.x
    ay1 = a.y
    ax2 = a.x + a.w
    ay2 = a.y + a.h

    bx1 = b.x
    by1 = b.y
    bx2 = b.x + b.w
    by2 = b.y + b.h

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)

    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(
        0,
        ix2 - ix1,
    )

    ih = max(
        0,
        iy2 - iy1,
    )

    intersection = (
        iw * ih
    )

    if intersection <= 0:
        return 0.0

    area_a = a.w * a.h
    area_b = b.w * b.h

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0:
        return 0.0

    return (
        intersection
        / float(union)
    )


def point_inside_detection(
    point,
    detection: Detection,
) -> bool:

    px, py = point

    return (
        detection.x <= px
        <= detection.x + detection.w

        and

        detection.y <= py
        <= detection.y + detection.h
    )


# ============================================================
# NMS
# ============================================================

def non_max_suppression(
    detections: List[Detection],
    iou_threshold: float,
) -> List[Detection]:

    detections = sorted(
        detections,
        key=lambda d: d.score,
        reverse=True,
    )

    kept = []

    for candidate in detections:

        duplicate = False

        for existing in kept:

            if (
                box_iou(
                    candidate,
                    existing,
                )
                >= iou_threshold
            ):
                duplicate = True
                break

        if not duplicate:
            kept.append(
                candidate
            )

    return kept


# ============================================================
# CARD DETECTOR
# ============================================================

class CardDetector:

    def __init__(self):
        pass


    # --------------------------------------------------------
    # CARD SURFACE MASK
    # --------------------------------------------------------

    def build_card_surface_mask(
        self,
        frame,
    ):
        """
        Detect bright, relatively low-saturation areas.

        This is intentionally based on appearance rather than
        absolute location.
        """

        hsv = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2HSV,
        )

        lower = np.array(
            [
                0,
                0,
                MIN_CARD_VALUE,
            ],
            dtype=np.uint8,
        )

        upper = np.array(
            [
                179,
                MAX_CARD_SATURATION,
                255,
            ],
            dtype=np.uint8,
        )

        mask = cv2.inRange(
            hsv,
            lower,
            upper,
        )

        kernel = np.ones(
            (3, 3),
            dtype=np.uint8,
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1,
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2,
        )

        return mask


    # --------------------------------------------------------
    # PRINTED INK MASK
    # --------------------------------------------------------

    def build_ink_mask(
        self,
        frame,
    ):
        """
        Detect black/dark rank glyphs plus red rank/suit glyphs.
        """

        hsv = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2HSV,
        )

        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        # Black / dark printing
        dark = (
            v < 115
        )

        # Red printing
        red = (
            (
                (h <= 12)
                |
                (h >= 168)
            )
            &
            (s >= 70)
            &
            (v >= 55)
        )

        ink = (
            dark
            |
            red
        ).astype(
            np.uint8
        ) * 255

        return ink


    # --------------------------------------------------------
    # FEATURE HELPERS
    # --------------------------------------------------------

    def white_ratio(
        self,
        surface_mask,
        x,
        y,
        w,
        h,
    ):

        crop = surface_mask[
            y:y + h,
            x:x + w
        ]

        if crop.size == 0:
            return 0.0

        return (
            cv2.countNonZero(
                crop
            )
            / float(crop.size)
        )


    def corner_ink_ratio(
        self,
        ink_mask,
        x,
        y,
        w,
        h,
    ):
        """
        Examine the upper-left region where rank/suit markings
        normally occur.
        """

        corner_w = max(
            1,
            int(w * 0.48),
        )

        corner_h = max(
            1,
            int(h * 0.48),
        )

        crop = ink_mask[
            y:y + corner_h,
            x:x + corner_w
        ]

        if crop.size == 0:
            return 0.0

        return (
            cv2.countNonZero(
                crop
            )
            / float(crop.size)
        )


    # --------------------------------------------------------
    # FULL / MOSTLY FULL CARD DETECTION
    # --------------------------------------------------------

    def detect_full_cards(
        self,
        frame,
        surface_mask,
        ink_mask,
    ) -> List[Detection]:

        frame_h, frame_w = (
            frame.shape[:2]
        )

        contours, _ = cv2.findContours(
            surface_mask,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        candidates = []

        for contour in contours:

            contour_area = (
                cv2.contourArea(
                    contour
                )
            )

            if contour_area <= 0:
                continue

            x, y, w, h = (
                cv2.boundingRect(
                    contour
                )
            )

            if h <= 0 or w <= 0:
                continue

            h_frac = (
                h / float(frame_h)
            )

            w_frac = (
                w / float(frame_w)
            )

            if not (
                MIN_CARD_HEIGHT_FRAC
                <= h_frac
                <= MAX_CARD_HEIGHT_FRAC
            ):
                continue

            if not (
                MIN_CARD_WIDTH_FRAC
                <= w_frac
                <= MAX_CARD_WIDTH_FRAC
            ):
                continue

            aspect = (
                w / float(h)
            )

            if not (
                MIN_CARD_ASPECT
                <= aspect
                <= MAX_CARD_ASPECT
            ):
                continue

            bounding_area = (
                w * h
            )

            rectangularity = (
                contour_area
                / float(bounding_area)
            )

            if (
                rectangularity
                < MIN_RECTANGULARITY
            ):
                continue

            white = self.white_ratio(
                surface_mask,
                x,
                y,
                w,
                h,
            )

            if (
                white
                < MIN_WHITE_RATIO
            ):
                continue

            ink = self.corner_ink_ratio(
                ink_mask,
                x,
                y,
                w,
                h,
            )

            if (
                ink
                < MIN_CORNER_INK_RATIO
            ):
                continue

            # -----------------------------------------------
            # SCORES
            # -----------------------------------------------

            aspect_error = (
                (
                    aspect
                    - IDEAL_CARD_ASPECT
                )
                / 0.22
            )

            aspect_score = math.exp(
                -(
                    aspect_error
                    ** 2
                )
            )

            white_score = min(
                1.0,
                white / 0.82,
            )

            rect_score = min(
                1.0,
                rectangularity / 0.85,
            )

            ink_score = min(
                1.0,
                ink / 0.08,
            )

            score = (
                0.33 * aspect_score
                + 0.27 * white_score
                + 0.25 * rect_score
                + 0.15 * ink_score
            )

            if (
                score
                < MIN_FULL_CARD_SCORE
            ):
                continue

            candidates.append(
                Detection(
                    kind="card",
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    score=score,
                    white_ratio=white,
                    rectangularity=rectangularity,
                    ink_ratio=ink,
                )
            )

        return non_max_suppression(
            candidates,
            FULL_CARD_NMS_IOU,
        )


    # --------------------------------------------------------
    # EXPOSED RANK-CORNER DETECTION
    # --------------------------------------------------------

    def detect_card_anchors(
        self,
        frame,
        surface_mask,
        ink_mask,
    ) -> List[Detection]:
        """
        Fallback for overlapping cards.

        Instead of requiring the complete card rectangle, detect
        compact rank/suit ink groups surrounded by bright card
        surface.

        This matters when only the upper-left portion of a card
        is visible in a pile.
        """

        frame_h, frame_w = (
            frame.shape[:2]
        )

        # Only consider ink that occurs close to a light surface.
        surface_context = cv2.dilate(
            surface_mask,
            np.ones(
                (11, 11),
                dtype=np.uint8,
            ),
            iterations=2,
        )

        candidate_ink = (
            cv2.bitwise_and(
                ink_mask,
                surface_context,
            )
        )

        # Join rank and suit components that are vertically close.
        join_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (5, 11),
        )

        grouped = cv2.morphologyEx(
            candidate_ink,
            cv2.MORPH_CLOSE,
            join_kernel,
            iterations=1,
        )

        contours, _ = cv2.findContours(
            grouped,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        anchors = []

        for contour in contours:

            x, y, w, h = (
                cv2.boundingRect(
                    contour
                )
            )

            if w <= 0 or h <= 0:
                continue

            h_frac = (
                h / float(frame_h)
            )

            w_frac = (
                w / float(frame_w)
            )

            if not (
                MIN_ANCHOR_HEIGHT_FRAC
                <= h_frac
                <= MAX_ANCHOR_HEIGHT_FRAC
            ):
                continue

            if not (
                MIN_ANCHOR_WIDTH_FRAC
                <= w_frac
                <= MAX_ANCHOR_WIDTH_FRAC
            ):
                continue

            # -----------------------------------------------
            # Inspect context around the glyph.
            # -----------------------------------------------

            pad_x = max(
                4,
                int(w * 1.2),
            )

            pad_y = max(
                4,
                int(h * 0.7),
            )

            cx1 = max(
                0,
                x - pad_x,
            )

            cy1 = max(
                0,
                y - pad_y,
            )

            cx2 = min(
                frame_w,
                x + w + pad_x,
            )

            cy2 = min(
                frame_h,
                y + h + pad_y,
            )

            context = surface_mask[
                cy1:cy2,
                cx1:cx2
            ]

            if context.size == 0:
                continue

            context_white = (
                cv2.countNonZero(
                    context
                )
                / float(context.size)
            )

            if (
                context_white
                < MIN_ANCHOR_WHITE_CONTEXT
            ):
                continue

            glyph = candidate_ink[
                y:y + h,
                x:x + w
            ]

            glyph_density = (
                cv2.countNonZero(
                    glyph
                )
                / float(glyph.size)
            )

            # Moderate glyph density is expected.
            density_score = min(
                1.0,
                glyph_density / 0.35,
            )

            context_score = min(
                1.0,
                context_white / 0.70,
            )

            # Penalize extremely wide components such as words.
            aspect = (
                w / float(h)
            )

            if aspect <= 0:
                continue

            shape_score = math.exp(
                -(
                    (
                        aspect - 0.55
                    )
                    / 0.70
                )
                ** 2
            )

            score = (
                0.45 * context_score
                + 0.35 * density_score
                + 0.20 * shape_score
            )

            if (
                score
                < MIN_ANCHOR_SCORE
            ):
                continue

            # Expand box slightly so the crop retains context
            # useful for later recognition.
            extra_x = max(
                2,
                int(w * 0.35),
            )

            extra_y = max(
                2,
                int(h * 0.20),
            )

            bx1 = max(
                0,
                x - extra_x,
            )

            by1 = max(
                0,
                y - extra_y,
            )

            bx2 = min(
                frame_w,
                x + w + extra_x,
            )

            by2 = min(
                frame_h,
                y + h + extra_y,
            )

            anchors.append(
                Detection(
                    kind="anchor",
                    x=bx1,
                    y=by1,
                    w=bx2 - bx1,
                    h=by2 - by1,
                    score=score,
                    white_ratio=context_white,
                    rectangularity=0.0,
                    ink_ratio=glyph_density,
                )
            )

        return non_max_suppression(
            anchors,
            ANCHOR_NMS_IOU,
        )


    # --------------------------------------------------------
    # MERGE FULL CARDS + ANCHORS
    # --------------------------------------------------------

    def merge_detections(
        self,
        cards: List[Detection],
        anchors: List[Detection],
    ) -> List[Detection]:

        remaining_anchors = []

        for anchor in anchors:

            center = anchor.center()

            belongs_to_card = False

            for card in cards:

                if point_inside_detection(
                    center,
                    card,
                ):
                    belongs_to_card = True

                    # Supporting anchor makes the full-card
                    # detection more trustworthy.
                    card.score = min(
                        1.0,
                        card.score + 0.08,
                    )

                    break

            if not belongs_to_card:
                remaining_anchors.append(
                    anchor
                )

        detections = (
            cards
            + remaining_anchors
        )

        detections.sort(
            key=lambda d: (
                d.y,
                d.x,
            )
        )

        return detections


    # --------------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------------

    def detect(
        self,
        frame,
    ) -> List[Detection]:

        surface_mask = (
            self.build_card_surface_mask(
                frame
            )
        )

        ink_mask = (
            self.build_ink_mask(
                frame
            )
        )

        cards = self.detect_full_cards(
            frame,
            surface_mask,
            ink_mask,
        )

        anchors = self.detect_card_anchors(
            frame,
            surface_mask,
            ink_mask,
        )

        return self.merge_detections(
            cards,
            anchors,
        )


# ============================================================
# CROP SAVING
# ============================================================

def save_detection_crop(
    frame,
    detection: Detection,
    frame_number: int,
    detection_number: int,
):

    x = detection.x
    y = detection.y
    w = detection.w
    h = detection.h

    crop = frame[
        y:y + h,
        x:x + w
    ]

    if crop.size == 0:
        return

    filename = (
        f"frame_{frame_number:06d}"
        f"_det_{detection_number:02d}"
        f"_{detection.kind}.png"
    )

    cv2.imwrite(
        str(
            CROPS_DIR / filename
        ),
        crop,
    )


# ============================================================
# DEBUG OVERLAY
# ============================================================

def draw_debug(
    frame,
    detections: List[Detection],
):

    debug = frame.copy()

    for index, detection in enumerate(
        detections,
        start=1,
    ):

        if detection.kind == "card":
            color = (
                0,
                255,
                0,
            )

        else:
            color = (
                0,
                255,
                255,
            )

        cv2.rectangle(
            debug,
            (
                detection.x,
                detection.y,
            ),
            (
                detection.x
                + detection.w,

                detection.y
                + detection.h,
            ),
            color,
            2,
        )

        label = (
            f"{index} "
            f"{detection.kind} "
            f"{detection.score:.2f}"
        )

        text_y = max(
            18,
            detection.y - 5,
        )

        cv2.putText(
            debug,
            label,
            (
                detection.x,
                text_y,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        debug,
        (
            f"detections="
            f"{len(detections)}"
        ),
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return debug


# ============================================================
# MAIN DATASET SCAN
# ============================================================

def main():

    capture_paths = (
        get_capture_paths()
    )

    if not capture_paths:

        print(
            "No images found in captures/"
        )

        return

    # --------------------------------------------------------
    # Reset output
    # --------------------------------------------------------

    if OUTPUT_DIR.exists():
        shutil.rmtree(
            OUTPUT_DIR
        )

    DEBUG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CROPS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    detector = CardDetector()

    detection_rows = []
    frame_rows = []

    previous_count = None

    total_detections = 0
    frames_with_detections = 0

    card_detections = 0
    anchor_detections = 0

    print(
        f"Frames found: "
        f"{len(capture_paths)}"
    )

    print()
    print(
        "Scanning full screenshots for cards..."
    )
    print()


    # ========================================================
    # SCAN
    # ========================================================

    for frame_number, path in enumerate(
        capture_paths,
        start=1,
    ):

        frame = cv2.imread(
            str(path)
        )

        if frame is None:
            print(
                f"Warning: couldn't read "
                f"{path.name}"
            )

            continue

        frame_h, frame_w = (
            frame.shape[:2]
        )

        detections = detector.detect(
            frame
        )

        count = len(
            detections
        )

        if count > 0:
            frames_with_detections += 1

        total_detections += count

        full_count = sum(
            1
            for d in detections
            if d.kind == "card"
        )

        anchor_count = sum(
            1
            for d in detections
            if d.kind == "anchor"
        )

        card_detections += full_count
        anchor_detections += anchor_count

        count_changed = (
            previous_count is not None
            and count != previous_count
        )

        frame_rows.append(
            {
                "frame_number":
                    frame_number,

                "filename":
                    path.name,

                "detections":
                    count,

                "full_cards":
                    full_count,

                "anchors":
                    anchor_count,
            }
        )


        # ----------------------------------------------------
        # Save individual detection rows/crops
        # ----------------------------------------------------

        for detection_number, detection in enumerate(
            detections,
            start=1,
        ):

            cx, cy = (
                detection.center()
            )

            detection_rows.append(
                {
                    "frame_number":
                        frame_number,

                    "filename":
                        path.name,

                    "detection_number":
                        detection_number,

                    "kind":
                        detection.kind,

                    "score":
                        f"{detection.score:.6f}",

                    "x":
                        detection.x,

                    "y":
                        detection.y,

                    "w":
                        detection.w,

                    "h":
                        detection.h,

                    # Normalized coordinates are what later
                    # tracking should primarily use.
                    "center_x_norm":
                        f"{cx / frame_w:.6f}",

                    "center_y_norm":
                        f"{cy / frame_h:.6f}",

                    "width_norm":
                        f"{detection.w / frame_w:.6f}",

                    "height_norm":
                        f"{detection.h / frame_h:.6f}",

                    "white_ratio":
                        f"{detection.white_ratio:.6f}",

                    "rectangularity":
                        f"{detection.rectangularity:.6f}",

                    "ink_ratio":
                        f"{detection.ink_ratio:.6f}",
                }
            )

            save_detection_crop(
                frame,
                detection,
                frame_number,
                detection_number,
            )


        # ----------------------------------------------------
        # Debug image
        #
        # Save every Nth frame, plus whenever the number of
        # detections changes.
        # ----------------------------------------------------

        should_save_debug = (
            frame_number == 1
            or frame_number
            % DEBUG_EVERY_N_FRAMES
            == 0
            or count_changed
        )

        if should_save_debug:

            debug = draw_debug(
                frame,
                detections,
            )

            debug_name = (
                f"frame_"
                f"{frame_number:06d}"
                f"_count_{count}.png"
            )

            cv2.imwrite(
                str(
                    DEBUG_DIR
                    / debug_name
                ),
                debug,
            )


        if (
            frame_number % 50 == 0
            or frame_number
            == len(capture_paths)
        ):

            print(
                f"Processed "
                f"{frame_number}/"
                f"{len(capture_paths)}"
            )

        previous_count = count


    # ========================================================
    # WRITE DETECTIONS CSV
    # ========================================================

    with open(
        DETECTIONS_CSV,
        "w",
        newline="",
    ) as file:

        fieldnames = [
            "frame_number",
            "filename",
            "detection_number",
            "kind",
            "score",
            "x",
            "y",
            "w",
            "h",
            "center_x_norm",
            "center_y_norm",
            "width_norm",
            "height_norm",
            "white_ratio",
            "rectangularity",
            "ink_ratio",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            detection_rows
        )


    # ========================================================
    # WRITE PER-FRAME COUNTS
    # ========================================================

    with open(
        FRAMES_CSV,
        "w",
        newline="",
    ) as file:

        fieldnames = [
            "frame_number",
            "filename",
            "detections",
            "full_cards",
            "anchors",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            frame_rows
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print(
        "--- CARD DETECTION COMPLETE ---"
    )

    print(
        f"Frames scanned: "
        f"{len(capture_paths)}"
    )

    print(
        f"Frames with detections: "
        f"{frames_with_detections}"
    )

    print(
        f"Total detections: "
        f"{total_detections}"
    )

    print(
        f"Full-card detections: "
        f"{card_detections}"
    )

    print(
        f"Exposed-corner detections: "
        f"{anchor_detections}"
    )

    if capture_paths:

        average = (
            total_detections
            / len(capture_paths)
        )

        print(
            f"Average detections/frame: "
            f"{average:.2f}"
        )

    print()
    print(
        f"Debug overlays: "
        f"{DEBUG_DIR.resolve()}"
    )

    print(
        f"Detection crops: "
        f"{CROPS_DIR.resolve()}"
    )

    print(
        f"Per-frame data: "
        f"{FRAMES_CSV.resolve()}"
    )

    print(
        f"Detection data: "
        f"{DETECTIONS_CSV.resolve()}"
    )


if __name__ == "__main__":
    main()