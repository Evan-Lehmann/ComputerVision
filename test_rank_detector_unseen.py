from pathlib import Path
import argparse
import csv
import math
import shutil

import cv2
import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    ROOT
    / "rank_detector_runs"
    / "rank_detector_v2"
    / "weights"
    / "best.pt"
)

DEFAULT_INPUT = ROOT / "captures"
DEFAULT_OUTPUT = ROOT / "unseen_detector_test"


# =========================================================
# TILE GEOMETRY
# =========================================================

TILE_W_FRAC = 0.40
TILE_H_FRAC = 0.50

STRIDE_X_FRAC = 0.30
STRIDE_Y_FRAC = 0.35

CONF_THRESHOLD = 0.25


# =========================================================
# DEDUPE PARAMETERS
# =========================================================

# Stage 1:
# remove duplicate predictions emitted inside one tile.
SAME_TILE_IOU_THRESHOLD = 0.45


# Stage 2:
# merge duplicate predictions produced by overlapping tiles.
DUP_IOU_THRESHOLD = 0.10
DUP_IOS_THRESHOLD = 0.45

DUP_CENTER_X = 0.45
DUP_CENTER_Y = 0.45

MAX_WIDTH_RATIO = 2.0
MAX_HEIGHT_RATIO = 2.0


# Stage 3:
# final safety NMS after cross-tile clustering.
#
# v3 audit:
# suspicious survivor IoU ~= 0.502
# next-highest final IoU ~= 0.315
#
# 0.40 therefore separates the known duplicate cleanly.
FINAL_IOU_THRESHOLD = 0.40


# =========================================================
# KNOWN REGRESSION CASES
# =========================================================

KNOWN_EXPECTED_COUNTS = {
    "frame_20261025_030012_556713_001025.png": 7,
}


# =========================================================
# BASIC GEOMETRY
# =========================================================

def positions(length, tile_size, stride):
    values = list(
        range(
            0,
            max(1, length - tile_size + 1),
            stride,
        )
    )

    final = max(
        0,
        length - tile_size,
    )

    if not values or values[-1] != final:
        values.append(final)

    return sorted(set(values))


def box_area(det):
    width = max(
        0.0,
        det["x2"] - det["x1"],
    )

    height = max(
        0.0,
        det["y2"] - det["y1"],
    )

    return width * height


def intersection_area(a, b):
    ix1 = max(
        a["x1"],
        b["x1"],
    )

    iy1 = max(
        a["y1"],
        b["y1"],
    )

    ix2 = min(
        a["x2"],
        b["x2"],
    )

    iy2 = min(
        a["y2"],
        b["y2"],
    )

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    return (
        (ix2 - ix1)
        * (iy2 - iy1)
    )


def box_iou(a, b):
    inter = intersection_area(
        a,
        b,
    )

    if inter <= 0:
        return 0.0

    area_a = box_area(a)
    area_b = box_area(b)

    union = (
        area_a
        + area_b
        - inter
    )

    if union <= 0:
        return 0.0

    return inter / union


def box_ios(a, b):
    """
    Intersection divided by the smaller box area.

    Useful for tiny detections where one tile produces
    a slightly larger box than another tile.
    """

    inter = intersection_area(
        a,
        b,
    )

    if inter <= 0:
        return 0.0

    smaller = min(
        box_area(a),
        box_area(b),
    )

    if smaller <= 0:
        return 0.0

    return inter / smaller


def box_center(det):
    return (
        (
            det["x1"]
            + det["x2"]
        ) / 2,
        (
            det["y1"]
            + det["y2"]
        ) / 2,
    )


def dimensions(det):
    return (
        det["x2"] - det["x1"],
        det["y2"] - det["y1"],
    )


# =========================================================
# STAGE 1:
# SAME-TILE NMS
# =========================================================

def same_tile_nms(
    detections,
    iou_threshold=SAME_TILE_IOU_THRESHOLD,
):
    """
    Remove duplicate detections emitted by the SAME
    inference tile.

    Keeps the highest-confidence box.
    """

    by_tile = {}

    for det in detections:
        by_tile.setdefault(
            det["tile_id"],
            [],
        ).append(det)

    cleaned = []
    removed_count = 0

    for tile_detections in by_tile.values():

        tile_detections = sorted(
            tile_detections,
            key=lambda d: d["confidence"],
            reverse=True,
        )

        kept = []

        while tile_detections:

            best = tile_detections.pop(0)
            kept.append(best)

            remaining = []

            for det in tile_detections:

                overlap = box_iou(
                    best,
                    det,
                )

                if overlap >= iou_threshold:
                    removed_count += 1
                else:
                    remaining.append(det)

            tile_detections = remaining

        cleaned.extend(kept)

    return (
        cleaned,
        removed_count,
    )


# =========================================================
# STAGE 2:
# CROSS-TILE DUPLICATE SCORING
# =========================================================

def duplicate_score(a, b):
    """
    Return a duplicate-likelihood score.

    None means the two boxes should not be joined.
    """

    if a["tile_id"] == b["tile_id"]:
        return None

    aw, ah = dimensions(a)
    bw, bh = dimensions(b)

    if min(
        aw,
        ah,
        bw,
        bh,
    ) <= 0:
        return None

    width_ratio = (
        max(aw, bw)
        / min(aw, bw)
    )

    height_ratio = (
        max(ah, bh)
        / min(ah, bh)
    )

    if width_ratio > MAX_WIDTH_RATIO:
        return None

    if height_ratio > MAX_HEIGHT_RATIO:
        return None

    iou = box_iou(
        a,
        b,
    )

    ios = box_ios(
        a,
        b,
    )

    ax, ay = box_center(a)
    bx, by = box_center(b)

    dx = abs(
        ax - bx
    )

    dy = abs(
        ay - by
    )

    avg_w = (
        aw + bw
    ) / 2

    avg_h = (
        ah + bh
    ) / 2

    x_close = (
        dx
        <= avg_w * DUP_CENTER_X
    )

    y_close = (
        dy
        <= avg_h * DUP_CENTER_Y
    )

    # Strong containment.
    if ios >= DUP_IOS_THRESHOLD:
        return (
            3.0
            + ios
            + iou
        )

    # Standard overlap.
    if iou >= DUP_IOU_THRESHOLD:
        return (
            2.0
            + iou
            + ios
        )

    # Tiny boxes may shift noticeably between tiles.
    if x_close and y_close:

        normalized_distance = math.sqrt(
            (
                dx
                / max(avg_w, 1)
            ) ** 2
            +
            (
                dy
                / max(avg_h, 1)
            ) ** 2
        )

        return (
            1.0
            + max(
                0.0,
                1.0 - normalized_distance,
            )
        )

    return None


# =========================================================
# UNION-FIND
# =========================================================

class UnionFind:
    def __init__(
        self,
        detections,
    ):
        count = len(detections)

        self.parent = list(
            range(count)
        )

        self.tile_sets = [
            {
                detections[i]["tile_id"]
            }
            for i in range(count)
        ]

    def find(self, x):
        while (
            self.parent[x]
            != x
        ):
            self.parent[x] = (
                self.parent[
                    self.parent[x]
                ]
            )

            x = self.parent[x]

        return x

    def can_union(
        self,
        a,
        b,
    ):
        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return False

        # Safety rule:
        # two clusters cannot be merged if both already
        # contain a prediction from the same tile.
        return (
            self.tile_sets[root_a]
            .isdisjoint(
                self.tile_sets[root_b]
            )
        )

    def union(
        self,
        a,
        b,
    ):
        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return

        self.parent[root_b] = root_a

        self.tile_sets[root_a].update(
            self.tile_sets[root_b]
        )


def merge_cross_tile_duplicates(
    detections,
):
    if not detections:
        return (
            [],
            [],
        )

    uf = UnionFind(
        detections
    )

    candidate_edges = []

    count = len(detections)

    for i in range(count):
        for j in range(
            i + 1,
            count,
        ):
            score = duplicate_score(
                detections[i],
                detections[j],
            )

            if score is None:
                continue

            candidate_edges.append(
                (
                    score,
                    i,
                    j,
                )
            )

    candidate_edges.sort(
        reverse=True,
        key=lambda item: item[0],
    )

    for score, i, j in candidate_edges:

        if not uf.can_union(
            i,
            j,
        ):
            continue

        uf.union(
            i,
            j,
        )

    groups = {}

    for i in range(count):

        root = uf.find(i)

        groups.setdefault(
            root,
            [],
        ).append(
            detections[i]
        )

    final = []
    duplicate_groups = []

    cluster_number = 0

    for cluster in groups.values():

        cluster_number += 1

        best = max(
            cluster,
            key=lambda d: d["confidence"],
        )

        best = best.copy()

        best["cluster_id"] = (
            cluster_number
        )

        best["cluster_size"] = (
            len(cluster)
        )

        final.append(best)

        if len(cluster) > 1:
            duplicate_groups.append(
                cluster
            )

    final.sort(
        key=lambda d: (
            d["y1"],
            d["x1"],
        )
    )

    return (
        final,
        duplicate_groups,
    )


# =========================================================
# STAGE 3:
# FINAL SAFETY NMS
# =========================================================

def final_nms(
    detections,
    iou_threshold=FINAL_IOU_THRESHOLD,
):
    """
    Final safety pass after tile-aware clustering.

    Strongly overlapping surviving boxes should represent
    the same physical rank.

    v3 audit showed one survivor pair around IoU 0.50,
    while the next-highest pair was around 0.31.
    """

    detections = sorted(
        detections,
        key=lambda d: d["confidence"],
        reverse=True,
    )

    kept = []
    removed = 0

    while detections:

        best = detections.pop(0)

        kept.append(best)

        remaining = []

        for det in detections:

            overlap = box_iou(
                best,
                det,
            )

            if overlap >= iou_threshold:
                removed += 1
            else:
                remaining.append(det)

        detections = remaining

    kept.sort(
        key=lambda d: (
            d["y1"],
            d["x1"],
        )
    )

    return (
        kept,
        removed,
    )


# =========================================================
# FULL-FRAME DETECTION
# =========================================================

def detect_full_frame(
    model,
    image,
):
    height, width = (
        image.shape[:2]
    )

    tile_w = int(
        width * TILE_W_FRAC
    )

    tile_h = int(
        height * TILE_H_FRAC
    )

    stride_x = max(
        1,
        int(
            width
            * STRIDE_X_FRAC
        ),
    )

    stride_y = max(
        1,
        int(
            height
            * STRIDE_Y_FRAC
        ),
    )

    xs = positions(
        width,
        tile_w,
        stride_x,
    )

    ys = positions(
        height,
        tile_h,
        stride_y,
    )

    raw_detections = []

    tile_number = 0
    detection_number = 0

    for tile_y1 in ys:
        for tile_x1 in xs:

            tile_number += 1

            tile_x2 = (
                tile_x1
                + tile_w
            )

            tile_y2 = (
                tile_y1
                + tile_h
            )

            tile = image[
                tile_y1:tile_y2,
                tile_x1:tile_x2
            ]

            results = model.predict(
                source=tile,
                imgsz=640,
                conf=CONF_THRESHOLD,
                verbose=False,
            )

            result = results[0]

            if result.boxes is None:
                continue

            for box in result.boxes:

                detection_number += 1

                confidence = float(
                    box.conf[0]
                )

                (
                    bx1,
                    by1,
                    bx2,
                    by2,
                ) = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                    .tolist()
                )

                raw_detections.append(
                    {
                        "detection_id":
                            detection_number,

                        "x1":
                            float(
                                bx1
                                + tile_x1
                            ),

                        "y1":
                            float(
                                by1
                                + tile_y1
                            ),

                        "x2":
                            float(
                                bx2
                                + tile_x1
                            ),

                        "y2":
                            float(
                                by2
                                + tile_y1
                            ),

                        "confidence":
                            confidence,

                        "tile_id":
                            tile_number,
                    }
                )

    # Stage 1:
    # same-tile duplicate suppression.
    (
        same_tile_cleaned,
        same_tile_removed,
    ) = same_tile_nms(
        raw_detections,
        iou_threshold=(
            SAME_TILE_IOU_THRESHOLD
        ),
    )

    # Stage 2:
    # cross-tile clustering.
    (
        clustered,
        duplicate_groups,
    ) = (
        merge_cross_tile_duplicates(
            same_tile_cleaned
        )
    )

    # Stage 3:
    # final surviving-overlap cleanup.
    (
        final_detections,
        final_overlap_removed,
    ) = final_nms(
        clustered,
        iou_threshold=(
            FINAL_IOU_THRESHOLD
        ),
    )

    return (
        raw_detections,
        same_tile_cleaned,
        clustered,
        final_detections,
        duplicate_groups,
        same_tile_removed,
        final_overlap_removed,
    )


# =========================================================
# DRAWING
# =========================================================

def choose_label_position(
    det,
    used_positions,
):
    label_x = int(
        det["x1"]
    )

    label_y = max(
        24,
        int(
            det["y1"]
        ) - 8,
    )

    while any(
        abs(
            label_x - px
        ) < 60
        and
        abs(
            label_y - py
        ) < 24
        for px, py
        in used_positions
    ):
        label_y -= 24

        if label_y < 24:

            label_y = (
                int(
                    det["y2"]
                )
                + 24
            )

            break

    used_positions.append(
        (
            label_x,
            label_y,
        )
    )

    return (
        label_x,
        label_y,
    )


def draw_final(
    image,
    detections,
):
    output = image.copy()

    used_positions = []

    for det in detections:

        x1 = int(
            det["x1"]
        )

        y1 = int(
            det["y1"]
        )

        x2 = int(
            det["x2"]
        )

        y2 = int(
            det["y2"]
        )

        cv2.rectangle(
            output,
            (
                x1,
                y1,
            ),
            (
                x2,
                y2,
            ),
            (
                0,
                255,
                0,
            ),
            2,
        )

        (
            label_x,
            label_y,
        ) = choose_label_position(
            det,
            used_positions,
        )

        label = (
            f'{det["confidence"]:.2f}'
        )

        cluster_size = (
            det.get(
                "cluster_size",
                1,
            )
        )

        if cluster_size > 1:
            label += (
                f" x{cluster_size}"
            )

        cv2.putText(
            output,
            label,
            (
                label_x,
                label_y,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (
                0,
                255,
                0,
            ),
            2,
            cv2.LINE_AA,
        )

    return output


def draw_raw(
    image,
    detections,
):
    output = image.copy()

    for det in detections:

        x1 = int(
            det["x1"]
        )

        y1 = int(
            det["y1"]
        )

        x2 = int(
            det["x2"]
        )

        y2 = int(
            det["y2"]
        )

        cv2.rectangle(
            output,
            (
                x1,
                y1,
            ),
            (
                x2,
                y2,
            ),
            (
                0,
                165,
                255,
            ),
            2,
        )

        text = (
            f'{det["confidence"]:.2f}'
            f' T{det["tile_id"]}'
        )

        cv2.putText(
            output,
            text,
            (
                x1,
                max(
                    20,
                    y1 - 5,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (
                0,
                165,
                255,
            ),
            1,
            cv2.LINE_AA,
        )

    return output


def draw_stage(
    image,
    detections,
    color,
):
    output = image.copy()

    for det in detections:

        cv2.rectangle(
            output,
            (
                int(det["x1"]),
                int(det["y1"]),
            ),
            (
                int(det["x2"]),
                int(det["y2"]),
            ),
            color,
            2,
        )

    return output


def make_debug_panel(
    raw_image,
    stage1_image,
    clustered_image,
    final_image,
):
    images = [
        raw_image,
        stage1_image,
        clustered_image,
        final_image,
    ]

    labels = [
        "RAW TILE OUTPUT",
        "SAME-TILE NMS",
        "CROSS-TILE CLUSTERS",
        "FINAL NMS",
    ]

    colors = [
        (0, 165, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 0),
    ]

    height = max(
        image.shape[0]
        for image in images
    )

    width = sum(
        image.shape[1]
        for image in images
    )

    canvas = np.ones(
        (
            height,
            width,
            3,
        ),
        dtype=np.uint8,
    ) * 255

    current_x = 0

    for image, label, color in zip(
        images,
        labels,
        colors,
    ):
        image_width = (
            image.shape[1]
        )

        canvas[
            :image.shape[0],
            current_x:
            current_x + image_width
        ] = image

        cv2.putText(
            canvas,
            label,
            (
                current_x + 20,
                35,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
        )

        current_x += (
            image_width
        )

    return canvas


# =========================================================
# MAIN
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        default=str(
            DEFAULT_INPUT
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(
            DEFAULT_OUTPUT
        ),
    )

    parser.add_argument(
        "--every",
        type=int,
        default=25,
    )

    args = parser.parse_args()

    input_dir = Path(
        args.input
    )

    output_dir = Path(
        args.output
    )

    debug_dir = (
        output_dir
        / "dedupe_debug"
    )

    if output_dir.exists():
        shutil.rmtree(
            output_dir
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    debug_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found:\n"
            f"{MODEL_PATH}"
        )

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input folder not found:\n"
            f"{input_dir}"
        )

    print()
    print(
        "Loading V2 rank detector..."
    )

    print(
        MODEL_PATH
    )

    model = YOLO(
        str(
            MODEL_PATH
        )
    )

    frames = sorted(
        list(
            input_dir.glob(
                "*.png"
            )
        )
        +
        list(
            input_dir.glob(
                "*.jpg"
            )
        )
        +
        list(
            input_dir.glob(
                "*.jpeg"
            )
        )
    )

    selected = frames[
        ::args.every
    ]

    print()

    print(
        f"Found {len(frames)} frames."
    )

    print(
        f"Testing {len(selected)} frames."
    )

    print()

    audit_rows = []

    raw_detection_rows = []
    final_detection_rows = []

    total_raw = 0
    total_stage1 = 0
    total_clustered = 0
    total_final = 0

    total_same_tile_removed = 0
    total_cross_tile_removed = 0
    total_final_overlap_removed = 0

    regression_failures = 0

    for index, frame_path in enumerate(
        selected,
        start=1,
    ):

        image = cv2.imread(
            str(
                frame_path
            )
        )

        if image is None:
            continue

        (
            raw,
            stage1,
            clustered,
            final,
            duplicate_groups,
            same_tile_removed,
            final_overlap_removed,
        ) = detect_full_frame(
            model,
            image,
        )

        cross_tile_removed = (
            len(stage1)
            - len(clustered)
        )

        total_raw += (
            len(raw)
        )

        total_stage1 += (
            len(stage1)
        )

        total_clustered += (
            len(clustered)
        )

        total_final += (
            len(final)
        )

        total_same_tile_removed += (
            same_tile_removed
        )

        total_cross_tile_removed += (
            cross_tile_removed
        )

        total_final_overlap_removed += (
            final_overlap_removed
        )

        final_view = draw_final(
            image,
            final,
        )

        cv2.imwrite(
            str(
                output_dir
                / frame_path.name
            ),
            final_view,
        )

        if raw:

            raw_view = draw_raw(
                image,
                raw,
            )

            stage1_view = draw_stage(
                image,
                stage1,
                (
                    255,
                    255,
                    0,
                ),
            )

            clustered_view = draw_stage(
                image,
                clustered,
                (
                    255,
                    0,
                    255,
                ),
            )

            debug_panel = (
                make_debug_panel(
                    raw_view,
                    stage1_view,
                    clustered_view,
                    final_view,
                )
            )

            cv2.imwrite(
                str(
                    debug_dir
                    / frame_path.name
                ),
                debug_panel,
            )

        final_confidences = [
            det["confidence"]
            for det in final
        ]

        min_confidence = (
            min(
                final_confidences
            )
            if final_confidences
            else ""
        )

        average_confidence = (
            sum(
                final_confidences
            )
            / len(
                final_confidences
            )
            if final_confidences
            else ""
        )

        expected_count = (
            KNOWN_EXPECTED_COUNTS
            .get(
                frame_path.name
            )
        )

        regression_status = ""

        if expected_count is not None:

            if (
                len(final)
                == expected_count
            ):
                regression_status = (
                    "PASS"
                )

            else:
                regression_status = (
                    "FAIL"
                )

                regression_failures += 1

        audit_rows.append(
            {
                "frame":
                    frame_path.name,

                "raw_detections":
                    len(raw),

                "after_same_tile_nms":
                    len(stage1),

                "after_cross_tile":
                    len(clustered),

                "final_detections":
                    len(final),

                "same_tile_removed":
                    same_tile_removed,

                "cross_tile_removed":
                    cross_tile_removed,

                "final_overlap_removed":
                    final_overlap_removed,

                "duplicate_groups":
                    len(
                        duplicate_groups
                    ),

                "min_confidence":
                    min_confidence,

                "average_confidence":
                    average_confidence,

                "expected_count":
                    (
                        expected_count
                        if expected_count
                        is not None
                        else ""
                    ),

                "regression":
                    regression_status,
            }
        )

        for det in raw:

            raw_detection_rows.append(
                {
                    "frame":
                        frame_path.name,

                    "detection_id":
                        det[
                            "detection_id"
                        ],

                    "tile_id":
                        det[
                            "tile_id"
                        ],

                    "confidence":
                        det[
                            "confidence"
                        ],

                    "x1":
                        det["x1"],

                    "y1":
                        det["y1"],

                    "x2":
                        det["x2"],

                    "y2":
                        det["y2"],
                }
            )

        for det in final:

            final_detection_rows.append(
                {
                    "frame":
                        frame_path.name,

                    "cluster_id":
                        det.get(
                            "cluster_id",
                            "",
                        ),

                    "cluster_size":
                        det.get(
                            "cluster_size",
                            1,
                        ),

                    "tile_id":
                        det[
                            "tile_id"
                        ],

                    "confidence":
                        det[
                            "confidence"
                        ],

                    "x1":
                        det["x1"],

                    "y1":
                        det["y1"],

                    "x2":
                        det["x2"],

                    "y2":
                        det["y2"],
                }
            )

        message = (
            f"{index:03d}/"
            f"{len(selected):03d} "
            f"{frame_path.name}: "
            f"{len(raw)} raw -> "
            f"{len(stage1)} stage1 -> "
            f"{len(clustered)} clustered -> "
            f"{len(final)} final"
        )

        if final_overlap_removed:

            message += (
                f" | final-NMS removed "
                f"{final_overlap_removed}"
            )

        if regression_status:

            message += (
                f" | regression "
                f"{regression_status} "
                f"expected="
                f"{expected_count}"
            )

        print(
            message
        )

    # =====================================================
    # AUDIT CSV
    # =====================================================

    audit_path = (
        output_dir
        / "audit.csv"
    )

    with audit_path.open(
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "frame",
                "raw_detections",
                "after_same_tile_nms",
                "after_cross_tile",
                "final_detections",
                "same_tile_removed",
                "cross_tile_removed",
                "final_overlap_removed",
                "duplicate_groups",
                "min_confidence",
                "average_confidence",
                "expected_count",
                "regression",
            ],
        )

        writer.writeheader()

        writer.writerows(
            audit_rows
        )

    # =====================================================
    # RAW DETECTIONS CSV
    # =====================================================

    raw_csv = (
        output_dir
        / "raw_detections.csv"
    )

    with raw_csv.open(
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "frame",
                "detection_id",
                "tile_id",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
            ],
        )

        writer.writeheader()

        writer.writerows(
            raw_detection_rows
        )

    # =====================================================
    # FINAL DETECTIONS CSV
    # =====================================================

    final_csv = (
        output_dir
        / "final_detections.csv"
    )

    with final_csv.open(
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "frame",
                "cluster_id",
                "cluster_size",
                "tile_id",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
            ],
        )

        writer.writeheader()

        writer.writerows(
            final_detection_rows
        )

    print()
    print(
        "================================"
    )

    print(
        "UNSEEN DETECTOR TEST COMPLETE"
    )

    print(
        "================================"
    )

    print()

    print(
        f"Frames tested:             "
        f"{len(selected)}"
    )

    print(
        f"Raw detections:            "
        f"{total_raw}"
    )

    print(
        f"After same-tile NMS:       "
        f"{total_stage1}"
    )

    print(
        f"After cross-tile merge:    "
        f"{total_clustered}"
    )

    print(
        f"Final detections:          "
        f"{total_final}"
    )

    print(
        f"Same-tile duplicates rm:   "
        f"{total_same_tile_removed}"
    )

    print(
        f"Cross-tile duplicates rm:  "
        f"{total_cross_tile_removed}"
    )

    print(
        f"Final-overlap duplicates:  "
        f"{total_final_overlap_removed}"
    )

    print(
        f"Regression failures:       "
        f"{regression_failures}"
    )

    print()

    print(
        "Audit CSV:"
    )

    print(
        audit_path
    )

    print()

    print(
        "Final detection CSV:"
    )

    print(
        final_csv
    )

    print()

    print(
        "Debug frames:"
    )

    print(
        debug_dir
    )


if __name__ == "__main__":
    main()