from pathlib import Path
import re
import sys

import cv2
import numpy as np


# ============================================================
# PROJECT IMPORTS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from detect_cards import CardDetector


# ============================================================
# CONFIG
# ============================================================

# Analyze every second captured screenshot.
# Capture is currently ~2 FPS, so this is roughly 1 analysis/sec.
FRAME_STEP = 2


# Require a rank to appear repeatedly before accepting it.
REQUIRED_CONFIRMATIONS = 2


# Similarity required before giving a rank instead of UNKNOWN.
MIN_RANK_SIMILARITY = 0.48


# When the same detected card remains in approximately the same
# location over successive frames, treat it as one card.
POSITION_MATCH_DISTANCE = 0.07


# Once a card event has been emitted from a location, don't
# emit it repeatedly while it remains on screen.
TRACK_MAX_MISSES = 3


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


# ============================================================
# REFERENCE DATA
# ============================================================

REFERENCE_DIRS = [
    PROJECT_ROOT / "samples" / "cards",
    PROJECT_ROOT / "output" / "processed_v2",
]


def infer_rank_from_text(text):
    """
    Try to infer rank from a filename or parent folder.

    This is deliberately permissive because the historical
    CardVision datasets used several naming conventions.
    """

    text = text.upper()

    # 10 first so it doesn't become 1.
    if re.search(
        r"(^|[^0-9])10([^0-9]|$)",
        text,
    ):
        return "10"

    aliases = {
        "ACE": "A",
        "KING": "K",
        "QUEEN": "Q",
        "JACK": "J",
    }

    for word, rank in aliases.items():
        if word in text:
            return rank

    for rank in (
        "A",
        "K",
        "Q",
        "J",
        "9",
        "8",
        "7",
        "6",
        "5",
        "4",
        "3",
        "2",
    ):
        pattern = (
            rf"(^|[^A-Z0-9])"
            rf"{re.escape(rank)}"
            rf"([^A-Z0-9]|$)"
        )

        if re.search(
            pattern,
            text,
        ):
            return rank

    return None


# ============================================================
# RANK NORMALIZATION
# ============================================================

def extract_rank_region(card_image):
    """
    Use the upper-left portion of a detected card.

    The historical reference images and runtime detections are
    both transformed through this same function.
    """

    if card_image is None:
        return None

    h, w = card_image.shape[:2]

    if h < 10 or w < 10:
        return None

    crop = card_image[
        0:max(1, int(h * 0.45)),
        0:max(1, int(w * 0.48)),
    ]

    return crop


def normalize_rank_image(image):
    """
    Convert a rank-corner crop into a normalized binary glyph.

    Output:
        64 x 64 float32 image
    """

    if image is None:
        return None

    if image.size == 0:
        return None

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0,
    )

    # Dark/red glyphs should become foreground.
    _, binary = cv2.threshold(
        gray,
        175,
        255,
        cv2.THRESH_BINARY_INV,
    )

    # Remove tiny isolated noise.
    kernel = np.ones(
        (2, 2),
        dtype=np.uint8,
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel,
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    useful = []

    for contour in contours:
        area = cv2.contourArea(
            contour
        )

        if area >= 5:
            useful.append(
                contour
            )

    if not useful:
        return None

    points = np.vstack(
        useful
    )

    x, y, w, h = cv2.boundingRect(
        points
    )

    glyph = binary[
        y:y + h,
        x:x + w,
    ]

    if glyph.size == 0:
        return None

    canvas_size = 64

    scale = min(
        52 / float(glyph.shape[1]),
        52 / float(glyph.shape[0]),
    )

    new_w = max(
        1,
        int(
            glyph.shape[1]
            * scale
        ),
    )

    new_h = max(
        1,
        int(
            glyph.shape[0]
            * scale
        ),
    )

    resized = cv2.resize(
        glyph,
        (
            new_w,
            new_h,
        ),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.zeros(
        (
            canvas_size,
            canvas_size,
        ),
        dtype=np.uint8,
    )

    x_offset = (
        canvas_size - new_w
    ) // 2

    y_offset = (
        canvas_size - new_h
    ) // 2

    canvas[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w,
    ] = resized

    return (
        canvas.astype(
            np.float32
        )
        / 255.0
    )


# ============================================================
# TEMPLATE LIBRARY
# ============================================================

def load_rank_templates():
    templates = []

    seen_paths = set()

    for reference_dir in REFERENCE_DIRS:

        if not reference_dir.exists():
            continue

        for path in reference_dir.rglob(
            "*.png"
        ):

            if path in seen_paths:
                continue

            seen_paths.add(
                path
            )

            rank = infer_rank_from_text(
                " ".join(
                    [
                        path.name,
                        path.parent.name,
                    ]
                )
            )

            if rank not in VALID_RANKS:
                continue

            image = cv2.imread(
                str(path)
            )

            if image is None:
                continue

            rank_region = (
                extract_rank_region(
                    image
                )
            )

            normalized = (
                normalize_rank_image(
                    rank_region
                )
            )

            if normalized is None:
                continue

            templates.append(
                {
                    "rank": rank,
                    "image": normalized,
                    "source": str(path),
                }
            )

    print(
        f"[legacy recognizer] "
        f"Loaded {len(templates)} "
        f"rank reference images."
    )

    return templates


# ============================================================
# CLASSIFICATION
# ============================================================

def cosine_similarity(a, b):
    a = a.flatten()
    b = b.flatten()

    a_norm = np.linalg.norm(
        a
    )

    b_norm = np.linalg.norm(
        b
    )

    if (
        a_norm < 1e-8
        or b_norm < 1e-8
    ):
        return 0.0

    return float(
        np.dot(a, b)
        / (
            a_norm
            * b_norm
        )
    )


def classify_rank(
    card_crop,
    templates,
):
    rank_region = (
        extract_rank_region(
            card_crop
        )
    )

    normalized = (
        normalize_rank_image(
            rank_region
        )
    )

    if normalized is None:
        return (
            "UNKNOWN",
            0.0,
        )

    scores_by_rank = {}

    for template in templates:

        score = cosine_similarity(
            normalized,
            template["image"],
        )

        rank = template[
            "rank"
        ]

        if (
            rank not in scores_by_rank
            or score
            > scores_by_rank[rank]
        ):
            scores_by_rank[
                rank
            ] = score

    if not scores_by_rank:
        return (
            "UNKNOWN",
            0.0,
        )

    best_rank = max(
        scores_by_rank,
        key=scores_by_rank.get,
    )

    best_score = scores_by_rank[
        best_rank
    ]

    if (
        best_score
        < MIN_RANK_SIMILARITY
    ):
        return (
            "UNKNOWN",
            best_score,
        )

    return (
        best_rank,
        best_score,
    )


# ============================================================
# TEMPORAL TRACK
# ============================================================

class Track:
    def __init__(
        self,
        x,
        y,
        rank,
        confidence,
    ):
        self.x = x
        self.y = y

        self.rank = rank
        self.confidence = confidence

        self.confirmations = 1
        self.misses = 0

        self.emitted = False


def normalized_distance(
    x1,
    y1,
    x2,
    y2,
):
    return float(
        np.sqrt(
            (
                x1 - x2
            ) ** 2
            +
            (
                y1 - y2
            ) ** 2
        )
    )


# ============================================================
# ROUND RECOGNITION
# ============================================================

def recognize_round(
    frame_paths
):
    templates = (
        load_rank_templates()
    )

    if not templates:
        print(
            "[legacy recognizer] "
            "No reference templates found."
        )

        return []

    detector = CardDetector()

    tracks = []
    events = []

    sampled_paths = frame_paths[
        ::FRAME_STEP
    ]

    print(
        f"[legacy recognizer] "
        f"Analyzing "
        f"{len(sampled_paths)} "
        f"of {len(frame_paths)} frames."
    )

    for frame_index, path in enumerate(
        sampled_paths
    ):
        frame = cv2.imread(
            str(path)
        )

        if frame is None:
            continue

        frame_h, frame_w = (
            frame.shape[:2]
        )

        detections = detector.detect(
            frame
        )

        # Existing heuristic had no useful anchor detections,
        # so use complete-card detections for this prototype.
        detections = [
            detection
            for detection in detections
            if detection.kind
            == "card"
        ]

        observations = []

        for detection in detections:

            x = detection.x
            y = detection.y
            w = detection.w
            h = detection.h

            crop = frame[
                y:y + h,
                x:x + w,
            ]

            if crop.size == 0:
                continue

            rank, confidence = (
                classify_rank(
                    crop,
                    templates,
                )
            )

            cx = (
                x + w / 2.0
            ) / frame_w

            cy = (
                y + h / 2.0
            ) / frame_h

            observations.append(
                {
                    "x": cx,
                    "y": cy,
                    "rank": rank,
                    "confidence":
                        confidence,
                }
            )

        matched_tracks = set()

        for observation in observations:

            best_track = None
            best_distance = None

            for track_index, track in enumerate(
                tracks
            ):

                distance = (
                    normalized_distance(
                        observation["x"],
                        observation["y"],
                        track.x,
                        track.y,
                    )
                )

                if (
                    distance
                    <= POSITION_MATCH_DISTANCE
                    and (
                        best_distance is None
                        or distance
                        < best_distance
                    )
                ):
                    best_distance = (
                        distance
                    )

                    best_track = (
                        track_index
                    )

            # -----------------------------------------------
            # Existing object
            # -----------------------------------------------

            if best_track is not None:

                track = tracks[
                    best_track
                ]

                matched_tracks.add(
                    best_track
                )

                track.x = (
                    observation["x"]
                )

                track.y = (
                    observation["y"]
                )

                track.misses = 0

                if (
                    observation["rank"]
                    == track.rank
                ):
                    track.confirmations += 1

                    track.confidence = max(
                        track.confidence,
                        observation[
                            "confidence"
                        ],
                    )

                else:
                    # Require the new prediction to stabilize.
                    track.rank = (
                        observation[
                            "rank"
                        ]
                    )

                    track.confidence = (
                        observation[
                            "confidence"
                        ]
                    )

                    track.confirmations = 1

                if (
                    not track.emitted
                    and track.confirmations
                    >= REQUIRED_CONFIRMATIONS
                ):
                    events.append(
                        {
                            "rank":
                                track.rank,

                            "confidence":
                                track.confidence,

                            "frame":
                                frame_index,
                        }
                    )

                    track.emitted = True

            # -----------------------------------------------
            # New object
            # -----------------------------------------------

            else:

                tracks.append(
                    Track(
                        observation[
                            "x"
                        ],
                        observation[
                            "y"
                        ],
                        observation[
                            "rank"
                        ],
                        observation[
                            "confidence"
                        ],
                    )
                )

        # -----------------------------------------------
        # Age unmatched tracks
        # -----------------------------------------------

        for track_index, track in enumerate(
            tracks
        ):
            if (
                track_index
                not in matched_tracks
            ):
                track.misses += 1

        tracks = [
            track
            for track in tracks
            if track.misses
            <= TRACK_MAX_MISSES
        ]

    events.sort(
        key=lambda event:
            event["frame"]
    )

    cards = [
        event["rank"]
        for event in events
    ]

    print(
        f"[legacy recognizer] "
        f"Events: {cards}"
    )

    return cards