from pathlib import Path
from collections import defaultdict
import csv

import cv2
import numpy as np
import torch
import torch.nn as nn

from ultralytics import YOLO

from test_rank_detector_unseen import detect_full_frame
from card_tracking import CardIdentityMatcher


# =========================================================
# CONFIG
# =========================================================

ROOT = Path(__file__).resolve().parent

CAPTURES_DIR = ROOT / "captures"

DETECTOR_PATH = (
    ROOT
    / "rank_detector_runs"
    / "rank_detector_v2"
    / "weights"
    / "best.pt"
)

CLASSIFIER_PATH = (
    ROOT
    / "rank_classifier_v2"
    / "best_rank_classifier.pt"
)

OUTPUT_DIR = (
    ROOT
    / "temporal_rank_test"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TRACK_CSV = (
    OUTPUT_DIR
    / "tracks.csv"
)

OBSERVATION_CSV = (
    OUTPUT_DIR
    / "observations.csv"
)

SUMMARY_IMAGE = (
    OUTPUT_DIR
    / "track_summary.jpg"
)


RANKS = [
    "A", "2", "3", "4", "5", "6",
    "7", "8", "9", "10", "J", "Q", "K"
]

IMAGE_SIZE = 64


# Process every fifth recorded frame.
EVERY_N_FRAMES = 5


# Detector crop padding. This matches how the
# rank-value candidate dataset was created.
PAD_FRAC = 0.20


# Tracking parameters.
MAX_GAP_SAMPLES = 3

MIN_TRACK_OBSERVATIONS = 3


# Acceptance thresholds.
#
# These are deliberately conservative.
MIN_FINAL_PROBABILITY = 0.42
MIN_FINAL_MARGIN = 0.10


# =========================================================
# DEVICE
# =========================================================

def choose_device():
    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


DEVICE = choose_device()


# =========================================================
# CNN
# =========================================================

class RankCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(
                (4, 4)
            ),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                128 * 4 * 4,
                256,
            ),

            nn.ReLU(),

            nn.Dropout(0.35),

            nn.Linear(
                256,
                len(RANKS),
            ),
        )

    def forward(self, x):
        x = self.features(x)

        return self.classifier(x)


# =========================================================
# CLASSIFIER PREPROCESSING
#
# Mirrors the current classifier training preprocessing.
# =========================================================

def preprocess_crop(image):
    if image is None or image.size == 0:
        return None

    h, w = image.shape[:2]

    trim_x = int(
        w * 0.11
    )

    trim_y = int(
        h * 0.11
    )

    if (
        w - 2 * trim_x > 2
        and h - 2 * trim_y > 2
    ):
        image = image[
            trim_y:h - trim_y,
            trim_x:w - trim_x,
        ]

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    gray = cv2.normalize(
        gray,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    )

    h, w = gray.shape[:2]

    available = (
        IMAGE_SIZE - 8
    )

    scale = min(
        available / max(w, 1),
        available / max(h, 1),
    )

    new_w = max(
        1,
        round(w * scale),
    )

    new_h = max(
        1,
        round(h * scale),
    )

    gray = cv2.resize(
        gray,
        (
            new_w,
            new_h,
        ),
        interpolation=cv2.INTER_CUBIC,
    )

    canvas = np.full(
        (
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        255,
        dtype=np.uint8,
    )

    x = (
        IMAGE_SIZE - new_w
    ) // 2

    y = (
        IMAGE_SIZE - new_h
    ) // 2

    canvas[
        y:y + new_h,
        x:x + new_w
    ] = gray

    canvas = (
        255.0
        - canvas.astype(
            np.float32
        )
    ) / 255.0

    tensor = torch.from_numpy(
        canvas
    ).unsqueeze(0).unsqueeze(0)

    return tensor


# =========================================================
# LOAD CLASSIFIER
# =========================================================

def load_classifier():
    checkpoint = torch.load(
        CLASSIFIER_PATH,
        map_location=DEVICE,
    )

    model = RankCNN().to(
        DEVICE
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    return model


# =========================================================
# DETECTION ADAPTER
#
# Handles several reasonable formats so we do not have
# to change your detector code.
# =========================================================

def parse_detection(det):
    if isinstance(det, dict):

        if all(
            key in det
            for key in [
                "x1",
                "y1",
                "x2",
                "y2",
            ]
        ):
            box = [
                det["x1"],
                det["y1"],
                det["x2"],
                det["y2"],
            ]

        elif "xyxy" in det:
            box = det[
                "xyxy"
            ]

        elif "box" in det:
            box = det[
                "box"
            ]

        elif "bbox" in det:
            box = det[
                "bbox"
            ]

        else:
            raise ValueError(
                f"Unknown detection format: {det}"
            )

        confidence = (
            det.get(
                "confidence",
                det.get(
                    "conf",
                    det.get(
                        "score",
                        1.0,
                    ),
                ),
            )
        )

    elif (
        isinstance(
            det,
            (list, tuple),
        )
        and len(det) >= 4
    ):
        box = det[:4]

        confidence = (
            det[4]
            if len(det) >= 5
            else 1.0
        )

    else:
        raise ValueError(
            f"Unknown detection type: {type(det)}"
        )

    box = [
        float(value)
        for value in box
    ]

    return (
        box,
        float(confidence),
    )


# =========================================================
# BOX HELPERS
# =========================================================

def box_center(box):
    x1, y1, x2, y2 = box

    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
    )


def box_size(box):
    x1, y1, x2, y2 = box

    return (
        max(
            x2 - x1,
            1.0,
        ),
        max(
            y2 - y1,
            1.0,
        ),
    )


def center_distance_score(
    box_a,
    box_b,
):
    ax, ay = box_center(
        box_a
    )

    bx, by = box_center(
        box_b
    )

    aw, ah = box_size(
        box_a
    )

    bw, bh = box_size(
        box_b
    )

    scale = max(
        (
            aw + bw
        ) / 2.0,
        (
            ah + bh
        ) / 2.0,
        1.0,
    )

    distance = (
        (
            ax - bx
        ) ** 2
        + (
            ay - by
        ) ** 2
    ) ** 0.5

    return (
        distance / scale
    )


# =========================================================
# CROP
# =========================================================

def crop_detection(
    image,
    box,
):
    x1, y1, x2, y2 = box

    width = (
        x2 - x1
    )

    height = (
        y2 - y1
    )

    pad_x = (
        width * PAD_FRAC
    )

    pad_y = (
        height * PAD_FRAC
    )

    h, w = image.shape[:2]

    x1 = max(
        0,
        int(
            x1 - pad_x
        ),
    )

    y1 = max(
        0,
        int(
            y1 - pad_y
        ),
    )

    x2 = min(
        w,
        int(
            x2 + pad_x
        ),
    )

    y2 = min(
        h,
        int(
            y2 + pad_y
        ),
    )

    if (
        x2 <= x1
        or y2 <= y1
    ):
        return None

    return image[
        y1:y2,
        x1:x2,
    ].copy()


# =========================================================
# CLASSIFY ONE OBSERVATION
# =========================================================

def classify_crop(
    model,
    crop,
):
    tensor = preprocess_crop(
        crop
    )

    if tensor is None:
        return None

    tensor = tensor.to(
        DEVICE
    )

    with torch.no_grad():

        logits = model(
            tensor
        )

        probabilities = (
            torch.softmax(
                logits,
                dim=1,
            )[0]
            .cpu()
            .numpy()
        )

    index = int(
        np.argmax(
            probabilities
        )
    )

    return {
        "predicted":
            RANKS[index],

        "confidence":
            float(
                probabilities[index]
            ),

        "probabilities":
            probabilities,
    }


# =========================================================
# TRACK
# =========================================================

class CardTrack:
    def __init__(
        self,
        track_id,
        box,
        sample_index,
    ):
        self.track_id = track_id

        self.last_box = box

        self.last_sample_index = (
            sample_index
        )

        self.observations = []

    def add(
        self,
        box,
        sample_index,
        frame_name,
        detector_confidence,
        classification,
        crop,
    ):
        self.last_box = box

        self.last_sample_index = (
            sample_index
        )

        self.observations.append(
            {
                "sample_index":
                    sample_index,

                "frame_name":
                    frame_name,

                "box":
                    box,

                "detector_confidence":
                    detector_confidence,

                "prediction":
                    classification[
                        "predicted"
                    ],

                "classifier_confidence":
                    classification[
                        "confidence"
                    ],

                "probabilities":
                    classification[
                        "probabilities"
                    ],

                "crop":
                    crop,
            }
        )

    def final_result(self):
        if not self.observations:
            return None

        probability_sum = (
            np.zeros(
                len(RANKS),
                dtype=np.float64,
            )
        )

        weight_sum = 0.0

        for observation in (
            self.observations
        ):

            # YOLO confidence determines how much
            # influence this observation receives.
            weight = max(
                observation[
                    "detector_confidence"
                ],
                0.05,
            )

            probability_sum += (
                observation[
                    "probabilities"
                ]
                * weight
            )

            weight_sum += weight

        average = (
            probability_sum
            / max(
                weight_sum,
                1e-8,
            )
        )

        order = np.argsort(
            average
        )[::-1]

        best_index = int(
            order[0]
        )

        second_index = int(
            order[1]
        )

        best_probability = float(
            average[
                best_index
            ]
        )

        second_probability = float(
            average[
                second_index
            ]
        )

        margin = (
            best_probability
            - second_probability
        )

        raw_rank = (
            RANKS[
                best_index
            ]
        )

        accepted = (
            len(
                self.observations
            )
            >= MIN_TRACK_OBSERVATIONS
            and best_probability
            >= MIN_FINAL_PROBABILITY
            and margin
            >= MIN_FINAL_MARGIN
        )

        final_rank = (
            raw_rank
            if accepted
            else "UNKNOWN"
        )

        return {
            "raw_rank":
                raw_rank,

            "final_rank":
                final_rank,

            "probability":
                best_probability,

            "margin":
                margin,

            "observations":
                len(
                    self.observations
                ),
        }


# =========================================================
# SUMMARY IMAGE
# =========================================================

def make_track_summary(tracks):
    """Show temporal evidence, including short tracks, rather than one crop."""
    tiles = []
    for track in tracks:
        result = track.final_result()
        if result is None:
            continue
        tile = np.full((190, 420, 3), 25, dtype=np.uint8)
        cv2.putText(tile, f"Track {track.track_id}: {result['final_rank']} "
                    f"({result['observations']} observations)", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)
        observations = track.observations
        selected = [observations[0], observations[len(observations) // 2], observations[-1]]
        for column, observation in enumerate(selected):
            crop = observation["crop"]
            x = column * 140
            if crop is None or crop.size == 0:
                continue
            h, w = crop.shape[:2]
            scale = min(120 / w, 100 / h)
            preview = cv2.resize(crop, (max(1, round(w * scale)), max(1, round(h * scale))),
                                 interpolation=cv2.INTER_NEAREST)
            tile[50:50 + preview.shape[0], x + 8:x + 8 + preview.shape[1]] = preview
            cv2.putText(tile, f"sample {observation['sample_index']}", (x + 8, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)
            cv2.putText(tile, f"pred: {observation['prediction']}", (x + 8, 175),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
        tiles.append(tile)
    if not tiles:
        return
    columns = 3
    while len(tiles) % columns:
        tiles.append(np.full_like(tiles[0], 25))
    image = np.vstack([np.hstack(tiles[i:i + columns])
                       for i in range(0, len(tiles), columns)])
    if not cv2.imwrite(str(SUMMARY_IMAGE), image):
        raise RuntimeError(f"Could not save review image: {SUMMARY_IMAGE}")


def main():
    frame_paths = sorted(
        CAPTURES_DIR.glob(
            "*.png"
        )
    )

    if not frame_paths:
        raise FileNotFoundError(f"No PNG frames found in {CAPTURES_DIR}")

    sampled_paths = (
        frame_paths[
            ::EVERY_N_FRAMES
        ]
    )

    print()
    print(
        f"Frames available: "
        f"{len(frame_paths)}"
    )

    print(
        f"Frames sampled: "
        f"{len(sampled_paths)}"
    )

    print(
        f"Device: {DEVICE}"
    )

    print()
    print(
        "Loading detector..."
    )

    detector = YOLO(
        str(
            DETECTOR_PATH
        )
    )

    print(
        "Loading rank classifier..."
    )

    classifier = (
        load_classifier()
    )

    tracks = []
    matcher = CardIdentityMatcher(
        max_gap_samples=MAX_GAP_SAMPLES,
    )

    next_track_id = 1

    observation_rows = []

    print()
    print(
        "Processing sequence..."
    )
    print()

    for sample_index, frame_path in enumerate(
        sampled_paths
    ):

        image = cv2.imread(
            str(
                frame_path
            )
        )

        if image is None:
            continue

        (
            _raw,
            _stage1,
            _clustered,
            final_detections,
            _duplicate_groups,
            _same_tile_removed,
            _final_overlap_removed,
        ) = detect_full_frame(
            detector,
            image,
        )

        frame_detections = []

        for det in final_detections:

            box, detector_confidence = (
                parse_detection(
                    det
                )
            )

            crop = crop_detection(
                image,
                box,
            )

            if crop is None:
                continue

            classification = (
                classify_crop(
                    classifier,
                    crop,
                )
            )

            if classification is None:
                continue

            frame_detections.append(
                {
                    "box":
                        box,

                    "detector_confidence":
                        detector_confidence,

                    "classification":
                        classification,

                    "crop":
                        crop,
                }
            )

        matches = (
            matcher.match(
                frame_detections,
                tracks,
                sample_index,
            )
        )

        for detection_index, detection in enumerate(
            frame_detections
        ):

            if (
                detection_index
                in matches
            ):
                track = (
                    matches[
                        detection_index
                    ]
                )

            else:
                track = CardTrack(
                    next_track_id,
                    detection[
                        "box"
                    ],
                    sample_index,
                )

                tracks.append(
                    track
                )

                next_track_id += 1

            track.add(
                box=detection[
                    "box"
                ],

                sample_index=
                    sample_index,

                frame_name=
                    frame_path.name,

                detector_confidence=
                    detection[
                        "detector_confidence"
                    ],

                classification=
                    detection[
                        "classification"
                    ],

                crop=
                    detection[
                        "crop"
                    ],
            )

            observation_rows.append(
                {
                    "track_id":
                        track.track_id,

                    "frame":
                        frame_path.name,

                    "prediction":
                        detection[
                            "classification"
                        ][
                            "predicted"
                        ],

                    "classifier_confidence":
                        detection[
                            "classification"
                        ][
                            "confidence"
                        ],

                    "detector_confidence":
                        detection[
                            "detector_confidence"
                        ],
                }
            )

        if (
            sample_index + 1
        ) % 25 == 0:

            print(
                f"{sample_index + 1}/"
                f"{len(sampled_paths)} "
                f"| tracks={len(tracks)}"
            )

    # =====================================================
    # RESULTS
    # =====================================================

    valid_tracks = []

    for track in tracks:
        result = (
            track.final_result()
        )

        if result is None:
            continue

        if (
            result[
                "observations"
            ]
            < MIN_TRACK_OBSERVATIONS
        ):
            continue

        valid_tracks.append(
            (
                track,
                result,
            )
        )

    accepted = sum(
        1
        for _, result
        in valid_tracks
        if (
            result[
                "final_rank"
            ]
            != "UNKNOWN"
        )
    )

    unknown = (
        len(valid_tracks)
        - accepted
    )

    print()
    print(
        "=============================="
    )

    print(
        "TEMPORAL TRACK RESULTS"
    )

    print(
        "=============================="
    )

    print()
    print(
        f"All tracks created: "
        f"{len(tracks)}"
    )

    print(
        f"Tracks with >= "
        f"{MIN_TRACK_OBSERVATIONS} "
        f"observations: "
        f"{len(valid_tracks)}"
    )

    print(
        f"Accepted rank: "
        f"{accepted}"
    )

    print(
        f"UNKNOWN: "
        f"{unknown}"
    )

    print()

    rows = []

    for track, result in (
        valid_tracks
    ):

        prediction_counts = (
            defaultdict(int)
        )

        for observation in (
            track.observations
        ):
            prediction_counts[
                observation[
                    "prediction"
                ]
            ] += 1

        votes = sorted(
            prediction_counts.items(),
            key=lambda item:
                item[1],
            reverse=True,
        )

        vote_text = ", ".join(
            f"{rank}:{count}"
            for rank, count
            in votes
        )

        print(
            f"Track "
            f"{track.track_id:>3} | "
            f"obs={result['observations']:>2} | "
            f"final={result['final_rank']:>7} | "
            f"raw={result['raw_rank']:>2} | "
            f"p={result['probability']:.3f} | "
            f"margin={result['margin']:.3f} | "
            f"{vote_text}"
        )

        rows.append(
            {
                "track_id":
                    track.track_id,

                "observations":
                    result[
                        "observations"
                    ],

                "final_rank":
                    result[
                        "final_rank"
                    ],

                "raw_rank":
                    result[
                        "raw_rank"
                    ],

                "probability":
                    result[
                        "probability"
                    ],

                "margin":
                    result[
                        "margin"
                    ],

                "votes":
                    vote_text,
            }
        )

    with TRACK_CSV.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "track_id",
                "observations",
                "final_rank",
                "raw_rank",
                "probability",
                "margin",
                "votes",
            ],
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    with OBSERVATION_CSV.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "track_id",
                "frame",
                "prediction",
                "classifier_confidence",
                "detector_confidence",
            ],
        )

        writer.writeheader()
        writer.writerows(
            observation_rows
        )

    # Retain short tracks explicitly; they may contain real briefly visible cards.
    all_track_path = OUTPUT_DIR / "all_tracks.csv"
    with all_track_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "track_id", "observations", "status", "final_rank", "first_frame", "last_frame",
        ])
        writer.writeheader()
        for track in tracks:
            result = track.final_result()
            if result is None:
                continue
            status = ("insufficient_observations" if result["observations"] < MIN_TRACK_OBSERVATIONS
                      else "unknown" if result["final_rank"] == "UNKNOWN" else "accepted")
            writer.writerow(dict(track_id=track.track_id,
                                 observations=result["observations"], status=status,
                                 final_rank=result["final_rank"],
                                 first_frame=track.observations[0]["frame_name"],
                                 last_frame=track.observations[-1]["frame_name"]))
    make_track_summary(tracks)
    print(f"Saved all tracks, including short tracks: {all_track_path}")

    print()
    print(
        f"Saved tracks: "
        f"{TRACK_CSV}"
    )

    print(
        f"Saved observations: "
        f"{OBSERVATION_CSV}"
    )

    print(
        f"Saved review image: "
        f"{SUMMARY_IMAGE}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Replay rank detection and appearance-based card tracking.")
    parser.add_argument("--input", type=Path, default=CAPTURES_DIR)
    parser.add_argument("--detector", type=Path, default=DETECTOR_PATH)
    parser.add_argument("--classifier", type=Path, default=CLASSIFIER_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--every", type=int, default=EVERY_N_FRAMES)
    parser.add_argument("--threads", type=int, default=None, help="Optional CPU thread limit")
    args = parser.parse_args()
    if args.every < 1 or (args.threads is not None and args.threads < 1):
        parser.error("--every and --threads must be positive")
    CAPTURES_DIR, DETECTOR_PATH, CLASSIFIER_PATH = args.input, args.detector, args.classifier
    OUTPUT_DIR = args.output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TRACK_CSV = OUTPUT_DIR / "tracks.csv"
    OBSERVATION_CSV = OUTPUT_DIR / "observations.csv"
    SUMMARY_IMAGE = OUTPUT_DIR / "track_summary.jpg"
    EVERY_N_FRAMES = args.every
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    main()
