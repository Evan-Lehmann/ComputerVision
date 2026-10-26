from pathlib import Path
import csv
import random
import re
from collections import Counter, defaultdict

import cv2
import numpy as np


# =========================================================
# CONFIG
# =========================================================

ROOT = Path(__file__).resolve().parent

ORIGINAL_DIR = ROOT / "rank_value_dataset"
ORIGINAL_CSV = ORIGINAL_DIR / "labels.csv"

CANDIDATE_DIR = ROOT / "rank_value_candidates"
CANDIDATE_CSV = CANDIDATE_DIR / "candidate_labels.csv"

OUTPUT_DIR = ROOT / "rank_template_classifier"
OUTPUT_DIR.mkdir(exist_ok=True)

PREDICTIONS_CSV = OUTPUT_DIR / "holdout_predictions.csv"
CONFUSION_CSV = OUTPUT_DIR / "confusion_matrix.csv"

RANKS = [
    "A", "2", "3", "4", "5", "6",
    "7", "8", "9", "10", "J", "Q", "K"
]

IMAGE_SIZE = 64
RANDOM_SEED = 42

# How many best matching templates from each class
# contribute to that class's final score.
TOP_K_PER_CLASS = 3

# Allow a few pixels of alignment correction.
SHIFT_RANGE = 3


# =========================================================
# DATA
# =========================================================

def source_group_from_name(name):
    match = re.search(
        r"(frame_(?:\d{8}_\d{6}_\d{6}_)?\d{6})",
        name,
    )

    if match:
        return match.group(1)

    return name


def load_label_file(image_dir, csv_path, source_name):
    samples = []

    if not csv_path.exists():
        return samples

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            rank = row["rank"].strip()

            if rank == "UNKNOWN":
                continue

            if rank not in RANKS:
                continue

            path = image_dir / row["crop"]

            if not path.exists():
                continue

            samples.append(
                {
                    "path": path,
                    "rank": rank,
                    "source": source_name,
                    "group": source_group_from_name(
                        row["crop"]
                    ),
                }
            )

    return samples


def load_all_samples():
    samples = []

    samples.extend(
        load_label_file(
            ORIGINAL_DIR,
            ORIGINAL_CSV,
            "original",
        )
    )

    samples.extend(
        load_label_file(
            CANDIDATE_DIR,
            CANDIDATE_CSV,
            "candidate",
        )
    )

    return samples


# =========================================================
# GROUPED TRAIN / HOLDOUT SPLIT
# =========================================================

def grouped_split(samples):
    rng = random.Random(RANDOM_SEED)

    by_rank = defaultdict(list)

    for sample in samples:
        by_rank[sample["rank"]].append(sample)

    train = []
    holdout = []
    unvalidated = []

    for rank in RANKS:
        rank_samples = by_rank.get(rank, [])

        groups = defaultdict(list)

        for sample in rank_samples:
            groups[sample["group"]].append(sample)

        group_names = list(groups.keys())

        rng.shuffle(group_names)

        sample_count = len(rank_samples)
        group_count = len(group_names)

        if (
            sample_count >= 8
            and group_count >= 3
        ):
            holdout_group_count = max(
                1,
                round(group_count * 0.20),
            )

        elif (
            sample_count >= 5
            and group_count >= 2
        ):
            holdout_group_count = 1

        else:
            holdout_group_count = 0

        if holdout_group_count == 0:
            train.extend(rank_samples)
            unvalidated.append(rank)
            continue

        holdout_groups = set(
            group_names[:holdout_group_count]
        )

        for sample in rank_samples:
            if sample["group"] in holdout_groups:
                holdout.append(sample)
            else:
                train.append(sample)

    return train, holdout, unvalidated


# =========================================================
# IMAGE PREPROCESSING
# =========================================================

def resize_with_padding(gray):
    h, w = gray.shape[:2]

    if h <= 0 or w <= 0:
        return None

    available = IMAGE_SIZE - 8

    scale = min(
        available / w,
        available / h,
    )

    new_w = max(
        1,
        round(w * scale),
    )

    new_h = max(
        1,
        round(h * scale),
    )

    resized = cv2.resize(
        gray,
        (new_w, new_h),
        interpolation=cv2.INTER_CUBIC,
    )

    canvas = np.full(
        (IMAGE_SIZE, IMAGE_SIZE),
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
    ] = resized

    return canvas


def preprocess(path):
    image = cv2.imread(str(path))

    if image is None:
        raise RuntimeError(
            f"Could not read {path}"
        )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    gray = resize_with_padding(gray)

    if gray is None:
        raise RuntimeError(
            f"Bad image {path}"
        )

    # Normalize contrast while preserving the
    # actual shape/background information.
    gray = cv2.normalize(
        gray,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    )

    # Dark rank becomes large positive value.
    intensity = (
        255.0
        - gray.astype(np.float32)
    ) / 255.0

    # Edge representation provides a second,
    # shape-focused comparison.
    edges = cv2.Canny(
        gray,
        50,
        150,
    ).astype(np.float32) / 255.0

    return intensity, edges


# =========================================================
# SIMILARITY
# =========================================================

def cosine_similarity(a, b):
    a = a.reshape(-1)
    b = b.reshape(-1)

    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominator < 1e-8:
        return 0.0

    return float(
        np.dot(a, b)
        / denominator
    )


def shift_image(image, dx, dy):
    matrix = np.float32(
        [
            [1, 0, dx],
            [0, 1, dy],
        ]
    )

    return cv2.warpAffine(
        image,
        matrix,
        (
            image.shape[1],
            image.shape[0],
        ),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def compare_pair(query, template):
    query_intensity, query_edges = query
    template_intensity, template_edges = template

    best_score = -1.0

    for dy in range(
        -SHIFT_RANGE,
        SHIFT_RANGE + 1,
    ):
        for dx in range(
            -SHIFT_RANGE,
            SHIFT_RANGE + 1,
        ):
            shifted_intensity = shift_image(
                query_intensity,
                dx,
                dy,
            )

            shifted_edges = shift_image(
                query_edges,
                dx,
                dy,
            )

            intensity_score = cosine_similarity(
                shifted_intensity,
                template_intensity,
            )

            edge_score = cosine_similarity(
                shifted_edges,
                template_edges,
            )

            score = (
                0.70 * intensity_score
                + 0.30 * edge_score
            )

            if score > best_score:
                best_score = score

    return best_score


# =========================================================
# TEMPLATE CLASSIFIER
# =========================================================

def build_templates(train_samples):
    templates = defaultdict(list)

    print()
    print("Preparing templates...")

    for i, sample in enumerate(
        train_samples,
        start=1,
    ):
        processed = preprocess(
            sample["path"]
        )

        templates[
            sample["rank"]
        ].append(
            {
                "image": processed,
                "path": sample["path"],
            }
        )

    return templates


def classify(query_path, templates):
    query = preprocess(query_path)

    class_scores = {}

    for rank in RANKS:
        rank_templates = templates.get(
            rank,
            [],
        )

        if not rank_templates:
            continue

        similarities = []

        for template in rank_templates:
            similarity = compare_pair(
                query,
                template["image"],
            )

            similarities.append(
                similarity
            )

        similarities.sort(
            reverse=True
        )

        top_scores = similarities[
            :min(
                TOP_K_PER_CLASS,
                len(similarities),
            )
        ]

        # Each class gets equal voting power regardless
        # of how many training examples it has.
        class_scores[rank] = float(
            np.mean(top_scores)
        )

    predicted = max(
        class_scores,
        key=class_scores.get,
    )

    sorted_scores = sorted(
        class_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    best_score = sorted_scores[0][1]

    second_score = (
        sorted_scores[1][1]
        if len(sorted_scores) > 1
        else 0.0
    )

    margin = (
        best_score - second_score
    )

    return (
        predicted,
        best_score,
        margin,
        class_scores,
    )


# =========================================================
# MAIN
# =========================================================

def main():
    samples = load_all_samples()

    print()
    print(
        f"Usable labeled crops: "
        f"{len(samples)}"
    )

    counts = Counter(
        sample["rank"]
        for sample in samples
    )

    print()
    print("Class distribution:")

    for rank in RANKS:
        print(
            f"{rank:>2}: "
            f"{counts.get(rank, 0)}"
        )

    (
        train,
        holdout,
        unvalidated,
    ) = grouped_split(samples)

    print()
    print(
        f"Training templates: "
        f"{len(train)}"
    )

    print(
        f"Holdout samples: "
        f"{len(holdout)}"
    )

    if unvalidated:
        print()
        print(
            "Not independently validated:"
        )

        print(
            ", ".join(unvalidated)
        )

    templates = build_templates(
        train
    )

    results = []

    print()
    print(
        "Testing holdout..."
    )
    print()

    correct = 0

    for index, sample in enumerate(
        holdout,
        start=1,
    ):
        (
            predicted,
            score,
            margin,
            class_scores,
        ) = classify(
            sample["path"],
            templates,
        )

        is_correct = (
            predicted
            == sample["rank"]
        )

        if is_correct:
            correct += 1

        results.append(
            {
                "path": str(
                    sample["path"]
                ),
                "actual": sample["rank"],
                "predicted": predicted,
                "score": score,
                "margin": margin,
                "correct": int(
                    is_correct
                ),
            }
        )

        print(
            f"{index:>2}/{len(holdout)} "
            f"actual={sample['rank']:>2} "
            f"pred={predicted:>2} "
            f"score={score:.3f} "
            f"margin={margin:.3f}"
        )

    accuracy = (
        correct
        / len(holdout)
        if holdout
        else 0.0
    )

    # =====================================================
    # PER-CLASS RESULTS
    # =====================================================

    class_total = Counter()
    class_correct = Counter()

    for result in results:
        actual = result["actual"]

        class_total[
            actual
        ] += 1

        if result["correct"]:
            class_correct[
                actual
            ] += 1

    print()
    print(
        "=============================="
    )

    print(
        "TEMPLATE CLASSIFIER RESULTS"
    )

    print(
        "=============================="
    )

    print()
    print(
        f"Holdout accuracy: "
        f"{accuracy:.3f}"
    )

    print()
    print(
        "Per-class holdout accuracy:"
    )

    for rank in RANKS:
        total = class_total.get(
            rank,
            0,
        )

        if total == 0:
            print(
                f"{rank:>2}: "
                f"not validated"
            )

            continue

        hits = class_correct.get(
            rank,
            0,
        )

        print(
            f"{rank:>2}: "
            f"{hits}/{total} "
            f"({hits / total:.3f})"
        )

    # =====================================================
    # MISTAKES
    # =====================================================

    print()
    print("Mistakes:")

    mistakes = [
        result
        for result in results
        if not result["correct"]
    ]

    if not mistakes:
        print("None.")

    else:
        for result in mistakes:
            print(
                f"{Path(result['path']).name}: "
                f"actual={result['actual']} "
                f"predicted={result['predicted']} "
                f"score={result['score']:.3f} "
                f"margin={result['margin']:.3f}"
            )

    # =====================================================
    # SAVE RESULTS
    # =====================================================

    with PREDICTIONS_CSV.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "path",
                "actual",
                "predicted",
                "score",
                "margin",
                "correct",
            ],
        )

        writer.writeheader()
        writer.writerows(results)

    matrix = np.zeros(
        (
            len(RANKS),
            len(RANKS),
        ),
        dtype=int,
    )

    for result in results:
        actual_index = RANKS.index(
            result["actual"]
        )

        predicted_index = RANKS.index(
            result["predicted"]
        )

        matrix[
            actual_index,
            predicted_index
        ] += 1

    with CONFUSION_CSV.open(
        "w",
        newline="",
    ) as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "actual/predicted",
                *RANKS,
            ]
        )

        for i, rank in enumerate(
            RANKS
        ):
            writer.writerow(
                [
                    rank,
                    *matrix[i].tolist(),
                ]
            )

    print()
    print(
        f"Saved predictions: "
        f"{PREDICTIONS_CSV}"
    )

    print(
        f"Saved confusion matrix: "
        f"{CONFUSION_CSV}"
    )


if __name__ == "__main__":
    main()