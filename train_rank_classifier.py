from pathlib import Path
import csv
import random

import cv2
import joblib
import numpy as np

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix


ROOT = Path(__file__).resolve().parent

ORIGINAL_DIR = ROOT / "rank_value_dataset"
ORIGINAL_CSV = ORIGINAL_DIR / "labels.csv"

CANDIDATE_DIR = ROOT / "rank_value_candidates"
CANDIDATE_CSV = CANDIDATE_DIR / "candidate_labels.csv"

OUTPUT_DIR = ROOT / "rank_classifier"
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "rank_classifier.joblib"

RANDOM_SEED = 42

RANKS = [
    "A", "2", "3", "4", "5", "6",
    "7", "8", "9", "10", "J", "Q", "K"
]


# ---------------------------------------------------------
# IMAGE NORMALIZATION
# ---------------------------------------------------------

def normalize_rank_crop(image):
    """
    Convert a rank crop into a consistent 48x64
    grayscale representation.
    """

    if image is None or image.size == 0:
        return None

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    # Local contrast normalization.
    gray = cv2.equalizeHist(gray)

    target_w = 48
    target_h = 64

    h, w = gray.shape[:2]

    scale = min(
        target_w / max(w, 1),
        target_h / max(h, 1),
    )

    new_w = max(
        1,
        int(w * scale),
    )

    new_h = max(
        1,
        int(h * scale),
    )

    resized = cv2.resize(
        gray,
        (new_w, new_h),
        interpolation=cv2.INTER_CUBIC,
    )

    canvas = np.zeros(
        (target_h, target_w),
        dtype=np.uint8,
    )

    x = (
        target_w - new_w
    ) // 2

    y = (
        target_h - new_h
    ) // 2

    canvas[
        y:y + new_h,
        x:x + new_w
    ] = resized

    return canvas


# ---------------------------------------------------------
# HOG FEATURE EXTRACTION
# ---------------------------------------------------------

HOG = cv2.HOGDescriptor(
    _winSize=(48, 64),
    _blockSize=(16, 16),
    _blockStride=(8, 8),
    _cellSize=(8, 8),
    _nbins=9,
)


def extract_features(image):
    normalized = normalize_rank_crop(
        image
    )

    if normalized is None:
        return None

    features = HOG.compute(
        normalized
    )

    return features.flatten()


# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------

def load_csv_dataset(
    image_dir,
    csv_path,
    source_name,
):
    samples = []

    if not csv_path.exists():
        return samples

    with csv_path.open(
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            rank = row["rank"].strip()

            if rank == "UNKNOWN":
                continue

            image_path = (
                image_dir
                / row["crop"]
            )

            if not image_path.exists():
                print(
                    f"Missing image: {image_path}"
                )
                continue

            image = cv2.imread(
                str(image_path)
            )

            features = extract_features(
                image
            )

            if features is None:
                continue

            samples.append(
                {
                    "features": features,
                    "rank": rank,
                    "source": source_name,
                    "name": row["crop"],
                }
            )

    return samples


# ---------------------------------------------------------
# TRAIN / TEST SPLIT
# ---------------------------------------------------------

def make_split(samples):
    """
    Hold out examples only for classes with enough
    samples to support a meaningful holdout.

    Tiny classes remain training-only for now and are
    explicitly reported as unvalidated.
    """

    rng = random.Random(
        RANDOM_SEED
    )

    by_rank = {}

    for sample in samples:
        by_rank.setdefault(
            sample["rank"],
            [],
        ).append(sample)

    train = []
    test = []
    unvalidated = []

    for rank in RANKS:
        group = by_rank.get(
            rank,
            []
        )

        rng.shuffle(group)

        count = len(group)

        if count >= 8:
            test_count = max(
                2,
                round(count * 0.25),
            )

        elif count >= 5:
            test_count = 1

        else:
            test_count = 0

        if test_count == 0:
            unvalidated.append(
                rank
            )

            train.extend(
                group
            )

        else:
            test.extend(
                group[:test_count]
            )

            train.extend(
                group[test_count:]
            )

    return (
        train,
        test,
        unvalidated,
    )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    samples = []

    samples.extend(
        load_csv_dataset(
            ORIGINAL_DIR,
            ORIGINAL_CSV,
            "original",
        )
    )

    samples.extend(
        load_csv_dataset(
            CANDIDATE_DIR,
            CANDIDATE_CSV,
            "candidate",
        )
    )

    print()
    print(
        f"Usable labeled crops: "
        f"{len(samples)}"
    )

    counts = {}

    for sample in samples:
        counts[
            sample["rank"]
        ] = (
            counts.get(
                sample["rank"],
                0,
            )
            + 1
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
        test,
        unvalidated,
    ) = make_split(
        samples
    )

    print()
    print(
        f"Training samples: "
        f"{len(train)}"
    )

    print(
        f"Holdout samples: "
        f"{len(test)}"
    )

    print()

    if unvalidated:
        print(
            "Not independently validated "
            "because of low sample count:"
        )

        print(
            ", ".join(
                unvalidated
            )
        )

        print()

    X_train = np.array(
        [
            sample["features"]
            for sample in train
        ]
    )

    y_train = np.array(
        [
            sample["rank"]
            for sample in train
        ]
    )

    model = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                SVC(
                    kernel="rbf",
                    C=5.0,
                    gamma="scale",
                    probability=True,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )

    print(
        "Training rank classifier..."
    )

    model.fit(
        X_train,
        y_train,
    )

    joblib.dump(
        {
            "model": model,
            "ranks": RANKS,
        },
        MODEL_PATH,
    )

    print()
    print(
        f"Saved model:"
    )

    print(
        MODEL_PATH
    )

    if not test:
        print()
        print(
            "No holdout examples available."
        )
        return

    X_test = np.array(
        [
            sample["features"]
            for sample in test
        ]
    )

    y_test = np.array(
        [
            sample["rank"]
            for sample in test
        ]
    )

    predictions = model.predict(
        X_test
    )

    accuracy = np.mean(
        predictions == y_test
    )

    print()
    print(
        "=============================="
    )

    print(
        "RANK CLASSIFIER RESULTS"
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

    labels_present = sorted(
        set(y_test),
        key=lambda value: (
            RANKS.index(value)
            if value in RANKS
            else 999
        ),
    )

    print(
        classification_report(
            y_test,
            predictions,
            labels=labels_present,
            zero_division=0,
        )
    )

    print(
        "Confusion matrix:"
    )

    print(
        "Labels:",
        labels_present,
    )

    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=labels_present,
        )
    )

    print()

    print(
        "Individual mistakes:"
    )

    mistakes = 0

    probabilities = (
        model.predict_proba(
            X_test
        )
    )

    classes = (
        model.classes_
    )

    for sample, actual, predicted, probs in zip(
        test,
        y_test,
        predictions,
        probabilities,
    ):
        if actual == predicted:
            continue

        mistakes += 1

        confidence = float(
            np.max(probs)
        )

        print(
            f"{sample['name']}: "
            f"actual={actual} "
            f"predicted={predicted} "
            f"confidence={confidence:.3f}"
        )

    if mistakes == 0:
        print(
            "None."
        )


if __name__ == "__main__":
    main()