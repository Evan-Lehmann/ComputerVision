from pathlib import Path
from collections import defaultdict, Counter
import csv

import cv2
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("output/processed_v2")
METADATA = DATA_DIR / "metadata.csv"

DECKS = ["deck1", "deck2", "original"]

hog = cv2.HOGDescriptor(
    (64, 64),
    (16, 16),
    (8, 8),
    (8, 8),
    9
)

# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():
    samples = []

    with METADATA.open(newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            path = Path(row["processed_file"])

            image = cv2.imread(
                str(path),
                cv2.IMREAD_GRAYSCALE
            )

            if image is None:
                continue

            if image.shape != (64, 64):
                image = cv2.resize(image, (64, 64))

            hog_features = hog.compute(image).flatten()

            raw_features = (
                image.astype(np.float32).flatten() / 255.0
            )

            samples.append({
                "deck": row["source_deck"],
                "rank": row["rank"],
                "filename": path.name,
                "image": image,
                "hog": hog_features,
                "raw": raw_features
            })

    return samples


# ============================================================
# MODEL
# ============================================================

def build_model():
    return make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale"
        )
    )


# ============================================================
# WITHIN-DECK TESTING
# ============================================================

def within_deck_test(samples, feature_type):
    print(
        f"\n===== WITHIN-DECK TEST: "
        f"{feature_type.upper()} ====="
    )

    for deck in DECKS:
        deck_samples = [
            s for s in samples
            if s["deck"] == deck
        ]

        if len(deck_samples) < 2:
            continue

        X = np.array([
            s[feature_type]
            for s in deck_samples
        ])

        y = np.array([
            s["rank"]
            for s in deck_samples
        ])

        loo = LeaveOneOut()

        predictions = []
        actual = []

        for train_index, test_index in loo.split(X):
            X_train = X[train_index]
            X_test = X[test_index]

            y_train = y[train_index]
            y_test = y[test_index]

            # If test rank has no remaining training example,
            # skip this prediction.
            if y_test[0] not in set(y_train):
                continue

            model = build_model()

            model.fit(
                X_train,
                y_train
            )

            prediction = model.predict(
                X_test
            )[0]

            predictions.append(prediction)
            actual.append(y_test[0])

        if not actual:
            print(f"{deck}: no valid predictions")
            continue

        accuracy = accuracy_score(
            actual,
            predictions
        )

        print(
            f"{deck:10} "
            f"{sum(a == p for a, p in zip(actual, predictions))}"
            f"/{len(actual)} "
            f"Accuracy: {accuracy * 100:.1f}%"
        )

        mistakes = Counter(
            (a, p)
            for a, p in zip(actual, predictions)
            if a != p
        )

        if mistakes:
            print("  Top confusions:")

            for (a, p), count in mistakes.most_common(5):
                print(
                    f"    {a} -> {p}: {count}"
                )


# ============================================================
# CROSS-DECK TESTING
# ============================================================

def cross_deck_test(samples, feature_type):
    print(
        f"\n===== CROSS-DECK TEST: "
        f"{feature_type.upper()} ====="
    )

    for test_deck in DECKS:
        train = [
            s for s in samples
            if s["deck"] != test_deck
        ]

        test = [
            s for s in samples
            if s["deck"] == test_deck
        ]

        available_ranks = {
            s["rank"] for s in train
        }

        test = [
            s for s in test
            if s["rank"] in available_ranks
        ]

        X_train = np.array([
            s[feature_type]
            for s in train
        ])

        y_train = np.array([
            s["rank"]
            for s in train
        ])

        X_test = np.array([
            s[feature_type]
            for s in test
        ])

        y_test = np.array([
            s["rank"]
            for s in test
        ])

        model = build_model()

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        print(
            f"Test {test_deck:10} "
            f"{sum(y_test == predictions)}"
            f"/{len(y_test)} "
            f"Accuracy: {accuracy * 100:.1f}%"
        )

        prediction_counts = Counter(
            predictions
        )

        print(
            "  Prediction distribution:"
        )

        for rank, count in prediction_counts.most_common():
            print(
                f"    {rank:6}: {count}"
            )


# ============================================================
# FEATURE SIMILARITY ANALYSIS
# ============================================================

def cosine_similarity(a, b):
    denom = (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    )

    if denom == 0:
        return 0.0

    return float(
        np.dot(a, b) / denom
    )


def feature_similarity_analysis(samples):
    print(
        "\n===== HOG FEATURE SIMILARITY ====="
    )

    for deck in DECKS:
        deck_samples = [
            s for s in samples
            if s["deck"] == deck
        ]

        same_rank_scores = []
        different_rank_scores = []

        for i in range(len(deck_samples)):
            for j in range(
                i + 1,
                len(deck_samples)
            ):
                a = deck_samples[i]
                b = deck_samples[j]

                score = cosine_similarity(
                    a["hog"],
                    b["hog"]
                )

                if a["rank"] == b["rank"]:
                    same_rank_scores.append(
                        score
                    )
                else:
                    different_rank_scores.append(
                        score
                    )

        print(f"\n{deck}:")

        if same_rank_scores:
            print(
                f"  Same-rank mean similarity: "
                f"{np.mean(same_rank_scores):.3f}"
            )

        if different_rank_scores:
            print(
                f"  Different-rank mean similarity: "
                f"{np.mean(different_rank_scores):.3f}"
            )

        if same_rank_scores and different_rank_scores:
            separation = (
                np.mean(same_rank_scores)
                - np.mean(different_rank_scores)
            )

            print(
                f"  Separation: {separation:.3f}"
            )


# ============================================================
# MAIN
# ============================================================

def main():
    samples = load_dataset()

    print(
        f"Loaded {len(samples)} images."
    )

    print("\nDataset breakdown:")

    by_deck = defaultdict(list)

    for sample in samples:
        by_deck[sample["deck"]].append(
            sample
        )

    for deck in DECKS:
        print(
            f"  {deck}: "
            f"{len(by_deck[deck])}"
        )

    # Compare HOG and raw-pixel SVMs
    within_deck_test(
        samples,
        "hog"
    )

    within_deck_test(
        samples,
        "raw"
    )

    cross_deck_test(
        samples,
        "hog"
    )

    cross_deck_test(
        samples,
        "raw"
    )

    # Determine whether HOG features are actually
    # separating ranks within each deck.
    feature_similarity_analysis(
        samples
    )


if __name__ == "__main__":
    main()