
from pathlib import Path
from collections import Counter
import csv

import cv2
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("output/processed_v2")
METADATA = DATA_DIR / "metadata.csv"
RESULTS_FILE = Path("output/hog_svm_results.csv")

DECKS = ["deck1", "deck2", "original"]

# HOG feature extractor for standardized 64x64 images
hog = cv2.HOGDescriptor(
    (64, 64),   # Window size
    (16, 16),   # Block size
    (8, 8),     # Block stride
    (8, 8),     # Cell size
    9           # Orientation bins
)


# ============================================================
# LOAD DATASET
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
                print(f"Skipping unreadable image: {path}")
                continue

            if image.shape != (64, 64):
                image = cv2.resize(image, (64, 64))

            features = hog.compute(image).flatten()

            samples.append({
                "deck": row["source_deck"],
                "rank": row["rank"],
                "filename": path.name,
                "features": features
            })

    return samples


# ============================================================
# TRAIN AND EVALUATE
# ============================================================

def evaluate_held_out_deck(samples, test_deck):

    train = [
        s for s in samples
        if s["deck"] != test_deck
    ]

    test = [
        s for s in samples
        if s["deck"] == test_deck
    ]

    # Only evaluate ranks represented in training
    available_ranks = {s["rank"] for s in train}

    test = [
        s for s in test
        if s["rank"] in available_ranks
    ]

    if not train or not test:
        print(f"No usable data for {test_deck}")
        return []

    X_train = np.array(
        [s["features"] for s in train]
    )

    y_train = np.array(
        [s["rank"] for s in train]
    )

    X_test = np.array(
        [s["features"] for s in test]
    )

    y_test = np.array(
        [s["rank"] for s in test]
    )

    # Standardize features using training data only
    # and train an SVM classifier.
    model = make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale"
        )
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test, predictions
    )

    print(f"\n--- TESTING ON {test_deck.upper()} ---")
    print(f"Training samples: {len(train)}")
    print(f"Testing samples: {len(test)}")
    print(f"Accuracy: {accuracy * 100:.1f}%")

    results = []

    for sample, prediction in zip(test, predictions):

        correct = prediction == sample["rank"]

        results.append({
            "test_deck": test_deck,
            "filename": sample["filename"],
            "actual": sample["rank"],
            "predicted": prediction,
            "correct": bool(correct)
        })

        if not correct:
            print(
                f"  FAIL: {sample['filename']} "
                f"Actual: {sample['rank']} "
                f"Predicted: {prediction}"
            )

    # Report the most frequent mistakes
    mistakes = Counter(
        (r["actual"], r["predicted"])
        for r in results
        if not r["correct"]
    )

    if mistakes:
        print("Most common confusions:")

        for (actual, predicted), count in (
            mistakes.most_common(5)
        ):
            print(
                f"  {actual} -> {predicted}: {count}"
            )

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    samples = load_dataset()

    print(f"Loaded {len(samples)} processed images.")

    all_results = []

    # Each deck gets a turn as the unseen test set
    for test_deck in DECKS:
        results = evaluate_held_out_deck(
            samples,
            test_deck
        )

        all_results.extend(results)

    if not all_results:
        print("No evaluation results.")
        return

    total_correct = sum(
        result["correct"]
        for result in all_results
    )

    overall = (
        total_correct / len(all_results) * 100
    )

    print("\n--- OVERALL RESULTS ---")
    print(f"Correct: {total_correct}/{len(all_results)}")
    print(f"Accuracy: {overall:.1f}%")

    RESULTS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with RESULTS_FILE.open(
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "test_deck",
                "filename",
                "actual",
                "predicted",
                "correct"
            ]
        )

        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
