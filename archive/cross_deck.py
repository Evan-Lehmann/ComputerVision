
from pathlib import Path
from collections import defaultdict
import csv

import cv2
import numpy as np

DATA_DIR = Path("output/processed_v2")
METADATA = DATA_DIR / "metadata.csv"
RESULTS_FILE = Path("output/cross_deck_results.csv")


def load_dataset():
    samples = []

    with METADATA.open(newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            path = Path(row["processed_file"])
            image = cv2.imread(
                str(path), cv2.IMREAD_GRAYSCALE
            )

            if image is None:
                continue

            samples.append({
                "rank": row["rank"],
                "deck": row["source_deck"],
                "filename": path.name,
                "image": image
            })

    return samples


def similarity(image_a, image_b):
    a = image_a.astype(np.float32).flatten()
    b = image_b.astype(np.float32).flatten()

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


def evaluate(train, test, train_name, test_name):
    results = []
    correct = 0

    available_ranks = {s["rank"] for s in train}

    for sample in test:
        if sample["rank"] not in available_ranks:
            print(
                f"Skipping {sample['filename']}: "
                "rank absent from reference deck"
            )
            continue

        best_score = -1
        best_rank = None

        for reference in train:
            score = similarity(
                sample["image"], reference["image"]
            )

            if score > best_score:
                best_score = score
                best_rank = reference["rank"]

        is_correct = best_rank == sample["rank"]
        correct += int(is_correct)

        results.append({
            "reference_deck": train_name,
            "test_deck": test_name,
            "filename": sample["filename"],
            "actual": sample["rank"],
            "predicted": best_rank,
            "score": round(best_score, 4),
            "correct": is_correct
        })

    accuracy = 100 * correct / len(results) if results else 0

    print(
        f"{train_name:10} -> {test_name:10} "
        f"{correct:3}/{len(results):3} "
        f"Accuracy: {accuracy:.1f}%"
    )

    return results


def main():
    samples = load_dataset()

    by_deck = defaultdict(list)

    for sample in samples:
        by_deck[sample["deck"]].append(sample)

    print("Loaded datasets:")
    for deck, items in sorted(by_deck.items()):
        print(f"  {deck}: {len(items)} images")

    results = []

    print("\n--- CROSS-DECK ACCURACY ---")

    decks = ["deck1", "deck2", "original"]

    for train_deck in decks:
        for test_deck in decks:
            if train_deck == test_deck:
                continue

            results.extend(
                evaluate(
                    by_deck[train_deck],
                    by_deck[test_deck],
                    train_deck,
                    test_deck
                )
            )

    RESULTS_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )

    with RESULTS_FILE.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "reference_deck",
                "test_deck",
                "filename",
                "actual",
                "predicted",
                "score",
                "correct"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved results to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
