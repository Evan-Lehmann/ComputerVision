
from pathlib import Path
from collections import defaultdict
import csv

import cv2
import numpy as np

DATA_DIR = Path("output/processed_v2")
METADATA = DATA_DIR / "metadata.csv"


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
                print(f"Skipping unreadable file: {path}")
                continue

            samples.append({
                "rank": row["rank"],
                "deck": row["source_deck"],
                "filename": path.name,
                "image": image
            })

    return samples


def similarity(a, b):
    a = a.astype(np.float32).flatten()
    b = b.astype(np.float32).flatten()

    denominator = (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


def evaluate(samples):
    results = []
    deck_stats = defaultdict(lambda: [0, 0])

    for i, test in enumerate(samples):
        # Leave the current image out of references
        references = [
            ref for j, ref in enumerate(samples)
            if j != i
        ]

        matches = [
            (
                similarity(test["image"], ref["image"]),
                ref["rank"]
            )
            for ref in references
        ]

        best_score, prediction = max(matches)

        correct = prediction == test["rank"]
        deck_stats[test["deck"]][1] += 1
        deck_stats[test["deck"]][0] += int(correct)

        results.append({
            "deck": test["deck"],
            "filename": test["filename"],
            "actual": test["rank"],
            "predicted": prediction,
            "score": round(best_score, 4),
            "correct": correct
        })

    return results, deck_stats


def main():
    samples = load_dataset()

    if len(samples) < 2:
        raise ValueError("Not enough images to evaluate")

    print(f"Loaded {len(samples)} images.")

    results, stats = evaluate(samples)

    print("\n--- ACCURACY BY SOURCE ---")

    for deck, (correct, total) in sorted(stats.items()):
        accuracy = 100 * correct / total

        print(
            f"{deck:10} "
            f"{correct:3}/{total:3} "
            f"Accuracy: {accuracy:.1f}%"
        )

    total_correct = sum(
        result["correct"] for result in results
    )

    overall = 100 * total_correct / len(results)

    print(f"\nOverall accuracy: {overall:.1f}%")

    failures = [
        result for result in results
        if not result["correct"]
    ]

    print(f"\nMisclassified images: {len(failures)}")

    for item in failures:
        print(
            f"{item['deck']:10} "
            f"{item['filename']:28} "
            f"Actual: {item['actual']:6} "
            f"Predicted: {item['predicted']:6} "
            f"Score: {item['score']:.3f}"
        )

    # Save results for future comparisons
    output = Path("output/benchmark_results.csv")

    with output.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "deck",
                "filename",
                "actual",
                "predicted",
                "score",
                "correct"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved results to: {output}")


if __name__ == "__main__":
    main()
