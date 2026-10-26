
from pathlib import Path
import cv2
import numpy as np

DATA_DIR = Path("output/processed_ranks")

def load_images():
    samples = []

    for folder in sorted(DATA_DIR.iterdir()):
        if not folder.is_dir():
            continue

        rank = folder.name

        for path in folder.glob("*.png"):
            img = cv2.imread(
                str(path), cv2.IMREAD_GRAYSCALE
            )

            if img is not None:
                samples.append((rank, path.name, img))

    return samples


def similarity(image_a, image_b):
    a = image_a.astype(np.float32).flatten()
    b = image_b.astype(np.float32).flatten()

    # Cosine similarity
    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


def main():
    samples = load_images()

    if len(samples) < 2:
        print("Not enough samples.")
        return

    correct = 0
    total = 0

    for actual_rank, filename, test_img in samples:

        # Exclude the test image itself
        references = [
            (rank, name, img)
            for rank, name, img in samples
            if not (rank == actual_rank and name == filename)
        ]

        scores = []

        for rank, name, reference in references:
            score = similarity(test_img, reference)
            scores.append((score, rank))

        scores.sort(reverse=True)

        best_score, predicted_rank = scores[0]
        is_correct = predicted_rank == actual_rank

        correct += int(is_correct)
        total += 1

        result = "PASS" if is_correct else "FAIL"

        print(
            f"{filename:22} "
            f"Actual: {actual_rank:6} "
            f"Predicted: {predicted_rank:6} "
            f"Score: {best_score:.3f} "
            f"{result}"
        )

    print("\n--- RESULTS ---")
    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {correct / total * 100:.1f}%")


if __name__ == "__main__":
    main()
