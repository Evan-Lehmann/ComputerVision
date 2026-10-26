from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent

DATASET = (
    ROOT
    / "detection_dataset_tiled"
    / "dataset.yaml"
)

OUTPUT_DIR = (
    ROOT
    / "rank_detector_runs"
)

RUN_NAME = "rank_detector_v2"


def main():
    if not DATASET.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found:\n{DATASET}"
        )

    print()
    print("===================================")
    print("CardVision Rank Detector - V2")
    print("===================================")
    print()
    print(f"Dataset: {DATASET}")
    print(f"Output:  {OUTPUT_DIR / RUN_NAME}")
    print()

    # Same pretrained model as V1 so the comparison is fair.
    model = YOLO("yolo11n.pt")

    print("Starting training...")
    print()

    model.train(
        data=str(DATASET),

        # Training
        epochs=100,
        patience=20,
        batch=8,

        # Each tile gets resized to 640 rather than
        # the entire original screen getting resized to 640.
        imgsz=640,

        # Output
        project=str(OUTPUT_DIR),
        name=RUN_NAME,
        exist_ok=True,

        # Start from pretrained weights.
        pretrained=True,

        # Save graphs/results.
        plots=True,

        verbose=True,
    )

    best_model = (
        OUTPUT_DIR
        / RUN_NAME
        / "weights"
        / "best.pt"
    )

    print()
    print("===================================")
    print("TRAINING COMPLETE")
    print("===================================")
    print()
    print(f"Best model: {best_model}")
    print()

    # Run an explicit validation pass on best.pt so
    # the final metrics are easy to compare with V1.
    print("Running final validation...")
    print()

    best = YOLO(str(best_model))

    metrics = best.val(
        data=str(DATASET),
        imgsz=640,
        batch=8,
        plots=True,
    )

    print()
    print("===================================")
    print("V2 FINAL METRICS")
    print("===================================")

    print(
        f"Precision:   "
        f"{metrics.box.mp:.6f}"
    )

    print(
        f"Recall:      "
        f"{metrics.box.mr:.6f}"
    )

    print(
        f"mAP50:       "
        f"{metrics.box.map50:.6f}"
    )

    print(
        f"mAP50-95:    "
        f"{metrics.box.map:.6f}"
    )

    print()
    print("V1 BASELINE")
    print("Precision:   0.002080")
    print("Recall:      0.455000")
    print("mAP50:       0.002940")
    print("mAP50-95:    0.001120")
    print()


if __name__ == "__main__":
    main()