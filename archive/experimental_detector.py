
from pathlib import Path
import cv2
import numpy as np

INPUT = Path("samples/test.png")
OUTPUT = Path("output")
OUTPUT.mkdir(exist_ok=True)

image = cv2.imread(str(INPUT))
if image is None:
    raise FileNotFoundError(INPUT)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Bright, low-saturation pixels
white_mask = cv2.inRange(
    hsv,
    np.array([0, 0, 180]),
    np.array([179, 80, 255])
)

# Saturated red pixels, allowing both hue ranges
red_low = cv2.inRange(
    hsv,
    np.array([0, 90, 75]),
    np.array([12, 255, 255])
)

red_high = cv2.inRange(
    hsv,
    np.array([170, 90, 75]),
    np.array([179, 255, 255])
)

red_mask = cv2.bitwise_or(red_low, red_high)

def find_candidates(mask, label):
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel
    )

    contours, _ = cv2.findContours(
        cleaned,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        area = w * h
        if area < 250:
            continue

        aspect = w / max(h, 1)

        if 0.35 <= aspect <= 1.2:
            candidates.append(
                (x, y, w, h, label)
            )

    return candidates

detections = (
    find_candidates(white_mask, "Face up")
    + find_candidates(red_mask, "Red candidate")
)

annotated = image.copy()

for x, y, w, h, label in detections:
    color = (0, 255, 0) if label == "Face up" else (255, 0, 255)

    cv2.rectangle(
        annotated,
        (x, y),
        (x + w, y + h),
        color,
        2
    )

    cv2.putText(
        annotated,
        label,
        (x, max(15, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2
    )

cv2.imwrite(
    str(OUTPUT / "experimental.png"),
    annotated
)

cv2.imwrite(
    str(OUTPUT / "red_mask.png"),
    red_mask
)

print(f"Found {len(detections)} candidate regions")
