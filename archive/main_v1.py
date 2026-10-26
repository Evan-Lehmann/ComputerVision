
from pathlib import Path
import cv2
import numpy as np

INPUT = Path("samples/test.png")
OUTPUT = Path("output")
OUTPUT.mkdir(exist_ok=True)

image = cv2.imread(str(INPUT))
if image is None:
    raise FileNotFoundError(f"Image not found: {INPUT}")

height, width = image.shape[:2]

# Identify light surfaces such as white playing cards
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(
    hsv,
    np.array([0, 0, 185]),
    np.array([179, 75, 255])
)

# Clean up noise and join small gaps
kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT, (3, 3)
)
mask = cv2.morphologyEx(
    mask, cv2.MORPH_CLOSE, kernel
)
mask = cv2.morphologyEx(
    mask, cv2.MORPH_OPEN, kernel
)

contours, _ = cv2.findContours(
    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

detections = []
image_area = width * height

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    box_area = w * h

    if box_area == 0:
        continue

    aspect = w / h
    rectangularity = cv2.contourArea(contour) / box_area
    relative_area = box_area / image_area

    # Exclude extremely small and large elements
    if not 0.0005 <= relative_area <= 0.04:
        continue

    # Standard upright playing-card proportions
    if not 0.45 <= aspect <= 0.85:
        continue

    # Reject irregular shapes and text fragments
    if rectangularity < 0.75:
        continue

    detections.append((x, y, w, h))

detections.sort(key=lambda b: (b[1], b[0]))

annotated = image.copy()

for i, (x, y, w, h) in enumerate(detections, 1):
    cv2.rectangle(
        annotated,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated,
        f"Card {i}",
        (x, max(20, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2
    )

    crop = image[y:y+h, x:x+w]
    cv2.imwrite(
        str(OUTPUT / f"card_{i}.png"), crop
    )

cv2.imwrite(str(OUTPUT / "detected.png"), annotated)
cv2.imwrite(str(OUTPUT / "debug_mask.png"), mask)

print(f"Detected {len(detections)} candidate cards")
print(f"Output directory: {OUTPUT.resolve()}")
