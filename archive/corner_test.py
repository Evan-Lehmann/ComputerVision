
from pathlib import Path
import cv2
import numpy as np

INPUT = Path("samples/overlapping_cards.png")
OUTPUT = Path("output")
OUTPUT.mkdir(exist_ok=True)

image = cv2.imread(str(INPUT))

if image is None:
    raise FileNotFoundError(INPUT)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Light blur to suppress image noise
blurred = cv2.GaussianBlur(gray, (3, 3), 0)

# Detect contrast boundaries
edges = cv2.Canny(blurred, 60, 150)

# Find contours from the edge image
contours, _ = cv2.findContours(
    edges,
    cv2.RETR_LIST,
    cv2.CHAIN_APPROX_SIMPLE
)

annotated = image.copy()
candidates = []

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    area = w * h

    # Initial filters for small visual elements
    if area < 50 or area > 3000:
        continue

    if w < 4 or h < 6:
        continue

    candidates.append((x, y, w, h))

# Draw candidate regions for visual inspection
for x, y, w, h in candidates:
    cv2.rectangle(
        annotated,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        1
    )

cv2.imwrite(
    str(OUTPUT / "corner_candidates.png"),
    annotated
)

cv2.imwrite(
    str(OUTPUT / "corner_edges.png"),
    edges
)

print(f"Found {len(candidates)} candidate regions")
print("Saved edge map and annotated image.")
