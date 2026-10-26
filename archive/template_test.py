
from pathlib import Path
import cv2
import numpy as np

SAMPLES = Path("samples")
OUTPUT = Path("output")
OUTPUT.mkdir(exist_ok=True)

template = cv2.imread(str(SAMPLES / "card_back.png"))
image = cv2.imread(str(SAMPLES / "test_table.png"))

if template is None or image is None:
    raise FileNotFoundError(
        "Check card_back.png and test_table.png"
    )

# Convert to grayscale
template_gray = cv2.cvtColor(
    template, cv2.COLOR_BGR2GRAY
)
image_gray = cv2.cvtColor(
    image, cv2.COLOR_BGR2GRAY
)

th, tw = template_gray.shape[:2]
ih, iw = image_gray.shape[:2]

if th > ih or tw > iw:
    raise ValueError(
        "Template must be smaller than the test image"
    )

# Compare the template against each possible location
scores = cv2.matchTemplate(
    image_gray,
    template_gray,
    cv2.TM_CCOEFF_NORMED
)

# Find the highest-scoring match
_, best_score, _, best_location = cv2.minMaxLoc(scores)

x, y = best_location

annotated = image.copy()

cv2.rectangle(
    annotated,
    (x, y),
    (x + tw, y + th),
    (0, 255, 0),
    2
)

cv2.putText(
    annotated,
    f"Match: {best_score:.3f}",
    (x, max(20, y - 10)),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    (0, 255, 0),
    2
)

cv2.imwrite(
    str(OUTPUT / "template_result.png"),
    annotated
)

# Save a visualization of the matching scores
heatmap = cv2.normalize(
    scores, None, 0, 255, cv2.NORM_MINMAX
).astype(np.uint8)

heatmap = cv2.applyColorMap(
    heatmap, cv2.COLORMAP_JET
)

cv2.imwrite(
    str(OUTPUT / "template_heatmap.png"),
    heatmap
)

print(f"Best match score: {best_score:.4f}")
print(f"Best match location: ({x}, {y})")
