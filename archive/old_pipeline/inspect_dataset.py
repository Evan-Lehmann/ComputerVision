
from pathlib import Path
from collections import Counter
import cv2

CARDS = Path("samples/cards")
EXTENSIONS = {".png", ".jpg", ".jpeg"}

files = [
    p for p in CARDS.rglob("*")
    if p.is_file() and p.suffix.lower() in EXTENSIONS
]

dimensions = Counter()
folders = Counter()
unreadable = []

for path in files:
    image = cv2.imread(str(path))

    if image is None:
        unreadable.append(path.name)
        continue

    height, width = image.shape[:2]
    dimensions[(width, height)] += 1
    folders[str(path.parent.relative_to(CARDS))] += 1

print(f"\nTotal image files: {len(files)}")

print("\nImage dimensions:")
for size, count in dimensions.most_common():
    print(f"  {size}: {count} files")

print("\nFolders:")
for folder, count in folders.items():
    print(f"  {folder}: {count} files")

print("\nUnreadable images:", unreadable)
