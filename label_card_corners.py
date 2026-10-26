from pathlib import Path
import csv
import cv2


DATASET_ROOT = Path("detection_dataset")
MANIFEST_PATH = DATASET_ROOT / "manifest.csv"
PROGRESS_PATH = DATASET_ROOT / "label_progress.csv"

WINDOW_NAME = "Label Center Card Ranks"

CLASS_ID = 0


# ============================================================
# LOAD DATASET
# ============================================================

def load_manifest():
    rows = []

    with open(
        MANIFEST_PATH,
        newline=""
    ) as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append(row)

    return rows


def get_image_path(row):
    return (
        DATASET_ROOT
        / row["split"]
        / "images"
        / row["dataset_filename"]
    )


def get_label_path(row):
    labels_dir = (
        DATASET_ROOT
        / row["split"]
        / "labels"
    )

    labels_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    return (
        labels_dir
        / (
            Path(
                row["dataset_filename"]
            ).stem
            + ".txt"
        )
    )


# ============================================================
# PROGRESS
# ============================================================

def load_completed():
    completed = set()

    if not PROGRESS_PATH.exists():
        return completed

    with open(
        PROGRESS_PATH,
        newline=""
    ) as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row["completed"] == "1":
                completed.add(
                    row["dataset_filename"]
                )

    return completed


def save_progress(
    rows,
    completed
):
    with open(
        PROGRESS_PATH,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset_filename",
                "split",
                "completed",
            ],
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    "dataset_filename":
                        row["dataset_filename"],

                    "split":
                        row["split"],

                    "completed":
                        "1"
                        if row["dataset_filename"]
                        in completed
                        else "0",
                }
            )


# ============================================================
# LABEL IO
# ============================================================

def load_existing_boxes(
    path,
    image_w,
    image_h
):
    boxes = []

    if not path.exists():
        return boxes

    for line in path.read_text().splitlines():

        parts = line.strip().split()

        if len(parts) != 5:
            continue

        _, xc, yc, bw, bh = map(
            float,
            parts
        )

        w = int(
            bw * image_w
        )

        h = int(
            bh * image_h
        )

        x = int(
            xc * image_w
            - w / 2
        )

        y = int(
            yc * image_h
            - h / 2
        )

        boxes.append(
            (
                x,
                y,
                w,
                h
            )
        )

    return boxes


def save_boxes(
    path,
    boxes,
    image_w,
    image_h
):
    lines = []

    for x, y, w, h in boxes:

        xc = (
            x + w / 2
        ) / image_w

        yc = (
            y + h / 2
        ) / image_h

        bw = (
            w / image_w
        )

        bh = (
            h / image_h
        )

        lines.append(
            f"{CLASS_ID} "
            f"{xc:.6f} "
            f"{yc:.6f} "
            f"{bw:.6f} "
            f"{bh:.6f}"
        )

    path.write_text(
        "\n".join(lines)
        + (
            "\n"
            if lines
            else ""
        )
    )


# ============================================================
# BOX DRAWER
# ============================================================

class BoxDrawer:

    def __init__(self):
        self.boxes = []

        self.drawing = False

        self.start_x = 0
        self.start_y = 0

        self.current_x = 0
        self.current_y = 0


    def callback(
        self,
        event,
        x,
        y,
        flags,
        param
    ):

        if event == cv2.EVENT_LBUTTONDOWN:

            self.drawing = True

            self.start_x = x
            self.start_y = y

            self.current_x = x
            self.current_y = y


        elif (
            event == cv2.EVENT_MOUSEMOVE
            and self.drawing
        ):

            self.current_x = x
            self.current_y = y


        elif event == cv2.EVENT_LBUTTONUP:

            self.drawing = False

            x1 = min(
                self.start_x,
                x
            )

            y1 = min(
                self.start_y,
                y
            )

            x2 = max(
                self.start_x,
                x
            )

            y2 = max(
                self.start_y,
                y
            )

            w = x2 - x1
            h = y2 - y1

            if (
                w >= 5
                and h >= 5
            ):
                self.boxes.append(
                    (
                        x1,
                        y1,
                        w,
                        h
                    )
                )


# ============================================================
# DISPLAY
# ============================================================

def draw_display(
    image,
    boxes,
    current_index,
    total,
    split
):
    display = image.copy()

    for index, (
        x,
        y,
        w,
        h
    ) in enumerate(
        boxes,
        start=1
    ):

        cv2.rectangle(
            display,
            (x, y),
            (
                x + w,
                y + h
            ),
            (0, 255, 0),
            2
        )

        cv2.putText(
            display,
            str(index),
            (
                x,
                max(
                    18,
                    y - 5
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    lines = [
        (
            f"{current_index}/{total} "
            f"| {split}"
        ),

        "LABEL: center/table TOP-LEFT RANK only",

        (
            "Ignore suit, bottom-right rank, "
            "and duplicate display card"
        ),

        (
            "N save+next | Z undo | "
            "R reset | P previous | Q quit"
        ),
    ]

    y = 28

    for line in lines:

        cv2.putText(
            display,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

        y += 30

    return display


# ============================================================
# MAIN
# ============================================================

def main():

    rows = load_manifest()

    if not rows:
        print(
            "No images found in manifest."
        )
        return

    completed = load_completed()

    # --------------------------------------------------------
    # Resume at first unfinished image
    # --------------------------------------------------------

    start_index = 0

    for i, row in enumerate(rows):

        if (
            row["dataset_filename"]
            not in completed
        ):
            start_index = i
            break

    else:
        print(
            "All images are already labeled."
        )
        return

    print(
        f"Resuming at image "
        f"{start_index + 1}/"
        f"{len(rows)}"
    )

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )

    index = start_index

    while (
        0 <= index < len(rows)
    ):

        row = rows[index]

        image_path = get_image_path(
            row
        )

        label_path = get_label_path(
            row
        )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            print(
                f"Could not load "
                f"{image_path}"
            )

            index += 1
            continue

        image_h, image_w = (
            image.shape[:2]
        )

        drawer = BoxDrawer()

        drawer.boxes = (
            load_existing_boxes(
                label_path,
                image_w,
                image_h
            )
        )

        cv2.setMouseCallback(
            WINDOW_NAME,
            drawer.callback
        )

        while True:

            display = draw_display(
                image,
                drawer.boxes,
                index + 1,
                len(rows),
                row["split"]
            )

            if drawer.drawing:

                cv2.rectangle(
                    display,
                    (
                        drawer.start_x,
                        drawer.start_y
                    ),
                    (
                        drawer.current_x,
                        drawer.current_y
                    ),
                    (255, 255, 0),
                    2
                )

            cv2.imshow(
                WINDOW_NAME,
                display
            )

            key = (
                cv2.waitKey(20)
                & 0xFF
            )

            # ------------------------------------------------
            # NEXT
            # ------------------------------------------------

            if key == ord("n"):

                save_boxes(
                    label_path,
                    drawer.boxes,
                    image_w,
                    image_h
                )

                completed.add(
                    row["dataset_filename"]
                )

                save_progress(
                    rows,
                    completed
                )

                print(
                    f"{index + 1:02d}/"
                    f"{len(rows)} "
                    f"{row['dataset_filename']} "
                    f"-> "
                    f"{len(drawer.boxes)} rank boxes"
                )

                index += 1
                break


            # ------------------------------------------------
            # PREVIOUS
            # ------------------------------------------------

            elif key == ord("p"):

                save_boxes(
                    label_path,
                    drawer.boxes,
                    image_w,
                    image_h
                )

                index = max(
                    0,
                    index - 1
                )

                break


            # ------------------------------------------------
            # UNDO
            # ------------------------------------------------

            elif key == ord("z"):

                if drawer.boxes:
                    drawer.boxes.pop()


            # ------------------------------------------------
            # RESET
            # ------------------------------------------------

            elif key == ord("r"):

                drawer.boxes = []


            # ------------------------------------------------
            # QUIT
            # ------------------------------------------------

            elif key == ord("q"):

                save_boxes(
                    label_path,
                    drawer.boxes,
                    image_w,
                    image_h
                )

                save_progress(
                    rows,
                    completed
                )

                cv2.destroyAllWindows()

                print()
                print(
                    "Progress saved."
                )

                return

    cv2.destroyAllWindows()

    print()
    print(
        "--- LABELING COMPLETE ---"
    )

    print(
        f"Images completed: "
        f"{len(completed)}"
    )


if __name__ == "__main__":
    main()