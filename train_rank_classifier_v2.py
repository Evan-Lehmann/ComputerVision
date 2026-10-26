from pathlib import Path
import csv
import random
import re
from collections import Counter, defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================================================
# CONFIG
# =========================================================

ROOT = Path(__file__).resolve().parent

ORIGINAL_DIR = ROOT / "rank_value_dataset"
ORIGINAL_CSV = ORIGINAL_DIR / "labels.csv"

CANDIDATE_DIR = ROOT / "rank_value_candidates"
CANDIDATE_CSV = CANDIDATE_DIR / "candidate_labels.csv"

OUTPUT_DIR = ROOT / "rank_classifier_v2"
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "best_rank_classifier.pt"
CONFUSION_CSV = OUTPUT_DIR / "confusion_matrix.csv"
PREDICTIONS_CSV = OUTPUT_DIR / "holdout_predictions.csv"

RANKS = [
    "A", "2", "3", "4", "5", "6",
    "7", "8", "9", "10", "J", "Q", "K"
]

RANK_TO_INDEX = {
    rank: i
    for i, rank in enumerate(RANKS)
}

INDEX_TO_RANK = {
    i: rank
    for rank, i in RANK_TO_INDEX.items()
}

IMAGE_SIZE = 64

RANDOM_SEED = 42

BATCH_SIZE = 16
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001


# =========================================================
# REPRODUCIBILITY
# =========================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# =========================================================
# DEVICE
# =========================================================

def choose_device():
    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


DEVICE = choose_device()


# =========================================================
# DATA LOADING
# =========================================================

def source_group_from_name(name):
    """
    Keep crops from the same original frame together
    during train/validation splitting.

    Examples:
      candidate_0039_frame_20261025_025548_247027_000500_01.png
      train_train_013_frame_000505_02.png
    """

    match = re.search(
        r"(frame_(?:\d{8}_\d{6}_\d{6}_)?\d{6})",
        name,
    )

    if match:
        return match.group(1)

    return name


def load_label_file(
    image_dir,
    csv_path,
    source_name,
):
    samples = []

    if not csv_path.exists():
        return samples

    with csv_path.open(
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            rank = row["rank"].strip()

            if rank == "UNKNOWN":
                continue

            if rank not in RANK_TO_INDEX:
                print(
                    f"Skipping invalid rank "
                    f"{rank}: {row['crop']}"
                )
                continue

            image_path = (
                image_dir
                / row["crop"]
            )

            if not image_path.exists():
                print(
                    f"Missing image: "
                    f"{image_path}"
                )
                continue

            samples.append(
                {
                    "path": image_path,
                    "rank": rank,
                    "source": source_name,
                    "group": source_group_from_name(
                        row["crop"]
                    ),
                }
            )

    return samples


def load_all_samples():
    samples = []

    samples.extend(
        load_label_file(
            ORIGINAL_DIR,
            ORIGINAL_CSV,
            "original",
        )
    )

    samples.extend(
        load_label_file(
            CANDIDATE_DIR,
            CANDIDATE_CSV,
            "candidate",
        )
    )

    return samples


# =========================================================
# GROUPED HOLDOUT SPLIT
# =========================================================

def grouped_split(samples):
    """
    Split by source frame rather than individual crop.

    This reduces the chance that two nearly identical
    crops from the same frame appear in both train and val.

    Very rare classes remain training-only and are
    explicitly reported as unvalidated.
    """

    rng = random.Random(
        RANDOM_SEED
    )

    by_rank = defaultdict(list)

    for sample in samples:
        by_rank[
            sample["rank"]
        ].append(sample)

    train = []
    val = []
    unvalidated = []

    for rank in RANKS:
        rank_samples = (
            by_rank.get(
                rank,
                []
            )
        )

        groups = defaultdict(list)

        for sample in rank_samples:
            groups[
                sample["group"]
            ].append(sample)

        group_keys = list(
            groups.keys()
        )

        rng.shuffle(
            group_keys
        )

        sample_count = len(
            rank_samples
        )

        group_count = len(
            group_keys
        )

        if (
            sample_count >= 8
            and group_count >= 3
        ):
            validation_groups = max(
                1,
                round(
                    group_count
                    * 0.20
                ),
            )

        elif (
            sample_count >= 5
            and group_count >= 2
        ):
            validation_groups = 1

        else:
            validation_groups = 0

        if validation_groups == 0:
            unvalidated.append(
                rank
            )

            train.extend(
                rank_samples
            )

            continue

        val_group_names = set(
            group_keys[
                :validation_groups
            ]
        )

        for sample in rank_samples:
            if (
                sample["group"]
                in val_group_names
            ):
                val.append(sample)
            else:
                train.append(sample)

    return (
        train,
        val,
        unvalidated,
    )


# =========================================================
# IMAGE PREPROCESSING
# =========================================================

def resize_with_padding(gray):
    """
    Preserve aspect ratio and place the glyph on
    a white 64x64 canvas.
    """

    height, width = gray.shape[:2]

    if (
        height <= 0
        or width <= 0
    ):
        return None

    available = (
        IMAGE_SIZE - 8
    )

    scale = min(
        available / width,
        available / height,
    )

    new_width = max(
        1,
        int(
            width * scale
        ),
    )

    new_height = max(
        1,
        int(
            height * scale
        ),
    )

    resized = cv2.resize(
        gray,
        (
            new_width,
            new_height,
        ),
        interpolation=cv2.INTER_CUBIC,
    )

    canvas = np.full(
        (
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        255,
        dtype=np.uint8,
    )

    x = (
        IMAGE_SIZE
        - new_width
    ) // 2

    y = (
        IMAGE_SIZE
        - new_height
    ) // 2

    canvas[
        y:y + new_height,
        x:x + new_width
    ] = resized

    return canvas


def augment_gray(image):
    """
    Mild augmentation matching the kinds of variation
    we expect from detection crops.

    We deliberately avoid extreme augmentation because
    the interface itself is visually stable.
    """

    height, width = (
        image.shape[:2]
    )

    # Small rotation.
    angle = random.uniform(
        -4.0,
        4.0,
    )

    # Small scale change.
    scale = random.uniform(
        0.90,
        1.10,
    )

    # Small translation.
    tx = random.uniform(
        -3.0,
        3.0,
    )

    ty = random.uniform(
        -3.0,
        3.0,
    )

    matrix = cv2.getRotationMatrix2D(
        (
            width / 2,
            height / 2,
        ),
        angle,
        scale,
    )

    matrix[0, 2] += tx
    matrix[1, 2] += ty

    image = cv2.warpAffine(
        image,
        matrix,
        (
            width,
            height,
        ),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )

    # Occasional slight blur.
    if random.random() < 0.20:
        image = cv2.GaussianBlur(
            image,
            (
                3,
                3,
            ),
            0,
        )

    # Slight brightness change.
    if random.random() < 0.50:
        delta = random.randint(
            -15,
            15,
        )

        image = np.clip(
            image.astype(
                np.int16
            ) + delta,
            0,
            255,
        ).astype(
            np.uint8
        )

    return image


def preprocess_image(
    path,
    augment=False,
):
    image = cv2.imread(str(path))

    if image is None:
        raise RuntimeError(
            f"Could not read: {path}"
        )

    h, w = image.shape[:2]

    # Undo most of the padding that was added
    # when rank candidates were saved.
    trim_x = int(w * 0.11)
    trim_y = int(h * 0.11)

    if (
        w - 2 * trim_x > 2
        and h - 2 * trim_y > 2
    ):
        image = image[
            trim_y:h - trim_y,
            trim_x:w - trim_x,
        ]

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    gray = cv2.normalize(
        gray,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    )

    h, w = gray.shape[:2]

    available = IMAGE_SIZE - 8

    scale = min(
        available / max(w, 1),
        available / max(h, 1),
    )

    new_w = max(
        1,
        round(w * scale),
    )

    new_h = max(
        1,
        round(h * scale),
    )

    gray = cv2.resize(
        gray,
        (new_w, new_h),
        interpolation=cv2.INTER_CUBIC,
    )

    canvas = np.full(
        (IMAGE_SIZE, IMAGE_SIZE),
        255,
        dtype=np.uint8,
    )

    x = (
        IMAGE_SIZE - new_w
    ) // 2

    y = (
        IMAGE_SIZE - new_h
    ) // 2

    canvas[
        y:y + new_h,
        x:x + new_w
    ] = gray

    if augment:
        canvas = augment_gray(canvas)

    # White background -> 0
    # dark glyph -> positive values
    canvas = (
        255.0
        - canvas.astype(np.float32)
    ) / 255.0

    return torch.from_numpy(
        canvas
    ).unsqueeze(0)

# =========================================================
# DATASET
# =========================================================

class RankDataset(Dataset):
    def __init__(
        self,
        samples,
        augment=False,
    ):
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(
            self.samples
        )

    def __getitem__(
        self,
        index,
    ):
        sample = (
            self.samples[index]
        )

        image = preprocess_image(
            sample["path"],
            augment=self.augment,
        )

        label = RANK_TO_INDEX[
            sample["rank"]
        ]

        return (
            image,
            label,
            str(
                sample["path"]
            ),
        )


# =========================================================
# CNN
# =========================================================

class RankCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(
                (
                    4,
                    4,
                )
            ),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                128 * 4 * 4,
                256,
            ),

            nn.ReLU(),

            nn.Dropout(
                0.35
            ),

            nn.Linear(
                256,
                len(RANKS),
            ),
        )

    def forward(
        self,
        x,
    ):
        x = self.features(
            x
        )

        return self.classifier(
            x
        )


# =========================================================
# CLASS WEIGHTS
# =========================================================

def make_class_weights(
    train_samples,
):
    counts = Counter(
        sample["rank"]
        for sample
        in train_samples
    )

    weights = []

    total = len(
        train_samples
    )

    for rank in RANKS:
        count = counts.get(
            rank,
            0,
        )

        if count == 0:
            weight = 0.0
        else:
            weight = (
                total
                / (
                    len(RANKS)
                    * count
                )
            )

        weights.append(
            weight
        )

    return torch.tensor(
        weights,
        dtype=torch.float32,
        device=DEVICE,
    )


# =========================================================
# EVALUATION
# =========================================================

def evaluate(
    model,
    loader,
):
    model.eval()

    actual = []
    predicted = []
    confidence_values = []
    paths = []

    with torch.no_grad():

        for (
            images,
            labels,
            batch_paths,
        ) in loader:

            images = images.to(
                DEVICE
            )

            logits = model(
                images
            )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            confidence, predictions = (
                probabilities.max(
                    dim=1
                )
            )

            actual.extend(
                labels.numpy().tolist()
            )

            predicted.extend(
                predictions
                .cpu()
                .numpy()
                .tolist()
            )

            confidence_values.extend(
                confidence
                .cpu()
                .numpy()
                .tolist()
            )

            paths.extend(
                batch_paths
            )

    return (
        actual,
        predicted,
        confidence_values,
        paths,
    )


def accuracy_score(
    actual,
    predicted,
):
    if not actual:
        return 0.0

    correct = sum(
        int(a == p)
        for a, p
        in zip(
            actual,
            predicted,
        )
    )

    return (
        correct
        / len(actual)
    )


# =========================================================
# TRAINING
# =========================================================

def main():
    samples = load_all_samples()

    print()
    print(
        f"Usable labeled crops: "
        f"{len(samples)}"
    )

    counts = Counter(
        sample["rank"]
        for sample
        in samples
    )

    print()
    print(
        "Class distribution:"
    )

    for rank in RANKS:
        print(
            f"{rank:>2}: "
            f"{counts.get(rank, 0)}"
        )

    (
        train_samples,
        val_samples,
        unvalidated,
    ) = grouped_split(
        samples
    )

    print()
    print(
        f"Training samples: "
        f"{len(train_samples)}"
    )

    print(
        f"Holdout samples: "
        f"{len(val_samples)}"
    )

    print()

    print(
        "Device:",
        DEVICE,
    )

    if unvalidated:
        print()
        print(
            "Training-only / not independently "
            "validated:"
        )

        print(
            ", ".join(
                unvalidated
            )
        )

    train_dataset = RankDataset(
        train_samples,
        augment=True,
    )

    val_dataset = RankDataset(
        val_samples,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = RankCNN().to(
        DEVICE
    )

    class_weights = make_class_weights(
        train_samples
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.05,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=5,
        )
    )

    best_accuracy = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    print()
    print(
        "Training CNN..."
    )
    print()

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        model.train()

        running_loss = 0.0
        training_correct = 0
        training_total = 0

        for (
            images,
            labels,
            _,
        ) in train_loader:

            images = images.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )

            optimizer.zero_grad()

            logits = model(
                images
            )

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = (
                logits.argmax(
                    dim=1
                )
            )

            training_correct += (
                predictions
                .eq(labels)
                .sum()
                .item()
            )

            training_total += (
                labels.size(0)
            )

        train_loss = (
            running_loss
            / max(
                training_total,
                1,
            )
        )

        train_accuracy = (
            training_correct
            / max(
                training_total,
                1,
            )
        )

        (
            val_actual,
            val_predicted,
            _,
            _,
        ) = evaluate(
            model,
            val_loader,
        )

        val_accuracy = accuracy_score(
            val_actual,
            val_predicted,
        )

        scheduler.step(
            val_accuracy
        )

        print(
            f"Epoch {epoch:03d} | "
            f"loss={train_loss:.4f} | "
            f"train_acc={train_accuracy:.3f} | "
            f"val_acc={val_accuracy:.3f}"
        )

        if (
            val_accuracy
            > best_accuracy
        ):

            best_accuracy = (
                val_accuracy
            )

            best_epoch = epoch

            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "ranks":
                        RANKS,

                    "image_size":
                        IMAGE_SIZE,

                    "best_val_accuracy":
                        best_accuracy,

                    "epoch":
                        best_epoch,
                },
                MODEL_PATH,
            )

        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):
            print()
            print(
                "Early stopping."
            )
            break

    # =====================================================
    # LOAD BEST MODEL
    # =====================================================

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    # =====================================================
    # FINAL HOLDOUT RESULTS
    # =====================================================

    (
        actual,
        predicted,
        confidences,
        paths,
    ) = evaluate(
        model,
        val_loader,
    )

    final_accuracy = accuracy_score(
        actual,
        predicted,
    )

    print()
    print(
        "=============================="
    )

    print(
        "CNN RANK CLASSIFIER RESULTS"
    )

    print(
        "=============================="
    )

    print()

    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    print(
        f"Holdout accuracy: "
        f"{final_accuracy:.3f}"
    )

    print()

    # =====================================================
    # PER-CLASS ACCURACY
    # =====================================================

    class_correct = Counter()
    class_total = Counter()

    for a, p in zip(
        actual,
        predicted,
    ):
        actual_rank = (
            INDEX_TO_RANK[a]
        )

        class_total[
            actual_rank
        ] += 1

        if a == p:
            class_correct[
                actual_rank
            ] += 1

    print(
        "Per-class holdout accuracy:"
    )

    for rank in RANKS:

        total = class_total.get(
            rank,
            0,
        )

        if total == 0:
            print(
                f"{rank:>2}: "
                f"not validated"
            )
            continue

        correct = class_correct.get(
            rank,
            0,
        )

        print(
            f"{rank:>2}: "
            f"{correct}/{total} "
            f"({correct / total:.3f})"
        )

    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    matrix = np.zeros(
        (
            len(RANKS),
            len(RANKS),
        ),
        dtype=int,
    )

    for a, p in zip(
        actual,
        predicted,
    ):
        matrix[a, p] += 1

    with CONFUSION_CSV.open(
        "w",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "actual/predicted",
                *RANKS,
            ]
        )

        for i, rank in enumerate(
            RANKS
        ):
            writer.writerow(
                [
                    rank,
                    *matrix[i].tolist(),
                ]
            )

    # =====================================================
    # INDIVIDUAL PREDICTIONS
    # =====================================================

    with PREDICTIONS_CSV.open(
        "w",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "path",
                "actual",
                "predicted",
                "confidence",
                "correct",
            ]
        )

        for (
            path,
            a,
            p,
            confidence,
        ) in zip(
            paths,
            actual,
            predicted,
            confidences,
        ):

            writer.writerow(
                [
                    path,
                    INDEX_TO_RANK[a],
                    INDEX_TO_RANK[p],
                    confidence,
                    int(a == p),
                ]
            )

    print()
    print(
        "Mistakes:"
    )

    mistakes = 0

    for (
        path,
        a,
        p,
        confidence,
    ) in zip(
        paths,
        actual,
        predicted,
        confidences,
    ):

        if a == p:
            continue

        mistakes += 1

        print(
            f"{Path(path).name}: "
            f"actual={INDEX_TO_RANK[a]} "
            f"predicted={INDEX_TO_RANK[p]} "
            f"confidence={confidence:.3f}"
        )

    if mistakes == 0:
        print(
            "None."
        )

    print()
    print(
        "Saved model:"
    )

    print(
        MODEL_PATH
    )

    print()
    print(
        "Saved confusion matrix:"
    )

    print(
        CONFUSION_CSV
    )

    print()
    print(
        "Saved holdout predictions:"
    )

    print(
        PREDICTIONS_CSV
    )


if __name__ == "__main__":
    main()