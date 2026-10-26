"""Runtime architecture and preprocessing for the supplied rank checkpoint."""
import cv2
import numpy as np
import torch
import torch.nn as nn

RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
IMAGE_SIZE = 64

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
                (4, 4)
            ),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                128 * 4 * 4,
                256,
            ),

            nn.ReLU(),

            nn.Dropout(0.35),

            nn.Linear(
                256,
                len(RANKS),
            ),
        )

    def forward(self, x):
        x = self.features(x)

        return self.classifier(x)

def preprocess_crop(image):
    if image is None or image.size == 0:
        return None

    h, w = image.shape[:2]

    trim_x = int(
        w * 0.11
    )

    trim_y = int(
        h * 0.11
    )

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

    available = (
        IMAGE_SIZE - 8
    )

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
        (
            new_w,
            new_h,
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
        IMAGE_SIZE - new_w
    ) // 2

    y = (
        IMAGE_SIZE - new_h
    ) // 2

    canvas[
        y:y + new_h,
        x:x + new_w
    ] = gray

    canvas = (
        255.0
        - canvas.astype(
            np.float32
        )
    ) / 255.0

    tensor = torch.from_numpy(
        canvas
    ).unsqueeze(0).unsqueeze(0)

    return tensor
