from pathlib import Path
from datetime import datetime
import threading
import time

import cv2
import numpy as np
from mss import mss


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROUNDS_DIR = PROJECT_ROOT / "round_captures"

ROUNDS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


class RoundCapture:
    def __init__(
        self,
        fps=2.0,
        monitor_index=1,
    ):
        self.fps = fps
        self.monitor_index = monitor_index

        self.recording = False
        self.thread = None

        self.round_dir = None
        self.frame_count = 0

        self.lock = threading.Lock()


    def start(self):
        with self.lock:
            if self.recording:
                raise RuntimeError(
                    "A round is already recording."
                )

            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            self.round_dir = (
                ROUNDS_DIR
                / f"round_{timestamp}"
            )

            self.round_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            self.frame_count = 0
            self.recording = True

            self.thread = threading.Thread(
                target=self._capture_loop,
                daemon=True,
            )

            self.thread.start()

            return self.round_dir


    def stop(self):
        with self.lock:
            if not self.recording:
                raise RuntimeError(
                    "No round is currently recording."
                )

            self.recording = False

        if self.thread:
            self.thread.join()

        return {
            "round_dir": str(
                self.round_dir
            ),
            "frames": self.frame_count,
        }


    def _capture_loop(self):
        frame_delay = (
            1.0 / self.fps
        )

        with mss() as sct:
            monitors = sct.monitors

            if (
                self.monitor_index
                >= len(monitors)
            ):
                raise RuntimeError(
                    f"Monitor "
                    f"{self.monitor_index} "
                    f"does not exist. "
                    f"Available monitors: "
                    f"{len(monitors) - 1}"
                )

            monitor = monitors[
                self.monitor_index
            ]

            while True:
                with self.lock:
                    if not self.recording:
                        break

                started = time.time()

                screenshot = sct.grab(
                    monitor
                )

                frame = np.array(
                    screenshot
                )

                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGRA2BGR,
                )

                self.frame_count += 1

                filename = (
                    self.round_dir
                    / (
                        f"frame_"
                        f"{self.frame_count:06d}"
                        f".png"
                    )
                )

                cv2.imwrite(
                    str(filename),
                    frame
                )

                elapsed = (
                    time.time()
                    - started
                )

                sleep_time = (
                    frame_delay
                    - elapsed
                )

                if sleep_time > 0:
                    time.sleep(
                        sleep_time
                    )