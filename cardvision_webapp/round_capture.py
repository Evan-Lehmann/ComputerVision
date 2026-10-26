"""Screen capture with startup readiness and visible worker errors."""
from datetime import datetime
from pathlib import Path
import threading
import time
import uuid
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROUNDS_DIR = PROJECT_ROOT / 'round_captures'


class RoundCapture:
    def __init__(self, fps=2.0, monitor_index=1, rounds_dir=None, screen_factory=None):
        if fps <= 0 or monitor_index < 1:
            raise ValueError('FPS must be positive and monitor index must be at least one.')
        self.fps, self.monitor_index = fps, monitor_index
        self.rounds_dir = Path(rounds_dir or ROUNDS_DIR)
        self.screen_factory = screen_factory
        self.lock = threading.RLock()
        self.thread = None
        self.round_dir = None
        self.frame_count = 0
        self.error = None
        self.recording = False
        self._stop = threading.Event()
        self._ready = threading.Event()

    def _screen(self):
        if self.screen_factory:
            return self.screen_factory()
        from mss import mss
        return mss()

    def _monitor(self, screen):
        if self.monitor_index >= len(screen.monitors):
            raise RuntimeError('Monitor %d is unavailable; found %d monitor(s).' %
                               (self.monitor_index, len(screen.monitors) - 1))
        return screen.monitors[self.monitor_index]

    def preview(self):
        with self._screen() as screen:
            return cv2.cvtColor(np.asarray(screen.grab(self._monitor(screen))), cv2.COLOR_BGRA2BGR)

    def start(self):
        with self.lock:
            if self.thread is not None and self.thread.is_alive():
                raise RuntimeError('A recording is already active.')
            self.rounds_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.round_dir = self.rounds_dir / ('round_' + stamp + '_' + uuid.uuid4().hex[:8])
            self.round_dir.mkdir()
            self.frame_count, self.error = 0, None
            self._stop.clear()
            self._ready.clear()
            self.recording = True
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
        if not self._ready.wait(8):
            self._stop.set()
            self.thread.join(timeout=3)
            raise RuntimeError('Screen capture did not become ready. Check screen-recording permission.')
        if self.error:
            raise RuntimeError(self.error)
        return self.round_dir

    def stop(self):
        with self.lock:
            if self.thread is None:
                raise RuntimeError('No recording has started.')
            self._stop.set()
            thread = self.thread
        thread.join(timeout=10)
        if thread.is_alive():
            raise RuntimeError('Capture has not stopped yet; please retry.')
        if self.error:
            raise RuntimeError(self.error)
        if self.frame_count == 0:
            raise RuntimeError('No screenshots were saved.')
        return dict(round_dir=str(self.round_dir), frames=self.frame_count)

    def _capture_loop(self):
        try:
            with self._screen() as screen:
                monitor = self._monitor(screen)
                while not self._stop.is_set():
                    started = time.monotonic()
                    image = cv2.cvtColor(np.asarray(screen.grab(monitor)), cv2.COLOR_BGRA2BGR)
                    with self.lock:
                        filename = self.round_dir / ('frame_%06d.png' % (self.frame_count + 1))
                        if not cv2.imwrite(str(filename), image):
                            raise OSError('Could not save screenshot: ' + str(filename))
                        self.frame_count += 1
                    self._ready.set()
                    self._stop.wait(max(0, 1 / self.fps - (time.monotonic() - started)))
        except Exception as exc:
            with self.lock:
                self.error = ('Screen capture failed: %s. On macOS, allow Screen Recording for '
                              'the app running Python, then restart that app.' % exc)
        finally:
            with self.lock:
                self.recording = False
            self._ready.set()
