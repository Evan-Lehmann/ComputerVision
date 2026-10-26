from pathlib import Path
from datetime import datetime
import threading
import time

import mss
import mss.tools
from pynput import keyboard


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = Path("captures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Screenshots per second
CAPTURE_FPS = 2
CAPTURE_INTERVAL = 1 / CAPTURE_FPS


# ============================================================
# STATE
# ============================================================

capturing = False
running = True
capture_number = 0


# ============================================================
# SCREEN CAPTURE
# ============================================================

def capture_loop():
    global capture_number

    with mss.mss() as sct:
        # Primary monitor
        monitor = sct.monitors[1]

        while running:
            if not capturing:
                time.sleep(0.05)
                continue

            start_time = time.time()

            screenshot = sct.grab(monitor)

            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S_%f"
            )

            filename = (
                OUTPUT_DIR
                / f"frame_{timestamp}_{capture_number:06d}.png"
            )

            mss.tools.to_png(
                screenshot.rgb,
                screenshot.size,
                output=str(filename)
            )

            capture_number += 1

            elapsed = time.time() - start_time

            time.sleep(
                max(0, CAPTURE_INTERVAL - elapsed)
            )


# ============================================================
# HOTKEY ACTIONS
# ============================================================

def toggle_capture():
    global capturing

    capturing = not capturing

    if capturing:
        print("\n>>> CAPTURING")
    else:
        print("\n>>> PAUSED")


def quit_program():
    global running
    global capturing

    capturing = False
    running = False

    print("\n>>> STOPPING")


# ============================================================
# MAIN
# ============================================================

def main():
    print("--------------------------------")
    print("CardVision Screenshot Capture")
    print("--------------------------------")
    print()
    print("Control + Option + S  -> Start / pause")
    print("Control + Option + Q  -> Quit")
    print()
    print(f"Capture rate: {CAPTURE_FPS} FPS")
    print(f"Output folder: {OUTPUT_DIR.resolve()}")
    print()
    print("Waiting for hotkey...")

    capture_thread = threading.Thread(
        target=capture_loop,
        daemon=True
    )

    capture_thread.start()

    hotkeys = keyboard.GlobalHotKeys({
        "<ctrl>+<alt>+s": toggle_capture,
        "<ctrl>+<alt>+q": quit_program,
    })

    with hotkeys:
        while running:
            time.sleep(0.1)

    capture_thread.join(timeout=1)

    print()
    print(
        f"Finished. Saved {capture_number} screenshots."
    )


if __name__ == "__main__":
    main()