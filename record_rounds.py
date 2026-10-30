"""Save separate screen recordings without running analysis or Flask."""
import argparse
from pathlib import Path
from queue import Empty, Queue
import time

from cardvision_webapp.round_capture import RoundCapture


def record_sessions(capture, countdown=3, prompt=input, emit=print, wait=time.sleep):
    """Enter starts/stops each recording; q quits between recordings."""
    emit('Capture only: no analysis runs. Each recording gets a separate folder.')
    while True:
        choice = prompt('\nPress Enter to prepare a recording, or q then Enter to quit: ').strip().lower()
        if choice == 'q':
            return
        if choice:
            emit('Use Enter to record, or q to quit.')
            continue
        for remaining in range(countdown, 0, -1):
            emit('Switch to the table. Recording starts in %d...' % remaining)
            wait(1)
        directory = capture.start()
        emit('Recording to ' + str(directory))
        try:
            prompt('Return here and press Enter to STOP and save this recording. ')
        finally:
            saved = capture.stop()
            emit('Saved %d frames to %s' % (saved['frames'], saved['round_dir']))


def record_hotkeys(capture, countdown=3, listener_factory=None, emit=print, clock=time.monotonic):
    """Global callbacks enqueue commands; only the main thread operates capture."""
    if listener_factory is None:
        try:
            from pynput.keyboard import GlobalHotKeys
        except ImportError as exc:
            raise RuntimeError('Hotkeys need pynput. Run: python -m pip install pynput') from exc
        listener_factory = GlobalHotKeys
    commands = Queue()
    last_toggle = [float('-inf')]

    def toggle():
        now = clock()
        if now - last_toggle[0] >= .35:
            last_toggle[0] = now
            commands.put('toggle')

    quit_recording = lambda: commands.put('quit')
    # macOS may deliver the Option-modified characters instead of s/q.
    bindings = {key: toggle for key in ['<alt>+s', '<alt>+ß', '<ctrl>+<alt>+s']}
    bindings.update({key: quit_recording for key in ['<alt>+q', '<alt>+œ', '<ctrl>+<alt>+q']})
    emit('Option+S: start/stop from any window. Option+Q: save and quit.')
    emit('Capture only; each recording gets its own folder in ' + str(capture.rounds_dir))
    active = False

    def save():
        saved = capture.stop()
        emit('Saved %d frames to %s' % (saved['frames'], saved['round_dir']))

    try:
        with listener_factory(bindings) as listener:
            while True:
                try:
                    command = commands.get(timeout=.1)
                except Empty:
                    if not listener.is_alive():
                        raise RuntimeError('Hotkey listener stopped. Check macOS Accessibility/Input Monitoring permission.')
                    continue
                if command == 'quit':
                    break
                if active:
                    active = False
                    save()
                    continue
                cancelled = False
                for remaining in range(countdown, 0, -1):
                    emit('Recording starts in %d... Option+S cancels.' % remaining)
                    try:
                        command = commands.get(timeout=1)
                        cancelled = True
                        break
                    except Empty:
                        pass
                if cancelled:
                    if command == 'quit':
                        break
                    emit('Countdown cancelled; nothing recorded.')
                    continue
                directory = capture.start()
                active = True
                emit('Recording to ' + str(directory))
    finally:
        if active:
            save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fps', type=float, default=2.0)
    parser.add_argument('--monitor', type=int, default=1)
    parser.add_argument('--countdown', type=int, default=3)
    parser.add_argument('--hotkeys', action='store_true', help='Use global Option+S / Option+Q instead of terminal Enter')
    parser.add_argument('--output', type=Path, help='Parent folder for separate recordings (default: round_captures)')
    args = parser.parse_args()
    if args.countdown < 0:
        parser.error('Countdown must be zero or greater.')
    try:
        capture = RoundCapture(fps=args.fps, monitor_index=args.monitor, rounds_dir=args.output)
        if args.hotkeys:
            record_hotkeys(capture, countdown=args.countdown)
        else:
            record_sessions(capture, countdown=args.countdown)
    except (KeyboardInterrupt, EOFError):
        print('\nRecorder closed.')
    except (RuntimeError, ValueError) as exc:
        parser.exit(1, str(exc) + '\n')


if __name__ == '__main__':
    main()
