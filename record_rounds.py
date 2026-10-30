"""Save separate screen recordings without running analysis or Flask."""
import argparse
from pathlib import Path
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fps', type=float, default=2.0)
    parser.add_argument('--monitor', type=int, default=1)
    parser.add_argument('--countdown', type=int, default=3)
    parser.add_argument('--output', type=Path, help='Parent folder for separate recordings (default: round_captures)')
    args = parser.parse_args()
    if args.countdown < 0:
        parser.error('Countdown must be zero or greater.')
    try:
        capture = RoundCapture(fps=args.fps, monitor_index=args.monitor, rounds_dir=args.output)
        record_sessions(capture, countdown=args.countdown)
    except (KeyboardInterrupt, EOFError):
        print('\nRecorder closed.')
    except (RuntimeError, ValueError) as exc:
        parser.exit(1, str(exc) + '\n')


if __name__ == '__main__':
    main()
