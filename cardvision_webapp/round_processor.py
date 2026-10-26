"""Adapter from saved screenshots to the current vision pipeline."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from round_pipeline import COUNT_VALUES, process_frames


def calculate_count_change(cards):
    return sum(COUNT_VALUES.get(rank, 0) for rank in cards)


def process_round(round_dir, output_dir=None, regions=None, progress=None, engine=None):
    round_dir = Path(round_dir)
    frames = sorted(p for p in round_dir.iterdir() if p.suffix.lower() in ['.png', '.jpg', '.jpeg'])
    return process_frames(frames, output_dir or round_dir/'analysis', engine=engine,
                          regions=regions, progress=progress)
