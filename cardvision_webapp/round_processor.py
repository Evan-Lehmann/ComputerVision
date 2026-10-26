from pathlib import Path

from legacy_round_recognizer import (
    recognize_round,
)


COUNT_VALUES = {
    "A": -1,

    "2": 1,
    "3": 1,
    "4": 1,
    "5": 1,
    "6": 1,

    "7": 0,
    "8": 0,
    "9": 0,

    "10": -1,
    "J": -1,
    "Q": -1,
    "K": -1,

    # Never alter the count for an uncertain card.
    "UNKNOWN": 0,
}


def calculate_count_change(cards):
    return sum(
        COUNT_VALUES.get(
            rank,
            0,
        )
        for rank in cards
    )


def process_round(round_dir):

    round_dir = Path(
        round_dir
    )

    frame_paths = sorted(
        round_dir.glob(
            "frame_*.png"
        )
    )

    print()
    print(
        ">>> PROCESSING ROUND"
    )

    print(
        f"Frames available: "
        f"{len(frame_paths)}"
    )

    cards = recognize_round(
        frame_paths
    )

    count_change = (
        calculate_count_change(
            cards
        )
    )

    print(
        f"Cards recognized: "
        f"{cards}"
    )

    print(
        f"Count change: "
        f"{count_change:+d}"
    )

    return {
        "cards": cards,

        "count_change":
            count_change,

        "frames_processed":
            len(frame_paths),
    }