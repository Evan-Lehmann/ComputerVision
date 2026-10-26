from tracker import CardTracker


tracker = CardTracker(
    min_confidence=0.85,
    confirmation_frames=3,
    cooldown_frames=8,
)


test_stream = [
    ("4", 0.95),
    ("4", 0.96),
    ("4", 0.97),

    ("4", 0.95),
    ("4", 0.96),
    ("4", 0.96),

    (None, 0.20),
    (None, 0.20),

    ("Q", 0.91),
    ("Q", 0.94),
    ("Q", 0.95),
]


for frame_number, (rank, confidence) in enumerate(
    test_stream,
    start=1,
):
    event = tracker.update(
        rank,
        confidence
    )

    print(
        f"frame={frame_number:02d} "
        f"prediction={rank} "
        f"confidence={confidence:.2f} "
        f"event={event}"
    )