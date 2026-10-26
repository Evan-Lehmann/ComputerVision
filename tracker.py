from collections import deque


class CardTracker:
    def __init__(
        self,
        min_confidence=0.85,
        confirmation_frames=3,
        cooldown_frames=8,
    ):
        self.min_confidence = min_confidence
        self.confirmation_frames = confirmation_frames
        self.cooldown_frames = cooldown_frames

        self.recent_predictions = deque(
            maxlen=confirmation_frames
        )

        self.last_emitted_rank = None
        self.frames_since_emit = cooldown_frames


    def update(self, rank, confidence):
        """
        Returns a card rank only when a new card has been
        confirmed strongly enough.

        Otherwise returns None.
        """

        self.frames_since_emit += 1

        # Ignore uncertain recognition
        if (
            rank is None
            or confidence < self.min_confidence
        ):
            self.recent_predictions.clear()
            return None

        self.recent_predictions.append(rank)

        # Need enough consecutive observations
        if len(self.recent_predictions) < self.confirmation_frames:
            return None

        # Require all recent predictions to agree
        if len(set(self.recent_predictions)) != 1:
            return None

        confirmed_rank = self.recent_predictions[0]

        # Prevent repeatedly emitting the same visible card
        if (
            confirmed_rank == self.last_emitted_rank
            and self.frames_since_emit < self.cooldown_frames
        ):
            return None

        self.last_emitted_rank = confirmed_rank
        self.frames_since_emit = 0

        self.recent_predictions.clear()

        return confirmed_rank