import unittest
from unittest.mock import Mock
from pathlib import Path
from record_rounds import record_sessions


class RecordOnlyTests(unittest.TestCase):
    def test_successive_recordings_stop_and_save_without_analysis(self):
        capture = Mock()
        capture.start.side_effect = [Path('round_1'), Path('round_2')]
        capture.stop.side_effect = [dict(frames=10, round_dir='round_1'), dict(frames=20, round_dir='round_2')]
        prompt = Mock(side_effect=['', '', '', '', 'q'])
        emit, wait = Mock(), Mock()
        record_sessions(capture, countdown=3, prompt=prompt, emit=emit, wait=wait)
        self.assertEqual(capture.start.call_count, 2)
        self.assertEqual(capture.stop.call_count, 2)
        self.assertEqual(wait.call_count, 6)
        emit.assert_any_call('Saved 10 frames to round_1')
        emit.assert_any_call('Saved 20 frames to round_2')

    def test_interrupt_during_recording_stops_and_preserves_frames(self):
        for error in [KeyboardInterrupt, EOFError]:
            capture = Mock()
            capture.stop.return_value = dict(frames=5, round_dir='saved')
            with self.assertRaises(error):
                record_sessions(capture, countdown=0, prompt=Mock(side_effect=['', error()]), emit=Mock())
            capture.stop.assert_called_once()

    def test_quit_does_not_start_capture(self):
        capture = Mock()
        record_sessions(capture, prompt=Mock(return_value='q'), emit=Mock())
        capture.start.assert_not_called()


if __name__ == '__main__':
    unittest.main()
