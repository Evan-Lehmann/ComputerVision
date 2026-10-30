import unittest
from unittest.mock import Mock
from pathlib import Path
from record_rounds import record_sessions, record_hotkeys


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

    def hotkey_run(self, keys, countdown=0, times=None):
        capture=Mock();capture.rounds_dir=Path('round_captures')
        capture.start.side_effect=[Path('round_1'),Path('round_2')]
        capture.stop.side_effect=[dict(frames=10,round_dir='round_1'),dict(frames=20,round_dir='round_2')]
        class Listener:
            def __init__(self,bindings):self.bindings=bindings
            def __enter__(self):
                for key in keys:self.bindings[key]()
                return self
            def __exit__(self,*args):pass
            def is_alive(self):return True
        emit=Mock()
        record_hotkeys(capture,countdown=countdown,listener_factory=Listener,emit=emit,
                       clock=Mock(side_effect=times or range(1,20)))
        return capture,emit

    def test_global_hotkeys_make_separate_recordings_and_quit_saves_active_one(self):
        capture,emit=self.hotkey_run(['<alt>+s','<alt>+s','<alt>+s','<alt>+q'])
        self.assertEqual(capture.start.call_count,2)
        self.assertEqual(capture.stop.call_count,2)
        emit.assert_any_call('Saved 20 frames to round_2')

    def test_macos_option_characters_and_legacy_shortcut_work(self):
        capture,_=self.hotkey_run(['<alt>+ß','<ctrl>+<alt>+s','<alt>+œ'])
        capture.start.assert_called_once();capture.stop.assert_called_once()

    def test_duplicate_shortcut_callbacks_do_not_start_and_immediately_stop(self):
        capture,_=self.hotkey_run(['<alt>+s','<ctrl>+<alt>+s','<alt>+q'],times=[1,1.01])
        capture.start.assert_called_once();capture.stop.assert_called_once()

    def test_toggle_or_quit_cancels_countdown_without_creating_a_recording(self):
        for keys in [['<alt>+s','<alt>+s','<alt>+q'],['<alt>+s','<alt>+q']]:
            capture,_=self.hotkey_run(keys,countdown=3)
            capture.start.assert_not_called();capture.stop.assert_not_called()


if __name__ == '__main__':
    unittest.main()
