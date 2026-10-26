from pathlib import Path
import tempfile
import unittest
import zipfile
from split_recording_upload import split_archive


class SplitRecordingTests(unittest.TestCase):
    def test_independent_parts_preserve_frames_analysis_and_source_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'round.zip'
            files = {'round/frame_%d.png' % i: bytes([i]) * 1000 for i in range(4)}
            files['round/analysis/result.json'] = b'{"count_change":1}'
            with zipfile.ZipFile(source, 'w') as archive:
                for name, data in files.items():
                    archive.writestr(name, data)
                archive.writestr('__MACOSX/._round', b'metadata')
            original = source.read_bytes()
            parts = split_archive(source, max_bytes=2500)
            self.assertGreater(len(parts), 1)
            restored = {}
            for part in parts:
                self.assertLessEqual(part.stat().st_size, 2500)
                with zipfile.ZipFile(part) as archive:
                    self.assertIsNone(archive.testzip())
                    for name in archive.namelist():
                        restored[name] = archive.read(name)
            self.assertEqual(restored, files)
            self.assertEqual(source.read_bytes(), original)
            self.assertNotEqual(parts[0].parent, split_archive(source, 2500)[0].parent)

    def test_oversized_entry_fails_before_creating_parts(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'round.zip'
            with zipfile.ZipFile(source, 'w') as archive:
                archive.writestr('large.png', b'x' * 4000)
            with self.assertRaisesRegex(ValueError, 'exceeds'):
                split_archive(source, 2500)
            self.assertEqual(list(Path(directory).iterdir()), [source])


if __name__ == '__main__':
    unittest.main()
