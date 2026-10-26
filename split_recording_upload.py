"""Split a recording ZIP into independent ZIPs small enough to upload."""
import argparse
from pathlib import Path
import shutil
import tempfile
import zipfile


def split_archive(source, max_bytes=24 * 1024**2):
    source = Path(source)
    with zipfile.ZipFile(source) as archive:
        entries = [item for item in archive.infolist()
                   if not item.is_dir() and '__MACOSX' not in item.filename.split('/')
                   and Path(item.filename).name != '.DS_Store']
        if not entries:
            raise ValueError('The archive contains no files to upload.')
        for item in entries:
            # Stored ZIP headers plus a conservative allowance for metadata.
            required = item.file_size + 2 * len(item.filename.encode('utf-8')) + 256
            if required + 128 > max_bytes:
                raise ValueError('One file exceeds the part limit: ' + item.filename)
        output = Path(tempfile.mkdtemp(prefix=source.stem + '_parts_', dir=source.parent))
        parts, target, size = [], None, 128
        try:
            for item in entries:
                required = item.file_size + 2 * len(item.filename.encode('utf-8')) + 256
                if target is None or size + required > max_bytes:
                    if target:
                        target.close()
                    path = output / ('part_%02d.zip' % (len(parts) + 1))
                    parts.append(path)
                    target = zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_STORED)
                    size = 128
                with archive.open(item) as reader, target.open(item.filename, 'w') as writer:
                    shutil.copyfileobj(reader, writer, length=512 * 1024)
                size += required
        finally:
            if target:
                target.close()
        return parts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    args = parser.parse_args()
    try:
        parts = split_archive(args.archive)
    except (ValueError, OSError, zipfile.BadZipFile) as exc:
        parser.exit(1, str(exc) + '\n')
    print('Original archive preserved. Upload these independent ZIPs:')
    for path in parts:
        print('%s (%.1f MiB)' % (path, path.stat().st_size / 1024**2))


if __name__ == '__main__':
    main()
