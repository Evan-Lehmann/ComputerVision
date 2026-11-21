"""Benchmark real tiled detection on this host; no settings are selected automatically."""
import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path
import cv2
import torch
from round_pipeline import VisionEngine


def benchmark(paths, repeats=3):
    images = [cv2.imread(str(p)) for p in paths]
    if not images or any(image is None for image in images):
        raise ValueError('Supply readable representative frames.')
    report = dict(host=platform.platform(), torch=torch.__version__,
                  mps_available=torch.backends.mps.is_available(), repeats=repeats,
                  frames=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                               shape=list(im.shape)) for p, im in zip(paths, images)], runs=[])
    baseline = None
    for device in ['cpu', 'mps']:
        for batch in [1, 2, 4, 8]:
            row = dict(device=device, tile_batch_size=batch)
            if device == 'mps' and not report['mps_available']:
                row['skipped'] = 'MPS unavailable to this process'
                report['runs'].append(row)
                continue
            try:
                engine = VisionEngine(device=device, tile_batch_size=batch)
                engine.detect(images[0], paths[0])
                timings, outputs = [], []
                for _ in range(repeats):
                    for path, image in zip(paths, images):
                        if device == 'mps': torch.mps.synchronize()
                        start = time.perf_counter()
                        detections = engine.detect(image, path)
                        if device == 'mps': torch.mps.synchronize()
                        timings.append(time.perf_counter() - start)
                        outputs.append([[round(d[k], 3) for k in ['x1','y1','x2','y2','confidence']] for d in detections])
                if baseline is None: baseline = outputs
                row.update(median_seconds=statistics.median(timings),
                           max_seconds=max(timings), samples_seconds=timings,
                           detection_counts=[len(x) for x in outputs], exact_rounded_baseline=outputs == baseline)
                if device == 'mps':
                    row['mps_allocated_bytes'] = torch.mps.driver_allocated_memory()
            except Exception as exc:
                row['error'] = str(exc)
            report['runs'].append(row)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('frames', nargs='+', type=Path)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, default=Path('validation/live_benchmark.json'))
    args = parser.parse_args()
    if args.repeats < 1: parser.error('repeats must be positive')
    result = benchmark(args.frames, args.repeats)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
