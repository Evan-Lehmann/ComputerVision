"""Compare CPU, MPS and incremental replay identities/counts on saved frames."""
import argparse
import json
import resource
import statistics
import time
from pathlib import Path
import numpy as np
from round_pipeline import VisionEngine, process_frames


def signature(result):
    return dict(cards=result['cards'], count_change=result['count_change'],
                known_count_change=result['known_count_change'], coverage_pending=result['coverage_pending'],
                boundaries=result['boundaries'], rounds=result['rounds'],
                events=[{k:e.get(k) for k in ['id','rank','suit','region','round_number',
                         'observation_frames','observation_boxes','merged_from','identity_status']}
                        for e in result['events']])


def compare(reference, candidate):
    # Numerical backend differences are allowed only in detector coordinates.
    a, b = signature(reference), signature(candidate)
    delta = 0.0
    if len(a['events']) != len(b['events']):
        return dict(matches=False, max_box_delta_pixels=None)
    for left, right in zip(a['events'], b['events']):
        x, y = left.pop('observation_boxes'), right.pop('observation_boxes')
        if x is None or y is None:
            if x != y:return dict(matches=False, max_box_delta_pixels=None)
        else:
            x, y = np.asarray(x), np.asarray(y)
            if x.shape != y.shape:return dict(matches=False, max_box_delta_pixels=None)
            if x.size:delta=max(delta,float(np.max(np.abs(x-y))))
    return dict(matches=a==b and delta<=.001, max_box_delta_pixels=delta,
                box_tolerance_pixels=.001)


def validate(folder, output):
    paths = sorted(folder.glob('frame_*.png'))
    report, results = dict(folder=str(folder), frames=len(paths), runs=[]), []
    for device, streaming in [('cpu',False),('mps',False),('mps',True)]:
        engine=VisionEngine(device=device)
        stamps=[]
        def publish(result):
            stamps.append(time.perf_counter())
            assert not result['count_verified']
            assert result['frames_processed']==len(stamps)
        start=time.perf_counter()
        result=process_frames([] if streaming else paths,output/(device+str(streaming)),engine=engine,
                              frame_stream=iter(paths) if streaming else None,
                              live_progress=publish if streaming else None)
        elapsed=time.perf_counter()-start
        results.append(result)
        durations=[b-a for a,b in zip([start]+stamps,stamps)]
        report['runs'].append(dict(device=device,streaming=streaming,seconds=elapsed,
            frames_per_second=len(paths)/elapsed, cards=result['cards'], count=result['count_change'],
            subtotal=result['known_count_change'], round_count=result['round_count'],
            snapshots=len(stamps), median_update_seconds=statistics.median(durations) if durations else None,
            max_update_seconds=max(durations) if durations else None,
            peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            matches_cpu=compare(results[0],result)['matches'],
            comparison=compare(results[0],result)))
        print(json.dumps(report['runs'][-1]),flush=True)
    report['all_match']=all(compare(results[0],r)['matches'] for r in results)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    parser.add_argument('--artifacts',type=Path,default=Path('round_captures/analysis/live_validation'))
    parser.add_argument('--report',type=Path,default=Path('validation/live_replay.json'))
    args=parser.parse_args()
    result=validate(args.folder,args.artifacts)
    args.report.write_text(json.dumps(result,indent=2)+'\n')
    if not result['all_match']:raise SystemExit('Replay parity failed')
