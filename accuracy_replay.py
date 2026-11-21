"""Replay development recordings with a content-addressed detector cache.

Caches and evidence stay in ignored round_captures/accuracy_work. Baseline and
candidate results are separate. Never treats an already inspected round as held out.
"""
import argparse
import hashlib
import json
import time
import subprocess
import sys
import types
from pathlib import Path
import cv2
from round_pipeline import VisionEngine, process_frames


class CachedEngine:
    def __init__(self, cache_dir, device='auto'):
        self.cache_dir=Path(cache_dir);self.cache_dir.mkdir(parents=True,exist_ok=True)
        self.engine=VisionEngine(device=device)
        context=dict(detector_sha256=hashlib.sha256(Path('models/rank_detector_v2.pt').read_bytes()).hexdigest(),
                     detector_code_sha256=hashlib.sha256(Path('test_rank_detector_unseen.py').read_bytes()).hexdigest(),
                     device=self.engine.device, tile_batch_size=self.engine.tile_batch_size)
        manifest=self.cache_dir/'context.json'
        if manifest.exists() and json.loads(manifest.read_text())!=context:
            raise ValueError('Detector cache context changed. Choose a fresh --work directory.')
        manifest.write_text(json.dumps(context,indent=2)+'\n')
    def detect(self,image,path):
        # All recordings in a work directory must retain their captured size.
        shape_file=self.cache_dir/'shape.json'
        shape=list(image.shape)
        if shape_file.exists() and json.loads(shape_file.read_text())!=shape:
            raise ValueError('Frame shape changed. Choose a fresh --work directory.')
        if not shape_file.exists():shape_file.write_text(json.dumps(shape))
        key=hashlib.sha256(image.tobytes()).hexdigest()
        file=self.cache_dir/(key+'.json')
        if file.exists():return json.loads(file.read_text())
        detections=self.engine.detect(image,path)
        file.write_text(json.dumps(detections))
        return detections
    def classify(self,crops):return self.engine.classify(crops)


def published_vision(revision):
    """Load the published association/face baseline without altering the checkout."""
    import round_pipeline
    for name,file in [('_baseline_faces','card_faces.py'),('_baseline_tracking','card_tracking.py')]:
        source=subprocess.check_output(['git','show',revision+':'+file],text=True)
        if file=='card_tracking.py':
            source=source.replace('from card_faces import','from _baseline_faces import')
        module=types.ModuleType(name)
        sys.modules[name]=module
        exec(compile(source,revision+':'+file,'exec'),module.__dict__)
    faces,tracking=sys.modules['_baseline_faces'],sys.modules['_baseline_tracking']
    round_pipeline.reconcile_faces=faces.reconcile_faces
    round_pipeline.CardIdentityMatcher=tracking.CardIdentityMatcher
    round_pipeline.merge_track_fragments=tracking.merge_track_fragments
    round_pipeline.face_rank_crop=lambda image,detection:None


def run(names, phase, work, device='auto', streaming=False, source=Path('round_captures/v3_new_rounds'), limit=None):
    engine=CachedEngine(work/'detections',device)
    rows=[]
    for name in names:
        frames=sorted((source/name).glob('frame_*.png'))
        if limit is not None:
            frames=frames[:limit]
        start=time.perf_counter()
        result=process_frames([] if streaming else frames,work/phase/name,engine=engine,
            frame_stream=iter(frames) if streaming else None,
            live_progress=(lambda result:None) if streaming else None)
        row=dict(recording=name,frames=len(frames),seconds=time.perf_counter()-start,
            candidates=len(result['events']),unknowns=result['unresolved'],cards=result['cards'],
            suits=[e['suit'] for e in result['events']],count=result['count_change'],
            known_subtotal=result['known_count_change'],coverage_pending=result['coverage_pending'])
        rows.append(row);print(json.dumps(row),flush=True)
    (work/(phase+'.json')).write_text(json.dumps(rows,indent=2)+'\n')
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',required=True)
    parser.add_argument('--source',type=Path,default=Path('round_captures/v3_new_rounds'))
    parser.add_argument('--baseline-ref',help='Use published face/tracker code from this Git revision')
    parser.add_argument('--rounds',nargs='+',default=['r'+str(i) for i in range(1,14)])
    parser.add_argument('--work',type=Path,default=Path('round_captures/accuracy_work'))
    parser.add_argument('--device',choices=['auto','cpu','mps'],default='auto')
    parser.add_argument('--streaming',action='store_true')
    parser.add_argument('--limit',type=int,help='Replay only the first N frames of each recording')
    parser.add_argument('--disable-visibility-check',action='store_true',help='Diagnostic baseline only: allow other windows to supply detections')
    args=parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if args.disable_visibility_check:
        import round_pipeline
        round_pipeline.visible_table=lambda image,regions:True
    if args.baseline_ref:published_vision(args.baseline_ref)
    run(args.rounds,args.phase,args.work,args.device,args.streaming,args.source,args.limit)
