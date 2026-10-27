"""Benchmark a local external 52-class card-corner YOLO model on recordings.

No downloads or production-model changes. Detailed predictions remain in the
chosen output directory; --summary writes aggregate metrics only. Annotations
use evaluate_card_result.py's schema. They need not cover every visible card,
so misses/coverage are explicitly relative to those annotations.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import time

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from round_pipeline import DEFAULT_REGIONS, validate_regions

SOURCE_URL = 'https://github.com/TeogopK/Playing-Cards-Object-Detection'
SOURCE_REVISION = '6da007263c51cf5a44a3c06ea840030ff72ce678'
SYNTHETIC_SHA256 = '0b408029b1241b2489217f89b63f153d81cb0fc6a035361a87bc6e7dac5c62c9'
VARIANTS = ('full640', 'full1280', 'regions640')


def card_label(label):
    match = re.fullmatch(r'(10|[2-9AJQK])([CDHS])', str(label).upper())
    if not match:
        raise ValueError('Expected a rank/suit class, received '+str(label))
    return match.group(1), match.group(2).lower()


def validate_classes(names):
    labels = [card_label(label) for label in names.values()]
    expected = {(rank, suit) for rank in ['A','2','3','4','5','6','7','8','9','10','J','Q','K'] for suit in 'cdhs'}
    if len(labels) != 52 or set(labels) != expected or set(names) != set(range(52)):
        raise ValueError('This benchmark requires all 52 rank/suit classes; a 13-class model is not interchangeable.')


def load_external_model(path):
    """Use PyTorch's restricted loader, permitting only known installed layers."""
    import torch
    from torch import nn
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.nn.modules import SPPF, DFL, Conv, C2f, Concat, Bottleneck, Detect
    if not hasattr(torch.serialization, 'get_unsafe_globals_in_checkpoint'):
        raise RuntimeError('External checkpoint inspection requires PyTorch 2.6+; the app itself does not.')
    allowed = [DetectionModel, SPPF, DFL, Conv, C2f, Concat, Bottleneck, Detect,
               nn.BatchNorm2d, nn.SiLU, nn.ModuleList, nn.Upsample, nn.Sequential, nn.Conv2d, nn.MaxPool2d]
    known = {layer.__module__+'.'+layer.__name__ for layer in allowed}
    unexpected = set(torch.serialization.get_unsafe_globals_in_checkpoint(path))-known
    if unexpected:
        raise ValueError('Unsupported checkpoint globals: '+', '.join(sorted(unexpected)))
    prior = os.environ.get('TORCH_FORCE_WEIGHTS_ONLY_LOAD')
    os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '1'
    try:
        with torch.serialization.safe_globals(allowed):
            model = YOLO(str(path), task='detect')
    finally:
        if prior is None:
            os.environ.pop('TORCH_FORCE_WEIGHTS_ONLY_LOAD', None)
        else:
            os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = prior
    validate_classes(model.names)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    return model


def corner_overlap(rank_box, corner_box):
    """Rank-only reference boxes are smaller than rank-plus-suit predictions."""
    a,b = rank_box,corner_box
    area_a=max(0,a[2]-a[0])*max(0,a[3]-a[1])
    area_b=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    if not area_a or not area_b or intersection/area_a < .5 or intersection/area_b < .1:
        return 0.0
    return intersection/area_a


def observation_metrics(reference, frames, threshold):
    cards={card['id']:card for card in reference['cards']}
    truth=defaultdict(list)
    for row in reference['observations']:
        if row['identity'] not in cards:
            raise ValueError('Unknown annotated identity.')
        truth[row['frame'],row['region']].append(row)
    matched=correct=wrong=unmatched=predictions=0
    correct_cards=set()
    for record in frames:
        grouped=defaultdict(list)
        for detection in record['predictions']:
            if detection['confidence'] >= threshold:
                grouped[detection['region']].append(detection)
        keys={region for frame,region in truth if frame==record['frame']}|set(grouped)
        for region in keys:
            a,b=truth[record['frame'],region],grouped[region]
            predictions+=len(b)
            pairs=[]
            if a and b:
                overlaps=np.array([[corner_overlap(row['box'],d['box']) for d in b] for row in a])
                # Break geometric ties by confidence, never by the true rank.
                confidence=np.array([d['confidence'] for d in b])
                rows,columns=linear_sum_assignment(1-overlaps-confidence[None,:]*1e-6)
                pairs=[(i,j) for i,j in zip(rows,columns) if overlaps[i,j] > 0]
            matched+=len(pairs);unmatched+=len(b)-len(pairs)
            for i,j in pairs:
                if b[j]['rank']==cards[a[i]['identity']]['rank']:
                    correct+=1;correct_cards.add(a[i]['identity'])
                else:
                    wrong+=1
    total=len(reference['observations'])
    return dict(confidence_threshold=threshold,annotated_observations=total,
        predictions_in_selected_regions=predictions,matched_observations=matched,
        missed_annotated_observations=total-matched,correct_rank_observations=correct,
        wrong_rank_observations=wrong,unmatched_predictions=unmatched,
        rank_accuracy_on_matches=correct/matched if matched else None,
        correctly_recognized_observation_fraction=correct/total if total else None,
        expected_physical_cards=len(cards),physical_cards_ever_correctly_recognized=len(correct_cards))


def predict_frame(model,image,regions,variant):
    h,w=image.shape[:2]
    jobs=[(image,0,0,1280 if variant=='full1280' else 640)]
    if variant=='regions640':
        jobs=[]
        for region in regions:
            a,b,c,d=region['rect'];x,y,xx,yy=int(a*w),int(b*h),int(c*w),int(d*h)
            jobs.append((image[y:yy,x:xx],x,y,640))
    predictions=[]
    for crop,x,y,size in jobs:
        result=model.predict(crop,imgsz=size,device='cpu',conf=.05,iou=.7,verbose=False,augment=False)[0]
        for box,label,confidence in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.cls.cpu().tolist(),result.boxes.conf.cpu().tolist()):
            box=[value+(x if i%2==0 else y) for i,value in enumerate(box)]
            cx,cy=(box[0]+box[2])/2/w,(box[1]+box[3])/2/h
            names=[r['name'] for r in regions if r['rect'][0]<=cx<=r['rect'][2] and r['rect'][1]<=cy<=r['rect'][3]]
            if names:
                rank,suit=card_label(model.names[int(label)])
                predictions.append(dict(region=names[0],box=box,rank=rank,suit=suit,confidence=float(confidence)))
    return predictions


def source_controls(model,dataset):
    """Operational check on a local YOLO test folder; not CardVision accuracy."""
    import yaml
    from evaluate_card_result import iou
    dataset=Path(dataset)
    metadata=yaml.safe_load((dataset.parent/'data.yaml').read_text())
    names=metadata['names']
    if isinstance(names,list):names=dict(enumerate(names))
    if ({int(k):str(v).lower() for k,v in names.items()}
            != {int(k):str(v).lower() for k,v in model.names.items()}):
        raise ValueError('Source control class IDs do not match model names.')
    reports=[]
    for path in sorted((dataset/'images').iterdir()):
        if path.suffix.lower() not in ['.png','.jpg','.jpeg']:continue
        image=cv2.imread(str(path))
        if image is None:raise ValueError('Unreadable source control '+path.name)
        h,w=image.shape[:2];truth=[]
        for line in (dataset/'labels'/(path.stem+'.txt')).read_text().splitlines():
            cls,x,y,bw,bh=map(float,line.split())
            truth.append((int(cls),[(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h]))
        result=model.predict(image,imgsz=640,device='cpu',conf=.25,iou=.7,verbose=False)[0]
        predicted=list(zip(result.boxes.cls.cpu().tolist(),result.boxes.xyxy.cpu().tolist()))
        pairs=[]
        if truth and predicted:
            overlaps=np.array([[iou(a[1],b[1]) for b in predicted] for a in truth])
            rows,columns=linear_sum_assignment(1-overlaps)
            pairs=[(i,j) for i,j in zip(rows,columns) if overlaps[i,j]>=.5]
        reports.append(dict(image=path.name,labelled_corners=len(truth),predictions=len(predicted),
            matched_at_iou_05=len(pairs),correct_classes=sum(truth[i][0]==int(predicted[j][0]) for i,j in pairs)))
    if not reports:raise ValueError('No source-domain control images found.')
    return dict(images=reports,selection='All images supplied in the local source-control folder',
        class_mapping_verified=True,imgsz=640,confidence=.25,
        limitation='Source-domain operational check; not an independent CardVision accuracy test.')


def benchmark(weights,frames_dir,reference_path,output,summary=None,baseline=None,control_dataset=None):
    weights,frames_dir,output=Path(weights),Path(frames_dir),Path(output)
    reference=json.loads(Path(reference_path).read_text())
    paths=sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in ['.png','.jpg','.jpeg'])
    if not paths or any(type(row['frame']) is not int or not 0<=row['frame']<len(paths) for row in reference['observations']):
        raise ValueError('Empty recording or annotation frame indices outside recording.')
    regions=validate_regions(DEFAULT_REGIONS)
    output.mkdir(parents=True,exist_ok=True)
    model=load_external_model(weights)
    digest=hashlib.sha256(weights.read_bytes()).hexdigest()
    import torch, ultralytics
    report=dict(source_url=SOURCE_URL,source_revision=SOURCE_REVISION,
        weights_sha256=digest,matches_inspected_synthetic_checkpoint=digest==SYNTHETIC_SHA256,
        runtime=dict(torch=torch.__version__,ultralytics=ultralytics.__version__,opencv=cv2.__version__),
        frames=len(paths),annotation_scope=reference['scope'],
        protocol=dict(variants=list(VARIANTS),inference_floor=.05,primary_confidence=.25,
                      diagnostic_confidence=.05,nms_iou=.7,device='cpu',regions=regions,
                      matching_tiebreak='Geometric overlap first, confidence for ties; never annotated rank.'),variants={})
    if baseline:
        from evaluate_card_result import evaluate
        report['current_pipeline']=evaluate(json.loads(Path(baseline).read_text()),reference)
    if control_dataset:
        report['source_controls']=source_controls(model,control_dataset)
    for variant in VARIANTS:
        results=[];times=[]
        for frame,path in enumerate(paths):
            image=cv2.imread(str(path))
            if image is None:raise ValueError('Unreadable image '+path.name)
            start=time.monotonic();predictions=predict_frame(model,image,regions,variant);times.append(time.monotonic()-start)
            results.append(dict(frame=frame,source=path.name,predictions=predictions))
            if (frame+1)%25==0 or frame+1==len(paths):print(variant,frame+1,len(paths),flush=True)
        metrics={str(threshold):observation_metrics(reference,results,threshold) for threshold in [.25,.05]}
        metrics['timing_seconds']=dict(total=sum(times),median=float(np.median(times)),p95=float(np.percentile(times,95)))
        report['variants'][variant]=metrics
        (output/(variant+'_predictions.json')).write_text(json.dumps(results,indent=2)+'\n')
        print(variant,json.dumps(metrics),flush=True)
    report['limitations']=[
        'Existing development recording; not an independent holdout or general accuracy estimate.',
        'Annotations cover selected main-view detector observations, not all visible cards or views.',
        'Reference boxes are rank-only; external boxes can include suit. Matching uses >=50% reference coverage and >=10% prediction coverage.',
        'Unmatched predictions can include legitimate other corners; they are not automatically false positives.',
        'Current pipeline uses temporal rank aggregation; external metrics are frame-level without tracking.',
        'No model training/fine-tuning, production replacement, or third-party weight redistribution.']
    text=json.dumps(report,indent=2)+'\n';(output/'report.json').write_text(text)
    if summary:
        target=Path(summary);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(text)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['weights','frames','reference','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--summary',type=Path)
    parser.add_argument('--baseline-result',type=Path)
    parser.add_argument('--control-dataset',type=Path,help='Optional source YOLO test folder with images/labels and ../data.yaml')
    args=parser.parse_args()
    try:benchmark(args.weights,args.frames,args.reference,args.output,args.summary,args.baseline_result,args.control_dataset)
    except (OSError,ValueError,RuntimeError,KeyError) as exc:parser.exit(1,str(exc)+'\n')


if __name__=='__main__':main()
