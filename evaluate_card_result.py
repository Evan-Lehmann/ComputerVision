"""Evaluate physical identities and ranks against manually labelled observations.

Reference JSON: cards [{id,rank,round_number}], observations
[{frame,region,box,identity}], scope, count_change. Frame indices start at zero.
Annotations can cover detector observations or all visible cards; report their
scope explicitly. Unlabelled parts of a recording are not verified by this tool.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(a,b):
    overlap=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-overlap
    return overlap/union if union>0 else 0


def evaluate(result,reference):
    truth={card['id']:card for card in reference['cards']}
    expected,predicted=defaultdict(list),defaultdict(list)
    for row in reference['observations']:
        if row['identity'] not in truth:
            raise ValueError('Observation references an unknown physical card.')
        expected[row['frame'],row['region']].append(row)
    events=[e for e in result['events'] if not e.get('excluded')]
    for event in events:
        frames,boxes=event['observation_frames'],event['observation_boxes']
        if len(frames)!=len(boxes):
            raise ValueError('Frame and box provenance lengths differ.')
        for frame,box in zip(frames,boxes):
            predicted[frame,event['region']].append((event,box))
    track_labels,card_tracks=defaultdict(set),defaultdict(set)
    matched=missing=extra=0
    for key in set(expected)|set(predicted):
        a,b=expected[key],predicted[key]
        pairs=[]
        if a and b:
            scores=np.array([[iou(row['box'],box) for _,box in b] for row in a])
            rows,columns=linear_sum_assignment(1-scores)
            pairs=[(i,j) for i,j in zip(rows,columns) if scores[i,j]>=.5]
        matched+=len(pairs);missing+=len(a)-len(pairs);extra+=len(b)-len(pairs)
        for i,j in pairs:
            identity,event=a[i]['identity'],b[j][0]
            track_labels[event['id']].add(identity)
            card_tracks[identity].add(event['id'])
    wrong_rank=wrong_round=unresolved=0
    for event in events:
        identities=track_labels[event['id']]
        if event['rank']=='UNKNOWN':
            unresolved+=1
        if len(identities)==1:
            card=truth[next(iter(identities))]
            wrong_rank+=event['rank']!='UNKNOWN' and event['rank']!=card['rank']
            wrong_round+=event.get('round_number',1)!=card['round_number']
    mixed=sum(len(ids)>1 for ids in track_labels.values())
    fragments=sum(max(0,len(ids)-1) for ids in card_tracks.values())
    missing_cards=len(set(truth)-set(card_tracks))
    extra_events=sum(not track_labels[e['id']] for e in events)
    exact=not any([missing,extra,mixed,fragments,missing_cards,extra_events,wrong_rank,wrong_round,unresolved])
    return dict(annotation_scope=reference['scope'],expected_physical_cards=len(truth),
        candidate_events=len(events),matched_observations=matched,missing_annotated_observations=missing,
        unannotated_output_observations=extra,mixed_identity_tracks=mixed,extra_identity_fragments=fragments,
        missing_physical_cards=missing_cards,unmatched_events=extra_events,
        wrong_accepted_rank_events=wrong_rank,wrong_round_events=wrong_round,unknown_events=unresolved,
        exact_annotated_card_list=exact,expected_count_change=reference['count_change'],
        count_change=result['count_change'],count_matches=result['count_change']==reference['count_change'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result',type=Path)
    parser.add_argument('reference',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--require-exact',action='store_true')
    args=parser.parse_args()
    try:
        report=evaluate(json.loads(args.result.read_text()),json.loads(args.reference.read_text()))
    except (OSError,ValueError,KeyError,TypeError) as exc:
        parser.exit(1,str(exc)+'\n')
    text=json.dumps(report,indent=2)+'\n'
    if args.output:
        args.output.write_text(text)
    print(text,end='')
    if args.require_exact and not (report['exact_annotated_card_list'] and report['count_matches']):
        parser.exit(1,'Acceptance failed.\n')


if __name__=='__main__':
    main()
