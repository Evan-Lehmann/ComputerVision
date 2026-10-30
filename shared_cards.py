"""Geometry and evidence for the configured compressed central card hand.

This monitor never infers a duplicate from equal ranks alone. It records full
append states; reconciliation is performed after primary tracks are resolved.
"""
from collections import Counter
import cv2
import numpy as np

DEFAULT_SHARED_VIEW=dict(name='central',rect=[.45,.68,.57,.75],linked_region='player')


def mini_hand(image, rect):
    h,w=image.shape[:2];x1,y1,x2,y2=[round(v*s) for v,s in zip(rect,[w,h,w,h])]
    crop=image[y1:y2,x1:x2]
    if not crop.size:return None
    gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
    mask=(gray>=200).astype(np.uint8)
    _,labels,stats,_=cv2.connectedComponentsWithStats(mask)
    candidates=[]
    for x,y,cw,ch,area in stats[1:]:
        if not (h*.022<=ch<=h*.05 and .9<=cw/ch<=7 and area/(cw*ch)>.65):continue
        outer=crop[max(0,y-3):min(crop.shape[0],y+ch+3),max(0,x-3):min(crop.shape[1],x+cw+3)]
        border=np.concatenate([outer[:2].reshape(-1,3),outer[-2:].reshape(-1,3),outer[:,:2].reshape(-1,3),outer[:,-2:].reshape(-1,3)]).astype(float)
        if np.mean((border[:,1]>=border[:,0]-2)&(border[:,1]>border[:,2]+5))<.35:continue
        body=gray[y:y+ch,x:x+cw]
        # A component can include one antialiased outer row. Require a flat
        # pair of paper rows at each edge, rather than counting that fringe
        # as half of the bottom border (which dropped otherwise intact hands).
        edge_fill=np.mean(body>=200,axis=1)
        if min(max((edge_fill[:3][:-1]+edge_fill[:3][1:])/2),
               max((edge_fill[-3:][:-1]+edge_fill[-3:][1:])/2))<.70:continue
        # A digital hand has a flat white empty middle band, unlike live-video
        # paper/text/shadows. Unknown layouts are not declared covered.
        middle=gray[y+int(ch*.42):y+int(ch*.59),x:x+cw]
        profile=np.median(middle,axis=0).astype(float)
        if np.median(profile)<248:continue
        # Interior dark seam troughs distinguish exposed card strips.
        lows=[]
        for i in range(2,cw-2):
            if profile[i]<242 and profile[i]<=profile[i-1] and profile[i]<=profile[i+1]:
                if not lows or i-lows[-1]>=ch*.16:lows.append(i)
                elif profile[i]<profile[lows[-1]]:lows[-1]=i
        starts=[0]+[i+1 for i in lows if ch*.15<i<cw-ch*.85]
        if len(starts)>7 or x<=0 or y<=0 or x+cw>=crop.shape[1] or y+ch>=crop.shape[0]:
            return dict(unsupported=True,box=[int(x+x1),int(y+y1),int(x+x1+cw),int(y+y1+ch)])
        widths=np.diff(starts+[cw]);last=widths[-1]
        if not (.95<=last/ch<=1.4):continue
        if any(not .18<=value/ch<=.45 for value in widths[:-1]):continue
        # White component edges may be antialiased by one pixel.
        face=[int(x+x1),int(y+y1),int(x+x1+cw),int(y+y1+ch)]
        candidates.append(dict(box=face,starts=[int(x+x1+s) for s in starts],count=len(starts)))
    return candidates[0] if len(candidates)==1 else None


def mini_rank_crop(image,hand):
    x=hand['starts'][-1];_,y,right,bottom=hand['box'];h=bottom-y
    # Match the actual detector's padded rank stream, without its tile slivers.
    box=[x+1,y-1,min(right,x+h*.44),y+h*.34]
    x1,y1,x2,y2=box;px,py=2,2
    return image[max(0,int(y1)-py):int(y2)+py,max(0,int(x1)-px):int(x2)+px].copy(),box


def suit_crops(image,face,compressed=False):
    x,y,right,bottom=map(int,face);h,w=bottom-y,right-x
    if h<=0 or w<=0:return []
    if compressed:
        if not .95<=w/h<=1.45:return []
    elif not .55<=w/h<=.85:return []
    upper=image[y:int(y+h*.46),int(x+w*.52):right]
    lower=image[int(y+h*.53):bottom,x:int(x+w*.49)]
    result=[]
    if upper.size:result.append(upper.copy())
    if lower.size:result.append(cv2.rotate(lower,cv2.ROTATE_180))
    if compressed:result=[cv2.resize(c,None,fx=1,fy=1.5,interpolation=cv2.INTER_LINEAR) for c in result]
    return result


def mini_last_face(hand):
    return [hand['starts'][-1],hand['box'][1],hand['box'][2],hand['box'][3]]


def prefix_patches(image,hand):
    """Visible rank strips, including colour, for every ordinal card.

    The same narrow area is visible before and after overlap. These patches
    establish source continuity; they are never classified or exported as
    isolated training crops.
    """
    y,bottom=hand['box'][1],hand['box'][3];height=bottom-y
    return [image[y+1:y+round(height*.38),x+2:x+round(height*.27)].copy()
            for x in hand['starts']]


def preserves_prefix(before,after):
    """Local motion plus matching visible strips; equal ranks alone cannot link."""
    if not 0<after['frame']-before['frame']<=3 or after['count']<before['count']:return False
    a,b=before.get('prefix_patches',[]),after.get('prefix_patches',[])
    if len(a)!=before['count'] or len(b)!=after['count']:return False
    ax,ay,ar,ab=before['hand_box'];bx,by,br,bb=after['hand_box']
    ah,bh=ab-ay,bb-by
    if min(ah,bh)<=0 or max(ah/bh,bh/ah)>1.2 or abs(ay-by)>max(ah,bh)*.2:return False
    if abs((ax+ar-bx-br)/2)>max(ah,bh)*.5:return False
    for old,new in zip(a,b):
        descriptors=[]
        for crop in [old,new]:
            if crop.size==0:return False
            gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY).astype(np.float32)
            white=max(float(np.percentile(gray,90)),1)
            ink=np.clip((white-gray)/white,0,1)
            if np.count_nonzero(ink>.25)<3:return False
            blue,green,red=cv2.split(crop.astype(float));dark=ink>.25
            colour=float(np.mean(((red>green+30)&(red>blue+30))[dark]))
            descriptors.append((cv2.resize(ink,(24,32)),colour))
        (aa,red_a),(ba,red_b)=descriptors
        if abs(red_a-red_b)>.45:return False
        if float(cv2.matchTemplate(np.pad(ba,3),aa,cv2.TM_CCORR_NORMED).max())<.84:return False
    return True


def continuation_frames(snapshots,primary,common):
    """Carry an established prefix forward through observed source motion.

    A long gap, replaced prefix, or shrinking hand breaks this chain. A new
    hand cannot inherit old identities just because its ordinal count grew.
    """
    if common<2:return set()
    anchor=set.intersection(*(set(e['observation_frames']) for e in primary[:common]))
    if len({s['frame'] for s in snapshots if s['count']>=common} & anchor)<2:return set()
    trusted=[];frames=set()
    for snapshot in snapshots:
        if snapshot['count']<common:continue
        if snapshot['frame'] in anchor:
            trusted.append(snapshot);frames.add(snapshot['frame'])
        elif any(preserves_prefix(previous,snapshot) for previous in trusted[-3:]):
            trusted.append(snapshot);frames.add(snapshot['frame'])
    return frames


class SharedMonitor:
    def __init__(self,view,rank_reader,suit_reader):
        self.view=view;self.rank_reader=rank_reader;self.suit_reader=suit_reader;self.snapshots={};self.primary_frames={};self.unreadable={}
    def observe(self,image,frame,source,round_number,primary_visible=True):
        if primary_visible:self.primary_frames.setdefault(round_number,[]).append(frame)
        hand=mini_hand(image,self.view['rect'])
        if hand is None:return
        if hand.get('unsupported'):
            self.unreadable.setdefault(round_number,[]).append(frame);return
        crop,box=mini_rank_crop(image,hand)
        self.snapshots.setdefault(round_number,[]).append(dict(frame=frame,source=source,
            count=hand['count'],crop=crop,box=box,face_box=mini_last_face(hand),
            hand_box=hand['box'],prefix_patches=prefix_patches(image,hand),
            hand_crop=image[hand['box'][1]:hand['box'][3],hand['box'][0]:hand['box'][2]].copy(),
            suit_crops=suit_crops(image,mini_last_face(hand),True)))
    def resolve(self,events,output_dir):
        from pathlib import Path
        output_dir=Path(output_dir);coverage=dict(scope='configured_views',status='not_observed',view=self.view,issues=[],links=[],slots=[],additional_events=[])
        numbers=sorted(set(e.get('round_number',1) for e in events)|set(self.snapshots)|set(self.unreadable))
        if not numbers:coverage['issues'].append(dict(reason='No reliable primary or shared cards were observed; check capture and calibration.'))
        for number in numbers:
            snapshots=self.snapshots.get(number,[])
            if self.unreadable.get(number):coverage['issues'].append(dict(round_number=number,frame=self.unreadable[number][0],reason='Shared hand is clipped or exceeds the supported layout; inspect the source and add any missed cards.'))
            if snapshots:
                present={s['frame'] for s in snapshots};missing=[f for f in self.primary_frames.get(number,[]) if f>=min(present) and f not in present]
                gap=[]
                for f in missing:
                    gap=gap+[f] if not gap or f==gap[-1]+1 else [f]
                    if len(gap)>=3:
                        coverage['issues'].append(dict(round_number=number,frame=gap[0],reason='Shared view became unavailable while primary cards remained; coverage needs review.'));break
            frequencies=Counter(s['count'] for s in snapshots)
            stable={count for count,n in frequencies.items() if n>=2}
            if not snapshots:
                coverage['issues'].append(dict(round_number=number,frame=min(self.primary_frames.get(number,[0])),reason='Shared view not observed reliably; check the recording and calibration.'))
                continue
            filtered=snapshots
            runs=[]
            for snap in snapshots:
                if runs and runs[-1][-1]['count']==snap['count'] and snap['frame']-runs[-1][-1]['frame']<=2:runs[-1].append(snap)
                else:runs.append([snap])
            stable_runs=[run for run in runs if len(run)>=2]
            reset=any(b[0]['count']<a[-1]['count'] for a,b in zip(stable_runs,stable_runs[1:]))
            if any(s['count'] not in stable for s in snapshots):coverage['issues'].append(dict(round_number=number,reason='A shared card was visible too briefly for confirmation; review its candidate.'))
            if reset:coverage['issues'].append(dict(round_number=number,reason='Shared hand changed or shrank without a verified round boundary.'))
            slots=[]
            for i in range(max(s['count'] for s in snapshots)):
                observations=[s for s in filtered if s['count']==i+1]
                present=[s['frame'] for s in filtered if s['count']>=i+1]
                rank=self.rank_reader.consensus(observations) if self.rank_reader else None
                suit=self.suit_reader.consensus(observations) if self.suit_reader else None
                slot=dict(id=f'shared-{number}-{i+1}',round_number=number,ordinal=i+1,
                    rank=rank['rank'] if rank else 'UNKNOWN',suit=suit['suit'] if suit else 'UNKNOWN',
                    confidence=rank['score'] if rank else 0,suit_confidence=suit['score'] if suit else 0,
                    first_frame=min(present),last_frame=max(present),present_frames=present,
                    evidence=[],observations=observations)
                # A slot already overlapped on entry has no isolated rank crop.
                # Preserve a hand-context image, explicitly ineligible for labels.
                display=observations or [dict(s,crop=s.get('hand_crop',s['crop'])) for s in filtered if s['count']>=i+1]
                for label,index in zip(['first','middle','last'],[0,len(display)//2,max(0,len(display)-1)]):
                    if not display:continue
                    o=display[index];name=f"{slot['id']}_{label}.png"
                    if not cv2.imwrite(str(output_dir/name),o['crop']):raise OSError('Could not save shared-view evidence.')
                    slot['evidence'].append(dict(image=name,frame=o['frame'],source=o['source'],rank_crop=bool(observations)))
                slots.append(slot)
            primary=sorted([e for e in events if e['region']==self.view['linked_region'] and e.get('round_number',1)==number],key=lambda e:(e['first_frame'],str(e['id'])))
            matched=[];lags=[];known_anchors=0
            if not reset:
                for slot in slots:
                    candidates=[e for e in primary if e not in matched
                        and e['rank']!='UNKNOWN' and slot['suit']!='UNKNOWN'
                        and slot['suit']==e.get('suit')
                        and (slot['rank']=='UNKNOWN' or slot['rank']==e['rank'])
                        and abs(e['first_frame']-slot['first_frame'])<=8
                        and len(set(slot['present_frames'])&set(e['observation_frames']))>=2]
                    if len(candidates)!=1:break
                    event=candidates[0]
                    if matched and event['first_frame']<matched[-1]['first_frame']:break
                    matched.append(event);lags.append(event['first_frame']-slot['first_frame'])
                    known_anchors+=slot['rank']!='UNKNOWN'
            # A weak miniature rank may borrow identity from its uniquely timed
            # main counterpart, but only with suit agreement and another known
            # rank anchor. Conflicting accepted ranks can never be overridden.
            common=len(matched)
            if common<2 or not known_anchors or max(abs(v-np.median(lags)) for v in lags)>2:
                common=0;matched=[]
            primary=matched+[e for e in primary if e not in matched]
            continued_frames=continuation_frames(snapshots,primary,common)
            for i,slot in enumerate(slots):
                if i<common:
                    event=primary[i];event['source_views']=sorted(set(event.get('source_views',[event['region']])+[self.view['name']]))
                    slot.update(status='linked',event_id=event['id'])
                    coverage['links'].append(dict(slot_id=slot['id'],event_id=event['id'],round_number=number,reason='ordered_rank_suit_birth_and_coview_prefix'))
                else:
                    linked_ids={e['id'] for e in primary[:common]}
                    unmatched=[e for e in events if e.get('round_number',1)==number and e['region']!=self.view['name'] and e['id'] not in linked_ids]
                    co_visible=common>=2 and all(len(set(slot['present_frames'])&set(e['observation_frames']))>=2 for e in primary[:common])
                    continued=common>=2 and len({o['frame'] for o in slot['observations']} & continued_frames)>=2
                    # Equal rank/suit can denote different cards in a shoe. Only
                    # unused primary tracks can compete for this shared slot.
                    competing=[e for e in unmatched if (e['rank']=='UNKNOWN' or slot['rank']=='UNKNOWN' or e['rank']==slot['rank'])
                        and (e.get('suit','UNKNOWN')=='UNKNOWN' or slot['suit']=='UNKNOWN' or e.get('suit')==slot['suit'])
                        and not (e['region']=='dealer' and e['rank']==slot['rank'] and e.get('suit')==slot['suit']
                            and e['first_frame']-slot['first_frame']>8
                            and len([f for f in slot['present_frames'] if f<e['first_frame']])>=2
                            and len(set(slot['present_frames'])&set(e['observation_frames']))>=2)]
                    distinct=bool(common>=2 and (co_visible or continued) and not competing and not reset)
                    slot.update(status='additional' if distinct else 'needs_review',event_id=slot['id'])
                    if not distinct:coverage['issues'].append(dict(round_number=number,slot_id=slot['id'],reason='Shared card identity is unresolved; keep it as distinct or exclude it after checking other views.'))
                    obs=slot['observations'];candidate=dict(id=slot['id'],region=self.view['name'],round_number=number,
                        rank=slot['rank'] if distinct else 'UNKNOWN',suit=slot['suit'],suggested_rank=slot['rank'],
                        confidence=slot['confidence'],suit_confidence=slot['suit_confidence'],
                        observations=len(obs),first_frame=slot['first_frame'],last_frame=slot['last_frame'],
                        observation_frames=[o['frame'] for o in obs],observation_boxes=[o['box'] for o in obs],
                        observation_rank_boxes=[o['box'] for o in obs],observation_faces=[o['face_box'] for o in obs],
                        observation_crop_sources=['shared_view']*len(obs),evidence=slot['evidence'],
                        recognition_source='mini_consensus' if distinct else 'shared_identity_unresolved',
                        identity_status='distinct' if distinct else 'needs_review',source_views=[self.view['name']],
                        identity_basis=('simultaneous_prefix' if co_visible else 'continuous_shared_hand') if distinct else None,
                        excluded=False,reviewed=False,alternatives=[],possible_duplicates=[e['id'] for e in competing] if not distinct else [])
                    events.append(candidate);coverage['additional_events'].append(candidate['id'])
                coverage['slots'].append({k:v for k,v in slot.items() if k not in ['observations','present_frames']})
        coverage['status']='needs_review' if coverage['issues'] else 'configured_views_reconciled'
        return coverage
