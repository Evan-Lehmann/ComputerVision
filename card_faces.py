"""Optional geometric evidence for upright digital card faces in a hand row.

This does not predict ranks or generate cards. If rectangular paper/edges cannot
be established, existing detections remain untouched. Suppression requires a
retained rank candidate on the very same visible card face.
"""
import cv2
import numpy as np
from functools import lru_cache
from glyph_features import glyph_signature


def _gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)


def estimate_face_height(image, detections):
    if not detections:
        return None
    gray = _gray(image)
    mask = ((gray >= 180) & (np.ptp(image.astype(np.int16),axis=2) < 55)).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    heights=[]
    for x,y,w,h,area in stats[1:]:
        if not (28 <= h <= image.shape[0]*.12 and .25 <= w/h <= 8 and area/(w*h) >= .65):
            continue
        if any(abs(y-d['box'][1]) <= h*.15 and x-h <= d['box'][0] <= x+w+h for d in detections):
            heights.append(h)
    return float(np.median(heights)) if heights else float(np.median([d['box'][3]-d['box'][1] for d in detections])*3)


def _face(gray, box, height):
    ih,iw=gray.shape
    x1,y1,x2,y2=box
    h=height
    if h < 28 or (y2-y1)/h < .18 or (y2-y1)/h > .55:
        return None
    ya,yb=int(y1+h*.36),int(y1+h*.60)
    xa,xb=max(1,int(x1-h*.65)),min(iw-1,int(x2+h))
    if ya<0 or yb>=ih or xb-xa<6:
        return None
    profile=np.median(gray[ya:yb,xa:xb],axis=0)
    delta=np.diff(profile)
    candidates=[i for i in range(len(delta)) if -h*.65 <= (xa+i+1)-x1 <= h*.17 and delta[i]>=8]
    if not candidates:
        return None
    selected=None
    for index in sorted(candidates,key=lambda i:abs((xa+i+1)-x1)):
        left=xa+index+1
        rights=[i for i in range(len(delta)) if h*.25 <= (xa+i+1)-left <= h*.85 and delta[i]<=-8]
        if not rights:
            continue
        right=xa+min(rights)+1
        if left <= (x1+x2)/2 < right:
            selected=(left,right)
            break
    if selected is None:
        return None
    left,right=selected
    ca,cb=int(left+(right-left)*.15),int(right-(right-left)*.15)
    ya,yb=max(1,int(y1-h*.18)),min(ih-1,int(y1+h*1.2))
    if cb<=ca:
        return None
    vertical=np.median(gray[ya:yb,ca:cb],axis=1)
    vd=np.diff(vertical)
    tops=[i for i in range(len(vd)) if abs((ya+i+1)-y1)<=h*.18 and vd[i]>=8]
    if not tops:
        return None
    top=ya+max(tops,key=lambda i:vd[i]-.3*abs((ya+i+1)-y1))+1
    bottoms=[i for i in range(len(vd)) if abs((ya+i+1)-(top+h))<=h*.13 and vd[i]<=-8]
    if not bottoms:
        return None
    bottom=ya+max(bottoms,key=lambda i:-vd[i]-.3*abs((ya+i+1)-(top+h)))+1
    actual_h=bottom-top
    if not (actual_h*.05 <= (y1+y2)/2-top <= actual_h*.38):
        return None
    return [int(left),int(top),int(right),int(bottom)]


def secondary_rank(image, face):
    """Upright opposite rank only on a fully exposed face, not a suit strip."""
    x1,y1,x2,y2=face
    h,w=y2-y1,x2-x1
    if not .55 <= w/h <= .85:
        return None
    crop=image[int(y1+h*.55):y2, int(x1+w*.48):x2]
    return cv2.rotate(crop,cv2.ROTATE_180) if crop.size else None


def secondary_similarity(a,b):
    ga,gb=glyph_signature(a),glyph_signature(b)
    if ga is None or gb is None or ga[2]!=gb[2] or max(ga[1]/gb[1],gb[1]/ga[1])>1.25:
        return 0.0
    return float(cv2.matchTemplate(np.pad(gb[0],2),ga[0],cv2.TM_CCORR_NORMED).max())


@lru_cache(maxsize=1)
def _suit_validator():
    from card_suits import SuitReader
    return SuitReader()


def validate_face_glyphs(image, detections, height):
    """Group glyphs by independently observed paper; keep the left rank once.

    Only a complete upright face with a separate left-corner candidate can
    explain a right-corner suit detection. Unsupported geometry is retained.
    """
    gray=_gray(image)
    mask=((gray>90)&(np.ptp(image.astype(np.int16),axis=2)<50)).astype(np.uint8)
    _,_,stats,_=cv2.connectedComponentsWithStats(mask)
    removed=set();audit=[]
    for x,y,w,h,area in stats[1:]:
        if not (28<=h<=image.shape[0]*.12 and .55<=w/h<=.85 and area/(w*h)>.70):continue
        members=[]
        for i,d in enumerate(detections):
            a,b,c,e=d['box'];cx,cy=(a+c)/2,(b+e)/2
            inside=max(0,min(c,x+w)-max(a,x))*max(0,min(e,y+h)-max(b,y))
            if inside/max((c-a)*(e-b),1)>.9 and y<=cy<=y+h*.4:members.append((i,d,cx))
        ranks=[row for row in members if row[2]<x+w*.4 and row[1]['box'][2]-row[1]['box'][0]>=h*.18]
        if len(ranks)!=1:continue
        ri,rank,_=ranks[0]
        suit=_suit_validator().read(image[y:int(y+h*.46),int(x+w*.52):x+w])
        if not suit or not suit['accepted']:continue
        for i,d,cx in members:
            if i==ri or i in removed:continue
            if cx>=x+w*.48 and d['box'][0]>=x+w*.40 and d['box'][0]>=rank['box'][2]-2:
                removed.add(i);audit.append(dict(box=d['box'],retained_box=rank['box'],face_box=[int(x),int(y),int(x+w),int(y+h)],reason='suit_position_on_same_physical_face'))
    return [d for i,d in enumerate(detections) if i not in removed],audit


def reconcile_faces(image, detections, height=None):
    """Return kept candidates with face evidence plus a suppression audit."""
    if not detections:
        return [],[]
    height=height or estimate_face_height(image,detections)
    detections,face_audit=validate_face_glyphs(image,detections,height)
    gray=_gray(image)
    faces=[]
    for d in detections:
        f=_face(gray,d['box'],height)
        if f is not None and not any(abs(f[0]-g[0])<height*.10 and abs(f[1]-g[1])<height*.10 for g in faces):
            faces.append(f)
    groups={i:[] for i in range(len(faces))};unassigned=[]
    for d in detections:
        x1,y1,x2,y2=d['box'];cx,cy=(x1+x2)/2,(y1+y2)/2
        possible=[i for i,f in enumerate(faces) if f[0]<=cx<f[2] and f[1]<=cy<f[1]+(f[3]-f[1])*.4]
        if len(possible)!=1:
            unassigned.append(d)
        else:
            groups[possible[0]].append(d)
    kept=list(unassigned);suppressed=list(face_audit)
    for i,ds in groups.items():
        if not ds:
            continue
        f=faces[i];h=f[3]-f[1]
        def quality(d):
            x1,y1,x2,y2=d['box'];cx=(x1+x2)/2
            return .4*min((x2-x1)/h,.4)-abs((cx-f[0])/h-.13)
        best=max(ds,key=quality)
        kept.append(dict(best,face_box=f,secondary_crop=secondary_rank(image,f)))
        for d in ds:
            if d is not best:
                a,b=d['box'],best['box']
                overlap=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
                contained=overlap/max((a[2]-a[0])*(a[3]-a[1]),1)
                partial=(contained >= .85 or (contained >= .5 and a[2]-a[0] <= (b[2]-b[0])*.85))
                # A narrow suit-edge sliver can be disjoint from the rank.
                # Require a full face and agreement of both printed rank corners.
                opposite=secondary_rank(image,f)
                suit_sliver=(a[2]-a[0] < h*.15 and b[2]-b[0] >= h*.20
                             and secondary_similarity(best.get('crop'),opposite) >= .94)
                if partial or suit_sliver:
                    suppressed.append(dict(box=d['box'],retained_box=best['box'],face_box=f,reason='same_visible_card_face'))
                else:
                    kept.append(d)
    return kept,suppressed


def secondary_animation_similarity(early, late):
    """Extra identity evidence for a dim, fully exposed moving card only."""
    a,b=early.get('secondary_crop'),late.get('secondary_crop')
    ga,gb=glyph_signature(a),glyph_signature(b)
    if ga is None or gb is None or ga[3]>=200 or gb[3]-ga[3]<35:
        return 0.0
    fa,fb=early.get('face_box'),late.get('face_box')
    if fa is None or fb is None:
        return 0.0
    ha,hb=fa[3]-fa[1],fb[3]-fb[1]
    if max(ha/hb,hb/ha)>1.15:
        return 0.0
    ca=np.array([(fa[0]+fa[2])/2,(fa[1]+fa[3])/2])
    cb=np.array([(fb[0]+fb[2])/2,(fb[1]+fb[3])/2])
    if abs(ca[1]-cb[1])>max(ha,hb)*.15 or np.linalg.norm(ca-cb)>max(ha,hb)*1.8:
        return 0.0
    return secondary_similarity(a,b)


def projected_rank_crop(image, detection, observation):
    """Read a previous complete corner at its face-relative position now."""
    fa,fb=observation.get('face_box'),detection.get('face_box')
    if fa is None or fb is None:
        return None
    ha,hb=fa[3]-fa[1],fb[3]-fb[1]
    box=observation.get('rank_box') or observation['box']
    if (max(ha/hb,hb/ha)>1.15 or abs(box[0]-fa[0])>ha*.08
            or not ha*.18 <= box[2]-box[0] <= ha*.45):
        return None
    scale=hb/ha
    shifted=[fb[0]+(box[0]-fa[0])*scale,fb[1]+(box[1]-fa[1])*scale,
             fb[0]+(box[2]-fa[0])*scale,fb[1]+(box[3]-fa[1])*scale]
    # Do not sample another card behind a newly overlapping face.
    if shifted[2]>fb[2]+2:
        return None
    x1,y1,x2,y2=shifted;px,py=max(2,int((x2-x1)*.2)),max(2,int((y2-y1)*.2))
    crop=image[max(0,int(y1)-py):min(image.shape[0],int(y2)+py),max(0,int(x1)-px):min(image.shape[1],int(x2)+px)].copy()
    return (crop,shifted) if crop.size else None
