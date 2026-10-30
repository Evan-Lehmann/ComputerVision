"""Local suit recognition with shape, ink-colour and temporal agreement gates."""
from pathlib import Path
import pickle
import cv2
import numpy as np
import torch
from torch import nn
from glyph_features import glyph_signature

SUITS=['clubs','diamonds','hearts','spades']


class SuitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32,32,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d((4,4)),
            nn.Flatten(),nn.Linear(512,64),nn.ReLU(),nn.Linear(64,4))
    def forward(self,x):return self.layers(x)


def suit_tensor(crop):
    if crop is None or not crop.size or min(crop.shape[:2])<4:return None
    pixels=crop.astype(np.float32);background=np.maximum(np.percentile(pixels,95,axis=(0,1)),1)
    # NumPy versions differ in whether percentile preserves float32. Keep
    # OpenCV's input depth explicit after division by that background.
    normalized=np.clip(pixels/background,0,1).astype(np.float32)
    ink=1-cv2.cvtColor(normalized,cv2.COLOR_BGR2GRAY)
    mask=(ink>max(.08,.2*float(ink.max()))).astype(np.uint8)
    _,labels,stats,_=cv2.connectedComponentsWithStats(mask)
    possible=[i for i,(x,y,w,h,area) in enumerate(stats[1:],1) if area>=5 and .55<=w/max(h,1)<=1.9]
    if not possible:return None
    selected=max(possible,key=lambda i:stats[i,4]);x,y,w,h,area=stats[selected]
    # Multiple substantial icons indicate a neighbouring-card contamination.
    if any(stats[i,4]>area*.4 for i in possible if i!=selected):return None
    clean=(labels==selected);yy,xx=np.where(clean)
    if len(xx)<6:return None
    tight=(ink*clean)[yy.min():yy.max()+1,xx.min():xx.max()+1]
    h,w=tight.shape;scale=28/max(h,w);nw,nh=max(1,round(w*scale)),max(1,round(h*scale))
    feature=np.zeros((32,32),np.float32);feature[(32-nh)//2:(32-nh)//2+nh,(32-nw)//2:(32-nw)//2+nw]=cv2.resize(tight,(nw,nh))
    colored=normalized[clean];red=bool(np.mean((colored[:,2]>colored[:,1]+.1)&(colored[:,2]>colored[:,0]+.1))>.5)
    return torch.from_numpy(feature).unsqueeze(0).unsqueeze(0),red


class SuitReader:
    def __init__(self,path=None):
        self.available=False;self.error=None
        try:
            data=torch.load(path or Path(__file__).parent/'models/suit_classifier_v1.pt',map_location='cpu',weights_only=True)
            if data['suits']!=SUITS:raise ValueError('Suit class order is incompatible.')
            self.model=SuitCNN();self.model.load_state_dict(data['model_state_dict']);self.model.eval();self.available=True
        except (OSError,ValueError,KeyError,RuntimeError,EOFError,pickle.UnpicklingError):self.error='Suit recognition unavailable; cross-view links need review.'
    def read(self,crop):
        if not self.available:return None
        item=suit_tensor(crop)
        if item is None:return None
        tensor,red=item
        with torch.no_grad():p=self.model(tensor).softmax(1)[0].numpy()
        order=np.argsort(p)[::-1];suit=SUITS[order[0]];score=float(p[order[0]]);margin=float(p[order[0]]-p[order[1]])
        mask=tensor[0,0].numpy()>.3
        yy,xx=np.where(mask)
        symmetry=0.0
        if len(xx):
            tight=mask[yy.min():yy.max()+1,xx.min():xx.max()+1]
            symmetry=float(np.logical_and(tight,np.flipud(tight)).sum()/max(np.logical_or(tight,np.flipud(tight)).sum(),1))
        # A compressed diamond can have a modest network score but retains
        # vertical reflection symmetry; the heart notch/tip is asymmetric.
        geometric=(red and suit=='diamonds' and score>=.6 and margin>=.2 and symmetry>=.75)
        accepted=((score>=.85 and margin>=.25) or geometric) and red==(suit in ['hearts','diamonds'])
        return dict(suit=suit,score=score,accepted=bool(accepted),method='network_and_diamond_geometry' if geometric else 'network',vertical_symmetry=symmetry)
    def consensus(self,observations):
        # One vote per distinct frame; conflicting accepted corners abstain.
        observations=[dict(o,suit_crops=[c for c in o.get('suit_crops',[]) if np.percentile(c,95)>=200]) for o in observations]
        observations=[o for o in observations if o['suit_crops']]
        if not self.available or len(observations)<2:return None
        indices=sorted({min(len(observations)-1,i*len(observations)//6) for i in range(7)})
        votes=[];frames=set()
        for i in indices:
            o=observations[i]
            if o['frame'] in frames:continue
            frames.add(o['frame']);read=[self.read(c) for c in o['suit_crops']];read=[r for r in read if r and r['accepted']]
            if len({r['suit'] for r in read})>1:return None
            if read:
                votes.append(dict(frame=o['frame'],suit=read[0]['suit'],score=min(r['score'] for r in read)))
        if len(votes)<2 or len({v['suit'] for v in votes})!=1:return None
        return dict(suit=votes[0]['suit'],score=min(v['score'] for v in votes),votes=votes)
