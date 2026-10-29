"""Separate reader for compressed shared-view ranks; primary V5 is unchanged."""
from pathlib import Path
import pickle
import numpy as np
import torch
from rank_model import RankCNN,preprocess_crop,RANKS
from glyph_features import glyph_signature


class MiniRankReader:
    def __init__(self,path=None):
        self.available=False;self.error=None
        try:
            data=torch.load(path or Path(__file__).parent/'models/rank_classifier_mini_v1.pt',map_location='cpu',weights_only=True)
            if data['ranks']!=RANKS or data['image_size']!=64:raise ValueError('Incompatible miniature rank model.')
            self.model=RankCNN();self.model.load_state_dict(data['model_state_dict']);self.model.eval();self.available=True
        except (OSError,ValueError,KeyError,RuntimeError,EOFError,pickle.UnpicklingError):self.error='Shared-view rank recognition unavailable; coverage needs review.'
    def consensus(self,observations):
        observations=[o for o in observations if glyph_signature(o['crop']) is not None]
        if not self.available or len({o['frame'] for o in observations})<2:return None
        indices=sorted({min(len(observations)-1,i*len(observations)//6) for i in range(7)})
        samples=[observations[i] for i in indices]
        with torch.no_grad():p=self.model(torch.cat([preprocess_crop(o['crop']) for o in samples])).softmax(1).numpy()
        mean=p.mean(0);order=np.argsort(mean)[::-1]
        # A strong contradictory view prevents an otherwise confident average.
        strong={int(v.argmax()) for v in p if v.max()>=.8 and np.sort(v)[-1]-np.sort(v)[-2]>=.2}
        if (mean[order[0]]<.8 or mean[order[0]]-mean[order[1]]<.2
                or len(strong)>1 or (strong and int(order[0]) not in strong)):return None
        return dict(rank=RANKS[order[0]],score=float(mean[order[0]]),frames=[o['frame'] for o in samples])
