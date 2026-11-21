"""Visual annotation of the chip selector. No input or wager actions."""
from pathlib import Path
from functools import lru_cache
import cv2
import numpy as np

@lru_cache(maxsize=1)
def template():
    return cv2.imread(str(Path(__file__).parent/'assets/chip_row.png'))

def locate_chip(image):
    """Require the complete 1/2/5/10 selector row, not a red circle alone."""
    h,w=image.shape[:2]
    scale=min(1.,1000/w)
    small=cv2.resize(image,None,fx=scale,fy=scale) if scale<1 else image
    sh,sw=small.shape[:2];x0,y0=int(sw*.2),int(sh*.35)
    area=small[y0:int(sh*.8),x0:int(sw*.85)]
    best=None
    for factor in np.linspace(.45,1.8,23):
        target=cv2.resize(template(),None,fx=float(factor),fy=float(factor))
        th,tw=target.shape[:2]
        if th>area.shape[0] or tw>area.shape[1]:continue
        _,score,_,point=cv2.minMaxLoc(cv2.matchTemplate(area,target,cv2.TM_CCOEFF_NORMED))
        if best is None or score>best[0]:best=(score,point,factor)
    if best is None or best[0]<.86:return None
    score,(x,y),factor=best
    # Only the first chip in the verified four-chip row is annotated.
    return dict(box=[round((x+x0)/scale),round((y+y0)/scale),
                     round((x+x0+36*factor)/scale),round((y+y0+36*factor)/scale)],
                score=round(float(score),3))
