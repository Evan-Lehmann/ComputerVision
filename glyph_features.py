"""Shared glyph geometry for association and reference matching; no rank labels."""
import cv2
import numpy as np


def glyph_signature(crop):
    """Glyph signature normalized against the card's tinted background.

    Remove spanning borders, retaining compact glyphs even at a crop edge (both
    digits of 10). The CNN still receives its original detector crop.
    """
    if crop is None or not crop.size:
        return None
    h, w = crop.shape[:2]
    tx, ty = int(w * .12), int(h * .12)
    inner = crop[ty:h-ty, tx:w-tx].astype(np.float32)
    if min(inner.shape[:2]) < 3:
        return None
    background = np.maximum(np.percentile(inner, 90, axis=(0, 1)), 1)
    normalized = np.clip(inner / background, 0, 1).astype(np.float32)
    ink = 1 - cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
    mask = (ink > max(.04, .18 * float(ink.max()))).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    clean = np.zeros_like(mask)
    for label, (x, y, width, height, area) in enumerate(stats[1:], 1):
        interior = (x > 0 and y > 0 and x+width < mask.shape[1] and y+height < mask.shape[0])
        compact_edge_glyph = (height >= 3 and height < mask.shape[0]*.85
                              and width < mask.shape[1]*.85)
        if area >= 3 and (interior or compact_edge_glyph):
            clean[labels == label] = 1
    yy, xx = np.where(clean)
    if len(xx) < 6:
        return None
    tight = (ink * clean)[yy.min():yy.max()+1, xx.min():xx.max()+1]
    h, w = tight.shape
    scale = 28 / max(h, w)
    nw, nh = max(1, round(w*scale)), max(1, round(h*scale))
    canvas = np.zeros((32, 32), dtype=np.float32)
    canvas[(32-nh)//2:(32-nh)//2+nh, (32-nw)//2:(32-nw)//2+nw] = cv2.resize(tight, (nw, nh))
    pixels = normalized[clean.astype(bool)]
    red = np.mean((pixels[:, 2] > pixels[:, 1]+.10)
                  & (pixels[:, 2] > pixels[:, 0]+.10)) > .5
    return canvas, w/h, bool(red), float(np.mean(background))

