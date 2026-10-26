"""Optional, local Tesseract cross-check for uncertain red rank glyphs.

Uses a pretrained OCR engine, not labels from the recording being analyzed.
No network calls are made. Missing/failed OCR leaves the rank unresolved.
"""
import csv
import os
import shutil
import subprocess
import cv2
import numpy as np

VALID_RANKS = set(['A','2','3','4','5','6','7','8','9','10','J','Q','K'])


def red_glyph_image(crop):
    if crop is None or crop.size == 0:
        return None
    b,g,r = cv2.split(crop.astype(np.float32))
    redness = np.minimum(r-g,r-b)
    if np.count_nonzero(redness > 45) < 5:
        return None
    # Black borders and black adjacent print have no positive red chroma.
    ink = np.clip((redness-25)/100,0,1)
    yy,xx = np.where(ink > .25)
    if len(xx) < 5:
        return None
    glyph = ink[yy.min():yy.max()+1,xx.min():xx.max()+1]
    h,w = glyph.shape
    if h < 3 or w < 2:
        return None
    # Preserve both disconnected digits of 10; no component deletion/erosion.
    scale = min(168/w,168/h)
    nw,nh = max(1,round(w*scale)),max(1,round(h*scale))
    image = np.full((208,208),255,np.uint8)
    x,y = (208-nw)//2,(208-nh)//2
    resized = cv2.resize(glyph,(nw,nh),interpolation=cv2.INTER_LINEAR)
    image[y:y+nh,x:x+nw] = np.clip((1-resized)*255,0,255).astype(np.uint8)
    return image


class TesseractRankReader:
    def __init__(self, executable=None, runner=None):
        self.executable = executable or shutil.which('tesseract')
        self.runner = runner or subprocess.run
        self.error = None

    @property
    def available(self):
        return bool(self.executable)

    def read(self, crop):
        if not self.available:
            return None
        image = red_glyph_image(crop)
        if image is None:
            return None
        ok,encoded = cv2.imencode('.png',image)
        if not ok:
            return None
        try:
            result = self.runner([self.executable,'stdin','stdout','-l','eng','--psm','10',
                '-c','tessedit_char_whitelist=A2345678910JQK','tsv'],
                input=encoded.tobytes(),capture_output=True,timeout=10,
                env=dict(os.environ,OMP_THREAD_LIMIT='1'))
            if result.returncode:
                self.error = 'OCR failed; check Tesseract and its English language data.'
                return None
            words = [row for row in csv.DictReader(result.stdout.decode('utf-8',errors='replace').splitlines(),delimiter='\t')
                     if row.get('text','').strip()]
            if len(words) != 1:
                return None
            text = words[0]['text'].strip().upper()
            confidence = float(words[0]['conf'])/100
            if text not in VALID_RANKS or not np.isfinite(confidence):
                return None
            return dict(rank=text,confidence=max(0,min(1,confidence)))
        except (OSError,subprocess.TimeoutExpired,ValueError,KeyError) as exc:
            self.error = 'OCR could not complete: ' + type(exc).__name__
            return None

    def consensus(self, observations):
        if not self.available or len(observations) < 2:
            return None
        indices = sorted(set([0,len(observations)//2,len(observations)-1]))
        frames,votes = set(),[]
        for index in indices:
            observation = observations[index]
            frame = observation['frame']
            if frame in frames:
                continue
            frames.add(frame)
            result = self.read(observation['crop'])
            if result and result['confidence'] >= .90:
                votes.append(dict(result,frame=frame))
        if len(votes) < 2 or len({vote['rank'] for vote in votes}) != 1:
            return None
        return dict(rank=votes[0]['rank'],confidence=min(vote['confidence'] for vote in votes),votes=votes)
