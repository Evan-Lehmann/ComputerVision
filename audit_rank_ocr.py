"""Measure the optional OCR reader on the existing labelled crop datasets.

This is a crop-level diagnostic, not a claim about round accuracy. Runtime uses
an additional multi-frame agreement gate and never overrides a confident CNN.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import subprocess
import cv2
from rank_ocr import TesseractRankReader, VALID_RANKS, red_glyph_image

ROOT = Path(__file__).resolve().parent


def audit(root=ROOT):
    reader=TesseractRankReader()
    if not reader.available:
        raise RuntimeError('Tesseract is not installed; install it before running the OCR audit.')
    datasets=[('rank_value_dataset','labels.csv'),('rank_value_candidates','candidate_labels.csv')]
    rows=[]
    for folder,csv_name in datasets:
        directory=Path(root)/folder
        with (directory/csv_name).open(newline='') as file:
            for row in csv.DictReader(file):
                rank=row['rank'].strip()
                if rank not in VALID_RANKS:
                    continue
                path=directory/row['crop']
                image=cv2.imread(str(path))
                if image is None:
                    raise ValueError('Could not read labelled crop: '+str(path))
                eligible=red_glyph_image(image) is not None
                prediction=reader.read(image) if eligible else None
                if reader.error:
                    raise RuntimeError(reader.error)
                accepted=bool(prediction and prediction['confidence']>=.90)
                rows.append(dict(crop=folder+'/'+row['crop'],actual=rank,red_eligible=eligible,
                    predicted=prediction['rank'] if prediction else None,
                    confidence=prediction['confidence'] if prediction else None,
                    high_confidence=accepted,
                    correct=prediction['rank']==rank if accepted else None))
    if not rows:
        raise ValueError('No labelled crops were evaluated.')
    high=[row for row in rows if row['high_confidence']]
    version=subprocess.run([reader.executable,'--version'],capture_output=True,text=True,check=True).stdout.splitlines()[0]
    return dict(scope='Existing labelled public crops; single-view OCR, not temporal/round accuracy',
        tesseract=version,confidence_gate=.90,total_crops=len(rows),
        red_eligible=sum(row['red_eligible'] for row in rows),
        high_confidence=len(high),correct_high_confidence=sum(row['correct'] for row in high),
        incorrect_high_confidence=sum(not row['correct'] for row in high),
        eligible_by_rank=dict(sorted(Counter(row['actual'] for row in rows if row['red_eligible']).items())),
        high_confidence_by_rank=dict(sorted(Counter(row['actual'] for row in high).items())),
        errors=[row for row in high if not row['correct']],predictions=rows,
        limitations=['Examples contain correlated views and severely imbalanced classes.',
                     'OCR scores are not calibrated probabilities of correctness.',
                     'Runtime additionally requires agreement in distinct frames.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    try:
        result=audit()
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
    except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as exc:
        parser.exit(1,str(exc)+'\n')
    print(json.dumps({key:result[key] for key in ['total_crops','red_eligible','high_confidence','correct_high_confidence','incorrect_high_confidence']},indent=2))


if __name__=='__main__':
    main()
