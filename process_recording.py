"""Analyze saved screenshots without starting the web dashboard."""
import argparse
import json
from pathlib import Path
from round_pipeline import DEFAULT_REGIONS, VisionEngine, apply_review, process_frames, save_result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('frames',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--detector',type=Path)
    parser.add_argument('--classifier',type=Path)
    parser.add_argument('--regions',type=Path,help='JSON array of named relative regions')
    parser.add_argument('--review',type=Path,help='JSON containing events and confirmed fields')
    parser.add_argument('--no-ocr',action='store_true',help='Disable optional Tesseract cross-check')
    parser.add_argument('--no-references',action='store_true',help='Disable uncertain-rank reference cross-check')
    args=parser.parse_args()
    if not args.frames.is_dir():
        parser.error('Frame directory does not exist.')
    regions=json.loads(args.regions.read_text()) if args.regions else DEFAULT_REGIONS
    engine=VisionEngine(args.detector,args.classifier)
    paths=sorted(p for p in args.frames.iterdir() if p.suffix.lower() in ['.png','.jpg','.jpeg'])
    def progress(done,total):
        if done%25==0 or done==total:print('%d/%d frames' % (done,total),flush=True)
    result=process_frames(paths,args.output,engine,regions,progress,ocr_reader=False if args.no_ocr else None,
                          reference_reader=False if args.no_references else None)
    if args.review:
        review=json.loads(args.review.read_text())
        result=apply_review(result,review['events'],review.get('confirmed',False),
                            review.get('rounds_confirmed',False))
        save_result(result,args.output)
    print(json.dumps(dict(cards=result['cards'],count_change=result['count_change'],
        known_count_change=result['known_count_change'],count_verified=result['count_verified'],
        rounds=result['rounds'],needs_round_review=result['needs_round_review']),indent=2))

if __name__=='__main__':main()
