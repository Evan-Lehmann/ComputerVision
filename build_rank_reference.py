"""Build/audit reference signatures from published V4 training crops only.

Never consumes newly uploaded recordings. Leave-group-out diagnostics contain
correlated observations and incomplete physical-card groups, not independent
acceptance accuracy.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from glyph_features import glyph_signature
from rank_reference import DEFAULT_BANK, ReferenceRankReader, RANKS


def build(output, report):
    from train_rank_classifier_v4 import load_data
    rows = load_data()['train']
    examples = [(row, glyph_signature(row['crop'])) for row in rows]
    usable = [(row,glyph) for row,glyph in examples if glyph is not None]
    if {row['rank'] for row,glyph in usable} != set(RANKS):
        raise ValueError('References must cover all 13 ranks.')
    output = Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('wb') as file:
        np.savez_compressed(file, features=np.stack([g[0] for _,g in usable]),
            aspects=np.array([g[1] for _,g in usable]),
            ranks=np.array([row['rank'] for row,_ in usable]),
            groups=np.array([row['group'] for row,_ in usable]),
            keys=np.array([row['key'] for row,_ in usable]), version=np.array('1'))
    reader = ReferenceRankReader(output)
    if not reader.available:
        raise ValueError(reader.error)
    predictions = []
    for row,_ in usable:
        vote = reader.read(row['crop'], excluded_group=row['group'])
        predictions.append(dict(key=row['key'], actual=row['rank'], group=row['group'], prediction=vote))
    accepted = [row for row in predictions if row['prediction'] and row['prediction']['accepted']]
    errors = [row for row in accepted if row['prediction']['rank'] != row['actual']]
    result = dict(source='Published V4 training split only; no new recording labels',
        version=1, total_training_crops=len(rows), usable_references=len(usable),
        references_by_rank=dict(Counter(row['rank'] for row,_ in usable)),
        similarity_gate=reader.min_similarity, margin_gate=reader.min_margin,
        leave_group_out=dict(total=len(predictions),accepted=len(accepted),incorrect=len(errors),
            accepted_by_rank=dict(Counter(row['actual'] for row in accepted)), errors=errors),
        limitations=['Groups contain correlated observations and are not all physical-card/recording groups.',
                     'Similarity is not a calibrated probability; diagnostics are not independent acceptance.',
                     'Runtime additionally requires distinct-frame agreement and the same CNN top rank.',
                     'Confident CNN results are preserved, including possible high-confidence errors.'],
        predictions=predictions)
    report = Path(report); report.parent.mkdir(parents=True,exist_ok=True)
    report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['predictions']},indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_BANK)
    parser.add_argument('--report',type=Path,default=Path('validation/reference_v1_audit.json'))
    args=parser.parse_args()
    build(args.output,args.report)


if __name__ == '__main__':
    main()
