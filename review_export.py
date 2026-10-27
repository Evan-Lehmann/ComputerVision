"""Compact, explicitly reviewed crop export; never trains or uploads anything."""
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

from round_pipeline import RANKS


def reviewed_crop_bundle(result, output_dir, recording_group):
    if not result.get('review_confirmed') or not result.get('count_verified'):
        raise ValueError('Confirm and save a fully resolved card review before exporting labels.')
    included = [event for event in result['events'] if not event.get('excluded')]
    if any(not event.get('reviewed') or event['rank'] not in RANKS for event in included):
        raise ValueError('Every included card must have a reviewed, resolved rank.')
    output_dir = Path(output_dir).resolve()
    rows, files, labelled_cards = [], {}, set()
    for event in included:
        seen = set()
        for evidence in event.get('evidence', []):
            name = evidence['image']
            if not isinstance(name, str) or Path(name).name != name:
                raise ValueError('Invalid evidence filename.')
            path = (output_dir / name).resolve()
            if not path.is_relative_to(output_dir):
                raise ValueError('Evidence must remain within the analysis directory.')
            data = path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            archive_name = 'crops/' + name
            files[archive_name] = data
            labelled_cards.add(event['id'])
            rows.append(dict(crop=archive_name, rank=event['rank'],
                card_group=str(recording_group) + ':' + str(event['id']),
                recording_group=recording_group, region=event['region'],
                source_frame=evidence.get('source', ''), frame_index=evidence.get('frame', ''),
                sha256=digest, annotation_source='user_confirmed_review'))
    if not rows:
        raise ValueError('There are no reviewed crop images to export.')
    csv_buffer = io.StringIO(newline='')
    writer = csv.DictWriter(csv_buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    metadata = dict(schema_version=1, recording_group=recording_group,
        count_scope='captured_cards', confirmed_cards=len(included),
        labelled_cards=len(labelled_cards), crop_count=len(rows),
        annotation_source='user_confirmed_review',
        notes=['Views in one card_group are correlated, not independent labelled cards.',
               'Keep recording/card groups together when splitting training and validation.',
               'Manually added cards without image evidence are not exported.'])
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('labels.csv', csv_buffer.getvalue())
        archive.writestr('metadata.json', json.dumps(metadata, indent=2))
        for name, data in files.items():
            archive.writestr(name, data)
    output.seek(0)
    return output
