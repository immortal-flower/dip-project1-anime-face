"""Convert teacher-style point objects into the shared region manifest format."""
import argparse
import json
import os
from pathlib import Path
from src.data_io import write_json, load_manifest
from src.landmark_schema import LANDMARK_ORDER, validate_landmark_order, validate_landmarks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    source, output = Path(a.input), Path(a.output)
    rows = json.loads(source.read_text(encoding='utf-8'))
    for r in rows:
        points = r['landmarks']
        if len(points) != 28:
            raise ValueError('Expected exactly 28 points')
        if isinstance(points[0], dict):
            r['visibility'] = [q['visibility'] for q in points]
            r['landmarks'] = [[q['x'], q['y']] for q in points]
        r.setdefault('landmark_order', LANDMARK_ORDER)
        validate_landmark_order(r['landmark_order'])
        validate_landmarks(r['landmarks'], r['visibility'])
        r['image'] = os.path.relpath((source.parent/r['image']).resolve(), output.parent.resolve())
        r['label'] = 1
        if 'source_id' not in r or 'split' not in r:
            raise ValueError('Specify source_id and fixed split before conversion')
    write_json(output, rows)
    list(load_manifest(output))
    print(f'Validated {len(rows)} landmark records with {LANDMARK_ORDER}.')


if __name__ == '__main__':
    main()
