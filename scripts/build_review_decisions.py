"""Record a completed context-sheet review as an auditable decision JSON."""
import argparse
import json
from pathlib import Path

from src.data_io import write_json


def build(index_path, output, rejected=()):
    rows = json.loads(Path(index_path).read_text('utf-8'))
    rejected = set(rejected)
    known = {row['review_index'] for row in rows}
    if not rejected <= known:
        raise ValueError(f'Unknown rejected indices: {sorted(rejected - known)}')
    decisions = []
    for row in rows:
        index = row['review_index']
        if index in rejected:
            status = 'rejected'
            note = 'possible complete missed face; excluded from negative training'
        else:
            status = 'accepted'
            note = ('confirmed non-face: hair/text/clothing/line/texture/object '
                    'or local eye/face fragment')
        decisions.append(dict(
            review_index=index, review_status=status,
            review_method='human_context_sheet_visual_review', note=note,
        ))
    write_json(output, decisions)
    return dict(total=len(decisions), accepted=len(decisions) - len(rejected),
                rejected=len(rejected))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--reject-indices', default='')
    args = parser.parse_args()
    rejected = [int(value) for value in args.reject_indices.split(',') if value]
    print(json.dumps(build(args.index, args.output, rejected), ensure_ascii=False))


if __name__ == '__main__':
    main()
