"""Create a compact GitHub-safe summary from B's local experiment outputs."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from src.data_io import write_json


def _file_identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return dict(file=str(path), size=path.stat().st_size, sha256=digest.hexdigest())


def _metrics(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1)


def _compact_run(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    pages = data['pages']
    return dict(
        file=str(path), split=data['split'], selection=data['selection'],
        page_count=data['metrics']['page_count'], metrics={
            key: data['metrics'][key]
            for key in ('tp', 'fp', 'fn', 'precision', 'recall', 'f1')
        },
        scan_config=data['scan_config'], seconds=data['seconds'],
        mean_page_seconds=data['seconds']/len(pages),
        total_windows=sum(item['windows'] for item in pages),
        mean_windows=sum(item['windows'] for item in pages)/len(pages),
        detections=sum(item['detections'] for item in pages),
    )


def summarize(test_result, validation_result, threshold_result, ablations, output):
    test = json.loads(Path(test_result).read_text(encoding='utf-8'))
    threshold = json.loads(Path(threshold_result).read_text(encoding='utf-8'))
    stage_totals = defaultdict(lambda: dict(evaluated=0, passed=0, rejected=0, seconds=0.0))
    for page in test['pages']:
        for level in page['scan']:
            for stage in level['stages']:
                total = stage_totals[stage['stage']]
                for key in ('evaluated', 'passed', 'rejected'):
                    total[key] += stage[key]
                total['seconds'] += stage['seconds']
    stages = []
    for index in sorted(stage_totals):
        item = dict(stage=index, **stage_totals[index])
        item['conditional_pass_rate'] = item['passed']/item['evaluated']
        item['mean_microseconds_per_evaluation'] = (
            item['seconds']/item['evaluated']*1e6
        )
        stages.append(item)

    page_to_source = {item['page_id']: item['source_id'] for item in test['pages']}
    by_source = defaultdict(lambda: [0, 0, 0, 0])
    for page_id, metrics in test['metrics']['per_page'].items():
        aggregate = by_source[page_to_source[page_id]]
        aggregate[0] += metrics['tp']
        aggregate[1] += metrics['fp']
        aggregate[2] += metrics['fn']
        aggregate[3] += 1
    sources = []
    for source_id, (tp, fp, fn, pages) in by_source.items():
        sources.append(dict(source_id=source_id, pages=pages, **_metrics(tp, fp, fn)))
    sources.sort(key=lambda item: (-item['fp'], item['source_id']))

    per_page = [
        dict(page_id=page_id, source_id=page_to_source[page_id], **metrics)
        for page_id, metrics in test['metrics']['per_page'].items()
    ]
    summary = dict(
        schema_version=1,
        statement='B final detector results; raw Manga109 images are not included',
        selected_model='baseline-region-c1024',
        validation=_compact_run(validation_result),
        threshold_selection=threshold['selected'],
        test=_compact_run(test_result),
        identity=dict(
            pages_sha256=test['pages_sha256'], detector_sha256=test['detector_sha256'],
            config_sha256=test['config_sha256'], missing_prediction_pages=test['metrics']['missing_prediction_pages'],
        ),
        local_artifacts=[
            _file_identity(path) for path in
            [validation_result, threshold_result, test_result, *ablations]
        ],
        stage_totals=stages,
        ablations=[_compact_run(path) for path in ablations],
        source_metrics=sources,
        highest_false_positive_pages=sorted(
            per_page, key=lambda item: (-item['fp'], item['page_id'])
        )[:10],
        highest_true_positive_pages=sorted(
            per_page, key=lambda item: (-item['tp'], item['page_id'])
        )[:10],
    )
    write_json(output, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-result', required=True)
    parser.add_argument('--validation-result', required=True)
    parser.add_argument('--threshold-result', required=True)
    parser.add_argument('--ablation', action='append', default=[])
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = summarize(
        args.test_result, args.validation_result, args.threshold_result,
        args.ablation, args.output,
    )
    print(json.dumps(result['test']['metrics'], ensure_ascii=False))


if __name__ == '__main__':
    main()
