"""A：把已确认的复核结果导出为新清单，保持标签、框、划分和原始数据不变。"""
import argparse
import csv
import json
import os
from pathlib import Path
from src.data_io import write_json


def apply_review(manifest,review,output):
    """仅导出明确 accepted 的行；pending 和 rejected 都不混入可用清单。"""
    manifest,review,output=Path(manifest),Path(review),Path(output)
    if output.exists():
        raise FileExistsError('Use a new output manifest')
    rows=json.loads(manifest.read_text(encoding='utf-8'))
    identifiers={r['image'] for r in rows}
    if len(identifiers)!=len(rows):
        raise ValueError('Duplicate image paths in manifest')
    with review.open(encoding='utf-8-sig',newline='') as stream:
        decisions=list(csv.DictReader(stream))
    seen=set()
    for decision in decisions:
        key=decision['image']
        if key not in identifiers or key in seen or decision['review_status'] not in ('accepted','rejected'):
            raise ValueError(f'Invalid or repeated decision: {key}')
        seen.add(key)
    decisions={d['image']:d for d in decisions}
    accepted=[]
    for original in rows:
        decision=decisions.get(original['image'])
        if decision is None or decision['review_status']!='accepted':
            continue
        row=dict(original)
        # 新清单位置可能不同，转换相对路径；其余标注字段全部原样保留。
        for field in ('image','source_image'):
            if field in row:
                path=(manifest.parent/row[field]).resolve()
                if not path.is_file():
                    raise FileNotFoundError(path)
                row[field]=os.path.relpath(path,output.parent.resolve()).replace('\\','/')
        row['review_status']='accepted'
        row['sample_review_note']=decision.get('note','')
        accepted.append(row)
    if not accepted:
        raise ValueError('No accepted samples to export')
    write_json(output,accepted)
    summary=dict(accepted=len(accepted),rejected=sum(d['review_status']=='rejected' for d in decisions.values()),
                 pending=len(rows)-len(decisions),split_unchanged=True)
    write_json(output.with_suffix('.review_summary.json'),summary)
    return summary


def main():
    """读取人工判断记录，导出新版本清单并报告仍未复核的数量。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--review',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    print(apply_review(args.manifest,args.review,args.output))


if __name__=='__main__':
    main()
