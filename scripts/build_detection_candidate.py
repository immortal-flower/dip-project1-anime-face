"""A：按明确质量规则导出候选清单，不把自动过滤和AI抽查冒充人工验收。"""
import argparse
import json
import os
from pathlib import Path
import cv2
from src.data_io import read_image,write_json


def build(dataset,observations,output):
    dataset,output=Path(dataset),Path(output)
    if output.exists(): raise FileExistsError('Use a new candidate manifest')
    rows=json.loads((dataset/'manifest.json').read_text(encoding='utf-8'))
    notes=json.loads(Path(observations).read_text(encoding='utf-8'))
    decisions={note['image']:note for note in notes}
    if len(decisions)!=len(notes) or set(decisions)-{r['image'] for r in rows}:
        raise ValueError('Observation image IDs must be unique and present in dataset')
    included,excluded=[],[]
    for row in rows:
        reason=[];note=decisions.get(row['image'])
        if note and note['recommendation']=='quarantine':
            reason.append(note['reason'])
        image=read_image(dataset/row['image'])
        if row['label']==-1 and cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).std()<12:
            reason.append('post_resize_gray_std_below_12')
        if reason:
            excluded.append(dict(image=row['image'],split=row['split'],label=row['label'],reasons=reason))
            continue
        row=dict(row)
        for key in ('image','source_image'):
            row[key]=os.path.relpath((dataset/row[key]).resolve(),output.parent.resolve()).replace('\\','/')
        row['review_status']='needs_review'
        included.append(row)
    write_json(output,included)
    write_json(output.with_suffix('.excluded.json'),excluded)
    write_json(output.with_suffix('.summary.json'),dict(total=len(rows),included=len(included),excluded=len(excluded),
               status='candidate only; remaining images not fully reviewed; evaluation pages unchanged',
               split_counts={s:{'positive':sum(r['split']==s and r['label']==1 for r in included),
                                'negative':sum(r['split']==s and r['label']==-1 for r in included)} for s in ('train','val','test')}))
    print(f'Candidate {len(included)}, quarantined {len(excluded)}; original manifest unchanged')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True);parser.add_argument('--observations',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();build(args.dataset,args.observations,args.output)


if __name__=='__main__': main()
