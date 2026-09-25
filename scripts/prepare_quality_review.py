"""A：全量生成可复核的质量队列，并按来源抽取正负例展示原图上下文。"""
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from src.data_io import read_image,write_image,write_json
from src.data_preparation import intersects
from scripts.review_detection_samples import fit_image


def run(dataset,output):
    dataset,output=Path(dataset),Path(output)
    if output.exists():
        raise FileExistsError('Use a new quality review directory')
    rows=json.loads((dataset/'manifest.json').read_text(encoding='utf-8'))
    pages={p['page_id']:p for p in json.loads((dataset/'pages.json').read_text(encoding='utf-8'))}
    groups=defaultdict(list); queue=[]
    for row in rows:
        patch=read_image(dataset/row['image'])
        std=float(cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY).std())
        page=pages[row['page_id']]; box=row['source_bbox']
        flags=[]
        if std<12: flags.append('low_texture')
        if box[0]==0 or box[1]==0 or box[2]==page['width'] or box[3]==page['height']:
            flags.append('touches_image_boundary')
        if row['label']==1:
            truth=row.get('source_gt_bbox',box)
            if any(intersects(truth,b) for b in page['bboxes'] if b!=truth):
                flags.append('preexisting_neighbor_overlap')
            ratio=(truth[2]-truth[0])/(truth[3]-truth[1])
            if ratio<.6 or ratio>1.6: flags.append('unusual_face_aspect')
        queue.append(dict(image=row['image'],label=row['label'],split=row['split'],
                          source_id=row['source_id'],flags=';'.join(flags),gray_std=round(std,3),
                          review_status='needs_review',pose='',occlusion='',content_type='',note=''))
        groups[(row['source_id'],row['label'])].append(row)
    output.mkdir(parents=True)
    with (output/'quality_queue.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(queue[0]));writer.writeheader();writer.writerows(queue)
    # 每个来源和类别选一个稳定代表，再分集合均衡抽取；用于观察，不代替全量验收。
    selected=[]
    for label in (1,-1):
        for split in ('train','val','test'):
            candidates=[sorted(items,key=lambda r:hashlib.sha256(r['image'].encode()).hexdigest())[0]
                        for (source,kind),items in groups.items() if kind==label and items[0]['split']==split]
            candidates.sort(key=lambda r:hashlib.sha256(('review42'+r['image']).encode()).hexdigest())
            selected.extend(candidates[:12])
    # 抽样数最多72（验证来源不足12时更少）；每格包含放大小图和真实像素上下文。
    evidence=[]; tiles=[]
    cached={}
    for index,row in enumerate(selected):
        key=row['source_image']
        if key not in cached: cached[key]=read_image(dataset/key)
        original=cached[key].copy(); x1,y1,x2,y2=row['source_bbox']
        cv2.rectangle(original,(x1,y1),(x2-1,y2-1),(0,180,0),2)
        margin=max(x2-x1,y2-y1)//2
        context=original[max(0,y1-margin):min(original.shape[0],y2+margin),max(0,x1-margin):min(original.shape[1],x2+margin)]
        tile=np.full((230,480,3),255,np.uint8)
        tile[:200,:160]=fit_image(read_image(dataset/row['image']),160,200)
        tile[:200,160:]=fit_image(context,320,200)
        cv2.putText(tile,f'{index+1:03d} label={row["label"]} {row["split"]}',(5,222),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1)
        tiles.append(tile)
        evidence.append(dict(index=index+1,**{k:row[k] for k in ('image','source_image','source_bbox','split','label','source_id')}))
    for start in range(0,len(tiles),8):
        batch=tiles[start:start+8]+[np.full((230,480,3),255,np.uint8)]*(8-len(tiles[start:start+8]))
        write_image(output/f'sheet_{start//8+1:02d}.jpg',np.vstack([np.hstack(batch[i:i+2]) for i in range(0,8,2)]))
    write_json(output/'sampled_review.json',evidence)
    report=dict(total=len(rows),sampled=len(selected),
                flags=dict(Counter(flag for r in queue for flag in r['flags'].split(';') if flag)),
                note='Flags are review prompts, not labels; category fields require visual observation')
    write_json(output/'report.json',report); print(report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.dataset,args.output)


if __name__=='__main__':
    main()
