"""A：人工查看样本和原图上下文，按键记录接受/剔除；不更改原始图片。"""
import argparse
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from src.data_io import read_image


def fit_image(image,width=500,height=450):
    """等比例放到白底画布，避免预览把人脸拉伸。"""
    scale=min(width/image.shape[1],height/image.shape[0])
    resized=cv2.resize(image,(max(1,round(image.shape[1]*scale)),max(1,round(image.shape[0]*scale))))
    canvas=np.full((height,width,3),255,np.uint8)
    y,x=(height-resized.shape[0])//2,(width-resized.shape[1])//2
    canvas[y:y+resized.shape[0],x:x+resized.shape[1]]=resized
    return canvas


def main():
    """A接受，R剔除，N跳过，P上一张，Q保存退出；每次判断后自动保存。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--output',required=True,help='独立CSV判断记录，支持继续复核')
    parser.add_argument('--label',choices=['all','negative','positive'],default='negative')
    args=parser.parse_args()
    manifest,output=Path(args.manifest),Path(args.output)
    rows=json.loads(manifest.read_text(encoding='utf-8'))
    rows=[r for r in rows if args.label=='all' or r['label']==(-1 if args.label=='negative' else 1)]
    decisions={}
    if output.exists():
        with output.open(encoding='utf-8-sig',newline='') as stream:
            existing=list(csv.DictReader(stream))
        if any(row.get('review_status') not in ('accepted','rejected') for row in existing):
            raise ValueError('Use a decisions CSV created by this tool, not the initial needs_review table')
        decisions={r['image']:r for r in existing}

    def save():
        # 先写临时文件再替换，减少意外退出造成的判断记录损坏。
        output.parent.mkdir(parents=True,exist_ok=True)
        temporary=output.with_suffix('.tmp')
        with temporary.open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['image','review_status','note'])
            writer.writeheader(); writer.writerows(decisions.values())
        temporary.replace(output)

    index=next((i for i,row in enumerate(rows) if row['image'] not in decisions),len(rows))
    while index<len(rows):
        row=rows[index]
        patch=read_image(manifest.parent/row['image'])
        original=read_image(manifest.parent/row['source_image'])
        x1,y1,x2,y2=row['source_bbox']
        # 绿色框标出实际来源，右图保留周边，帮助判断是否误收局部人脸。
        marked=original.copy()
        cv2.rectangle(marked,(x1,y1),(x2-1,y2-1),(0,200,0),max(1,round(max(original.shape[:2])/500)))
        margin=max(x2-x1,y2-y1)
        context=marked[max(0,y1-margin):min(original.shape[0],y2+margin),max(0,x1-margin):min(original.shape[1],x2+margin)]
        display=np.vstack((np.full((70,1000,3),255,np.uint8),np.hstack((fit_image(patch),fit_image(context)))))
        status=decisions.get(row['image'],{}).get('review_status','pending')
        cv2.putText(display,f'{index+1}/{len(rows)} label={row["label"]} {row["split"]} {status}',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),1)
        cv2.putText(display,'A: accept  R: reject  N: skip  P: previous  Q: save & exit',(10,55),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,0),1)
        cv2.imshow('Sample review: patch | original context',display)
        key=cv2.waitKey(0)&0xff
        if key in (ord('a'),ord('r')):
            decisions[row['image']]=dict(image=row['image'],review_status='accepted' if key==ord('a') else 'rejected',note='human visual review')
            save(); index+=1
        elif key==ord('n'):
            index+=1
        elif key==ord('p'):
            index=max(0,index-1)
        elif key in (ord('q'),27):
            break
    save()
    cv2.destroyAllWindows()
    print(f'Saved {len(decisions)} decisions; dataset manifest unchanged')


if __name__=='__main__':
    main()
