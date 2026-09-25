"""A：自动检查样本可读性、像素重复、来源划分及负样本与人脸标注的交集。"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import cv2
from src.data_io import load_manifest,write_json
from src.data_preparation import audit_rows,expanded_faces,intersects


def audit(manifest,pages_path):
    """自动通过只说明清单与已有标注一致，不代表人工确认了全部图片。"""
    manifest=Path(manifest)
    pages_list=json.loads(Path(pages_path).read_text(encoding='utf-8'))
    pages={page['page_id']:page for page in pages_list}
    if len(pages)!=len(pages_list):
        raise ValueError('Duplicate page_id in full-page manifest')
    checked=[]
    issues=[]
    low_texture=[]
    for row in load_manifest(manifest):
        image=row.pop('_image')
        if image.shape!=(24,24,3):
            issues.append(dict(image=row['image'],reason='patch_shape_mismatch'))
        digest=hashlib.sha256(image.tobytes()).hexdigest()
        if row.get('pixel_sha256')!=digest:
            issues.append(dict(image=row['image'],reason='pixel_fingerprint_mismatch'))
        # 必须使用实际像素指纹检查跨集合，不盲信清单里的旧指纹。
        row['pixel_sha256']=digest
        page=pages.get(row.get('page_id'))
        if page is None or row['split']!=page['split'] or row['source_id']!=page['source_id']:
            issues.append(dict(image=row['image'],reason='page_or_split_mismatch'))
        else:
            if not (manifest.parent/row['source_image']).is_file():
                issues.append(dict(image=row['image'],reason='missing_original'))
            forbidden=expanded_faces(page['bboxes'],page['width'],page['height'])
            if row['label']==-1 and any(intersects(row['source_bbox'],face) for face in forbidden):
                issues.append(dict(image=row['image'],reason='negative_overlaps_face'))
        # 缩成24×24之后纹理还可能变弱；只列出待复核，不擅自删除样本。
        std=float(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).std())
        if row['label']==-1 and std<12:
            low_texture.append(dict(image=row['image'],std=std))
        checked.append(row)
    audit_rows(checked)
    counts={split:dict(Counter('positive' if r['label']==1 else 'negative' for r in checked if r['split']==split))
            for split in ('train','val','test')}
    return dict(ok=not issues,total=len(checked),split_counts=counts,issues=issues,
                low_texture_after_resize=low_texture,
                review_counts=dict(Counter(r.get('review_status','missing') for r in checked)),
                note='Automatic consistency audit; human review and near-duplicate review remain separate')


def main():
    """清单和整页标注各传一个路径，结果另存 JSON。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--pages',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    report=audit(args.manifest,args.pages)
    write_json(args.output,report)
    print(f'checked={report["total"]}, ok={report["ok"]}, low_texture={len(report["low_texture_after_resize"])}')
    if not report['ok']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
