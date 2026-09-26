"""A：接入C的28点交付，保留原坐标和split，生成可复核的联合候选清单。"""
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path,PureWindowsPath
import shutil
import zipfile
import zlib
import numpy as np
from src.data_io import read_image,write_json,load_manifest


def landmark_issues(row,width,height):
    """格式错误拒绝导入；可见点越界报告给C，不擅自截断或改成不可见。"""
    if row.get('landmark_order')!='hysts28-v1' or row.get('label')!=1:
        raise ValueError('Expected positive hysts28-v1 landmark annotations')
    if row.get('split') not in ('train','val','test'):
        raise ValueError('Invalid existing C split')
    points=np.asarray(row['landmarks'],dtype=float);visibility=np.asarray(row['visibility'])
    if points.shape!=(28,2) or not np.isfinite(points).all() or visibility.shape!=(28,) or not np.isin(visibility,[0,1]).all():
        raise ValueError('Expected 28 finite points and binary visibility')
    box=row['bbox']
    if len(box)!=4 or not all(isinstance(v,int) for v in box) or not (0<=box[0]<box[2]<=width and 0<=box[1]<box[3]<=height):
        raise ValueError('Invalid C bbox')
    outside=(points[:,0]<0)|(points[:,0]>=width)|(points[:,1]<0)|(points[:,1]>=height)
    return [dict(kind='visible_point_outside_image',point_index=int(i),xy=points[i].tolist(),width=width,height=height)
            for i in np.where(outside & (visibility==1))[0]]


def near_components(inventory,pairs,anchors):
    """以完全重复像素组为节点连起近似候选，检查含多个C split的潜在冲突。"""
    parent={}
    def find(key):
        parent.setdefault(key,key)
        if parent[key]!=key: parent[key]=find(parent[key])
        return parent[key]
    for pair in pairs:
        a,b=inventory[pair['left']]['pixel_sha256'],inventory[pair['right']]['pixel_sha256']
        ra,rb=find(a),find(b)
        if ra!=rb: parent[max(ra,rb)]=min(ra,rb)
    memberships={key:find(key) for key in list(parent)}
    attached=defaultdict(list)
    for digest,rows in anchors.items():
        attached[memberships.get(digest,digest)].extend(rows)
    conflicts=[dict(component=component,anchors=rows)
               for component,rows in attached.items() if len({r['split'] for r in rows})>1]
    return memberships,attached,conflicts


def integrate(manifest,archive,image_root,inventory_path,near_path,manga_manifest,output):
    manifest,archive,image_root,inventory_path,near_path,manga_manifest,output=map(Path,
        (manifest,archive,image_root,inventory_path,near_path,manga_manifest,output))
    if output.exists(): raise FileExistsError('Use a new integration version directory')
    original=manifest.read_bytes();source_rows=json.loads(original.decode('utf-8-sig'))
    with inventory_path.open(encoding='utf-8-sig',newline='') as stream:
        inventory={r['image']:r for r in csv.DictReader(stream)}
    pairs=json.loads(near_path.read_text(encoding='utf-8'))
    rows,issues,anchors=[],[],defaultdict(list)
    names=set();source_groups={};pixel_splits={}
    output.mkdir(parents=True);(output/'images').mkdir()
    with zipfile.ZipFile(archive) as bundle:
        for upstream in source_rows:
            name=PureWindowsPath(upstream['image']).name
            if name in names: raise ValueError(f'Duplicate annotated image name: {name}')
            names.add(name)
            path=image_root/name;raw=path.read_bytes();entry=bundle.getinfo('images/'+name)
            if len(raw)!=entry.file_size or zlib.crc32(raw)&0xffffffff!=entry.CRC:
                raise ValueError(f'C archive and local image differ: {name}')
            image=read_image(path);height,width=image.shape[:2]
            problems=landmark_issues(upstream,width,height)
            # 沿用AnimeFace清点时的指纹格式：BGR像素后接高度、宽度，不能混用纯像素哈希。
            digest=hashlib.sha256(image.tobytes()+f'{height},{width}'.encode()).hexdigest()
            if digest!=inventory['images/'+name]['pixel_sha256']:
                raise ValueError(f'Image changed since inventory: {name}')
            for mapping,key in ((source_groups,upstream['source_id']),(pixel_splits,digest)):
                if mapping.setdefault(key,upstream['split'])!=upstream['split']:
                    raise ValueError(f'Confirmed cross-split C leakage: {name}')
            anchors[digest].append(dict(image=name,split=upstream['split']))
            row=dict(upstream)
            row.update(image='images/'+name,upstream_image=upstream['image'],pixel_sha256=digest,
                       pixel_hash_policy='sha256(BGR_bytes + utf8(height,width))',
                       integration_status='needs_C_confirmation' if problems else 'format_and_bounds_checked',
                       annotation_provenance='human_reviewed declared by C; retained as supplied')
            shutil.copy2(path,output/row['image']);rows.append(row)
            issues.extend(dict(image=row['image'],split=row['split'],**problem) for problem in problems)
    membership,attached,conflicts=near_components(inventory,pairs,anchors)
    conflict_digests={digest for digest,component in membership.items() if any(c['component']==component for c in conflicts)}
    valid=[r for r in rows if r['integration_status']=='format_and_bounds_checked' and r['pixel_sha256'] not in conflict_digests]
    write_json(output/'landmarks_imported.json',rows)
    write_json(output/'landmarks_validated.json',valid)
    write_json(output/'annotation_issues.json',issues)
    write_json(output/'near_split_conflicts.json',conflicts)
    # 所有相同像素的别名都绑定C原split；近似关系只是保留候选，不做自动标签传播。
    registry=[]
    for name,item in inventory.items():
        digest=item['pixel_sha256'];component=membership.get(digest,digest)
        locked={r['split'] for r in anchors.get(digest,[])}
        suggested={r['split'] for r in attached.get(component,[])}
        registry.append(dict(image=name,pixel_sha256=digest,locked_split=next(iter(locked)) if locked else '',
                             near_component=component,near_anchor_splits=';'.join(sorted(suggested)),
                             status='exact_alias_locked' if locked else 'near_candidate_hold' if suggested else 'unassigned'))
    with (output/'animeface_split_registry.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(registry[0]));writer.writeheader();writer.writerows(registry)
    # 区域训练可合并，整页评价仍单独使用Manga109 pages，不把头像全图框当场景真值。
    manga=json.loads(manga_manifest.read_text(encoding='utf-8'))
    rebased=[]
    for original_row in manga:
        row=dict(original_row)
        for key in ('image','source_image'):
            if key in row:row[key]=os.path.relpath((manga_manifest.parent/row[key]).resolve(),output.resolve()).replace('\\','/')
        rebased.append(row)
    joint=rebased+valid
    write_json(output/'joint_manifest_candidate.json',joint)
    # 使用团队公共读取器完整走一遍，验证路径、bbox、点格式和来源split。
    checked=sum(1 for _ in load_manifest(output/'joint_manifest_candidate.json'))
    report=dict(source_manifest_sha256=hashlib.sha256(original).hexdigest(),source_count=len(source_rows),
                imported_split_counts=dict(Counter(r['split'] for r in rows)),
                validated_count=len(valid),validated_split_counts=dict(Counter(r['split'] for r in valid)),
                point_issue_count=len(issues),images_with_point_issues=len({r['image'] for r in issues}),
                landmark_order='hysts28-v1',near_cross_split_components=len(conflicts),
                duplicate_pixel_groups_in_C=sum(len(group)>1 for group in anchors.values()),
                registry_status_counts=dict(Counter(r['status'] for r in registry)),
                joint_count=checked,joint_split_counts={s:{'positive':sum(r['split']==s and r['label']==1 for r in joint),
                                                        'negative':sum(r['split']==s and r['label']==-1 for r in joint)} for s in ('train','val','test')},
                source_unchanged=(manifest.read_bytes()==original),
                status='integration candidate; Manga quality review and C point exceptions remain pending')
    write_json(output/'integration_report.json',report)
    print(report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True);parser.add_argument('--archive',required=True)
    parser.add_argument('--image-root',default='data/animeface/images')
    parser.add_argument('--inventory',default='data/inspection/animeface/inventory.csv')
    parser.add_argument('--near-pairs',default='results/near_duplicates_anime_v1/pairs.json')
    parser.add_argument('--manga-manifest',default='data/processed/manga109_detection_v2_margin10/manifest_candidate.json')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    integrate(args.manifest,args.archive,args.image_root,args.inventory,args.near_pairs,args.manga_manifest,args.output)


if __name__=='__main__':main()
