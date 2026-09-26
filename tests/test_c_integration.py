"""C交付对接的保护测试：不改原点，隔离越界例，识别间接近似关系冲突。"""
import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
import numpy as np
from src.data_io import write_image,write_json
from scripts.integrate_c_landmarks import landmark_issues,near_components,integrate


class CIntegrationTests(unittest.TestCase):
    def test_visible_out_of_bounds_is_reported_not_clipped(self):
        row=dict(label=1,split='train',bbox=[0,0,20,20],landmark_order='hysts28-v1',
                 landmarks=[[10.,10.] for _ in range(28)],visibility=[1]*28)
        row['landmarks'][2]=[10,20.2]
        original=copy.deepcopy(row)
        issues=landmark_issues(row,20,20)
        self.assertEqual(issues[0]['point_index'],2)
        self.assertEqual(row,original)
        row['visibility'][2]=0
        self.assertEqual(landmark_issues(row,20,20),[])

    def test_transitive_near_relation_finds_anchor_conflict(self):
        inventory={name:dict(pixel_sha256=name) for name in ('a','b','c')}
        pairs=[dict(left='a',right='b'),dict(left='b',right='c')]
        anchors={'a':[dict(image='a',split='train')],'c':[dict(image='c',split='test')]}
        membership,attached,conflicts=near_components(inventory,pairs,anchors)
        self.assertEqual(membership['a'],membership['c'])
        self.assertEqual(len(conflicts),1)

    def test_import_preserves_upstream_and_candidate_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);images=root/'raw';images.mkdir()
            rows=[];inventory=[]
            for i,split in enumerate(('train','val','test','train')):
                image=np.random.default_rng(i).integers(0,256,(24,24,3),dtype=np.uint8)
                write_image(images/f'{i}.png',image)
                inventory.append(dict(image=f'images/{i}.png',pixel_sha256=hashlib.sha256(image.tobytes()+b'24,24').hexdigest()))
                rows.append(dict(image=f'D:\\teammate\\{i}.png',source_id=f'c:{i}',split=split,label=1,
                                 bbox=[0,0,24,24],landmark_order='hysts28-v1',landmarks=[[12.,12.] for _ in range(28)],
                                 visibility=[1]*28,annotation_status='human_reviewed'))
            rows[-1]['landmarks'][2]=[12,24.5]
            write_json(root/'c.json',rows);original=(root/'c.json').read_bytes()
            with zipfile.ZipFile(root/'images.zip','w') as z:
                for path in images.glob('*.png'):z.write(path,'images/'+path.name)
            with (root/'inventory.csv').open('w',newline='',encoding='utf-8') as stream:
                writer=csv.DictWriter(stream,fieldnames=['image','pixel_sha256']);writer.writeheader();writer.writerows(inventory)
            write_json(root/'near.json',[])
            write_image(root/'negative.png',np.zeros((24,24,3),np.uint8))
            write_json(root/'manga.json',[dict(image='negative.png',source_id='manga:a',label=-1,split='train',bbox=[0,0,24,24])])
            output=root/'joint'
            report=integrate(root/'c.json',root/'images.zip',images,root/'inventory.csv',root/'near.json',root/'manga.json',output)
            self.assertEqual(report['validated_count'],3)
            self.assertEqual(report['joint_count'],4)
            self.assertEqual((root/'c.json').read_bytes(),original)
            imported=json.loads((output/'landmarks_imported.json').read_text())
            for before,after in zip(rows,imported):
                for key in ('landmarks','visibility','bbox','split','source_id','annotation_status'):
                    self.assertEqual(before[key],after[key])
            self.assertTrue(all((output/r['image']).exists() for r in imported))
            with self.assertRaises(FileExistsError):
                integrate(root/'c.json',root/'images.zip',images,root/'inventory.csv',root/'near.json',root/'manga.json',output)


if __name__=='__main__':unittest.main()
