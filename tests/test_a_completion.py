"""验证扩边不污染真值、数据版本不改变划分，以及感知候选索引不漏边界。"""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from src.data_io import write_image,write_json
from src.data_preparation import positive_crop_box
from scripts.revise_detection_crops import revise
from scripts.find_near_duplicates import candidate_pairs


class ACompletionTests(unittest.TestCase):
    def test_exported_feature_definition_matches_computation(self):
        from src.channels11 import compute_11_channels
        definition=json.loads((Path(__file__).resolve().parents[1]/'configs/feature_definition.json').read_text())
        gray=np.random.default_rng(72).integers(0,256,(12,14),dtype=np.uint8)
        reference=[]
        for spec in definition['channels']:
            if spec['operation']=='identity':
                reference.append(gray.copy());continue
            plane=np.zeros_like(gray);source=reference[spec['source']];m=spec['invalid_right_bottom']
            for y in range(gray.shape[0]-m):
                for x in range(gray.shape[1]-m):
                    if spec['operation']=='average4':
                        value=(sum(int(source[y+dy,x+dx]) for dx,dy in spec['offsets_xy'])+2)//4
                    else:
                        ax,ay=spec['minuend_xy'];bx,by=spec['subtrahend_xy']
                        value=(int(source[y+ay,x+ax])-int(source[y+by,x+bx])+255)//2
                    plane[y,x]=value
            reference.append(plane)
        for a,b in zip(reference,compute_11_channels(gray)):
            np.testing.assert_array_equal(a,b)

    def test_margin_is_clipped_and_avoids_new_neighbor(self):
        self.assertEqual(positive_crop_box([0,0,20,20],40,40,.1)[0],[0,0,22,22])
        crop,used=positive_crop_box([10,10,30,30],60,60,.1,[[31,10,50,30]])
        self.assertLess(used,.1)
        self.assertLessEqual(crop[2],31)
        self.assertLessEqual(crop[0],10)
        self.assertGreaterEqual(crop[2],30)
        with self.assertRaises(ValueError):
            positive_crop_box([0,0,20,20],40,40,float('nan'))

    def test_version_keeps_truth_split_and_negative_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'v1';output=root/'v2'
            image=np.random.default_rng(8).integers(0,256,(60,60,3),dtype=np.uint8)
            write_image(source/'original.png',image)
            write_image(source/'positive.png',image[10:34,10:34])
            write_image(source/'negative.png',image[35:59,35:59])
            rows=[dict(image='positive.png',source_image='original.png',source_id='one',page_id='p',
                       split='train',label=1,bbox=[0,0,24,24],source_bbox=[10,10,34,34]),
                  dict(image='negative.png',source_image='original.png',source_id='one',page_id='p',
                       split='train',label=-1,bbox=[0,0,24,24],source_bbox=[35,35,59,59])]
            pages=[dict(page_id='p',bboxes=[[10,10,34,34]],image='original.png')]
            write_json(source/'manifest.json',rows);write_json(source/'pages.json',pages)
            write_json(source/'book_splits.json',dict(seed=42,books={'one':'train'}))
            before=(source/'manifest.json').read_bytes()
            revise(source,output)
            new=json.loads((output/'manifest.json').read_text())
            self.assertEqual(new[0]['source_gt_bbox'],rows[0]['source_bbox'])
            self.assertEqual(new[0]['source_bbox'],[7,7,37,37])
            self.assertEqual(new[0]['split'],'train')
            self.assertEqual(json.loads((output/'pages.json').read_text())[0]['bboxes'],pages[0]['bboxes'])
            self.assertEqual((source/'manifest.json').read_bytes(),before)
            self.assertEqual((output/'negative.png').read_bytes(),(source/'negative.png').read_bytes())
            with self.assertRaises(FileExistsError): revise(source,output)

    def test_near_duplicate_four_bit_boundary_and_split_scope(self):
        thumb=np.arange(1024,dtype=np.int16).reshape(32,32)%256
        rows=[dict(image='a',code=0,thumbnail=thumb,aspect=1,split='train'),
              dict(image='b',code=(1<<0)|(1<<13)|(1<<26)|(1<<39),thumbnail=thumb+1,aspect=1,split='test'),
              dict(image='c',code=0,thumbnail=thumb,aspect=1,split='train')]
        pairs=candidate_pairs(rows)
        self.assertTrue(any(p['left']=='a' and p['right']=='b' and p['hamming']==4 for p in pairs))
        self.assertFalse(any(p['left']=='a' and p['right']=='c' for p in pairs))
        rows[1]['thumbnail']=thumb+50
        self.assertEqual(candidate_pairs(rows),[])


if __name__=='__main__': unittest.main()
