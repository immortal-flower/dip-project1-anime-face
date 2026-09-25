"""A：测试浮点通道对照、整集评价和人工复核导出，重点保护数据含义。"""
import csv
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from src.channels11 import compute_11_channels,compute_11_channels_float
from src.detection_metrics import evaluate_dataset,evaluate_detection
from src.data_io import write_json,write_image
from scripts.apply_sample_review import apply_review


class AWorkflowTests(unittest.TestCase):
    def test_float_equivalence_and_brightness_invariance(self):
        """同取整浮点应完全相等；不发生饱和时整体加亮不应改变方向差分。"""
        gray=np.random.default_rng(11).integers(0,200,(31,37),dtype=np.uint8)
        integer=compute_11_channels(gray)
        floating=compute_11_channels_float(gray,True)
        brighter=compute_11_channels((gray+20).astype(np.uint8))
        for a,b in zip(integer,floating):
            np.testing.assert_array_equal(a,b)
        for index in range(3,11):
            np.testing.assert_array_equal(integer[index],brighter[index])
        # 常量区零梯度显示为127；无效边框才是0。
        constant=compute_11_channels(np.full((20,20),255,np.uint8))
        self.assertEqual(int(constant[7][0,0]),127)
        self.assertEqual(int(constant[7][-1,-1]),0)

    def test_dataset_counts_are_micro_not_mean_of_page_f1(self):
        """100张脸的一页与1张脸的一页权重应来自TP/FP/FN，不能平均两个F1。"""
        boxes=[[i*20,0,i*20+10,10] for i in range(100)]
        pages=[dict(page_id='many',bboxes=boxes),dict(page_id='one',bboxes=[[0,0,10,10]])]
        results=evaluate_dataset(pages,{'many':[dict(bbox=b,score=1) for b in boxes]})
        self.assertEqual((results['tp'],results['fp'],results['fn']),(100,0,1))
        self.assertAlmostEqual(results['f1'],200/201)
        self.assertEqual(results['missing_prediction_pages'],['one'])
        with self.assertRaises(ValueError):
            evaluate_dataset(pages,{'unknown':[]})
        with self.assertRaises(ValueError):
            evaluate_detection([dict(bbox=[0,0,10,10],score=float('nan'))],[])

    def test_review_export_keeps_split_and_original(self):
        """接受记录可迁移路径；剔除和未复核样本不会混进新清单。"""
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            rows=[]
            for i in range(3):
                write_image(root/f'{i}.png',np.full((24,24,3),i,np.uint8))
                rows.append(dict(image=f'{i}.png',source_image=f'{i}.png',source_id=str(i),
                                 label=-1,bbox=[0,0,24,24],split='test',review_status='needs_review'))
            manifest=root/'manifest.json'
            write_json(manifest,rows)
            original=manifest.read_bytes()
            review=root/'decisions.csv'
            with review.open('w',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=['image','review_status','note'])
                writer.writeheader()
                writer.writerows([dict(image='0.png',review_status='accepted',note='checked'),
                                  dict(image='1.png',review_status='rejected',note='face fragment')])
            output=root/'new/manifest.json'
            result=apply_review(manifest,review,output)
            exported=json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual((result['accepted'],result['rejected'],result['pending']),(1,1,1))
            self.assertEqual(exported[0]['split'],'test')
            self.assertEqual(exported[0]['label'],-1)
            self.assertEqual((output.parent/exported[0]['image']).resolve(),(root/'0.png').resolve())
            self.assertEqual(manifest.read_bytes(),original)
            with self.assertRaises(FileExistsError):
                apply_review(manifest,review,output)


if __name__=='__main__':
    unittest.main()
