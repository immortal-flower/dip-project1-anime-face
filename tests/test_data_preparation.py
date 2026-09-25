# A 的数据准备测试：检查同源分组、人脸排除、数据泄漏与解压保护。
import random
import tempfile
import unittest
from pathlib import Path
import zipfile
import numpy as np
from src.data_preparation import (book_group, split_groups, expanded_faces,
                                  sample_background, intersects, audit_rows)
from scripts.verify_dataset_archive import verify


class DataPreparationTests(unittest.TestCase):
    # 检查输入列表顺序变化不会改变来源分配，多卷同系列分组一致。
    def test_group_split_is_order_independent(self):
        books = [f'Book{i}' for i in range(100)]
        a = split_groups(books)
        self.assertEqual(a, split_groups(list(reversed(books))))
        self.assertEqual([list(a.values()).count(s) for s in ('train','val','test')], [75,10,15])
        self.assertEqual(book_group('Series_vol01'), book_group('Series_vol20'))

    # 检查负样本不会包含人脸；大框内的小脸也必须排除。
    def test_background_excludes_contained_face_and_margin(self):
        gray = np.random.default_rng(9).integers(0,256,(256,256),dtype=np.uint8)
        faces = [[50,50,170,170]]
        result = sample_background(gray,faces,random.Random(42),count=12)
        self.assertEqual(len(result),12)
        exclusion = expanded_faces(faces,256,256)[0]
        for box in result:
            self.assertFalse(intersects(box,exclusion))
            self.assertTrue(0 <= box[0] < box[2] <= 256)
            self.assertTrue(0 <= box[1] < box[3] <= 256)
        # A tiny contained face must be rejected even though IoU would be low.
        self.assertTrue(intersects([0,0,200,200],[90,90,95,95]))
        self.assertEqual(sample_background(np.zeros((80,80),np.uint8),[],random.Random(1)),[])

    # 发现跨集合重复时报告冲突，而不是静默改掉队友已有划分。
    def test_leakage_fails_without_changing_c_splits(self):
        rows = [dict(source_id='one',split='train',pixel_sha256='same'),
                dict(source_id='two',split='test',pixel_sha256='same')]
        with self.assertRaises(ValueError):
            audit_rows(rows)
        self.assertEqual(rows[1]['split'],'test')
        rows[1]['pixel_sha256']='different'
        self.assertTrue(audit_rows(rows))

    # 检查补解压不会覆盖已有不同内容，并拒绝目录穿越路径。
    def test_archive_verification_preserves_mismatched_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            archive=root/'sample.zip'
            with zipfile.ZipFile(archive,'w') as z:
                z.writestr('dataset/a.txt','correct')
            destination=root/'out'
            first=verify(archive,destination,extract_missing=True)
            self.assertTrue(first['ok'])
            target=destination/'dataset/a.txt'
            target.write_text('changed')
            second=verify(archive,destination,extract_missing=True)
            self.assertFalse(second['ok'])
            self.assertEqual(target.read_text(),'changed')
            with zipfile.ZipFile(archive,'w') as z:
                z.writestr('../escape.txt','bad')
            with self.assertRaises(ValueError):
                verify(archive,destination,extract_missing=True)


if __name__ == '__main__':
    unittest.main()
