# 基础合同测试：用已知答案验证坐标、通道、匹配和弱树行为。
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from src.channels11 import compute_11_channels
from src.detection_metrics import iou, evaluate_detection
from src.grouping import nms
from src.landmark_metrics import nme
from src.depth2_tree import fit_tree, predict_tree
from src.shape_regression import predict_shape
from src.cascade import calibrate_threshold, sample_features, train_cascade
from src.data_io import load_manifest, write_image, write_json
from scripts.mine_hard_negatives import _select_pages, mine_hard_negatives
from scripts.review_hard_negatives import apply_decisions
from scripts.evaluate_page_detector import _source_spread
from src.sliding_window import _batch_features, scan_image


class ContractTests(unittest.TestCase):
    # 验证同一区域单独裁剪和放在整图中计算的有效特征一致。
    def test_crop_and_full_image_features_agree(self):
        gray = np.random.default_rng(19).integers(0, 256, (48, 60), dtype=np.uint8)
        full = compute_11_channels(gray)
        crop = compute_11_channels(gray[9:33, 13:37])
        features = [[c, x, y, 16, 16] for c in range(11) for x in range(17) for y in range(17)]
        np.testing.assert_array_equal(sample_features(full, features, 13, 9), sample_features(crop, features))

    # 用逐像素公式作为独立参考，检查向量化切片、边界与整数取整。
    def test_channels_against_scalar_reference(self):
        gray = np.random.default_rng(7).integers(0, 256, (13, 15), dtype=np.uint8)
        expected = [gray.copy()] + [np.zeros_like(gray) for _ in range(10)]
        for k, src, gap, margin in [(1, 0, 1, 1), (2, 1, 2, 3)]:
            for y in range(13-margin):
                for x in range(15-margin):
                    expected[k][y, x] = (sum(int(expected[src][y+dy, x+dx]) for dy in (0, gap) for dx in (0, gap))+2)//4
        for k, src, gap, margin in [(3, 1, 2, 3), (7, 2, 4, 7)]:
            for y in range(13-margin):
                for x in range(15-margin):
                    a, b, c, d = [int(expected[src][yy, xx]) for yy, xx in [(y,x), (y,x+gap), (y+gap,x), (y+gap,x+gap)]]
                    for j, delta in enumerate((b-a, c-a, d-a, c-b)):
                        expected[k+j][y, x] = (delta+255)//2
        for actual, reference in zip(compute_11_channels(gray), expected):
            np.testing.assert_array_equal(actual, reference)
        self.assertTrue(all(c.shape == (1, 1) for c in compute_11_channels(np.zeros((1, 1), np.uint8))))

    # 两个预测重复命中一张脸只能算一个 TP；其余算误检。
    def test_matching_and_nms(self):
        box = [0, 0, 10, 10]
        self.assertEqual(iou(box, [10, 0, 20, 10]), 0)
        predictions = [dict(bbox=box, score=2), dict(bbox=box, score=1)]
        result = evaluate_detection(predictions, [box])
        self.assertEqual((result['tp'], result['fp'], result['fn']), (1, 1, 0))
        self.assertEqual(len(nms(predictions)), 1)

    # XOR 需要两层判断，用来确认实现并非只有单个决策桩。
    def test_tree_can_express_xor(self):
        x = np.array([[0,0], [0,1], [1,0], [1,1]])
        y = np.array([-1, 1, 1, -1])
        tree = fit_tree(x, y, np.ones(4)/4)
        np.testing.assert_array_equal(predict_tree(tree, x), y)

    # 检查不可见点不影响 NME，以及归一化点能正确还原到原图。
    def test_landmark_mask_and_coordinates(self):
        truth = np.zeros((28, 2))
        prediction = np.full((28, 2), 100.0)
        prediction[0] = [3, 4]
        self.assertEqual(nme(prediction, truth, [1]+[0]*27, 10), .5)
        self.assertIsNone(nme(prediction, truth, [0]*28, 10))
        model = dict(mean_shape=np.full((28, 2), .5), offsets=np.zeros((28, 2, 2, 2)), weights=[])
        result = predict_shape(model, np.zeros((50, 50), np.uint8), [10, 20, 30, 40])
        np.testing.assert_array_equal(result, np.tile([20, 30], (28, 1)))

    # 阈值取满足目标召回率的最高正样本分数，并报告同一阈值下的误检率。
    def test_target_recall_threshold(self):
        threshold, recall, false_positive_rate = calibrate_threshold(
            np.array([4., 3., 2., 1., 3.5, 2.5]),
            np.array([1, 1, 1, 1, -1, -1]),
            .75,
        )
        self.assertEqual(threshold, 2.)
        self.assertEqual(recall, .75)
        self.assertEqual(false_positive_rate, 1.)

    # 负样本被前级全部拒绝后必须安全早停，不能重新使用已淘汰的训练池。
    def test_cascade_stops_when_negative_pool_is_exhausted(self):
        y, x = np.mgrid[:24, :24]
        positive = (4 * x + 3 * y).astype(np.uint8)
        negative = np.fliplr(positive).copy()
        train = [positive.copy() for _ in range(6)] + [negative.copy() for _ in range(6)]
        labels = np.array([1] * 6 + [-1] * 6)
        model = train_cascade(
            train, labels, train, labels, seed=4, candidates=64,
            stages=3, rounds=3, target_recall=1.0,
            target_false_positive_rate=.5,
            sample_weights=np.array([2.] + [1.] * 11),
        )
        summary = model['training_summary']
        self.assertTrue(summary['weighted_training'])
        self.assertTrue(summary['stopped_early'])
        self.assertIn('negatives_exhausted', summary['stop_reason'])
        self.assertEqual(len(model['training_log']), summary['trained_stages'])
        self.assertEqual(model['training_log'][0]['train']['pass_negative'], 0)

    # 困难负样本只能来自训练原页；重复运行不能再次写入同一个原页框。
    def test_hard_negative_mining_uses_pages_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            page = np.full((48, 48, 3), 127, dtype=np.uint8)
            write_image(root/'train.png', page)
            write_image(root/'val.png', page)
            manifest = [
                dict(image='train.png', source_id='book-train', split='train',
                     label=1, bbox=[0, 0, 24, 24]),
                dict(image='train.png', source_id='book-train', split='train',
                     label=-1, bbox=[24, 24, 48, 48]),
                dict(image='val.png', source_id='book-val', split='val',
                     label=1, bbox=[0, 0, 24, 24]),
                dict(image='val.png', source_id='book-val', split='val',
                     label=-1, bbox=[24, 24, 48, 48]),
            ]
            pages = [
                dict(image='train.png', source_id='book-train', page_id='train-1',
                     split='train', bboxes=[[0, 0, 24, 24]], width=48, height=48),
                dict(image='val.png', source_id='book-val', page_id='val-1',
                     split='val', bboxes=[], width=48, height=48),
            ]
            write_json(root/'manifest.json', manifest)
            write_json(root/'pages.json', pages)
            model_dir = root/'model'
            write_json(model_dir/'detector.json', dict(
                features=[[0, 0, 0, 1, 0]],
                stages=[dict(trees=[dict(alpha=1., tree={'leaf': 1})], threshold=0.)],
            ))
            write_json(model_dir/'config.json', dict(
                step=24, scale_factor=10., nms_threshold=.3,
            ))

            first = root/'round-1'
            result = mine_hard_negatives(
                root/'manifest.json', root/'pages.json', model_dir, first,
                max_per_page=10, max_total=10, gallery_size=6,
            )
            self.assertEqual((result['scanned_pages'], result['mined']), (1, 3))
            scan_summary = result['pages'][0]['scan'][-1]['scan_summary']
            self.assertEqual(
                (scan_summary['raw_candidates'], scan_summary['detections_after_nms']),
                (4, 4),
            )
            mined = json.loads((first/'mined_manifest.json').read_text(encoding='utf-8'))
            self.assertTrue(all(row['split'] == 'train' and row['label'] == -1 for row in mined))
            self.assertTrue(all(row['page_id'] == 'train-1' for row in mined))
            augmented = list(load_manifest(first/'augmented_manifest.json'))
            self.assertEqual(len(augmented), len(manifest) + 3)
            self.assertTrue((first/'gallery.png').is_file())

            second = root/'round-2'
            repeated = mine_hard_negatives(
                first/'augmented_manifest.json', root/'pages.json', model_dir, second,
                mining_round=2, max_per_page=10, max_total=10,
            )
            self.assertEqual(repeated['mined'], 0)
            self.assertEqual(repeated['duplicate_rejected'], 3)

    # 缺少原页时在创建输出前失败，避免生成看似成功的不完整挖掘结果。
    def test_hard_negative_mining_preflights_page_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root/'manifest.json', [dict(
                image='patch.png', source_id='book', split='train',
                label=-1, bbox=[0, 0, 24, 24],
            )])
            write_json(root/'pages.json', [dict(
                image='missing.png', source_id='book', page_id='book-1',
                split='train', bboxes=[], width=24, height=24,
            )])
            output = root/'output'
            with self.assertRaisesRegex(FileNotFoundError, 'Manga109'):
                mine_hard_negatives(root/'manifest.json', root/'pages.json',
                                    root/'model', output)
            self.assertFalse(output.exists())

    # 跨来源轮询应先覆盖更多书目，再从同一本书取第二页。
    def test_hard_negative_page_selection_spreads_sources(self):
        items = [
            (dict(source_id='a', page_id='a1'), None),
            (dict(source_id='a', page_id='a2'), None),
            (dict(source_id='b', page_id='b1'), None),
            (dict(source_id='b', page_id='b2'), None),
            (dict(source_id='c', page_id='c1'), None),
        ]
        selected = _select_pages(items, 4, 'source-spread')
        self.assertEqual(
            [item[0]['page_id'] for item in selected],
            ['a1', 'b1', 'c1', 'a2'],
        )

    # 固定整页预览也应优先跨来源选页，避免一本漫画主导测试结果。
    def test_page_evaluation_selection_spreads_sources(self):
        rows = [
            dict(source_id='a', page_id='a1'),
            dict(source_id='a', page_id='a2'),
            dict(source_id='b', page_id='b1'),
            dict(source_id='b', page_id='b2'),
            dict(source_id='c', page_id='c1'),
        ]
        self.assertEqual(
            [row['page_id'] for row in _source_spread(rows, 4)],
            ['a1', 'b1', 'c1', 'a2'],
        )

    # 批量扫描必须与逐窗口特征读取逐项一致，才能安全替代慢速循环。
    def test_batch_window_features_match_scalar_sampling(self):
        image = np.random.default_rng(31).integers(0, 256, (48, 52), dtype=np.uint8)
        channels = compute_11_channels(image)
        features = [
            [0, 0, 0, 16, 16], [3, 2, 4, 10, 11],
            [7, 1, 2, 12, 14], [10, 16, 0, 0, 16],
        ]
        xs = np.array([0, 3, 9, 21])
        ys = np.array([0, 5, 11, 19])
        expected = np.stack([
            sample_features(channels, features, int(x), int(y))
            for x, y in zip(xs, ys)
        ])
        np.testing.assert_array_equal(
            _batch_features(channels, features, xs, ys), expected
        )

    # 最终配置中的统一分数阈值必须由公共扫描接口执行，而不只存在于评价脚本。
    def test_scan_score_threshold_filters_after_nms(self):
        model = dict(
            features=[[0, 0, 0, 1, 0]],
            stages=[dict(
                trees=[dict(alpha=1.0, tree={'leaf': 1})], threshold=0.0,
            ) for _ in range(3)],
        )
        image = np.zeros((48, 48, 3), dtype=np.uint8)
        detections, logs = scan_image(image, model, dict(
            step=24, scale_factor=10.0, nms_threshold=0.3,
            pre_nms_limit=500, score_threshold=4.0,
        ))
        self.assertEqual(detections, [])
        summary = logs[-1]['scan_summary']
        self.assertEqual(summary['detections_after_nms'], 4)
        self.assertEqual(summary['detections_after_score_filter'], 0)

    # 人工决定必须逐项覆盖；增强清单只能加入明确接受的困难负样本。
    def test_hard_negative_review_requires_complete_decisions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = np.zeros((24, 24, 3), dtype=np.uint8)
            for name in ('base.png', 'one.png', 'two.png'):
                write_image(root/name, image)
            write_json(root/'base.json', [dict(
                image='base.png', source_id='book', split='train',
                label=1, bbox=[0, 0, 24, 24],
            )])
            candidates = [dict(
                image=name, source_id='book', page_id='page', split='train',
                label=-1, bbox=[0, 0, 24, 24], hard_negative=True,
                review_status='needs_review',
            ) for name in ('one.png', 'two.png')]
            write_json(root/'candidates.json', candidates)
            write_json(root/'incomplete.json', [dict(
                review_index=1, review_status='accepted',
            )])
            with self.assertRaisesRegex(ValueError, 'cover every item'):
                apply_decisions(root/'candidates.json', root/'incomplete.json',
                                root/'incomplete-output', root/'base.json')

            write_json(root/'decisions.json', [
                dict(review_index=1, review_status='accepted', note='background'),
                dict(review_index=2, review_status='rejected', note='partial face'),
            ])
            summary = apply_decisions(
                root/'candidates.json', root/'decisions.json', root/'review',
                root/'base.json',
            )
            self.assertEqual(
                (summary['reviewed'], summary['accepted'], summary['rejected']),
                (2, 1, 1),
            )
            augmented = list(load_manifest(root/'review'/'accepted_augmented_manifest.json'))
            self.assertEqual(len(augmented), 2)
            self.assertEqual(sum(bool(row.get('hard_negative')) for row in augmented), 1)


if __name__ == '__main__':
    unittest.main()
