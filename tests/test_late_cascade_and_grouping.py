import copy
import unittest

import numpy as np

from src.cascade import continue_cascade
from src.grouping import calibrate_box, weighted_nms


class GroupingTests(unittest.TestCase):
    def test_weighted_nms_reports_support_and_merges(self):
        result = weighted_nms([
            {'bbox': [0, 0, 10, 10], 'score': 3.0},
            {'bbox': [1, 0, 11, 10], 'score': 2.0},
            {'bbox': [30, 30, 40, 40], 'score': 1.0},
        ], .3, min_support=2)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['support'], 2)
        self.assertGreater(result[0]['bbox'][0], 0)
        self.assertLess(result[0]['bbox'][0], 1)

    def test_box_calibration_changes_center_and_size(self):
        result = calibrate_box([0, 0, 10, 10], {
            'dx': .1, 'dy': 0, 'scale_x': 1.2, 'scale_y': 1,
        })
        np.testing.assert_allclose(result, [0, 0, 12, 10], atol=1e-12)


class ContinuationTests(unittest.TestCase):
    def test_existing_stage_is_preserved(self):
        rng = np.random.default_rng(7)
        features = [[0, 0, 0, 1, 1], [1, 2, 2, 3, 3]]
        base = {'features': features, 'stages': [{
            'trees': [{'tree': {'feature': 0, 'threshold': 0.0,
                                'left': {'leaf': -1},
                                'right': {'leaf': 1}},
                       'alpha': 1.0}], 'threshold': -2.0,
        }]}
        before = copy.deepcopy(base)
        patches = rng.integers(0, 256, (20, 24, 24), dtype=np.uint8)
        labels = np.asarray([1] * 10 + [-1] * 10)
        result = continue_cascade(base, patches, labels, patches, labels,
                                  added_stages=1, rounds=2,
                                  target_recall=.8,
                                  target_false_positive_rate=.9)
        self.assertEqual(result['features'], before['features'])
        self.assertEqual(result['stages'][0], before['stages'][0])
        self.assertEqual(len(result['stages']), 2)


if __name__ == '__main__':
    unittest.main()
