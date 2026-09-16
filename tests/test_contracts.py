import unittest
import numpy as np
from src.channels11 import compute_11_channels
from src.detection_metrics import iou, evaluate_detection
from src.grouping import nms
from src.landmark_metrics import nme
from src.depth2_tree import fit_tree, predict_tree
from src.shape_regression import predict_shape
from src.cascade import sample_features


class ContractTests(unittest.TestCase):
    def test_crop_and_full_image_features_agree(self):
        gray = np.random.default_rng(19).integers(0, 256, (48, 60), dtype=np.uint8)
        full = compute_11_channels(gray)
        crop = compute_11_channels(gray[9:33, 13:37])
        features = [[c, x, y, 16, 16] for c in range(11) for x in range(17) for y in range(17)]
        np.testing.assert_array_equal(sample_features(full, features, 13, 9), sample_features(crop, features))

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

    def test_matching_and_nms(self):
        box = [0, 0, 10, 10]
        self.assertEqual(iou(box, [10, 0, 20, 10]), 0)
        predictions = [dict(bbox=box, score=2), dict(bbox=box, score=1)]
        result = evaluate_detection(predictions, [box])
        self.assertEqual((result['tp'], result['fp'], result['fn']), (1, 1, 0))
        self.assertEqual(len(nms(predictions)), 1)

    def test_tree_can_express_xor(self):
        x = np.array([[0,0], [0,1], [1,0], [1,1]])
        y = np.array([-1, 1, 1, -1])
        tree = fit_tree(x, y, np.ones(4)/4)
        np.testing.assert_array_equal(predict_tree(tree, x), y)

    def test_landmark_mask_and_coordinates(self):
        truth = np.zeros((28, 2))
        prediction = np.full((28, 2), 100.0)
        prediction[0] = [3, 4]
        self.assertEqual(nme(prediction, truth, [1]+[0]*27, 10), .5)
        self.assertIsNone(nme(prediction, truth, [0]*28, 10))
        model = dict(mean_shape=np.full((28, 2), .5), offsets=np.zeros((28, 2, 2, 2)), weights=[])
        result = predict_shape(model, np.zeros((50, 50), np.uint8), [10, 20, 30, 40])
        np.testing.assert_array_equal(result, np.tile([20, 30], (28, 1)))


if __name__ == '__main__':
    unittest.main()
