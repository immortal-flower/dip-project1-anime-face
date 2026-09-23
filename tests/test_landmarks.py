import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

from scripts.correct_landmarks import nearest_landmark, validate_correction_row
from scripts.evaluate_landmark import evaluate
from src.detector import LandmarkRegressor
from src.landmark_metrics import interocular_distance, nme_details
from src.landmark_schema import FLIP_PAIRS, LANDMARK_ORDER
from src.shape_regression import train_shape


class LandmarkTests(unittest.TestCase):
    def test_schema_flip_pairs_are_disjoint(self):
        flattened = [index for pair in FLIP_PAIRS for index in pair]
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertTrue(all(0 <= index < 28 for index in flattened))

    def test_interocular_nme_and_bbox_fallback(self):
        truth = np.zeros((28, 2), dtype=float)
        truth[11:17, 0] = 5
        truth[17:23, 0] = 15
        visible = np.ones(28, dtype=int)
        self.assertEqual(interocular_distance(truth, visible), 10)
        prediction = truth + [1, 0]
        details = nme_details(prediction, truth, visible, [0, 0, 30, 40])
        self.assertAlmostEqual(details['nme'], .1)
        self.assertEqual(details['normalization'], 'interocular')

        visible[11:23] = 0
        details = nme_details(prediction, truth, visible, [0, 0, 30, 40])
        self.assertAlmostEqual(details['normalizer'], 50)
        self.assertEqual(details['normalization'], 'bbox_diagonal')

    def test_correction_helpers(self):
        points = np.zeros((28, 2), dtype=float)
        points[3] = [10, 12]
        self.assertEqual(nearest_landmark(points, 10, 12, 1), 3)
        self.assertIsNone(nearest_landmark(points, 100, 100, 1))
        row = dict(landmark_order=LANDMARK_ORDER, landmarks=points.tolist(),
                   visibility=[1] * 28, bbox=[0, 0, 20, 20])
        validate_correction_row(row)
        row['landmark_order'] = 'course28-v1'
        with self.assertRaises(ValueError):
            validate_correction_row(row)

    def test_landmark_regressor_handles_multiple_and_empty_boxes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mean = np.full((28, 2), .5)
            offsets = np.zeros((28, 2, 2, 2))
            weights = np.empty((0, 113, 56))
            np.savez_compressed(root/'landmark.npz', mean_shape=mean,
                                offsets=offsets, weights=weights)
            (root/'config.json').write_text(json.dumps({'landmark_order': LANDMARK_ORDER}))
            regressor = LandmarkRegressor(root)
            image = np.zeros((40, 50, 3), dtype=np.uint8)
            self.assertEqual(regressor.predict(image, []), [])
            result = regressor.predict(image, [[0, 0, 20, 20], [10, 10, 30, 40]])
            self.assertEqual(len(result), 2)
            np.testing.assert_allclose(result[0]['landmarks'], np.tile([10, 10], (28, 1)))
            np.testing.assert_allclose(result[1]['landmarks'], np.tile([20, 25], (28, 1)))
            with self.assertRaises(ValueError):
                regressor.predict(image, [[-1, 0, 20, 20]])

    def test_training_rejects_nonbinary_visibility(self):
        images = [np.zeros((20, 20), dtype=np.uint8)]
        points = [np.full((28, 2), 10.0)]
        with self.assertRaises(ValueError):
            train_shape(images, [[0, 0, 20, 20]], points, [[2] * 28])

    def test_test_evaluation_requires_human_review(self):
        row = dict(split='test', label=1, landmarks=np.zeros((28, 2)).tolist(),
                   visibility=[1] * 28, landmark_order=LANDMARK_ORDER,
                   source_id='unreviewed', bbox=[0, 0, 20, 20])
        with patch('scripts.evaluate_landmark.load_manifest', return_value=iter([row])):
            with self.assertRaisesRegex(ValueError, 'human review'):
                evaluate('unused.json', 'unused.npz', 'unused-output')


if __name__ == '__main__':
    unittest.main()
