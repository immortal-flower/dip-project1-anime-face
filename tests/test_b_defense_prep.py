import unittest

from scripts.prepare_b_defense import aggregate_scan, match_detections


class DefensePreparationTests(unittest.TestCase):
    def test_scan_aggregation_keeps_stage_and_candidate_counts(self):
        result = {"pages": [{
            "windows": 10, "detections": 2,
            "scan": [{
                "stages": [
                    {"evaluated": 10, "passed": 4, "rejected": 6, "seconds": .1},
                    {"evaluated": 4, "passed": 3, "rejected": 1, "seconds": .2},
                ],
                "scan_summary": {"raw_candidates": 3, "detections_after_nms": 2},
            }],
        }]}
        summary = aggregate_scan(result)
        self.assertEqual(summary["total_windows"], 10)
        self.assertEqual(summary["raw_candidates"], 3)
        self.assertEqual(summary["detections_after_nms"], 2)
        self.assertEqual(summary["detections_after_score_filter"], 2)
        self.assertEqual(summary["stage_totals"][0]["rejected"], 6)

    def test_case_matching_is_score_ordered_one_to_one(self):
        predictions = [
            {"bbox": [0, 0, 10, 10], "score": 2.0},
            {"bbox": [1, 0, 11, 10], "score": 1.0},
        ]
        matches, unmatched_truth = match_detections(
            predictions, [[0, 0, 10, 10]], threshold=.5)
        self.assertEqual(matches, {0: (0, 1.0)})
        self.assertEqual(unmatched_truth, set())


if __name__ == "__main__":
    unittest.main()
