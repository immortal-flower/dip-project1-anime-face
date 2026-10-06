import unittest

from scripts.build_b_defense_assets import read_hnm_numbers
from scripts.prepare_b_defense import OLD_MODEL, OLD_RESULT, V2_MODEL, V2_RESULT, summarize_frozen_result


class BDefenseAssetsTest(unittest.TestCase):
    @unittest.skipUnless(OLD_RESULT.is_file() and V2_RESULT.is_file(),
                         "local frozen evaluation artifacts are not available")
    def test_frozen_test_metrics_match_reported_counts(self):
        old = summarize_frozen_result(OLD_RESULT, OLD_MODEL)
        v2 = summarize_frozen_result(V2_RESULT, V2_MODEL)
        self.assertEqual(tuple(old["metrics"][key] for key in ("tp", "fp", "fn")),
                         (69, 960, 1360))
        self.assertEqual(tuple(v2["metrics"][key] for key in ("tp", "fp", "fn")),
                         (247, 2647, 1182))
        self.assertAlmostEqual(old["stage0_early_rejection_rate"], 0.6890386282583261)

    @unittest.skipUnless((OLD_RESULT.parents[1] / "hard-negatives").is_dir(),
                         "local hard-negative artifacts are not available")
    def test_hnm_split_boundary_and_totals(self):
        hnm = read_hnm_numbers()
        self.assertEqual(hnm["first_two_rounds_accepted"], 215)
        self.assertEqual(hnm["v2_train"]["accepted"], 287)
        self.assertEqual(hnm["total_train_accepted"], 502)
        self.assertEqual(hnm["v2_val"]["accepted"], 88)
        self.assertEqual(hnm["test_hard_negatives"], 0)
        self.assertEqual(hnm["validation_fpr"]["source_type"], "document")


if __name__ == "__main__":
    unittest.main()
