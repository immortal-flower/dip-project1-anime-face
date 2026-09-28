import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "review_local_face_boxes.py"
SPEC = importlib.util.spec_from_file_location("review_local_face_boxes", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class LocalFaceBoxReviewTests(unittest.TestCase):
    def record(self):
        return {
            "source_id": "animeface:test",
            "image": "images/test.jpg",
            "split": "test",
            "bbox": [0, 0, 100, 80],
            "landmarks": [[20, 30], [80, 30], [50, 65]],
            "visibility": [1, 1, 1],
        }

    def test_suggestion_is_local_valid_and_contains_visible_points(self):
        record = self.record()
        box = MODULE.suggested_local_face_bbox(record)
        self.assertEqual(box, MODULE.validate_box(box, 100, 80))
        for x, y in record["landmarks"]:
            self.assertLessEqual(box[0], x)
            self.assertLessEqual(box[1], y)
            self.assertGreaterEqual(box[2], x)
            self.assertGreaterEqual(box[3], y)
        self.assertNotEqual(box, record["bbox"])

    def test_invalid_box_is_rejected(self):
        with self.assertRaises(ValueError):
            MODULE.validate_box([-1, 0, 20, 20], 100, 80)
        with self.assertRaises(ValueError):
            MODULE.validate_box([10, 10, 10, 20], 100, 80)
        with self.assertRaises(ValueError):
            MODULE.validate_box([10, 10, 101, 20], 100, 80)

    def test_finalize_requires_every_approval_and_preserves_original(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text("[]", encoding="utf-8")
            record = self.record()
            item = {
                "source_id": record["source_id"],
                "image": record["image"],
                "image_path": str(root / "test.jpg"),
                "width": 100,
                "height": 80,
                "original_bbox": record["bbox"],
                "suggested_bbox": [10, 10, 90, 75],
                "landmarks": record["landmarks"],
                "visibility": record["visibility"],
                "record": record,
            }
            store = MODULE.ReviewStore(manifest, [item], root / "out", "test")
            with self.assertRaises(ValueError):
                store.finalize()
            store.save(
                {
                    "index": 0,
                    "bbox": [12, 8, 92, 76],
                    "status": "approved",
                    "note": "manual",
                }
            )
            summary = store.finalize()
            output = MODULE.read_json(store.final_path)
            self.assertEqual(summary["records"], 1)
            self.assertEqual(output[0]["bbox"], [12, 8, 92, 76])
            self.assertEqual(output[0]["original_bbox"], [0, 0, 100, 80])
            self.assertEqual(output[0]["bbox_review_status"], "approved")


if __name__ == "__main__":
    unittest.main()
