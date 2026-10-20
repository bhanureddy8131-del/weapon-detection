import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import yaml

from scripts.audit_dataset import audit_dataset, parse_yolo_label
from src.config import EXPECTED_DETECTOR_CLASSES


class DatasetAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.data_yaml = self.root / "data.yaml"
        self._create_split("train")
        self._create_split("valid")
        self._create_split("test")
        self.data_yaml.write_text(
            yaml.safe_dump({
                "path": str(self.root),
                "train": "train/images",
                "val": "valid/images",
                "test": "test/images",
                "nc": 6,
                "names": {
                    index: name
                    for index, name in enumerate(EXPECTED_DETECTOR_CLASSES)
                },
            }),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _create_split(self, split):
        images = self.root / split / "images"
        labels = self.root / split / "labels"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        image = np.full(
            (32, 32, 3),
            sum(ord(character) for character in split) % 255,
            dtype=np.uint8,
        )
        if not cv2.imwrite(str(images / "sample.jpg"), image):
            raise RuntimeError("Test fixture image could not be written.")
        (labels / "sample.txt").write_text(
            "2 0.5 0.5 0.25 0.25\n",
            encoding="utf-8",
        )
        return images / "sample.jpg", labels / "sample.txt"

    def test_box_and_polygon_annotations_are_parsed_without_rewriting(self):
        label = self.root / "mixed.txt"
        label.write_text(
            "2 0.5 0.5 0.4 0.2\n"
            "4 0.1 0.1 0.4 0.1 0.3 0.5\n",
            encoding="utf-8",
        )

        rows, errors, box_count, polygon_count = parse_yolo_label(label, 6)

        self.assertEqual(errors, [])
        self.assertEqual(box_count, 1)
        self.assertEqual(polygon_count, 1)
        self.assertEqual([row["geometry"] for row in rows], ["box", "polygon"])

    def test_out_of_range_and_malformed_labels_are_reported(self):
        label = self.root / "invalid.txt"
        label.write_text(
            "9 0.5 0.5 0.2 0.2\n"
            "2 0.5 0.5 -0.2 0.2\n"
            "knife not-a-number\n",
            encoding="utf-8",
        )

        rows, errors, _, _ = parse_yolo_label(label, 6)

        self.assertEqual(rows, [])
        self.assertEqual(len(errors), 3)

    def test_six_decimal_boundary_rounding_is_reported_and_normalized(self):
        label = self.root / "rounded-boundary.txt"
        label.write_text("5 0.5 0.6000005 0.2 0.8\n", encoding="utf-8")

        rows, errors, _, _ = parse_yolo_label(label, 6)

        self.assertEqual(errors, [])
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["boundary_rounding_clipped"])
        self.assertEqual(rows[0]["box"][3], 1.0)

    def test_box_more_than_rounding_tolerance_outside_image_is_invalid(self):
        label = self.root / "outside-boundary.txt"
        label.write_text("5 0.5 0.60001 0.2 0.8\n", encoding="utf-8")

        rows, errors, _, _ = parse_yolo_label(label, 6)

        self.assertEqual(rows, [])
        self.assertIn("derived box extends outside the image", errors[0])

    def test_audit_tolerates_and_documents_only_rounding_scale_overshoot(self):
        (self.root / "train" / "labels" / "sample.txt").write_text(
            "2 0.5 0.6000005 0.2 0.8\n",
            encoding="utf-8",
        )
        report, _ = audit_dataset(
            self.data_yaml,
            output_directory=self.root / "rounding-audit",
            samples_per_class=0,
        )

        self.assertTrue(report["audit_passed"])
        self.assertEqual(
            report["splits"]["train"]["rounding_boundary_rows"],
            1,
        )
        self.assertTrue(any("rounding tolerance" in warning for warning in report["warnings"]))

    def test_empty_labels_are_reported_as_empty_not_malformed(self):
        label = self.root / "empty.txt"
        label.write_text("\n", encoding="utf-8")

        rows, errors, box_count, polygon_count = parse_yolo_label(label, 6)

        self.assertEqual(rows, [])
        self.assertEqual(errors, [])
        self.assertEqual((box_count, polygon_count), (0, 0))

    def test_cross_split_exact_image_leakage_fails_audit(self):
        train_image, _ = self._create_split("train-duplicate")
        test_image = self.root / "test" / "images" / "sample.jpg"
        test_image.write_bytes(train_image.read_bytes())
        # Point the training split at the duplicate fixture while leaving the
        # original test split intact.
        config = yaml.safe_load(self.data_yaml.read_text(encoding="utf-8"))
        config["train"] = "train-duplicate/images"
        self.data_yaml.write_text(
            yaml.safe_dump(config),
            encoding="utf-8",
        )

        report, _ = audit_dataset(
            self.data_yaml,
            output_directory=self.root / "audit",
            samples_per_class=0,
        )

        self.assertFalse(report["audit_passed"])
        self.assertTrue(report["duplicates"]["exact_cross_split_leakage"])

    def test_audit_reports_split_and_per_class_counts_and_visualizations(self):
        report, report_path = audit_dataset(
            self.data_yaml,
            output_directory=self.root / "audit",
            samples_per_class=1,
        )

        self.assertTrue(report_path.is_file())
        self.assertEqual(report["splits"]["train"]["image_count"], 1)
        self.assertEqual(report["splits"]["train"]["valid_label_files"], 1)
        self.assertEqual(report["splits"]["train"]["valid_annotation_rows"], 1)
        self.assertEqual(report["splits"]["val"]["classes"]["gun"]["instances"], 1)
        self.assertEqual(report["splits"]["test"]["classes"]["gun"]["images"], 1)
        self.assertTrue(report["visualizations"])
        self.assertTrue(Path(report["visualizations"][0]["visualization"]).is_file())


if __name__ == "__main__":
    unittest.main()
