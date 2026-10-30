"""Regression checks for data auditing, CLI output and report provenance."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts import run_grouped_evaluation as evaluation


class TestEvaluationCLI(unittest.TestCase):
    def test_duplicate_content_rejected_across_names_and_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.jpg").write_bytes(b"identical image bytes")
            (root / "b.jpg").write_bytes(b"identical image bytes")
            labels = pd.DataFrame({
                "path": ["a.jpg", "b.jpg"], "label": ["real", "fake"],
                "group_id": ["g1", "g2"],
            })
            with self.assertRaisesRegex(ValueError, "a.jpg.*b.jpg"):
                evaluation.validate_image_hashes(labels, root)

    def test_source_digest_changes_without_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").mkdir()
            source = root / "src" / "module.py"
            source.write_text("value = 1\n", encoding="utf-8")
            before = evaluation.collect_source_identity(root)
            source.write_text("value = 2\n", encoding="utf-8")
            after = evaluation.collect_source_identity(root)
            self.assertNotEqual(before["source_tree_sha256"], after["source_tree_sha256"])
            self.assertIn("src/module.py", after["source_files_sha256"])

    def test_cli_creates_new_directory_and_consistent_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            n = 12
            labels = pd.DataFrame({
                "path": [f"image_{i}.jpg" for i in range(n)],
                "label": ["real", "fake"] * 6,
                "fraud_type": ["none", "edit_text"] * 6,
                "group_id": np.repeat(["g1", "g2", "g3"], 4),
            })
            for i, path in enumerate(labels["path"]):
                (root / path).write_bytes(f"unique image {i}".encode())
            labels.to_csv(root / "labels.csv", index=False)
            rng = np.random.default_rng(42)
            y = np.array([0, 1] * 6)
            np.savez(
                root / "features.npz", X=rng.normal(size=(n, 60)) + y[:, None],
                y=y, groups=labels["group_id"].to_numpy(dtype=str),
                fraud_types=labels["fraud_type"].to_numpy(dtype=str),
            )
            argv = ["evaluation", "--features", "features.npz", "--labels", "labels.csv",
                    "--model", "svm", "--output-dir", "report/evaluation", "--skip-sensitivity"]
            with patch.object(evaluation, "ROOT_DIR", root), patch("sys.argv", argv), \
                    patch.object(evaluation, "save_evaluation_plots"), \
                    contextlib.redirect_stdout(io.StringIO()):
                evaluation.main()
            output = root / "report" / "evaluation"
            metrics = json.loads((output / "metrics_svm.json").read_text(encoding="utf-8"))
            manifest = json.loads((output / "evaluation_manifest.json").read_text(encoding="utf-8"))
            brief = (root / "report" / "REPORT_BRIEF_FOR_CLAUDE.md").read_text(encoding="utf-8")
            summary = (output / "evaluation_summary.md").read_text(encoding="utf-8")
            self.assertTrue((output / "fold_metrics_svm.csv").is_file())
            self.assertTrue((output / "oof_predictions_svm.csv").is_file())
            self.assertEqual(len(manifest["image_files_sha256"]), n)
            self.assertIn("git_dirty_at_start", manifest)
            rate = evaluation.format_rate(metrics["pooled_oof_metrics"]["recall_fake"])
            self.assertIn(rate, brief)
            self.assertIn(rate, summary)
            self.assertIn(manifest["source_tree_sha256"], brief)
            self.assertIn(manifest["source_tree_sha256"], summary)


if __name__ == "__main__":
    unittest.main()
