"""Checks for patch coordinates and a clear synthetic texture anomaly."""

import unittest

import numpy as np

from src.localization import TextureLocalizer, grid_boxes, render_heatmap


class TestLocalization(unittest.TestCase):
    def test_grid_covers_normalized_image(self):
        boxes = grid_boxes((540, 856))
        self.assertEqual(len(boxes), 24)
        coverage = np.zeros((540, 856), dtype=np.uint8)
        for x1, y1, x2, y2 in boxes:
            coverage[y1:y2, x1:x2] += 1
        self.assertTrue(np.all(coverage == 1))

    def test_synthetic_modified_cell_ranks_first(self):
        rng = np.random.default_rng(42)
        base = np.full((540, 856), 128, dtype=np.uint8)
        references = [base.copy(), base.copy(), base.copy()]
        target = base.copy()
        chosen = 8
        x1, y1, x2, y2 = grid_boxes(target.shape)[chosen]
        target[y1:y2, x1:x2] = rng.integers(
            0, 256, size=(y2 - y1, x2 - x1), dtype=np.uint8
        )
        scores = TextureLocalizer.fit(references).score(target)
        self.assertEqual(scores.shape, (4, 6))
        self.assertTrue(np.isfinite(scores).all())
        self.assertEqual(int(np.argmax(scores)), chosen)
        self.assertEqual(render_heatmap(target, scores).shape, (540, 856, 3))


if __name__ == "__main__":
    unittest.main()
