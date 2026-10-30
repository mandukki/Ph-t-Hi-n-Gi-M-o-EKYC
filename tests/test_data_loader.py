"""
Unit tests cho src/data_loader.py
Kiểm tra tính đúng đắn của việc nạp metadata, kiểm tra schema, strict mode,
và kiểm tra tính hợp lệ của feature bundle.
"""

import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from src.data_loader import (
    REQUIRED_COLUMNS,
    ALLOWED_FRAUD_TYPES,
    validate_metadata_df,
    validate_feature_bundle,
    load_labels,
    build_dataset,
)


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

        # Tạo file ảnh dummy
        (self.root / "data" / "real").mkdir(parents=True, exist_ok=True)
        (self.root / "data" / "fake").mkdir(parents=True, exist_ok=True)
        self.dummy_real = self.root / "data" / "real" / "card_01.jpg"
        self.dummy_fake = self.root / "data" / "fake" / "card_01_fake.jpg"
        self.dummy_real.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
        self.dummy_fake.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_validate_metadata_df_valid(self):
        """Metadata hợp lệ phải vượt qua validation mà không ném ngoại lệ."""
        df = pd.DataFrame({
            "path": ["data/real/card_01.jpg", "data/fake/card_01_fake.jpg"],
            "label": ["real", "fake"],
            "fraud_type": ["none", "edit_text"],
            "group_id": ["card_01", "card_01"],
        })
        checked = validate_metadata_df(df, project_root=self.root)
        self.assertEqual(len(checked), 2)
        self.assertIn("path", checked.columns)

    def test_validate_metadata_df_missing_columns(self):
        """Thiếu cột bắt buộc phải ném ValueError."""
        df = pd.DataFrame({
            "path": ["data/real/card_01.jpg"],
            "label": ["real"],
        })
        with self.assertRaises(ValueError) as ctx:
            validate_metadata_df(df, project_root=self.root)
        self.assertIn("Thiếu các cột bắt buộc", str(ctx.exception))

    def test_validate_metadata_df_invalid_label(self):
        """Nhãn ngoài [real, fake, 0, 1] phải ném ValueError."""
        df = pd.DataFrame({
            "path": ["data/real/card_01.jpg"],
            "label": ["unknown_label"],
            "fraud_type": ["none"],
            "group_id": ["card_01"],
        })
        with self.assertRaises(ValueError) as ctx:
            validate_metadata_df(df, project_root=self.root)
        self.assertIn("Nhãn không hợp lệ", str(ctx.exception))

    def test_validate_metadata_df_invalid_fraud_type(self):
        """Fraud type không nằm trong danh mục cho phép phải ném ValueError."""
        df = pd.DataFrame({
            "path": ["data/fake/card_01_fake.jpg"],
            "label": ["fake"],
            "fraud_type": ["deepfake_magic"],
            "group_id": ["card_01"],
        })
        with self.assertRaises(ValueError) as ctx:
            validate_metadata_df(df, project_root=self.root)
        self.assertIn("fraud_type không hợp lệ", str(ctx.exception))

    def test_validate_metadata_inconsistent_real_fraud_type(self):
        """Nhãn real nhưng fraud_type khác none phải ném ValueError."""
        df = pd.DataFrame({
            "path": ["data/real/card_01.jpg"],
            "label": ["real"],
            "fraud_type": ["replace_photo"],
            "group_id": ["card_01"],
        })
        with self.assertRaises(ValueError) as ctx:
            validate_metadata_df(df, project_root=self.root)
        self.assertIn("không nhất quán", str(ctx.exception))

    def test_validate_feature_bundle_valid(self):
        """Feature bundle khớp số lượng và không có NaN phải vượt qua kiểm tra."""
        x = np.ones((5, 60))
        y = np.array([0, 1, 0, 1, 1])
        groups = np.array(["g1", "g1", "g2", "g2", "g3"])
        fraud_types = np.array(["none", "replace_photo", "none", "reprint", "splice"])
        # Không ném ngoại lệ là thành công
        validate_feature_bundle(x, y, groups, fraud_types, expected_dim=60)

    def test_validate_feature_bundle_nan_detected(self):
        """Feature bundle chứa NaN phải ném ValueError."""
        x = np.ones((4, 60))
        x[2, 5] = np.nan
        y = np.array([0, 1, 0, 1])
        groups = np.array(["g1", "g1", "g2", "g2"])
        fraud_types = np.array(["none", "replace_photo", "none", "reprint"])
        with self.assertRaises(ValueError) as ctx:
            validate_feature_bundle(x, y, groups, fraud_types, expected_dim=60)
        self.assertIn("NaN", str(ctx.exception))

    def test_validate_feature_bundle_length_mismatch(self):
        """Kích thước x và y không khớp nhau phải ném ValueError."""
        x = np.ones((4, 60))
        y = np.array([0, 1, 0])
        groups = np.array(["g1", "g1", "g2"])
        fraud_types = np.array(["none", "replace_photo", "none"])
        with self.assertRaises(ValueError) as ctx:
            validate_feature_bundle(x, y, groups, fraud_types, expected_dim=60)
        self.assertIn("không đồng nhất", str(ctx.exception))

    def test_strict_mode_raises_on_missing_file(self):
        """Ở chế độ strict=True, nạp metadata trỏ tới file không tồn tại phải ném FileNotFoundError."""
        csv_path = self.root / "labels.csv"
        df = pd.DataFrame({
            "path": ["data/real/non_existent_card.jpg"],
            "label": ["real"],
            "fraud_type": ["none"],
            "group_id": ["card_99"],
        })
        df.to_csv(csv_path, index=False)
        with self.assertRaises(FileNotFoundError):
            build_dataset(csv_path=csv_path, project_root=self.root, strict=True)


if __name__ == "__main__":
    unittest.main()
