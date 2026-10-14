"""
Module kết hợp đặc trưng đa miền (Feature Fusion):
LBP (Micro-texture) + GLCM (Haralick Spatial Co-occurrence) + 2D FFT (Frequency Domain)
"""

import numpy as np
from src.features.lbp_extractor import LBPExtractor
from src.features.glcm_extractor import GLCMExtractor
from src.features.fft_extractor import FFTExtractor


class MultiFeatureExtractor:
    def __init__(self, use_lbp=True, use_glcm=True, use_fft=True):
        self.use_lbp = use_lbp
        self.use_glcm = use_glcm
        self.use_fft = use_fft

        self.lbp = LBPExtractor() if use_lbp else None
        self.glcm = GLCMExtractor() if use_glcm else None
        self.fft = FFTExtractor() if use_fft else None

    def extract_patch_features(self, gray_patch):
        """
        Trích xuất vector kết hợp cho một khối ảnh (patch).
        """
        feats = []
        if self.use_lbp:
            feats.append(self.lbp.extract_histogram(gray_patch))
        if self.use_glcm:
            feats.append(self.glcm.extract_features(gray_patch))
        if self.use_fft:
            feats.append(self.fft.extract_features(gray_patch))

        return np.concatenate(feats)

    def extract_card_features(self, rois_dict):
        """
        Trích xuất vector đặc trưng toàn diện cho toàn bộ thẻ CCCD dựa trên các vùng ROI chính:
        portrait, id_number, name, dob_gender, security_pattern.
        """
        card_features = []
        for roi_name in ["portrait", "id_number", "name", "dob_gender", "security_pattern"]:
            if roi_name in rois_dict:
                patch = rois_dict[roi_name]
                feat = self.extract_patch_features(patch)
                card_features.append(feat)

        return np.concatenate(card_features)
