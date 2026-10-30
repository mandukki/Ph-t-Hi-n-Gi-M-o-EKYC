"""Exploratory patch-level texture anomaly map for a normalized card image.

The reference bank is fitted on genuine cards only. This module does not
claim pixel-level forgery localization: it ranks fixed regions by how unusual
their texture is compared with genuine cards at the same position.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from sklearn.ensemble import RandomForestClassifier

from src.features.fft import extract_fft
from src.features.lbp import extract_lbp


GRID_ROWS = 4
GRID_COLS = 6


def grid_boxes(shape: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    """Return (x1, y1, x2, y2) for a 4x6 non-overlapping grid."""
    height, width = shape
    if height < GRID_ROWS or width < GRID_COLS:
        raise ValueError("Image is smaller than the localization grid")
    ys = np.linspace(0, height, GRID_ROWS + 1, dtype=int)
    xs = np.linspace(0, width, GRID_COLS + 1, dtype=int)
    return [
        (int(xs[col]), int(ys[row]), int(xs[col + 1]), int(ys[row + 1]))
        for row in range(GRID_ROWS) for col in range(GRID_COLS)
    ]


def extract_patch_features(patch: np.ndarray) -> np.ndarray:
    """LBP + quantized GLCM + FFT features for one grayscale patch."""
    if patch.ndim != 2 or patch.dtype != np.uint8:
        raise ValueError("Expected a 2D uint8 grayscale patch")
    quantized = (patch // 16).astype(np.uint8)
    matrix = graycomatrix(
        quantized, distances=[1, 3], angles=[0, np.pi / 2],
        levels=16, symmetric=True, normed=True,
    )
    glcm = np.concatenate([
        graycoprops(matrix, name).ravel()
        for name in ("contrast", "dissimilarity", "homogeneity", "energy")
    ])
    return np.concatenate((extract_lbp(patch), glcm, extract_fft(patch, n_bins=8)))


def extract_grid_features(image: np.ndarray) -> np.ndarray:
    """Shape: 24 grid cells x 34 features."""
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("Expected a 2D uint8 grayscale card")
    return np.stack([
        extract_patch_features(image[y1:y2, x1:x2])
        for x1, y1, x2, y2 in grid_boxes(image.shape)
    ])


class WeakPatchClassifier:
    """Patch classifier fitted on generator-derived weak edit boxes."""

    def __init__(self) -> None:
        self.model = RandomForestClassifier(
            n_estimators=200, min_samples_leaf=3,
            class_weight="balanced_subsample", random_state=42, n_jobs=1,
        )

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "WeakPatchClassifier":
        if features.ndim != 3 or features.shape[1:] != (24, 34):
            raise ValueError("Expected n x 24 x 34 patch features")
        if labels.shape != features.shape[:2] or set(np.unique(labels)) != {0, 1}:
            raise ValueError("Expected binary labels for every patch")
        self.model.fit(features.reshape(-1, 34), labels.ravel())
        return self

    def score_features(self, features: np.ndarray) -> np.ndarray:
        if features.shape != (24, 34):
            raise ValueError("Expected 24 x 34 patch features")
        return self.model.predict_proba(features)[:, 1].reshape(GRID_ROWS, GRID_COLS)

    def score(self, image: np.ndarray) -> np.ndarray:
        return self.score_features(extract_grid_features(image))


@dataclass
class TextureLocalizer:
    """Nearest genuine texture at each fixed card region."""

    reference_features: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, real_images: list[np.ndarray]) -> "TextureLocalizer":
        if len(real_images) < 2:
            raise ValueError("At least two genuine reference images are required")
        shape = real_images[0].shape
        if any(img.shape != shape for img in real_images):
            raise ValueError("All reference images must share the normalized shape")
        bank = np.stack([extract_grid_features(img) for img in real_images])
        flat = bank.reshape(-1, bank.shape[-1])
        q25, q75 = np.percentile(flat, [25, 75], axis=0)
        scale = np.maximum(q75 - q25, 1e-4)
        return cls(bank, scale)

    def score(self, image: np.ndarray) -> np.ndarray:
        """Return a 4x6 anomaly map; larger means less like known genuine patches."""
        if image.shape != (540, 856):
            raise ValueError("Expected a normalized 540x856 card image")
        features = extract_grid_features(image)
        distances = np.linalg.norm(
            (self.reference_features - features[None, :, :]) / self.scale,
            axis=2,
        )
        return distances.min(axis=0).reshape(GRID_ROWS, GRID_COLS)


def render_heatmap(
    image: np.ndarray, scores: np.ndarray, top_k: int = 3,
    fixed_zero_one_scale: bool = False,
) -> np.ndarray:
    """Draw top cells; use a fixed color scale for classifier scores in [0, 1]."""
    if scores.shape != (GRID_ROWS, GRID_COLS):
        raise ValueError("Invalid anomaly map shape")
    if not 1 <= top_k <= scores.size:
        raise ValueError("top_k must be between 1 and the number of grid cells")
    color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    expanded = cv2.resize(
        scores.astype(np.float32), (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    if fixed_zero_one_scale:
        normalized = np.uint8(np.clip(expanded * 255, 0, 255))
    else:
        low, high = float(np.min(scores)), float(np.max(scores))
        normalized = np.zeros_like(expanded, dtype=np.uint8)
        if high > low:
            normalized = np.uint8(np.clip((expanded - low) / (high - low) * 255, 0, 255))
    heat = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(color, 0.55, heat, 0.45, 0)
    boxes = grid_boxes(image.shape)
    for rank, index in enumerate(np.argsort(scores.ravel())[-top_k:][::-1], start=1):
        x1, y1, x2, y2 = boxes[int(index)]
        cv2.rectangle(overlay, (x1, y1), (x2 - 1, y2 - 1), (255, 255, 255), 2)
        cv2.putText(overlay, str(rank), (x1 + 5, y1 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return overlay
