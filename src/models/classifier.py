import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import warnings
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.models.evaluation import evaluate_all, evaluate_by_fraud_type

warnings.filterwarnings("ignore", category=FutureWarning, message=".*probability.*")


def build_svm_classifier(
    kernel: str = "rbf",
    c: float = 1.0,
    gamma: Union[str, float] = "scale",
    class_weight: Optional[str] = "balanced",
    probability: bool = True,
    random_state: int = 42,
) -> Pipeline:
    scaler = StandardScaler()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*probability.*")
        svm = SVC(
            C=c,
            kernel=kernel,
            gamma=gamma,
            class_weight=class_weight,
            probability=probability,
            random_state=random_state,
        )
    return Pipeline(steps=[("scaler", scaler), ("svm", svm)])


def build_rf_classifier(
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    class_weight: Optional[str] = "balanced",
    random_state: int = 42,
    n_jobs: int = -1,
) -> Pipeline:
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    return Pipeline(steps=[("scaler", StandardScaler()), ("rf", rf)])


def save_model(model: Any, filepath: Union[str, Path]) -> str:
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)
    return str(filepath.resolve())


def load_model(filepath: Union[str, Path]) -> Any:
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Không tìm thấy file mô hình tại: {filepath}")
    return joblib.load(filepath)


class TextureFraudClassifier:
    def __init__(
        self,
        model_type: str = "svm",
        threshold: float = 0.5,
        **model_kwargs,
    ):
        self.model_type = model_type.lower()
        self.threshold = threshold
        self.model_kwargs = model_kwargs
        self.pipeline: Optional[Pipeline] = None
        self.is_fitted: bool = False

        self._init_pipeline()

    def _init_pipeline(self) -> None:
        if self.model_type == "svm":
            self.pipeline = build_svm_classifier(**self.model_kwargs)
        elif self.model_type in ("random_forest", "rf"):
            self.pipeline = build_rf_classifier(**self.model_kwargs)
        else:
            raise ValueError(
                f"Loại mô hình không được hỗ trợ: '{self.model_type}'. "
                f"Hỗ trợ: 'svm', 'random_forest', 'rf'."
            )

    def fit(
        self,
        x: Union[np.ndarray, list],
        y: Union[np.ndarray, list],
    ) -> "TextureFraudClassifier":
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)

        if self.pipeline is None:
            self._init_pipeline()

        self.pipeline.fit(x_arr, y_arr)
        self.is_fitted = True
        return self

    def predict_proba(self, x: Union[np.ndarray, list]) -> np.ndarray:
        if not self.is_fitted or self.pipeline is None:
            raise RuntimeError("Mô hình chưa được huấn luyện. Hãy gọi .fit() trước.")

        x_arr = np.asarray(x)
        proba_matrix = self.pipeline.predict_proba(x_arr)

        classes = self.pipeline.classes_
        fake_idx = np.where(classes == 1)[0]
        if len(fake_idx) > 0:
            return proba_matrix[:, fake_idx[0]].astype(np.float64)

        return proba_matrix[:, -1].astype(np.float64)

    def predict(
        self,
        x: Union[np.ndarray, list],
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        cutoff = self.threshold if threshold is None else threshold
        scores = self.predict_proba(x)
        return (scores >= cutoff).astype(int)

    def evaluate(
        self,
        x: Union[np.ndarray, list],
        y: Union[np.ndarray, list],
        threshold: Optional[float] = None,
        update_best_threshold: bool = False,
    ) -> Dict[str, Any]:
        scores = self.predict_proba(x)
        metrics = evaluate_all(y_true=y, y_scores=scores, threshold=threshold)

        if update_best_threshold:
            self.threshold = float(metrics["eer_threshold"])

        return metrics

    def evaluate_fraud_types(
        self,
        x: Union[np.ndarray, list],
        y: Union[np.ndarray, list],
        fraud_types: Union[np.ndarray, list],
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        scores = self.predict_proba(x)
        thresh = threshold if threshold is not None else self.threshold
        y_pred = (scores >= thresh).astype(int)
        return evaluate_by_fraud_type(
            y_true=y,
            y_pred=y_pred,
            fraud_types=fraud_types,
            y_scores=scores,
        )

    def save(self, filepath: Union[str, Path]) -> str:
        state = {
            "model_type": self.model_type,
            "threshold": self.threshold,
            "model_kwargs": self.model_kwargs,
            "pipeline": self.pipeline,
            "is_fitted": self.is_fitted,
        }
        return save_model(state, filepath)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "TextureFraudClassifier":
        state = load_model(filepath)
        if not isinstance(state, dict) or "model_type" not in state:
            raise ValueError("File mô hình không đúng định dạng của TextureFraudClassifier.")

        instance = cls(
            model_type=state["model_type"],
            threshold=state.get("threshold", 0.5),
            **state.get("model_kwargs", {}),
        )
        instance.pipeline = state.get("pipeline")
        instance.is_fitted = state.get("is_fitted", False)
        return instance
