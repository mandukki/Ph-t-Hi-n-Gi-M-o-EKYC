"""
Định nghĩa và quản lý các mô hình phân loại gian lận kết cấu (Texture Fraud Classifier).

Hỗ trợ hai kiến trúc chính:
- Support Vector Machine (RBF/Linear kernel) với StandardScaler. Mặc định sử dụng
  decision_function (không dùng Platt scaling nội bộ) để tránh rò rỉ cross-validation
  ngẫu nhiên và đảm bảo tính độc lập theo nhóm (group-aware).
- Random Forest Classifier với StandardScaler.

Tất cả các mô hình cung cấp giao diện chấm điểm thống nhất qua .predict_score(x),
trong đó điểm số cao hơn luôn đại diện cho khả năng gian lận (lớp 1 - FAKE).
"""

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


def build_svm_classifier(
    kernel: str = "rbf",
    c: float = 1.0,
    gamma: Union[str, float] = "scale",
    class_weight: Optional[str] = "balanced",
    probability: bool = False,
    random_state: int = 42,
) -> Pipeline:
    """Khởi tạo pipeline chuẩn hóa + SVM.

    Tham số:
        kernel: Loại kernel ('rbf', 'linear',...). Mặc định 'rbf'.
        c: Hệ số phạt C. Mặc định 1.0.
        gamma: Hệ số kernel. Mặc định 'scale'.
        class_weight: Trọng số lớp, mặc định 'balanced' để xử lý mất cân bằng real/fake.
        probability: Mặc định False để sử dụng decision_function (tránh Platt scaling nội
                     bộ chạy k-fold ngẫu nhiên không có group). Nếu True, cho phép tính
                     xác suất nhưng cần lưu ý rò rỉ nhóm khi hiệu chuẩn.
        random_state: Seed ngẫu nhiên.

    Trả về:
        Pipeline(StandardScaler -> SVC).
    """
    scaler = StandardScaler()
    svm_kwargs: Dict[str, Any] = {
        "C": c,
        "kernel": kernel,
        "gamma": gamma,
        "class_weight": class_weight,
        "random_state": random_state,
    }
    # Trong scikit-learn 1.9+, tham số `probability` có giá trị mặc định là 'deprecated' (tương đương False).
    # Chỉ truyền `probability=True` khi người dùng yêu cầu rõ ràng để tránh FutureWarning không cần thiết.
    if probability:
        svm_kwargs["probability"] = True

    svm = SVC(**svm_kwargs)
    return Pipeline(steps=[("scaler", scaler), ("svm", svm)])


def build_rf_classifier(
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    class_weight: Optional[str] = "balanced",
    random_state: int = 42,
    n_jobs: int = 1,
) -> Pipeline:
    """Khởi tạo pipeline chuẩn hóa + Random Forest.

    n_jobs được đặt mặc định là 1 để tránh lỗi deadlock đa tiến trình trên Windows.
    """
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
    """Lưu mô hình ra đĩa dưới định dạng joblib."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)
    return str(filepath.resolve())


def load_model(filepath: Union[str, Path]) -> Any:
    """Tải mô hình từ file joblib."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Không tìm thấy file mô hình tại: {filepath}")
    return joblib.load(filepath)


class TextureFraudClassifier:
    """Bộ phân loại phát hiện gian lận eKYC dựa trên kết cấu bề mặt.

    Đóng gói thống nhất logic huấn luyện, chấm điểm (score), dự đoán nhãn theo ngưỡng,
    và đánh giá chi tiết theo từng loại can thiệp.
    """

    def __init__(
        self,
        model_type: str = "svm",
        threshold: Optional[float] = None,
        **model_kwargs,
    ):
        self.model_type = model_type.lower()
        self.model_kwargs = model_kwargs
        self.pipeline: Optional[Pipeline] = None
        self.is_fitted: bool = False

        self._init_pipeline()

        # Ngưỡng mặc định: 0.0 cho decision_function của SVM; 0.5 cho xác suất (RF hoặc SVM probability=True)
        if threshold is not None:
            self.threshold = float(threshold)
        else:
            if self.model_type == "svm" and not self.model_kwargs.get("probability", False):
                self.threshold = 0.0
            else:
                self.threshold = 0.5

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

    @property
    def score_type(self) -> str:
        """Kiểu điểm số đầu ra của mô hình: 'decision_function' hoặc 'probability'."""
        if self.model_type == "svm" and not self.model_kwargs.get("probability", False):
            return "decision_function"
        return "probability"

    def fit(
        self,
        x: Union[np.ndarray, list],
        y: Union[np.ndarray, list],
    ) -> "TextureFraudClassifier":
        """Huấn luyện pipeline trên dữ liệu x, y."""
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=int)

        if self.pipeline is None:
            self._init_pipeline()

        self.pipeline.fit(x_arr, y_arr)
        self.is_fitted = True
        return self

    def predict_score(self, x: Union[np.ndarray, list]) -> np.ndarray:
        """Trả về vector điểm số 1D đại diện cho xác suất hoặc khoảng cách tới lề (Fake = lớp 1).

        Quy ước thống nhất: Điểm số càng lớn thì khả năng là ảnh GIẢ (FAKE) càng cao.
        - Với SVM (probability=False): dùng decision_function (giá trị > 0 nghiêng về Fake).
        - Với RF hoặc SVM (probability=True): dùng P(y=1|x) thuộc [0, 1].
        """
        if not self.is_fitted or self.pipeline is None:
            raise RuntimeError("Mô hình chưa được huấn luyện. Hãy gọi .fit() trước.")

        x_arr = np.asarray(x, dtype=np.float64)

        if self.score_type == "decision_function":
            scores = self.pipeline.decision_function(x_arr)
            # Trong scikit-learn binary SVC với classes_=[0, 1],
            # decision_function là 1D array với f(x) > 0 biểu thị lớp 1.
            classes = self.pipeline.classes_
            if len(classes) == 2 and classes[0] == 1 and classes[1] == 0:
                scores = -scores
            return np.asarray(scores, dtype=np.float64).ravel()

        # Trường hợp mô hình có predict_proba
        proba_matrix = self.pipeline.predict_proba(x_arr)
        classes = self.pipeline.classes_
        fake_idx = np.where(classes == 1)[0]
        if len(fake_idx) > 0:
            return np.asarray(proba_matrix[:, fake_idx[0]], dtype=np.float64)
        return np.asarray(proba_matrix[:, -1], dtype=np.float64)

    def predict_proba(self, x: Union[np.ndarray, list]) -> np.ndarray:
        """Trả về xác suất dự đoán cho lớp 1 (Fake) nếu mô hình hỗ trợ.

        Nếu mô hình SVM dùng decision_function, hàm sẽ báo lỗi rõ ràng hướng dẫn dùng .predict_score().
        """
        if self.score_type == "decision_function":
            raise AttributeError(
                "Mô hình SVM đang sử dụng decision_function (để chống rò rỉ group cross-validation). "
                "Hãy dùng .predict_score(x) để lấy điểm số quyết định thay vì .predict_proba(x)."
            )
        return self.predict_score(x)

    def predict(
        self,
        x: Union[np.ndarray, list],
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """Dự đoán nhãn nhị phân (0=Real, 1=Fake) dựa trên ngưỡng (cutoff)."""
        cutoff = self.threshold if threshold is None else threshold
        scores = self.predict_score(x)
        return (scores >= cutoff).astype(int)

    def evaluate(
        self,
        x: Union[np.ndarray, list],
        y: Union[np.ndarray, list],
        threshold: Optional[float] = None,
        update_best_threshold: bool = False,
    ) -> Dict[str, Any]:
        """Đánh giá toàn diện các chỉ số phân loại.

        LƯU Ý: update_best_threshold mặc định là False. Không được đặt True trên tập Test
        vì sẽ gây rò rỉ thông tin nhãn test vào ngưỡng quyết định.
        """
        cutoff = self.threshold if threshold is None else threshold
        scores = self.predict_score(x)
        metrics = evaluate_all(y_true=y, y_scores=scores, threshold=cutoff)

        if update_best_threshold:
            warnings.warn(
                "update_best_threshold=True đang cập nhật self.threshold từ tập dữ liệu hiện tại. "
                "Tuyệt đối không gọi tham số này trên tập Test/Validation ngoài fold để tránh rò rỉ.",
                UserWarning,
                stacklevel=2,
            )
            self.threshold = float(metrics["eer_threshold"])

        metrics["score_type"] = self.score_type
        return metrics

    def evaluate_fraud_types(
        self,
        x: Union[np.ndarray, list],
        y: Union[np.ndarray, list],
        fraud_types: Union[np.ndarray, list],
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Đánh giá chi tiết tỷ lệ phát hiện theo từng loại can thiệp giả mạo."""
        scores = self.predict_score(x)
        thresh = threshold if threshold is not None else self.threshold
        y_pred = (scores >= thresh).astype(int)
        return evaluate_by_fraud_type(
            y_true=y,
            y_pred=y_pred,
            fraud_types=fraud_types,
            y_scores=scores,
        )

    def save(self, filepath: Union[str, Path]) -> str:
        """Lưu trạng thái của classifier."""
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
        """Tải trạng thái classifier từ file."""
        state = load_model(filepath)
        if not isinstance(state, dict) or "model_type" not in state:
            raise ValueError("File mô hình không đúng định dạng của TextureFraudClassifier.")

        instance = cls(
            model_type=state["model_type"],
            threshold=state.get("threshold", 0.0 if state["model_type"] == "svm" else 0.5),
            **state.get("model_kwargs", {}),
        )
        instance.pipeline = state.get("pipeline")
        instance.is_fitted = state.get("is_fitted", False)
        return instance
