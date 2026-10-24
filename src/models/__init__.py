from src.models.classifier import (
    TextureFraudClassifier,
    build_rf_classifier,
    build_svm_classifier,
    load_model,
    save_model,
)
from src.models.evaluation import (
    calc_accuracy,
    calc_auc_roc,
    calc_eer,
    evaluate_all,
    evaluate_by_fraud_type,
    plot_fraud_type_report,
    print_fraud_type_report,
)

__all__ = [
    "TextureFraudClassifier",
    "build_svm_classifier",
    "build_rf_classifier",
    "save_model",
    "load_model",
    "split_dataset_by_group",
    "split_indices_by_group",
    "estimate_eer_threshold_oof",
    "cross_validate_grouped",
    "tune_hyperparameters",
    "train_and_evaluate_pipeline",
    "calc_accuracy",
    "calc_auc_roc",
    "calc_eer",
    "evaluate_all",
    "evaluate_by_fraud_type",
    "print_fraud_type_report",
    "plot_fraud_type_report",
]


def __getattr__(name: str):
    if name in (
        "split_dataset_by_group",
        "split_indices_by_group",
        "estimate_eer_threshold_oof",
        "cross_validate_grouped",
        "tune_hyperparameters",
        "train_and_evaluate_pipeline",
    ):
        import src.models.train as train_mod

        return getattr(train_mod, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
