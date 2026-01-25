import mlflow
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, log_loss,
    precision_recall_curve, RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay
)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

class MetricsTools:
    """
    Classe pour calculer, optimiser et loguer toutes les métriques principales,
    avec MLflow : sous-dossiers 'classification/' et 'probability/'.
    Compatible backend interactif par défaut, sans Tkinter.
    """

    def __init__(self, y_true, y_pred_proba=None, params=None):
        self.y_true = y_true
        self.y_pred_proba = y_pred_proba
        self.best_threshold = 0.5
        self.y_pred = None
        self.classification_metrics = {}
        self.probability_metrics = {}
        self.params = params if params is not None else {}

    def optimize_threshold(self):
        if self.y_pred_proba is None:
            raise ValueError("y_pred_proba est requis pour optimiser le seuil.")
        precisions, recalls, thresholds = precision_recall_curve(self.y_true, self.y_pred_proba)
        f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
        best_idx = np.argmax(f1_scores)
        self.best_threshold = thresholds[best_idx]
        self.y_pred = (self.y_pred_proba >= self.best_threshold).astype(int)
        return self.best_threshold

    def compute_metrics(self):
        if self.y_pred is None:
            if self.y_pred_proba is not None:
                self.y_pred = (self.y_pred_proba >= self.best_threshold).astype(int)
            else:
                self.y_pred = self.y_true

        self.classification_metrics = {
            "accuracy": accuracy_score(self.y_true, self.y_pred),
            "precision": precision_score(self.y_true, self.y_pred),
            "recall": recall_score(self.y_true, self.y_pred),
            "f1_score": f1_score(self.y_true, self.y_pred)
        }

        if self.y_pred_proba is not None:
            self.probability_metrics = {
                "mse": mean_squared_error(self.y_true, self.y_pred_proba),
                "mae": mean_absolute_error(self.y_true, self.y_pred_proba),
                "log_loss": log_loss(self.y_true, self.y_pred_proba)
            }

        return self.get_metrics()

    def log_mlflow(self):
        # Log params
        for k, v in self.params.items():
            if v is not None:
                mlflow.log_param(k, v)

        # Log classification metrics
        for k, v in self.classification_metrics.items():
            mlflow.log_metric(f"classification/{k}", v)

        # Log probability metrics
        for k, v in self.probability_metrics.items():
            mlflow.log_metric(f"probability/{k}", v)

        # Log threshold si disponible
        if self.y_pred_proba is not None:
            mlflow.log_param("best_threshold", self.best_threshold)

    def log_figures_mlflow(self):
        # Precision-Recall Curve
        if self.y_pred_proba is not None:
            fig_pr, ax_pr = plt.subplots()
            PrecisionRecallDisplay.from_predictions(self.y_true, self.y_pred_proba, ax=ax_pr)
            ax_pr.set_title("Precision-Recall Curve")
            mlflow.log_figure(fig_pr, "precision_recall_curve.png")
            plt.close(fig_pr)

            # ROC Curve
            fig_roc, ax_roc = plt.subplots()
            RocCurveDisplay.from_predictions(self.y_true, self.y_pred_proba, ax=ax_roc)
            ax_roc.set_title("ROC Curve")
            mlflow.log_figure(fig_roc, "roc_curve.png")
            plt.close(fig_roc)

        # Confusion Matrix
        fig_cm, ax_cm = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(self.y_true, self.y_pred, ax=ax_cm)
        ax_cm.set_title("Confusion Matrix")
        mlflow.log_figure(fig_cm, "confusion_matrix.png")
        plt.close(fig_cm)

    def summary(self):
        print("=== Classification Metrics ===")
        for k, v in self.classification_metrics.items():
            print(f"{k}: {v:.4f}")
        if self.probability_metrics:
            print("\n=== Probability Metrics ===")
            for k, v in self.probability_metrics.items():
                print(f"{k}: {v:.4f}")
        if self.y_pred_proba is not None:
            print(f"\nBest threshold used: {self.best_threshold:.3f}")

    def get_params(self):
        return self.params

    def get_metrics(self):
        return {**self.classification_metrics, **self.probability_metrics}

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            self.params[k] = v
        return self.params
