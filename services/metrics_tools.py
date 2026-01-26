import mlflow
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)

class MetricsTools:
    """
    Classe universelle pour calculer et logger les métriques, totalement paramétrable.
    On passe tous les inputs (y_true, y_pred, y_pred_proba, type de tâche) via params.
    """
    def __init__(self, params=None):
        """
        Params attendus :
        - y_true : array-like
        - y_pred : array-like
        - y_pred_proba : array-like (optionnel, pour classification probabiliste)
        - task : "classification" ou "regression"
        - autres paramètres MLflow (model, features, etc.)
        """
        self.params = params if params is not None else {}
        self.y_true = self.params.get("y_true", None)
        self.y_pred = self.params.get("y_pred", None)
        self.y_pred_proba = self.params.get("y_pred_proba", None)
        self.task = self.params.get("task", "regression")
        self.metrics = {}

        if self.y_true is None or self.y_pred is None:
            raise ValueError("y_true et y_pred doivent être fournis dans params.")

    def compute_metrics(self):
        if self.task == "classification":
            # Classification
            self.metrics = {
                "accuracy": accuracy_score(self.y_true, self.y_pred),
                "precision": precision_score(self.y_true, self.y_pred, average="weighted", zero_division=0),
                "recall": recall_score(self.y_true, self.y_pred, average="weighted", zero_division=0),
                "f1_score": f1_score(self.y_true, self.y_pred, average="weighted", zero_division=0)
            }
            # Probabilités si disponibles
            if self.y_pred_proba is not None:
                from sklearn.metrics import log_loss
                self.metrics.update({
                    "log_loss": log_loss(self.y_true, self.y_pred_proba),
                    "mse_proba": mean_squared_error(self.y_true, self.y_pred_proba),
                    "mae_proba": mean_absolute_error(self.y_true, self.y_pred_proba)
                })
        elif self.task == "regression":
            # Régression : calcul RMSE manuellement pour compatibilité
            mse = mean_squared_error(self.y_true, self.y_pred)
            rmse = np.sqrt(mse)
            self.metrics = {
                "MSE": mse,
                "RMSE": rmse,
                "MAE": mean_absolute_error(self.y_true, self.y_pred),
                "R2": r2_score(self.y_true, self.y_pred)
            }
        else:
            raise ValueError(f"Tâche inconnue : {self.task}")
        return self.metrics

    def summary(self):
        print(f"=== Metrics Summary ({self.task}) ===")
        for k, v in self.metrics.items():
            print(f"{k}: {v:.4f}")

    def log_mlflow(self):
        # Log de tous les paramètres
        for k, v in self.params.items():
            mlflow.log_param(k, v)
        # Log des métriques
        for k, v in self.metrics.items():
            mlflow.log_metric(f"{self.task}/{k}", v)

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            self.params[k] = v
        # Mettre à jour les attributs si pertinents
        if "y_true" in kwargs: self.y_true = kwargs["y_true"]
        if "y_pred" in kwargs: self.y_pred = kwargs["y_pred"]
        if "y_pred_proba" in kwargs: self.y_pred_proba = kwargs["y_pred_proba"]
        if "task" in kwargs: self.task = kwargs["task"]
        return self.params

    def get_metrics(self):
        return self.metrics

    def get_params(self):
        return self.params
