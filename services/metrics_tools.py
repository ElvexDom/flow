import mlflow
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    log_loss,
    roc_auc_score
)


class MetricsTools:
    """
    Classe universelle pour calculer et logger les métriques
    - Classification
    - Régression
    - Clustering
    """

    def __init__(self, params=None):
        self.params = params if params is not None else {}

        self.y_true = self.params.get("y_true", None)
        self.y_pred = self.params.get("y_pred", None)
        self.y_pred_proba = self.params.get("y_pred_proba", None)
        self.task = self.params.get("task", "regression")

        self.metrics = {}

        # Validation des entrées
        if self.task == "clustering":
            if self.y_pred is None:
                raise ValueError("Pour le clustering, y_pred (labels de clusters) est requis.")
        else:
            if self.y_true is None or self.y_pred is None:
                raise ValueError("y_true et y_pred sont requis pour classification et régression.")

    # --------------------------------------------------
    # Calcul des métriques
    # --------------------------------------------------
    def compute_metrics(self):
        if self.task == "classification":
            self.metrics = {}

            self.metrics["precision_score"] = precision_score(
                self.y_true, self.y_pred, average="weighted", zero_division=0
            )
            self.metrics["recall_score"] = recall_score(
                self.y_true, self.y_pred, average="weighted", zero_division=0
            )
            self.metrics["f1_score"] = f1_score(
                self.y_true, self.y_pred, average="weighted", zero_division=0
            )
            self.metrics["accuracy_score"] = accuracy_score(self.y_true, self.y_pred)

            if self.y_pred_proba is not None:
                y_proba = self.y_pred_proba
                if y_proba.ndim == 1:
                    y_proba = np.vstack([1 - y_proba, y_proba]).T

                self.metrics["log_loss"] = float(log_loss(self.y_true, y_proba))

                n_classes = len(np.unique(self.y_true))
                if n_classes > 2:
                    self.metrics["roc_auc"] = float(
                        roc_auc_score(self.y_true, y_proba, multi_class="ovr")
                    )
                else:
                    self.metrics["roc_auc"] = float(roc_auc_score(self.y_true, y_proba[:, 1]))
            else:
                self.metrics["log_loss"] = None
                self.metrics["roc_auc"] = None

            self.metrics["score"] = accuracy_score(self.y_true, self.y_pred)

        elif self.task == "regression":
            self.metrics = {}
            self.metrics["mean_squared_error"] = mean_squared_error(self.y_true, self.y_pred)
            self.metrics["mean_absolute_error"] = mean_absolute_error(self.y_true, self.y_pred)
            self.metrics["r2_score"] = r2_score(self.y_true, self.y_pred)
            self.metrics["root_mean_squared_error"] = np.sqrt(mean_squared_error(self.y_true, self.y_pred))
            self.metrics["score"] = r2_score(self.y_true, self.y_pred)

        elif self.task == "clustering":
            X = self.params.get("X", None)
            if X is None:
                raise ValueError("Pour le clustering, les features doivent être fournies via params['X']")

            labels = self.y_pred
            n_clusters = len(np.unique(labels))
            if n_clusters < 2:
                raise ValueError("Le clustering doit contenir au moins 2 clusters pour calculer les métriques.")

            self.metrics = {
                "silhouette_score": silhouette_score(X, labels),
                "davies_bouldin_score": davies_bouldin_score(X, labels),
                "calinski_harabasz_score": calinski_harabasz_score(X, labels),
                "n_clusters": int(n_clusters)
            }

        else:
            raise ValueError(f"Tâche inconnue : {self.task}")

        return self.metrics

    # --------------------------------------------------
    # Résumé console
    # --------------------------------------------------
    def summary(self):
        print(f"\n=== Metrics Summary ({self.task.upper()}) ===")
        if not self.metrics:
            print("Aucune métrique disponible.")
        for k, v in self.metrics.items():
            if isinstance(v, (int, float)):
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v}")

    # --------------------------------------------------
    # Logging MLflow
    # --------------------------------------------------
    def log_mlflow(self, prefix=None):
        # Log paramètres
        for k, v in self.params.items():
            if isinstance(v, (str, int, float, bool)):
                mlflow.log_param(k, v)
            else:
                mlflow.log_param(k, str(v))

        # Log métriques
        for k in self.metrics.keys():
            v = self.metrics[k]
            if isinstance(v, (int, float)):
                metric_name = f"{prefix}_{k}" if prefix else k
                mlflow.log_metric(metric_name, float(v))

    # --------------------------------------------------
    # Mise à jour params
    # --------------------------------------------------
    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            self.params[k] = v

        if "y_true" in kwargs:
            self.y_true = kwargs["y_true"]
        if "y_pred" in kwargs:
            self.y_pred = kwargs["y_pred"]
        if "y_pred_proba" in kwargs:
            self.y_pred_proba = kwargs["y_pred_proba"]
        if "task" in kwargs:
            self.task = kwargs["task"]

        return self.params

    # --------------------------------------------------
    # Getters
    # --------------------------------------------------
    def get_metrics(self):
        return self.metrics

    def get_params(self):
        return self.params
