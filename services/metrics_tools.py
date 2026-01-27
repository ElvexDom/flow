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
    calinski_harabasz_score
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

        # Dictionnaire final des métriques
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
            self.metrics.update({
                "accuracy": accuracy_score(self.y_true, self.y_pred),
                "precision": precision_score(
                    self.y_true, self.y_pred, average="weighted", zero_division=0
                ),
                "recall": recall_score(
                    self.y_true, self.y_pred, average="weighted", zero_division=0
                ),
                "f1_score": f1_score(
                    self.y_true, self.y_pred, average="weighted", zero_division=0
                )
            })

        elif self.task == "regression":
            mse = mean_squared_error(self.y_true, self.y_pred)
            self.metrics.update({
                "MSE": mse,
                "RMSE": np.sqrt(mse),
                "MAE": mean_absolute_error(self.y_true, self.y_pred),
                "R2": r2_score(self.y_true, self.y_pred)
            })

        elif self.task == "clustering":
            X = self.params.get("X", None)

            if X is None:
                raise ValueError(
                    "Pour le clustering, les features doivent être fournies via params['X']"
                )

            labels = self.y_pred
            n_clusters = len(np.unique(labels))

            if n_clusters < 2:
                raise ValueError(
                    "Le clustering doit contenir au moins 2 clusters pour calculer les métriques."
                )

            self.metrics.update({
                "silhouette_score": silhouette_score(X, labels),
                "davies_bouldin_score": davies_bouldin_score(X, labels),
                "calinski_harabasz_score": calinski_harabasz_score(X, labels),
                "n_clusters": int(n_clusters)
            })

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
            print(f"{k}: {v:.4f}")

    # --------------------------------------------------
    # Logging MLflow
    # --------------------------------------------------
    def log_mlflow(self):
        # Log paramètres (sécurisé)
        for k, v in self.params.items():
            if isinstance(v, (str, int, float, bool)):
                mlflow.log_param(k, v)
            else:
                mlflow.log_param(k, str(v))

        # Log métriques
        for k, v in self.metrics.items():
            mlflow.log_metric(f"{self.task}/{k}", float(v))

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
