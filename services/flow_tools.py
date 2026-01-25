import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, ConfusionMatrixDisplay

class FlowTools:
    """
    Classe utilitaire pour gérer un workflow MLflow complet :
    logging dataset et figures de métriques (Precision-Recall, ROC, Confusion Matrix).
    """

    def __init__(self, experiment_name: str = None):
        """
        Initialise la classe et configure l'expérience MLflow.
        """
        if experiment_name:
            self.experiment = mlflow.get_experiment_by_name(experiment_name)
            if self.experiment is None:
                self.experiment_id = mlflow.create_experiment(experiment_name)
                self.experiment = mlflow.get_experiment(self.experiment_id)
            else:
                self.experiment_id = self.experiment.experiment_id
        else:
            self.experiment = None
            self.experiment_id = None

    def log_dataset(self, df: pd.DataFrame, artifact_path: str = "datasets"):
        """
        Log un dataset (DataFrame) dans MLflow.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df doit être un DataFrame pandas.")

        df.to_csv("temp_dataset.csv", index=False)
        mlflow.log_artifact("temp_dataset.csv", artifact_path=artifact_path)
        print(f"Dataset loggé dans MLflow sous '{artifact_path}'")
        return artifact_path

    def log_metrics_figures(self, y_true, y_pred, prefix: str = "metrics"):
        """
        Log les figures : Precision-Recall, ROC, Confusion Matrix.
        """
        # Precision-Recall Curve
        fig_pr = plt.figure()
        PrecisionRecallDisplay.from_predictions(y_true, y_pred, ax=plt.gca())
        plt.title("Precision-Recall Curve")
        mlflow.log_figure(fig_pr, f"{prefix}/precision_recall_curve.png")
        plt.close()

        # ROC Curve
        fig_roc = plt.figure()
        RocCurveDisplay.from_predictions(y_true, y_pred, ax=plt.gca())
        plt.title("ROC Curve")
        mlflow.log_figure(fig_roc, f"{prefix}/roc_curve.png")
        plt.close()

        # Confusion Matrix
        fig_cm = plt.figure()
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=plt.gca())
        plt.title("Confusion Matrix")
        mlflow.log_figure(fig_cm, f"{prefix}/confusion_matrix.png")
        plt.close()

        print("Figures de métriques loggées dans MLflow")
