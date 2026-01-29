import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, ConfusionMatrixDisplay

class FlowTools:
    """
    Classe utilitaire pour gérer un workflow MLflow complet :
    - Logging dataset
    - Logging figures de classification (Precision-Recall, ROC, Confusion Matrix)
    - Logging figures de régression (y_true vs y_pred, histogramme des résidus)
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

    # -----------------------------------
    # Dataset
    # -----------------------------------
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

    # -----------------------------------
    # Classification
    # -----------------------------------
    def log_classification_figures(self, y_true, y_pred, prefix: str = "metrics"):
        """
        Log les figures de classification : Precision-Recall, ROC, Confusion Matrix.
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

        print("Figures de classification loggées dans MLflow")

    # -----------------------------------
    # Régression
    # -----------------------------------
    def log_regression_figures(self, y_true, y_pred, prefix: str = "regression"):
        """
        Log les figures de régression :
        - y_true vs y_pred
        - histogramme des résidus
        """
        # y_true vs y_pred
        fig1, ax1 = plt.subplots(figsize=(6,6))
        ax1.scatter(y_true, y_pred, alpha=0.5)
        ax1.plot([y_true.min(), y_true.max()],
                 [y_true.min(), y_true.max()], 'r--')
        ax1.set_xlabel("y_true")
        ax1.set_ylabel("y_pred")
        ax1.set_title("Régression : y_true vs y_pred")
        mlflow.log_figure(fig1, f"{prefix}/y_true_vs_y_pred.png")
        plt.close(fig1)

        # Histogramme des résidus
        residus = y_true - y_pred
        fig2, ax2 = plt.subplots(figsize=(6,4))
        sns.histplot(residus, bins=30, kde=True, ax=ax2)
        ax2.set_xlabel("Erreur (y_true - y_pred)")
        ax2.set_title("Distribution des résidus")
        mlflow.log_figure(fig2, f"{prefix}/residuals_histogram.png")
        plt.close(fig2)

        print("Figures de régression loggées dans MLflow")
