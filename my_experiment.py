from services.flow_tools import FlowTools
from services.pipeline_tools import PipelineTools
from services.metrics_tools import MetricsTools
from sklearn.model_selection import train_test_split
import pandas as pd
import mlflow
import mlflow.sklearn
import numpy as np

# ------------------------------
# Charger le dataset
# ------------------------------
df = pd.read_csv("data/churn.csv")
df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])
X = df.drop("Exited", axis=1)
y = df["Exited"]

# ------------------------------
# Définition des colonnes
# ------------------------------
num_features = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]
cat_features = ["Geography"]
ordinal_features = ["Gender"]

# Convertir les colonnes int en float pour éviter warning MLflow
int_cols = ["CreditScore", "Age", "Tenure", "NumOfProducts"]
X[int_cols] = X[int_cols].astype(float)

# ------------------------------
# Instanciation des outils
# ------------------------------
tools = FlowTools(experiment_name="My Experiment")
pipeline_tools = PipelineTools(num_features, cat_features, ordinal_features)

# ------------------------------
# Autolog sklearn
# ------------------------------
mlflow.sklearn.autolog(log_input_examples=True, log_models=True, silent=True)

# ------------------------------
# Récupération des params dynamiques depuis le pipeline
# ------------------------------
params = pipeline_tools.get_params()
params["test_size"] = 0.2  # ajout du paramètre test_size pour logging

# ------------------------------
# Run MLflow
# ------------------------------
with mlflow.start_run(run_name="pipeline_run", experiment_id=tools.experiment_id) as run:
    # Log du dataset
    tools.log_dataset(df)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["test_size"], random_state=42, stratify=y
    )

    # Entraînement du pipeline
    pipeline_tools.fit(X_train, y_train)

    # Prédictions probabilistes si supportées
    try:
        y_pred_proba = pipeline_tools.predict_proba(X_test)[:, 1]
    except AttributeError:
        y_pred_proba = None

    # Instanciation de MetricsTools
    metrics_tool = MetricsTools(y_true=y_test, y_pred_proba=y_pred_proba, params=params)

    # Optimisation du seuil si possible
    if y_pred_proba is not None:
        best_threshold = metrics_tool.optimize_threshold()
        print(f"Seuil optimal pour F1 : {best_threshold:.3f}")
    else:
        metrics_tool.y_pred = pipeline_tools.predict(X_test)

    # Calcul des métriques
    metrics = metrics_tool.compute_metrics()
    metrics_tool.summary()

    # Log params et metrics dans MLflow
    metrics_tool.log_mlflow()

    # Log des figures : PR, ROC, Confusion Matrix
    metrics_tool.log_figures_mlflow()

    # Affichage des IDs du run
    print(f"Run ID: {run.info.run_id}")
    print(f"Experiment ID: {run.info.experiment_id}")
