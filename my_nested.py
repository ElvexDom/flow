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

# Vérification de l'experiment_id
print("Experiment ID utilisé :", tools.experiment_id)

# ------------------------------
# Autolog sklearn
# ------------------------------
mlflow.sklearn.autolog(log_input_examples=True, log_models=True, silent=True)

# ------------------------------
# Paramètres à tester
# ------------------------------
base_params = pipeline_tools.get_params()
test_sizes = [0.2, 0.3]
random_states = [42, 7]

# ------------------------------
# Run MLflow parent
# ------------------------------
with mlflow.start_run(run_name="parent_pipeline_run", experiment_id=tools.experiment_id) as parent:
    tools.log_dataset(df)
    mlflow.log_param("dataset_rows", df.shape[0])

    # Boucle sur toutes les combinaisons de paramètres
    for ts in test_sizes:
        for rs in random_states:
            run_name = f"child_ts{ts}_rs{rs}"

            # Nested run : passer experiment_id explicitement
            with mlflow.start_run(run_name=run_name, experiment_id=tools.experiment_id, nested=True) as child:
                # Paramètres pour ce run
                params = base_params.copy()
                params["test_size"] = ts
                params["random_state"] = rs
                mlflow.log_params(params)

                # Split train/test
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=ts, random_state=rs, stratify=y
                )

                # Entraînement du pipeline
                pipeline_tools.fit(X_train, y_train)

                # Prédictions probabilistes si supportées
                try:
                    y_pred_proba = pipeline_tools.predict_proba(X_test)[:, 1]
                except AttributeError:
                    y_pred_proba = None

                # Instanciation MetricsTools
                metrics_tool = MetricsTools(y_true=y_test, y_pred_proba=y_pred_proba, params=params)

                # Optimisation du seuil si applicable
                # if y_pred_proba is not None:
                #     best_threshold = metrics_tool.optimize_threshold()
                #     mlflow.log_metric("best_threshold", best_threshold)
                # else:
                #     metrics_tool.y_pred = pipeline_tools.predict(X_test)
                metrics_tool.y_pred = pipeline_tools.predict(X_test)

                # Calcul des métriques
                metrics = metrics_tool.compute_metrics()
                metrics_tool.summary()
                metrics_tool.log_mlflow()
                metrics_tool.log_figures_mlflow()

                print(f"Child run {child.info.run_id} terminé pour test_size={ts}, random_state={rs}")

    print(f"Parent run ID: {parent.info.run_id}")
