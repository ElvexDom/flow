import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

from services.flow_tools import FlowTools
from services.pipeline_tools import PipelineTools
from services.metrics_tools import MetricsTools

# ------------------------------
# Charger dataset
# ------------------------------
df = pd.read_csv("data/car.csv")

# Nettoyage éventuel
for col in ["RowNumber", "CustomerId", "Surname"]:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

# Cible et features
target_col = "selling_price"
X = df.drop(columns=[target_col])
y = df[target_col]

# ------------------------------
# Colonnes
# ------------------------------
num_features = ["year", "km_driven"]
cat_features = ["fuel", "seller_type", "transmission"]
ordinal_features = ["owner"]

# Regrouper les marques rares en "Autre"
if "name" in X.columns:
    counts = X["name"].value_counts()
    rare_brands = counts[counts < 5].index
    X["name"] = X["name"].replace(rare_brands, "Autre")
    cat_features.append("name")

# Convertir int -> float pour MLflow
int_cols = [c for c in num_features if np.issubdtype(X[c].dtype, np.integer)]
X[int_cols] = X[int_cols].astype(float)

# ------------------------------
# Instanciation PipelineTools
# ------------------------------
pipeline_tools = PipelineTools(
    num_features=num_features,
    cat_features=cat_features,
    ordinal_features=ordinal_features,
    model_class=LinearRegression  # remplacer par un autre modèle si souhaité
)

# ------------------------------
# Split train/test
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------
# MLflow et FlowTools
# ------------------------------
tools = FlowTools(experiment_name="Car Price Regression")
mlflow.sklearn.autolog(log_input_examples=True, log_models=True, silent=True)

with mlflow.start_run(run_name="CarPrice_Regression", experiment_id=tools.experiment_id) as run:
    # Log dataset
    tools.log_dataset(df)

    # Entraînement pipeline
    pipeline_tools.fit(X_train, y_train)

    # Prédictions
    y_pred = pipeline_tools.predict(X_test)
    y_pred_proba = None  # pas de probabilités pour la régression

    # ------------------------------
    # MetricsTools avec tous les params dynamiques
    # ------------------------------
    params = pipeline_tools.get_params()
    # Ajouter infos dynamiques au params
    params.update({
        "task": "regression",          # "classification" si problème de classification
        "y_true": y_test.to_numpy(),
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba
    })

    metrics_tool = MetricsTools(params=params)

    # Calcul et résumé
    metrics_tool.compute_metrics()
    metrics_tool.summary()

    # Récupérer les métriques si besoin
    metrics_dict = metrics_tool.get_metrics()
    print("Metrics dict:", metrics_dict)

    # Log dans MLflow
    metrics_tool.log_mlflow()

    # ------------------------------
    # Figure optionnelle y_true vs y_pred
    # ------------------------------
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(y_test, y_pred, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel("y_true")
    ax.set_ylabel("y_pred")
    ax.set_title("y_true vs y_pred")
    plt.tight_layout()
    mlflow.log_figure(fig, "ytrue_ypred.png")
    plt.close(fig)

    # ------------------------------
    # IDs run
    print(f"Run ID: {run.info.run_id}")
    print(f"Experiment ID: {run.info.experiment_id}")
