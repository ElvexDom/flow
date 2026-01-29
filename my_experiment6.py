import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import mlflow
import mlflow.sklearn

# Import des outils custom
from services.flow_tools import FlowTools
from services.pipeline_tools import PipelineTools
from services.metrics_tools import MetricsTools

# ------------------------------
# 1. Charger le dataset
# ------------------------------
df = pd.read_csv("data/housing.csv")

TARGET = "median_house_value"

# Supprimer le plafond artificiel
df = df[df[TARGET] < 500000]

X = df.drop(columns=[TARGET])
y = df[TARGET]

# ------------------------------
# 2. Définition des colonnes
# ------------------------------
num_features = [
    "longitude", "latitude",
    "housing_median_age",
    "total_rooms", "total_bedrooms",
    "population", "households",
    "median_income"
]

cat_features = ["ocean_proximity"]
ordinal_features = []  # Pas de colonnes ordinales

# ------------------------------
# 3a. Convertir certaines colonnes int en float pour MLflow
# ------------------------------
int_cols = ["housing_median_age", "total_rooms", "total_bedrooms", "population", "households"]
X[int_cols] = X[int_cols].astype(float)

# ------------------------------
# 3b. Remplacer les NaN par la médiane
# ------------------------------
for col in num_features:
    median_value = X[col].median()
    X[col] = X[col].fillna(median_value)

# ------------------------------
# 4. Instanciation des outils
# ------------------------------
tools = FlowTools(experiment_name="California Housing Regression")

pipeline_tools = PipelineTools(
    num_features=num_features,
    cat_features=cat_features,
    ordinal_features=ordinal_features,
    model_class=LinearRegression
)

# ------------------------------
# 5. Autolog sklearn
# ------------------------------
mlflow.sklearn.autolog(log_input_examples=True, log_models=True, silent=True)

# ------------------------------
# 6. Récupération des params dynamiques depuis le pipeline
# ------------------------------
params = pipeline_tools.get_params()
params["test_size"] = 0.2

# ------------------------------
# 7. Run MLflow
# ------------------------------
with mlflow.start_run(run_name="pipeline_run", experiment_id=tools.experiment_id) as run:
    # Log du dataset
    tools.log_dataset(df)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["test_size"], random_state=42
    )

    # Entraînement du pipeline
    pipeline_tools.fit(X_train, y_train)

    # Prédictions
    y_pred = pipeline_tools.predict(X_test)

    # ------------------------------
    # 8. Calcul et logging des métriques
    # ------------------------------
    metrics_tool = MetricsTools(
        params={**params, "y_true": y_test, "y_pred": y_pred, "task": "regression"}
    )
    metrics_tool.compute_metrics()
    metrics_tool.summary()
    metrics_tool.log_mlflow(prefix="testing")

    # ------------------------------
    # 9. Log figure : y_true vs y_pred
    # ------------------------------
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(y_test, y_pred, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel("y_true")
    ax.set_ylabel("y_pred")
    ax.set_title("Régression : y_true vs y_pred")
    mlflow.log_figure(fig, "figures/y_true_vs_y_pred.png")
    plt.close(fig)

    print("✅ Figures loggées dans MLflow")

    # ------------------------------
    # 10. IDs du run
    # ------------------------------
    print(f"Run ID: {run.info.run_id}")
    print(f"Experiment ID: {run.info.experiment_id}")
