import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from services.flow_tools import FlowTools
from services.pipeline_tools import PipelineTools
from services.metrics_tools import MetricsTools

# ------------------------------
# 1. Charger dataset
# ------------------------------
df = pd.read_csv("data/housing.csv")
TARGET = "median_house_value"
df = df[df[TARGET] < 500000]

X = df.drop(columns=[TARGET])
y = df[TARGET]

# ------------------------------
# 2. Colonnes
# ------------------------------
num_features = [
    "longitude", "latitude",
    "housing_median_age",
    "total_rooms", "total_bedrooms",
    "population", "households",
    "median_income"
]
cat_features = ["ocean_proximity"]
ordinal_features = []

# Convertir int -> float
int_cols = [c for c in num_features if np.issubdtype(X[c].dtype, np.integer)]
X[int_cols] = X[int_cols].astype(float)

# Remplacer NaN par la médiane
for col in num_features:
    X[col] = X[col].fillna(X[col].median())

# ------------------------------
# 3. Modèles
# ------------------------------
regressors = {
    "LinearRegression": (LinearRegression, {}),
    "RandomForest": (
        RandomForestRegressor,
        {
            "n_estimators": 120,
            "max_depth": 12,
            "min_samples_split": 12,
            "min_samples_leaf": 6,
            "random_state": 42,
        },
    ),
    "SVR": (SVR, {"kernel": "rbf", "C": 1.0, "gamma": "scale"}),
    "XGBoost": (XGBRegressor, {"tree_method": "hist", "n_estimators": 100, "random_state": 42}),
    "LightGBM": (
        LGBMRegressor,
        {
            "n_estimators": 100,
            "learning_rate": 0.05,
            "random_state": 42,
            "force_col_wise": True,
            "verbose": -1
        }
    )
}

# ------------------------------
# 4. MLflow + FlowTools
# ------------------------------
tools = FlowTools(experiment_name="California Housing Benchmark")
mlflow.sklearn.autolog(log_models=True, silent=True)

results = []

# Normalisation target pour SVR
y_scaler = StandardScaler()
y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1)).ravel()

# ------------------------------
# 5. Boucle sur les modèles
# ------------------------------
for name, (model_class, model_params) in regressors.items():
    with mlflow.start_run(run_name=name, experiment_id=tools.experiment_id):
        print(f"⏳ Training {name}...")

        # ------------------------------
        # Config dict pour PipelineTools
        # ------------------------------
        config = {
            "num_features": num_features,
            "cat_features": cat_features,
            "ordinal_features": ordinal_features,
            "model_class": model_class,
            "model_params": model_params
        }

        # Pipeline
        pipe_tools = PipelineTools(
            X=X,
            y=y_scaled if name == "SVR" else y,
            config=config
        )

        # Train / Test
        pipe_tools.train(test_size=0.2, random_state=42)

        # ------------------------------
        # Prédictions
        # ------------------------------
        preds_train = pipe_tools.predict_train()
        preds_test = pipe_tools.predict_test()

        if name == "SVR":
            preds_train = y_scaler.inverse_transform(preds_train.reshape(-1, 1)).ravel()
            preds_test = y_scaler.inverse_transform(preds_test.reshape(-1, 1)).ravel()

        # ------------------------------
        # Metrics TRAIN
        # ------------------------------
        train_metrics_tool = MetricsTools(
            params={
                "y_true": pipe_tools.y_train,
                "y_pred": preds_train,
                "task": "regression",
            }
        )
        train_metrics = train_metrics_tool.compute_metrics()
        train_metrics_tool.log_mlflow(prefix="training")

        # ------------------------------
        # Metrics TEST
        # ------------------------------
        test_metrics_tool = MetricsTools(
            params={
                "y_true": pipe_tools.y_test,
                "y_pred": preds_test,
                "task": "regression",
            }
        )
        test_metrics = test_metrics_tool.compute_metrics()
        test_metrics_tool.log_mlflow(prefix="testing")
        test_metrics_tool.summary()

        results.append({"model": name, **test_metrics})

        # ------------------------------
        # Figures MLflow
        # ------------------------------
        tools.log_regression_figures(
            pipe_tools.y_train, preds_train, prefix="regression_train"
        )
        tools.log_regression_figures(
            pipe_tools.y_test, preds_test, prefix="regression_test"
        )

        # ------------------------------
        # Log dataset
        # ------------------------------
        tools.log_dataset(df, artifact_path="datasets")

    print(f"✅ Modèle {name} terminé et loggé dans MLflow")

# ------------------------------
# 6. Résultats finaux
# ------------------------------
results_df = pd.DataFrame(results)
print("\n=== Résultats finaux ===")
print(results_df)
