import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns

from services.flow_tools import FlowTools
from services.metrics_tools import MetricsTools
from services.pipeline_tools import PipelineTools  # ton pipeline avec @monitor_performance

# ------------------------------
# Charger dataset
# ------------------------------
df = pd.read_csv("data/car.csv")

# Nettoyage éventuel
for col in ["RowNumber", "CustomerId", "Surname"]:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

TARGET = "selling_price"
X = df.drop(columns=[TARGET])
y = df[TARGET]

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
# Split train/test
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------
# Régression – Modèles à tester
# ------------------------------
regressors = {
    "LinearRegression": (LinearRegression, {}),
    "RandomForest": (RandomForestRegressor, {"n_estimators": 50, "max_depth": 10, "n_jobs": -1, "random_state": 42}),
    "SVR": (SVR, {"kernel": "rbf", "C": 1.0, "gamma": "scale", "max_iter": 5000}),
    "XGBoost": (XGBRegressor, {"tree_method": "hist", "n_estimators": 100, "random_state": 42}),
    "LightGBM": (LGBMRegressor, {"n_estimators": 100, "learning_rate": 0.05, "random_state": 42})
}

tools = FlowTools(experiment_name="my test car")
mlflow.sklearn.autolog(log_models=True, silent=True)

results = []

# Normalisation target pour SVR uniquement
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).ravel()

for name, (model_class, model_params) in regressors.items():
    with mlflow.start_run(run_name=name, experiment_id=tools.experiment_id):
        print(f"⏳ Training {name}...")

        # Créer le pipeline avec ton PipelineTools
        pipe_tools = PipelineTools(
            num_features=num_features,
            cat_features=cat_features,
            ordinal_features=ordinal_features,
            model_class=model_class,
            model_params=model_params
        )

        # SVR : utiliser target normalisée
        if name == "SVR":
            pipe_tools.fit(X_train, y_train_scaled)
            preds_scaled = pipe_tools.predict(X_test)
            preds = y_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
        else:
            pipe_tools.fit(X_train, y_train)
            preds = pipe_tools.predict(X_test)

        # ------------------------------
        # Metrics avec MetricsTools
        # ------------------------------
        metrics_tool = MetricsTools(params={
            "y_true": y_test,
            "y_pred": preds,
            "task": "regression"
        })
        metrics = metrics_tool.compute_metrics()
        metrics_tool.summary()
        metrics_tool.log_mlflow(prefix="testing")

        # Récupération directe du dict
        results.append({"model": name, **metrics})

# ------------------------------
# Visualisation
# ------------------------------
res_df = pd.DataFrame(results)
fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(data=res_df, x='model', y='r2_score', ax=ax)
ax.set_title('Régression – Benchmark Car Prices')
plt.xticks(rotation=30)
plt.tight_layout()
mlflow.log_figure(fig, 'regression_benchmark_car.png')
plt.close(fig)
