import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns

from services.flow_tools import FlowTools
from services.metrics_tools import MetricsTools

# ------------------------------
# Charger dataset
# ------------------------------
df = pd.read_csv("data/weather.csv")
df.replace('NA', np.nan, inplace=True)
df = df.dropna(subset=['RainTomorrow'])
df['RainToday'] = df['RainToday'].map({'Yes': 1, 'No': 0})
df['RainTomorrow'] = df['RainTomorrow'].map({'Yes': 1, 'No': 0})

num_cols = ['MinTemp','MaxTemp','Rainfall','Evaporation','Sunshine','WindGustSpeed',
            'WindSpeed9am','WindSpeed3pm','Humidity9am','Humidity3pm','Pressure9am',
            'Pressure3pm','Cloud9am','Cloud3pm','Temp9am','Temp3pm']
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')
df['Date'] = pd.to_datetime(df['Date'])

target = 'RainTomorrow'
features = [c for c in df.columns if c != target and c != 'Date']
X, y = df[features], df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------
# Préprocessing
# ------------------------------
numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_features = X_train.select_dtypes(exclude=np.number).columns.tolist()

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
]) if categorical_features else 'passthrough'

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# ------------------------------
# Modèle
# ------------------------------
classifier = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(
        max_depth=10,
        n_estimators=100,
        n_jobs=-1,
        random_state=42
    ))
])

# ------------------------------
# MLflow
# ------------------------------
tools = FlowTools(experiment_name="Weather_RainTomorrow")
mlflow.autolog(log_models=True, silent=True)

with mlflow.start_run(run_name="RainTomorrow_Classification", experiment_id=tools.experiment_id) as run:

    print("⏳ Training model...")
    classifier.fit(X_train, y_train)
    print("✅ Model trained")

    # Prédiction sur test set
    y_pred = classifier.predict(X_test)
    y_pred_proba = classifier.predict_proba(X_test)
    print("✅ Prediction done")

    # MetricsTools → test set
    metrics_tool = MetricsTools(
        params={
            "task": "classification",
            "y_true": y_test,
            "y_pred": y_pred,
            "y_pred_proba": y_pred_proba,
            "model": repr(classifier.named_steps['model']),
            "X_shape": X_test.shape
        }
    )
    metrics_tool.compute_metrics()
    metrics_tool.summary()
    metrics_tool.log_mlflow(prefix="test")
    print("✅ Metrics logged")

    # Visualisation RainTomorrow
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(x=y_test, ax=ax)
    ax.set_title("Distribution RainTomorrow (Test Set)")
    plt.tight_layout()
    mlflow.log_figure(fig, "rain_distribution.png")
    plt.close(fig)
    print("✅ Figure logged")

    print(f"Run ID: {run.info.run_id}")
    print(f"Experiment ID: {run.info.experiment_id}")
