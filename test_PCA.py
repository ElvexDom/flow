# ===================================================
# 1️⃣ Imports
# ===================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from services.pipeline_tools import PipelineTools
from services.metrics_tools import MetricsTools
import mlflow

# ===================================================
# 2️⃣ Chargement et préparation du dataset
# ===================================================
df = pd.read_csv("data/weather.csv")
df.replace("NA", np.nan, inplace=True)
df = df.dropna(subset=['RainTomorrow'])

# Target binaire
df['RainToday'] = df['RainToday'].map({'Yes': 1, 'No': 0})
df['RainTomorrow'] = df['RainTomorrow'].map({'Yes': 1, 'No': 0})

# Colonnes numériques
numeric_features = [
    'MinTemp','MaxTemp','Rainfall','Evaporation','Sunshine','WindGustSpeed',
    'WindSpeed9am','WindSpeed3pm','Humidity9am','Humidity3pm',
    'Pressure9am','Pressure3pm','Cloud9am','Cloud3pm','Temp9am','Temp3pm'
]

df[numeric_features] = df[numeric_features].apply(pd.to_numeric, errors='coerce')

# Colonnes catégorielles
categorical_features = ['Location','WindGustDir','WindDir9am','WindDir3pm']

# Features et target
target = 'RainTomorrow'
features = [c for c in df.columns if c != target and c != 'Date']

X = df[features]
y = df[target]

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===================================================
# 3️⃣ Fonction de comparaison PCA / no PCA
# ===================================================
def compare_pca(
    X_train, X_test, y_train, y_test,
    numeric_features, categorical_features,
    model_class=RandomForestClassifier,
    model_params=None,
    pca_variance=0.99
):
    model_params = model_params or {}
    results = {}
    
    for use_pca in [False, True]:
        suffix = "pca" if use_pca else "no_pca"
        print(f"\n=== Run: {suffix} ===")
        
        # Créer pipeline
        pipe = PipelineTools(
            num_features=numeric_features,
            cat_features=categorical_features,
            model_class=model_class,
            model_params=model_params,
            use_pca=use_pca,
            pca_variance=pca_variance
        )
        
        # Entraînement
        pipe.fit(X_train, y_train)
        
        # Prédiction
        y_pred = pipe.predict(X_test)
        try:
            y_proba = pipe.predict_proba(X_test)
        except AttributeError:
            y_proba = None
        
        # Calcul métriques
        metrics_tool = MetricsTools(
            params={
                "task": "classification",
                "y_true": y_test,
                "y_pred": y_pred,
                "y_pred_proba": y_proba,
                "X_shape": X_test.shape
            }
        )
        metrics_tool.compute_metrics()
        metrics_tool.summary()
        
        # Stockage
        results[suffix] = metrics_tool.get_metrics()
        
        # MLflow logging
        for k, v in metrics_tool.get_metrics().items():
            mlflow.log_metric(f"{suffix}_{k}", float(v))
        
    return results

# ===================================================
# 4️⃣ Lancer comparaison PCA / no PCA
# ===================================================
mlflow.set_experiment("Weather_RainTomorrow_PCA_Comparison")

with mlflow.start_run(run_name="PCA_vs_noPCA"):
    results = compare_pca(
        X_train, X_test, y_train, y_test,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        model_class=RandomForestClassifier,
        model_params={
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42,
            "n_jobs": -1
        },
        pca_variance=1.0
    )

# ===================================================
# 5️⃣ Comparaison visuelle rapide
# ===================================================
import pandas as pd
df_metrics = pd.DataFrame(results).T
print("\n=== Tableau comparatif PCA vs no PCA ===")
print(df_metrics)
