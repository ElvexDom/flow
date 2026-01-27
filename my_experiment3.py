import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import mlflow
import seaborn as sns

from services.flow_tools import FlowTools
from services.metrics_tools import MetricsTools

# ------------------------------
# Charger dataset
# ------------------------------
df = pd.read_csv("data/Mall_Customers.csv")
if "CustomerID" in df.columns:
    df.drop(columns=["CustomerID"], inplace=True)

# ------------------------------
# Préparer features pour clustering
# ------------------------------
X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------
# Méthode du coude
# ------------------------------
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Log figure du coude directement dans MLflow
fig, ax = plt.subplots()
ax.plot(K, inertia, 'bx-')
ax.set_xlabel('Nombre de clusters k')
ax.set_ylabel('Inertia')
ax.set_title('Méthode du coude')
plt.tight_layout()

tools = FlowTools(experiment_name="Customer Segmentation")
mlflow.autolog(log_models=True, silent=True)

with mlflow.start_run(run_name="KMeans_Segmentation", experiment_id=tools.experiment_id) as run:

    tools.log_dataset(df)
    mlflow.log_figure(fig, "elbow_method.png")
    plt.close(fig)  # plus besoin d'afficher

    # ------------------------------
    # KMeans
    # ------------------------------
    k_optimal = 5
    kmeans = KMeans(n_clusters=k_optimal, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    df["Cluster"] = clusters

    # ------------------------------
    # Visualisation clusters → MLflow
    # ------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    scatter = ax.scatter(
        X_scaled[:, 0],
        X_scaled[:, 1],
        c=clusters,
        cmap='Set1',
        alpha=0.6
    )
    ax.set_xlabel("Annual Income (scaled)")
    ax.set_ylabel("Spending Score (scaled)")
    ax.set_title(f"Clients segmentés ({k_optimal} clusters)")
    plt.colorbar(scatter, ax=ax, label="Cluster")
    plt.tight_layout()
    mlflow.log_figure(fig, "clusters.png")
    plt.close(fig)  # ferme sans afficher

    # ------------------------------
    # MetricsTools → clustering
    # ------------------------------
    metrics_tool = MetricsTools(
        params={
            "task": "clustering",
            "y_pred": clusters,
            "X": X_scaled,
            "model": "KMeans",
            "k": k_optimal,
            "inertia": kmeans.inertia_,
            "X_shape": X_scaled.shape
        }
    )
    metrics_tool.compute_metrics()
    metrics_tool.summary()
    metrics_tool.log_mlflow()

    print(f"Run ID: {run.info.run_id}")
    print(f"Experiment ID: {run.info.experiment_id}")
