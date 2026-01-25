import mlflow
import logging
from datetime import datetime

# ==================== Configuration du logger principal ====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ==================== Réduire le bruit des logs MLflow/Alembic ====================
logging.getLogger("mlflow").setLevel(logging.WARNING)
logging.getLogger("alembic").setLevel(logging.WARNING)

# ==================== Fonction principale ====================
def main():
    # Infos globales MLflow
    logger.info("=== MLflow global info ===")
    logger.info(f"Tracking URI: {mlflow.get_tracking_uri()}")

    # Définir l'experiment
    experiment = mlflow.set_experiment("My Experiment")
    logger.info("=== Experiment info ===")
    logger.info(f"Experiment name: {experiment.name}")
    logger.info(f"Experiment ID: {experiment.experiment_id}")
    logger.info(f"Artifact location: {experiment.artifact_location}")
    logger.info(f"Lifecycle stage: {experiment.lifecycle_stage}")

    # Dictionnaires
    params = {
        "batch_size": 32,
        "epochs": 10,
        "learning_rate": 0.01,
        "loss_function": "mse",
        "optimizer": "adam"
        }

    metrics = {
        "mse": 0.12,
        "mae": 0.08,
        "rmse": 0.15,
        "r2": 0.85
        }

    # Démarrer un run
    with mlflow.start_run(run_name="mlflow_runs") as run:
        # Log params et metrics
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

    # Infos du run
    logger.info("=== Run info ===")
    logger.info(f"Run ID: {run.info.run_id}")
    logger.info(f"Experiment ID: {run.info.experiment_id}")
    logger.info(f"Status: {run.info.status}")
    logger.info(f"User ID: {run.info.user_id}")
    logger.info(f"Lifecycle stage: {run.info.lifecycle_stage}")

    # Timing
    start_time = datetime.fromtimestamp(run.info.start_time / 1000) if run.info.start_time else None
    end_time = datetime.fromtimestamp(run.info.end_time / 1000) if run.info.end_time else None
    logger.info("=== Timing ===")
    logger.info(f"Start time: {start_time}")
    logger.info(f"End time: {end_time}")

    # Données loggées
    logger.info("=== Logged data ===")
    logger.info(f"Params: {run.data.params}")
    logger.info(f"Metrics: {run.data.metrics}")
    logger.info(f"Tags: {run.data.tags}")

    # Artifacts
    logger.info("=== Artifacts ===")
    logger.info(f"Artifact URI: {run.info.artifact_uri}")

    logger.info("✅ MLflow run terminé avec succès")


if __name__ == "__main__":
    main()
