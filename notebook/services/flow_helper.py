import mlflow
import logging
import sys
from typing import Dict, Any

# ==================== Logger ====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)
logging.getLogger("mlflow").setLevel(logging.WARNING)
logging.getLogger("alembic").setLevel(logging.WARNING)

# ==================== Classe FlowHelper ====================
class FlowHelper:
    """
    Classe pour gérer un experiment MLflow :
    - Création idempotente de l'experiment
    - Démarrage et fin de run
    - Logging de params/metrics depuis dictionnaires
    - Logging de modèles pyfunc avec pip>=23.0 et python détecté automatiquement
    """

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        self.run = None
        logger.info(f"Experiment '{experiment_name}' sélectionné ou créé.")

    def start_run(self, run_name: str):
        """Démarre un run MLflow"""
        self.run = mlflow.start_run(run_name=run_name)
        logger.info(f"Run '{run_name}' démarré (ID={self.run.info.run_id}).")
        return self.run

    def log_params(self, params: Dict[str, Any], synchronous: bool = True):
        """Log des paramètres depuis un dictionnaire"""
        mlflow.log_params(params, synchronous=synchronous)
        logger.info(f"Params logged: {params}")

    def log_metrics(self, metrics: Dict[str, float], synchronous: bool = True):
        """Log des métriques depuis un dictionnaire"""
        mlflow.log_metrics(metrics, synchronous=synchronous)
        logger.info(f"Metrics logged: {metrics}")

    def log_model(self, model, name: str = "model"):
        """
        Log un modèle Python avec mlflow.pyfunc.log_model
        conda_env fixe pip>=23.0 et python détecté automatiquement
        """
        # Détecter Python
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

        conda_env = {
            'name': 'mlflow-env',
            'channels': ['defaults'],
            'dependencies': [
                f'python={python_version}',
                'pip>=23.0',
            ]
        }

        mlflow.pyfunc.log_model(
            name=name,
            python_model=model,
            conda_env=conda_env
        )
        logger.info(f"Model logged at name='{name}' with Python {python_version} and pip>=23.0")

    def end_run(self):
        """Termine le run MLflow"""
        mlflow.end_run()
        logger.info("Run terminé.")


# ==================== Exemple d'utilisation ====================
if __name__ == "__main__":
    import mlflow.pyfunc

    class MyModel(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input):
            return model_input * 2

    flow = FlowHelper(experiment_name="Demo Experiment")
    run = flow.start_run(run_name="Demo Run")

    flow.log_params({"learning_rate": 0.01, "batch_size": 32})
    flow.log_metrics({"accuracy": 0.95, "loss": 0.12})

    model = MyModel()
    flow.log_model(model, name="my_model")

    flow.end_run()
