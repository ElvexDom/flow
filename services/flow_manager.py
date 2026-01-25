import logging
from typing import Optional, Dict, Any
from datetime import datetime
import mlflow
from mlflow.exceptions import MlflowException

# Configuration de logging par défaut
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FlowManager:
    """
    Gestion centralisée des experiments MLflow :
    - Création idempotente avec tags et logs réduits
    - Récupération par experiment_name ou experiment_id avec tags et timestamp
    - Suppression d'experiments
    """

    @staticmethod
    def create_experiment(
        experiment_name: str,
        artifact_location: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        silence_logs: bool = True
    ) -> str:
        """
        Crée un experiment MLflow avec tags s'il n'existe pas déjà.
        Retourne l'experiment_id.
        """
        if silence_logs:
            logging.getLogger("mlflow").setLevel(logging.WARNING)
            logging.getLogger("alembic").setLevel(logging.WARNING)

        client = mlflow.tracking.MlflowClient()

        # Vérifier si l'experiment existe déjà
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is not None:
            experiment_id = experiment.experiment_id
            logger.info(f"Experiment '{experiment_name}' existe déjà avec l'ID: {experiment_id}")

            # Ajout / mise à jour des tags seulement si nécessaire
            if tags:
                existing_tags = client.get_experiment(experiment_id).tags
                for key, value in tags.items():
                    if key not in existing_tags or existing_tags[key] != value:
                        client.set_experiment_tag(experiment_id, key, value)

            # Définir l'experiment actif
            mlflow.set_experiment(experiment_name)
            return experiment_id

        # Création de l'experiment
        try:
            experiment_id = mlflow.create_experiment(
                name=experiment_name,
                artifact_location=artifact_location,
                tags=tags
            )
            mlflow.set_experiment(experiment_name)
            logger.info(f"Experiment '{experiment_name}' créé avec l'ID: {experiment_id}")
            return experiment_id
        except MlflowException as e:
            logger.error(f"Erreur lors de la création de l'experiment '{experiment_name}': {e}")
            raise RuntimeError(
                f"Erreur lors de la création de l'experiment MLflow '{experiment_name}'"
            ) from e

    @staticmethod
    def get_experiment(
        experiment_name: Optional[str] = None,
        experiment_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Récupère un experiment MLflow par son nom ou son ID.
        Retourne un dictionnaire complet ou None si introuvable.
        """
        client = mlflow.tracking.MlflowClient()

        try:
            if experiment_id:
                exp_obj = client.get_experiment(experiment_id)
            elif experiment_name:
                exp_obj_raw = mlflow.get_experiment_by_name(experiment_name)
                if exp_obj_raw is None:
                    return None
                experiment_id = exp_obj_raw.experiment_id
                exp_obj = client.get_experiment(experiment_id)
            else:
                return None
        except MlflowException:
            return None

        creation_time = datetime.fromtimestamp(exp_obj.creation_time / 1000)

        return {
            "id": exp_obj.experiment_id,
            "name": exp_obj.name,
            "artifact_location": exp_obj.artifact_location,
            "lifecycle_stage": exp_obj.lifecycle_stage,
            "creation_time": creation_time,
            "tags": exp_obj.tags
        }

    @staticmethod
    def delete_experiment(
        experiment_id: Optional[str] = None,
        experiment_name: Optional[str] = None
    ) -> bool:
        """
        Supprime un experiment MLflow (met en lifecycle_stage 'deleted').
        Retourne True si succès, False sinon.
        La suppression est idempotente : True si l'experiment n'existait déjà pas.
        """
        client = mlflow.tracking.MlflowClient()

        if experiment_id is None and experiment_name:
            exp_obj = mlflow.get_experiment_by_name(experiment_name)
            if exp_obj is None:
                logger.warning(f"Experiment '{experiment_name}' introuvable pour suppression.")
                return True  # Déjà supprimé ou inexistant
            experiment_id = exp_obj.experiment_id
        elif experiment_id is None:
            logger.warning("Aucun identifiant fourni pour supprimer un experiment.")
            return False

        try:
            client.delete_experiment(experiment_id)
            logger.info(f"Experiment supprimé (ID={experiment_id})")
            return True
        except MlflowException as e:
            logger.error(f"Échec de la suppression de l'experiment (ID={experiment_id}): {e}")
            return False
