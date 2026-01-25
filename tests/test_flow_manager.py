import pytest
from types import SimpleNamespace
from services.flow_manager import FlowManager

EXPERIMENT_NAME = "LocalTestFlowManager"

# ==================== Fonctions factices ====================

def fake_get_experiment_by_name(name):
    return None  # Simule qu'aucun experiment n'existe

def fake_create_experiment(name, artifact_location=None, tags=None):
    return "mocked_experiment_id"

def fake_set_experiment(name):
    return None

def fake_client_get_experiment(exp_id):
    # Retourne un objet simulé pour get_experiment
    return SimpleNamespace(
        experiment_id=exp_id,
        name=EXPERIMENT_NAME,
        artifact_location="mock_artifacts",  # Nom arbitraire
        lifecycle_stage="active",
        creation_time=1234567890000,
        tags={"project": "pytest"}
    )

def fake_delete_experiment(exp_id):
    return True

# ==================== Tests ====================

def test_create_experiment(monkeypatch):
    monkeypatch.setattr("mlflow.get_experiment_by_name", fake_get_experiment_by_name)
    monkeypatch.setattr("mlflow.create_experiment", fake_create_experiment)
    monkeypatch.setattr("mlflow.set_experiment", fake_set_experiment)

    experiment_id = FlowManager.create_experiment(
        experiment_name=EXPERIMENT_NAME,
        artifact_location="mock_artifacts",  # Nom arbitraire
        tags={"project": "pytest"}
    )

    assert experiment_id == "mocked_experiment_id"

def test_get_experiment(monkeypatch):
    class MockClient:
        def get_experiment(self, exp_id):
            return fake_client_get_experiment(exp_id)

    monkeypatch.setattr("mlflow.tracking.MlflowClient", lambda: MockClient())
    monkeypatch.setattr("mlflow.get_experiment_by_name", lambda name: SimpleNamespace(experiment_id="mocked_experiment_id"))

    info = FlowManager.get_experiment(experiment_name=EXPERIMENT_NAME)
    assert info is not None
    assert info["experiment_id"] == "mocked_experiment_id"
    assert info["name"] == EXPERIMENT_NAME
    assert info["tags"]["project"] == "pytest"
    assert info["artifact_location"] == "mock_artifacts"

def test_delete_experiment(monkeypatch):
    class MockClient:
        def delete_experiment(self, exp_id):
            return fake_delete_experiment(exp_id)

    monkeypatch.setattr("mlflow.tracking.MlflowClient", lambda: MockClient())
    monkeypatch.setattr("mlflow.get_experiment_by_name", lambda name: SimpleNamespace(experiment_id="mocked_experiment_id"))

    result = FlowManager.delete_experiment(experiment_name=EXPERIMENT_NAME)
    assert result is True
