from services.flow_manager import FlowManager

def main():
    FlowManager.create_experiment(
        experiment_name="My Experiment",
        artifact_location="mlflow-artifacts",
        tags={"owner": "data-team", "project": "mlflow-integration"}
    )

if __name__ == "__main__":
    main()
