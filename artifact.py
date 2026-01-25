import mlflow
from services.flow_manager import FlowManager
if __name__=="__main__":

    experiment = FlowManager.get_experiment(experiment_name="My Experiment")

    print("Name: {}".format(experiment["name"]))

    with mlflow.start_run(run_name="logging_artifacts", experiment_id=experiment["id"]) as run:

        # your machine learning code goes here

        mlflow.log_artifacts(local_dir="./data/",artifact_path="dataset")

        # print run info
        print("run_id: {}".format(run.info.run_id))
        print("experiment_id: {}".format(run.info.experiment_id))
        print("status: {}".format(run.info.status))
        print("start_time: {}".format(run.info.start_time))
        print("end_time: {}".format(run.info.end_time))
        print("lifecycle_stage: {}".format(run.info.lifecycle_stage))   