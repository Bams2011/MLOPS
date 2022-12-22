from array import array
import json
import tempfile
from unittest import result
import warnings
from argparse import Namespace
from pathlib import Path
from typing import Dict

import joblib
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
#import optuna
import pandas as pd
import typer
from numpyencoder import NumpyEncoder

from config import config
from config.config import logger
from fraud_detection import predict, train, utils

warnings.filterwarnings("ignore")

# Initialize Typer CLI app
app = typer.Typer()


def load_artifacts(run_id: str = None) -> Dict:
    """Load artifacts for a given run_id.

    Args:
        run_id (str): id of run to load artifacts from.

    Returns:
        Dict: run's artifacts.
    """
    if not run_id:
        run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()

    # Locate specifics artifacts directory
    experiment_id = mlflow.get_run(run_id=run_id).info.experiment_id
    artifacts_dir = Path(config.MODEL_REGISTRY, experiment_id, run_id, "artifacts")

    # Load objects from run
    args = Namespace(**utils.load_dict(filepath=Path(artifacts_dir, "args.json")))
    model = joblib.load(Path(artifacts_dir, "model.pkl"))
    performance = utils.load_dict(filepath=Path(artifacts_dir, "performance.json"))

    return {
        "args": args,
        "model": model,
        "performance": performance,
    }

def evaluate_model(experiment_name: str = "baselines") -> str:
    """Evaluate model compared to past experiments

        Args:
            experiment_name (str): name of the experimentations.

        Returns:
            Dict: run's artifacts.
    """

    logger.info('Running Model Evaluation...')
    artifacts = load_artifacts()
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    try:
        
        best_run_id = mlflow.search_runs(experiment_ids=experiment_id, order_by=["overall.f1 DESC"]).query("run_id!=@run_id").iloc[0].run_id
        best_run_metrics = mlflow.get_run(run_id=best_run_id).data.metrics
    except:
        best_run_metrics = {}
        best_run_metrics['f1'] = 0.80
    print(f"Best run metrics from previous runs from experiments : experiment name:{experiment_name} --- metrics: {best_run_metrics} " )
    if artifacts.metrics["overall"]["f1"]> best_run_metrics['f1']:
            return 'improved'

    return 'regressed'

def register_model(
    registered_model_name: str = "baselines",
    registered_model_stage: str = "fraud_detection"
) -> None:
    """Register  a model to a given stage

    Args:
        registered_model_name (str): name of the model to register.
        registered_model_stage (str): name of the stage to register the model.

    Returns:
        None
    """

    logger.info("Registring Model on Mlflow...")
    artifacts = load_artifacts()
    #signature = infer_signature(xTest, model.predict(xTest))
    mlflow.sklearn.log_model(artifacts["model"], "models",registered_model_name=registered_model_name)#, signature=signature)
    client = MlflowClient()
    #Transition model to Staging if not done yet
    modelversion = client.get_registered_model(registered_model_name).latest_versions[0].version
    print("Model latest Version: {}".format(modelversion))
    client.transition_model_version_stage(
        name="fraud-detection",
        version=int(modelversion),
        stage=registered_model_stage
    )
    logger.info("Registring Model done.")

@app.command()
def etl_data():
    """Extract, load and transform our data assets."""
    # Extract
    #fraud_data = pd.read_csv(config.TRAIN_DATA)
    logger.info(f"Train data path: {config.TRAIN_DATA}")
    logger.info("✅ Saved data!")


@app.command()
def train_model(
    args_fp: str = "config/args.json",
    experiment_name: str = "baselines",
    run_name: str = "rf",
    test_run: bool = False,
) -> None:
    """Train a model with given arguments.

    Args:
        args_fp (str): location of args.
        experiment_name (str): name of experiment.
        run_name (str): name of specific run in experiment.
        test_run (bool, optional): If True, artifacts will not be saved. Defaults to False.
    """
    # Load data
    df = pd.read_csv(Path(config.TRAIN_DATA))

    # Train
    args = Namespace(**utils.load_dict(filepath=args_fp))
    mlflow.set_experiment(experiment_name=experiment_name)
    with mlflow.start_run(run_name=run_name):
        run_id = mlflow.active_run().info.run_id
        logger.info(f"Run ID: {run_id}")
        artifacts = train.train(df=df, args=args)
        performance = artifacts["performance"]
        logger.info(json.dumps(performance, indent=2))

        # Log metrics and parameters
        mlflow.log_metrics({"precision": performance["overall"]["precision"]})
        mlflow.log_metrics({"recall": performance["overall"]["recall"]})
        mlflow.log_metrics({"f1": performance["overall"]["f1"]})
        mlflow.log_params(vars(artifacts["args"]))

        # Log artifacts
        with tempfile.TemporaryDirectory() as dp:
            utils.save_dict(vars(artifacts["args"]), Path(dp, "args.json"), cls=NumpyEncoder)
            joblib.dump(artifacts["model"], Path(dp, "model.pkl"))
            utils.save_dict(performance, Path(dp, "performance.json"))
            mlflow.log_artifacts(dp)

    # Save to config
    if not test_run:  # pragma: no cover, actual run
        open(Path(config.CONFIG_DIR, "run_id.txt"), "w").write(run_id)
        utils.save_dict(performance, Path(config.CONFIG_DIR, "performance.json"))


@app.command()
def predict_fraud(data: array, run_id: str = None) -> None:
    """Predict fraud or non fraud  for transaction.

    Args:
        data (array): transaction data
        run_id (str, optional): run id to load artifacts for prediction. Defaults to None.
    """
    if not run_id:
        run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
    artifacts = load_artifacts(run_id=run_id)
    prediction = predict.predict(data=data, artifacts=artifacts)
    logger.info(json.dumps(prediction, indent=2))
    return prediction


if __name__ == "__main__":
    app()  # pragma: no cover, live app
