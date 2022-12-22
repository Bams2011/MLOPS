import json
from argparse import Namespace
from typing import Dict
from sklearn.ensemble import RandomForestClassifier
from config import config
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config.config import logger
from fraud_detection import evaluate, predict, utils


def train(args: Namespace, df: pd.DataFrame) -> Dict:
    """Train model on data.

    Args:
        args (Namespace): arguments to use for training.
        df (pd.DataFrame): data for training.
    Returns:
        Dict: artifacts from the run.
    """

    # Setup
    xTrain, X_val, yTrain, y_val = train_test_split(df, df[config.CLASS_NAME], test_size = 0.2, random_state = 42)

    # Model
    model =  RandomForestClassifier(n_jobs=-1)

    # Training
    model.fit(xTrain, yTrain)


    # Evaluation
    y_pred = model.predict(X_val)
    performance = evaluate.get_metrics(
        y_true=y_val, y_pred=y_pred, classes=[0,1]
    )

    return {
        "args": args,
        "model": model,
        "performance": performance,
    }


