from array import array
from distutils.command.config import config
from turtle import pd
from typing import Dict, List

from config import config

def predict(data: pd.DataFrame, artifacts: Dict) -> List:
    """Predict class for transactions

    Args:
        data (DataFrame): transaction data
        artifacts (Dict): artifacts from a run.

    Returns:
        int: class for transactions
    """
    class_name = config.CLASS_NAME
    #x = data.drop(class_name,axis=1)
    if class_name in data.columns:
        data.drop(class_name,axis=1,inplace=True)
    y_pred = artifacts["model"].predict(data),
    return y_pred
