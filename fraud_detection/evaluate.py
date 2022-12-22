from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support




def get_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, classes: List, df: pd.DataFrame = None
) -> Dict:
    """Performance metrics using ground truths and predictions.

    Args:
        y_true (np.ndarray): true labels.
        y_pred (np.ndarray): predicted labels.
        classes (List): list of class labels.
        df (pd.DataFrame, optional): dataframe to generate slice metrics on. Defaults to None.

    Returns:
        Dict: performance metrics.
    """
    # Performance
    metrics = {"overall": {}, "class": {}}

    # Overall metrics
    overall_metrics = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    metrics["overall"]["precision"] = overall_metrics[0]
    metrics["overall"]["recall"] = overall_metrics[1]
    metrics["overall"]["f1"] = overall_metrics[2]
    metrics["overall"]["num_samples"] = np.float64(len(y_true))

    # Per-class metrics
    class_metrics = precision_recall_fscore_support(y_true, y_pred, average=None)
    for i, _class in enumerate(classes):
        metrics["class"][_class] = {
            "precision": class_metrics[0][i],
            "recall": class_metrics[1][i],
            "f1": class_metrics[2][i],
            "num_samples": np.float64(class_metrics[3][i]),
        }


    return metrics
