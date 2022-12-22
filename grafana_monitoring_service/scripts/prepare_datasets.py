#!/usr/bin/env python3

import argparse
import io
import json
from mlflow.tracking import MlflowClient
import pandas as pd
import yaml
import mlflow



# suppress SettingWithCopyWarning: warning
pd.options.mode.chained_assignment = None


DATA_SOURCE_URL = "/home/airflow/data/raw/"
DATASET_NAME = "CCFD dataset"
DATA_SOURCES = (DATASET_NAME, DATASET_NAME)
REGISTERED_MODEL_NAME = "fraud-detection"
MODEL_STAGE = "Production"
mlflow.set_tracking_uri("http://localhost:5001")
base_configuration = {
    "data_format": {
        "separator": ",",
        "header": True,
    },
    "column_mapping": {},
    "service": {
        "reference_path": "./reference.csv",
        "min_reference_size": 100,
        "use_reference": True,
        "moving_reference": False,
        "window_size": 100,
        "calculation_period_sec": 300,
    },
}


def get_data() -> (pd.DataFrame, pd.DataFrame):
    print(f"Load data for dataset: {DATASET_NAME}")

    
    reference_data = pd.read_csv(DATA_SOURCE_URL+"train_data.csv")
    production_data = pd.read_csv(DATA_SOURCE_URL+"prod_data.csv")

    target = "Class"
    numerical_features = reference_data.drop(target,axis=1).columns.to_list()
    #["Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V20","V21","V22","V23","V24","V25","V26", "V27", "V28", "Amount"]
    #categorical_features = 

    features = numerical_features

    print(f"num features: {json.dumps(numerical_features)}")
    #print(f"cat features: {json.dumps(categorical_features)}")
    print(f"target: {target}")
    print(f"number total of features: {len(features)}")
    #Normalize target values
    reference_data.reset_index(inplace=True, drop=True)
    #reference_data[target] = reference_data[target].apply(lambda x: x.decode("utf8"))
    production_data.reset_index(inplace=True, drop=True)
    #production_data[target] = production_data[target].apply(lambda x: x.decode("utf8"))
    model = mlflow.pyfunc.load_model(
        model_uri=f"models:/{REGISTERED_MODEL_NAME}/{MODEL_STAGE}"
    )
    reference_data["prediction"] = model.predict(reference_data[features])
    production_data["prediction"] = model.predict(production_data[features])

    # setup service configuration
    configuration = base_configuration
    configuration["column_mapping"]["target"] = target
    configuration["column_mapping"]["numerical_features"] = numerical_features
    #configuration["column_mapping"]["categorical_features"] = categorical_features
    configuration["service"]["monitors"] = ["data_drift", "classification_performance"]

    return reference_data[features + [target, "prediction"]], production_data, configuration




def main() -> None:
    #print(f'Generate test data for dataset "{dataset}"')

    ref_data, prod_data, configuration = get_data()
    ref_data.to_csv("reference.csv", index=False)
    prod_data.to_csv("production.csv", index=False)

    print("Generate config file...")
    with open("config.yaml", "w", encoding="utf8") as conf_file:
        yaml.dump(configuration, conf_file)
    print("Done.")

    print(f"Reference dataset was created with {ref_data.shape[0]} rows")
    print(f"Production dataset was created with {prod_data.shape[0]} rows")


if __name__ == "__main__":
    main()
