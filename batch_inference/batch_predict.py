import os
import pickle

import click
import numpy as np
import pandas as pd
from boto3 import client
from sklearn.pipeline import Pipeline


def load_object(path: str) -> Pipeline:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_data(s3_bucket: str, remote_path: str, local_path: str):
    s3 = client("s3")
    s3.download_file(s3_bucket, remote_path, local_path)


def upload_predictions(s3_bucket: str, local_path: str, s3_path: str):
    s3 = client("s3")
    s3.upload_file(local_path, s3_bucket, s3_path)


def batch_predict(
    s3_bucket: str,
    remote_model_path: str,
    path_to_data: str,
    output: str,
    local_output: str = "predicts.csv",
    local_model_path: str = "model.pkl",
    local_data_path: str = "data.csv",
):
    load_data(s3_bucket, path_to_data, local_model_path)
    load_data(s3_bucket, remote_model_path, local_data_path)

    model = load_object(local_model_path)
    data = pd.read_csv(local_data_path)
    ids = data["Id"]

    predicts = np.exp(model.predict(data))
    predict_df = pd.DataFrame(list(zip(ids, predicts)), columns=["Id", "Predict"])
    predict_df.to_csv(local_output, index=False)
    upload_predictions(s3_bucket, local_output, output)


@click.command(name="batch_predict")
@click.argument("PATH_TO_DATA", default=os.getenv("PATH_TO_DATA"))
@click.argument("PATH_TO_MODEL", default=os.getenv("PATH_TO_MODEL"))
@click.argument("OUTPUT", default=os.getenv("OUTPUT"))
@click.argument("S3_BUCKET", default="for-dvc")
def batch_predict_command(
    path_to_data: str, path_to_model: str, output: str, s3_bucket: str
):
    batch_predict(s3_bucket, path_to_data, path_to_model, output)


if __name__ == "__main__":
    batch_predict_command()
