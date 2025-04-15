import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.metrics import log_loss, f1_score
from pathlib import Path
import pycaret.classification as pc
import pycaret.regression as pr

import mlflow
from mlflow.models.signature import infer_signature
from mlflow.models.signature import ModelSignature
from mlflow.tracking import MlflowClient

import matplotlib.pyplot as plt


import streamlit
import os

# Para usar o sqlite como repositorio
mlflow.set_tracking_uri('sqlite:///../data/mlruns.db')

experiment_name = 'Projeto Kobe'
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    experiment_id = mlflow.create_experiment(experiment_name)
    experiment = mlflow.get_experiment(experiment_id)
experiment_id = experiment.experiment_id

columns = ['lat', 'lon', 'minutes_remaining', 'period', 'playoffs', 'shot_distance'] #, 'shot_made_flag'

with mlflow.start_run(experiment_id=experiment_id, run_name='PipelineAplicacao'):

    model_uri = f"models:/Kobe Predict@staging"
    loaded_model = mlflow.sklearn.load_model(model_uri)
    data_prod = pd.read_parquet('../data/raw/dataset_kobe_prod.parquet')
    data_prod.dropna(subset=['shot_made_flag'], inplace=True)

    Y = loaded_model.predict_proba(data_prod[columns])[:, 1]
    data_prod['predict_score'] = (Y >= 0.5).astype(int)

    data_prod.to_parquet('../data/processed/prediction_prod.parquet')
    mlflow.log_artifact('../data/processed/prediction_prod.parquet')

    mlflow.log_metrics({
        'log_loss_prod': log_loss(data_prod['shot_made_flag'], data_prod['predict_score']),
        'f1_score_prod': f1_score(data_prod['shot_made_flag'], data_prod['predict_score'])})

