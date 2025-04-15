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

import leitura_dados


def preparacao_dados():
    df_prod = leitura_dados.leitura_arquivo('../data/raw/dataset_kobe_prod.parquet')
    columns = ['lat', 'lon', 'minutes_remaining', 'period', 'playoffs', 'shot_distance', 'shot_made_flag']

    # Para usar o sqlite como repositorio
    mlflow.set_tracking_uri('sqlite:///../data/mlruns.db')
    experiment_name = 'Projeto Kobe'
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        experiment = mlflow.get_experiment(experiment_id)
    experiment_id = experiment.experiment_id


    # pipeline run 'PreparacaoDados"
    with mlflow.start_run(experiment_id=experiment_id, run_name='PreparacaoDados'):

        df_dev = pd.read_parquet('../data/raw/dataset_kobe_dev.parquet')

        df_dev.dropna(subset=['shot_made_flag'], inplace=True) # Somente a coluna "shot_made_flag" tinha dados faltantes
        df_dev = df_dev[columns] # shape 20285, 5
        df_dev.to_parquet('../data/processed/data_filtered.parquet')

        df_dev_columns = df_dev[['lat', 'lon', 'minutes_remaining', 'period', 'playoffs', 'shot_distance']]
        df_dev_target = df_dev['shot_made_flag']
        train_perc = 0.8

        xtrain, xtest, ytrain, ytest = train_test_split(df_dev_columns,
                                                        df_dev_target,
                                                        train_size=train_perc,
                                                        stratify=df_dev_target)

        xtrain['shot_made_flag'] = ytrain
        xtest['shot_made_flag'] = ytest

        xtrain.to_parquet('../data/processed/base_train.parquet')
        xtest.to_parquet('../data/processed/base_test.parquet')

        mlflow.log_params({'perc_test': 1 - train_perc, 'select_columns': columns})
        mlflow.log_metrics({'qtde_linhas_train': xtrain.shape[0], 'qtde_linhas_test': xtest.shape[0]})

        return xtrain, xtest, ytrain, ytest
