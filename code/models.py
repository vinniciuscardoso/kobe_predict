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

import leitura_dados, data_prep, plot

def model_train():

    registered_model_name = 'Kobe Predict'
    n_example = 5
    model_version = -1

    experiment_name = 'Projeto Kobe'
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id

    xtrain, xtest, ytrain, ytest = data_prep.preparacao_dados()

    with mlflow.start_run(experiment_id=experiment_id, run_name='Treinamento'):

        exp = pc.setup(
            data=xtrain,
            target='shot_made_flag',
            test_data=xtest,
            normalize=True,
            log_experiment=False
        )
        list_models = exp.compare_models(['lr', 'dt'], n_select=2, sort='f1')

        # Regressão Logística
        exp.plot_model(list_models[1], plot='vc', save=True) #
        yhat_test = exp.predict_model(list_models[1])

        plot.plot_parameter_validation_curve(xtrain.drop('shot_made_flag', axis=1),
                                             ytrain,
                                             'C',
                                             {'C': [0.001, 0.01, 0.1, 1, 10]},
                                             list_models[1],
                                             'Regressão Logística',
                                             'f1',
                                             logx=True)

        mlflow.log_metrics({
            'lr_log_loss': log_loss(yhat_test['shot_made_flag'], yhat_test['prediction_label']),
            'lr_f1': f1_score(yhat_test['shot_made_flag'], yhat_test['prediction_label'])
        })

        plt.savefig('rl_validation_curve.png')
        mlflow.log_artifact('rl_validation_curve.png')


        ## Arvore de Descisão
        exp.plot_model(list_models[0], plot='vc',save=True) #
        yhat_test = exp.predict_model(list_models[0])

        plot.plot_parameter_validation_curve(xtrain.drop('shot_made_flag', axis=1),
                                            ytrain,
                                            'max_depth',
                                            {'max_depth':[2, 3, 4, 5, 6, 7, 8]},
                                            list_models[0],
                                            'Árvore de Descisão',
                                            'f1',
                                            logx=False)

        mlflow.log_metrics({
            'lr_log_loss': log_loss(yhat_test['shot_made_flag'], yhat_test['prediction_label']),
            'lr_f1': f1_score(yhat_test['shot_made_flag'], yhat_test['prediction_label'])
        })

        plt.savefig('dt_validation_curve.png')
        mlflow.log_artifact('dt_validation_curve.png')


        # Finalização do Modelo
        tune_model = exp.tune_model(list_models[0],
                                    optimize='f1',
                                    search_library='scikit-learn',
                                    search_algorithm='random',
                                    n_iter=4)

        yhat_test = exp.predict_model(tune_model, raw_score=True)
        mlflow.log_metrics({
            'final_model_log_loss': log_loss(yhat_test['shot_made_flag'], yhat_test['prediction_label']),
            'final_model_f1': f1_score(yhat_test['shot_made_flag'], yhat_test['prediction_label'])
        })
        yhat_test.to_parquet('../data/processed/prediction_test.parquet')
        mlflow.log_artifact('../data/processed/prediction_test.parquet')

        final_model = exp.finalize_model(tune_model)

        # Exportação para Log e registro do modelo
        exp.save_model(final_model, f'./{registered_model_name}')
        # Carrega novamente o pipeline + bestmodel
        model_pipe = exp.load_model(f'./{registered_model_name}')
        # Assinatura do Modelo Inferida pelo MLFlow
        model_features = list(xtrain.drop('shot_made_flag', axis=1).columns)
        inf_signature = infer_signature(xtrain[model_features],
                                        model_pipe.predict(xtrain.drop('shot_made_flag', axis=1)))
        # Exemplo de entrada para o MLmodel
        input_example = {x: xtrain[x].values[:n_example] for x in model_features}
        # Log do pipeline de modelagem do sklearn e registrar como uma nova versao
        mlflow.sklearn.log_model(
            sk_model=model_pipe,
            artifact_path="sklearn-model",
            registered_model_name=registered_model_name,
            signature = inf_signature,
            input_example = input_example
        )
        # Criacao do cliente do servico MLFlow e atualizacao versao modelo
        client = MlflowClient()
        if model_version == -1:
            model_version = client.get_latest_versions(registered_model_name)[-1].version
        # Registrar o modelo como staging
        client.set_registered_model_alias(
            name=registered_model_name,
            alias='staging',
            version=model_version
        )

    return True
