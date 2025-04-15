import pandas as pd


def leitura_arquivo(path):
    df_prod = pd.read_parquet(path)

    return df_prod
