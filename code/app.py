import pandas as pd
import streamlit as st
import joblib
import pandas
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics


prod_file = '../data/processed/prediction_prod.parquet'
dev_file = '../data/processed/prediction_test.parquet'

############################################ SIDE BAR TITLE
st.sidebar.title('Painel de Controle')
st.sidebar.markdown(f"""
Controle do tipo de vinho analisado e entrada de variáveis para avaliação de novos vinhos.
""")

df_prod = pd.read_parquet(prod_file)
df_dev = pd.read_parquet(dev_file)

fignum = plt.figure(figsize=(6, 4))

# Saida do modelo dados dev
sns.distplot(df_dev.prediction_score_1,
             label='Teste',
             ax=plt.gca())

# Saida do modelo dados prod
sns.distplot(df_prod.predict_score,
             label='Produção',
             ax=plt.gca())

# User wine

plt.title('Modelo de Predição Kobe Bryant')
plt.ylabel('')
plt.xlabel('Probabilidade de Acertos')
plt.xlim((0, 1))
plt.grid(True)
plt.legend(loc='best')
st.pyplot(fignum)


report_dict = metrics.classification_report(df_dev['shot_made_flag'], df_dev['prediction_label'], output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_df = report_df.round(2)
st.subheader("🔍 Relatório de Classificação")
st.dataframe(report_df.style.format(precision=2))
st.write()




