# Kobe Predict


### 3. Como as ferramentas Streamlit, MLFlow, PyCaret e Scikit-Learn auxiliam na construção dos pipelines descritos anteriormente? A resposta deve abranger os seguintes aspectos:

Essas ferramentas atuam de forma complementar na construção de pipelines de ML. O PyCaret é o responsavel por automatizar o treinamento e 
comparação de modelos com integração nativa ao MLflow, que este é o que registra tudo o que ocorre e realiza os versionamentos e atualizações do modelo . 
O Scikit-learn é uma biblioteca que possue funções para treinamento e ajustes dos modelos. 
Já o Streamlit viabiliza o deployment rápido por meio de interfaces interativas, que também podem ser usadas para monitoramento da saúde do modelo em produção.
Em resumo, essas ferramentas facilitam a reprodutibilidade e agilidade do modelo, atraves do rastreamento, atualização e a disponibilização.

### 7.1 O modelo é aderente a essa nova base? O que mudou entre uma base e outra? Justifique.
Aparentemente, sim, pois modelo apresentou uma variação menor no intervalo de 0 e 1, o que indica uma boa classificação.
Entretanto, vale uma ressalva que o modelo possa estar superajustado. 




### 7.2 Descreva como podemos monitorar a saúde do modelo no cenário com e sem a disponibilidade da variável resposta para o modelo em operação.
Para monitorar a saúde do modelo com a variável resposta pode se utilizar o f1-score, log_loss e accuracy.

Já para modelos sem a variável resposta, pode se usar distribuições de score, data drift, análise de outliers etc.

### 7.3 Descreva as estratégias reativa e preditiva de retreinamento para o modelo em operação.

Na estratégia reativa o modelo é reavaliado e retreinado após uma queda nas métricas de desempenho observadas, ou após a coleta de novos dados rotulados

Já no caso de preditiva o modelo é retreinado de forma automática, com base em data drift, mudança de distribuição.

