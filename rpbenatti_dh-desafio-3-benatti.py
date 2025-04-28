# # SANTANDER CUSTOMER TRANSACTION PREDICTION
# 
# ## Entendendo o Problema
# 
# At Santander our mission is to help people and businesses prosper. We are always looking for ways to help our customers understand their financial health and identify which products and services might help them achieve their monetary goals.
# 
# Our data science team is continually challenging our machine learning algorithms, working with the global data science community to make sure we can more accurately identify new ways to solve our most common challenge, binary classification problems such as: is a customer satisfied? Will a customer buy this product? Can a customer pay this loan?
# 
# **In this challenge, we invite Kagglers to help us identify which customers will make a specific transaction in the future, irrespective of the amount of money transacted. The data provided for this competition has the same structure as the real data we have available to solve this problem.**
# 
# Ref.: https://www.kaggle.com/c/santander-customer-transaction-prediction


# Importando as bibliotecas
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import statsmodels.api as sm

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# ## Pegando os Dados


# Configurando opção para mostrar todas as colunas do dataset
pd.set_option("display.max_columns", None)


# Lendo e verificando os dados de treino e de teste
df_train = pd.read_csv(r'/kaggle/input/santander-customer-transaction-prediction/train.csv')
df_test = pd.read_csv(r'/kaggle/input/santander-customer-transaction-prediction/test.csv')


# ## Pre-processamento dos Dados


df_train.info()


df_test.info()


df_train.head()


df_test.head()


df_train.describe()


df_test.describe()


# ### Tratando os nulos


df_train.isnull().values.any()


df_test.isnull().values.any()


# **Conclusão:** não existem valores nulos nos datasets.


# ### Procurando por observações duplicadas


df_train.duplicated().value_counts()


df_test.duplicated().value_counts()


# **Conclusão:** não existem valores duplicados nos datasets.


# ### Verificando as distribuições das variáveis


df_train.iloc[:,2:102].hist(figsize = (20,20));


df_train.iloc[:,102:202].hist(figsize = (20,20));


df_test.iloc[:,1:101].hist(figsize = (20,20));


df_test.iloc[:,101:201].hist(figsize = (20,20));


from scipy import stats

skewed_feats = df_train.iloc[:,2:].apply(lambda x: stats.skew(x.dropna())).sort_values(ascending=False)
print("\nAssimetria: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head()


from scipy import stats

skewed_feats = df_test.iloc[:,1:].apply(lambda x: stats.skew(x.dropna())).sort_values(ascending=False)
print("\nAssimetria: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head()


# **Conclusão:** as variáveis explicativas apresentam distribuições normais.


# ## Exploratory Data Analysis (EDA)


# ### Verificando as correlações e plotando um HeatMap


# Verificando as correlações no Dataset de Treino
df_train_corr = df_train.corr()
df_train_corr


# Verificando as correlações no Dataset de Teste
df_test_corr = df_test.corr()
df_test_corr


# **Conclusão:** As variáveis possuem fraca correlação entre si.


# Identificando as 10 variáveis que mais estão positivamente correlacionadas com o target
df_train_most_pos_corr = df_train_corr.nlargest(11, 'target')['target']
# Identificando as 10 variáveis que mais estão negativamente correlacionadas com o target
df_train_most_neg_corr = df_train_corr.nsmallest(10, 'target')['target']
# Concatenando as 20 variáveis
df_train_most_corr = pd.concat([df_train_most_pos_corr,df_train_most_neg_corr])
df_train_most_corr


# Plotando um mapa de calor com as 20 variáveis mais correlacionadas (10+ positivamente e 10+ negativamente)
cols = df_train_most_corr.index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.15)
f, ax = plt.subplots(figsize=(20, 20))
hm = sns.heatmap(cm, 
                 cbar=True, 
                 annot=True, 
                 square=True, 
                 fmt='.2f', 
                 annot_kws={'size': 10}, 
                 yticklabels=cols.values, 
                 xticklabels=cols.values)


# ### Plotando um Pair-Plot


# Plotando o PAIPLOT para as variáveis mais correlacionadas com um sample (100 amostras)
sample = df_train[cols].sample(100)
sns.pairplot(sample)


# Pelo resultado do pair-plot, não é possivel identificar nenhum agrupamento ou linearidade entre as variáveis.
# 
# Além disso, os histogramas da diagonal mostram novamente que as variáveis se aproximam de distribuições normais.


# ## Feature Selection


# ### Feature Importance - Variancethreshold


# ***"This method removes features with variation below a certain cutoff.
# The idea is when a feature doesn’t vary much within itself, it generally has very little predictive power.***
# 
# ***Variance Threshold doesn’t consider the relationship of features with the target variable.***
# 
# Ref.: https://towardsdatascience.com/why-how-and-when-to-apply-feature-selection-e9c69adfabf2


from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(20.0)


# Realizando Feature Selection das variáveis de Treino
selected_trainX = selector.fit_transform(df_train.drop(['ID_code','target'], axis=1))
selected_trainX = pd.DataFrame(selected_trainX)
selected_trainX.rename(columns=('var_' + pd.Series(selector.get_support(indices=True)).astype(str)), inplace=True)
selected_trainX.describe()


# Realizando Feature Selection das variáveis de Teste
selected_testX = selector.fit_transform(df_test.drop(['ID_code'], axis=1))
selected_testX = pd.DataFrame(selected_testX)
selected_testX.rename(columns=('var_' + pd.Series(selector.get_support(indices=True)).astype(str)), inplace=True)
selected_testX.describe()


print(selected_trainX.shape, selected_testX.shape)


# ### Padronizando os Dados


# Transformando os dados através de uma operação de "scaling".
# 
# Desta forma será possível realizar a comparação entre os resultados, pois os cálculos de alguns modelos são baseados em distâncias e portanto os dados precisam estar padronizados.


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# Padronizando os dados de Treino com Standard Scaler
scaled_trainX = scaler.fit_transform(selected_trainX)
scaled_trainX = pd.DataFrame(scaled_trainX)
scaled_trainX.columns = selected_trainX.columns
scaled_trainX.describe()


# Nota:
# 
# Após a padronização, o desvio padrão (std) das variáveis deve ser igual a 1.0 e a média (mean) deve ser igual a 0.0, o que pode ser confirmado no comando describe acima.


# Padronizando os dados de Teste com Standard Scaler
scaled_testX = scaler.fit_transform(selected_testX)
scaled_testX = pd.DataFrame(scaled_testX)
scaled_testX.columns = selected_testX.columns
scaled_testX.describe()


# Nota:
# 
# Após a padronização, o desvio padrão (std) das variáveis deve ser igual a 1.0 e a média (mean) deve ser igual a 0.0, o que pode ser confirmado no comando describe acima.


print(scaled_trainX.shape, scaled_testX.shape)


scaled_train = pd.concat([df_train.target, scaled_trainX], axis=1)
scaled_trainY = scaled_train.target
scaled_trainY.value_counts()


# Nota:
#     
# O dataset de Test não inclui a variável target, pois é ela que queremos prever.
# 
# Sendo assim, vamos considerar que scaled_test = scaled_testX


# ### Balaceamento do Dataset de Treino


# De acordo com os resultados, a quantidade de dados obtidos para a variável target mostra um dataset não balanceado.
# 
# Para corrigir este desbalanceamento, utilizada a técnica de oversampling.
# 
# Ref.: https://towardsdatascience.com/having-an-imbalanced-dataset-here-is-how-you-can-solve-it-1640568947eb


from imblearn.over_sampling import SMOTE

# Resample the minority class.
smt = SMOTE(sampling_strategy='minority', random_state=42)

# Fit the model to generate the data.
%time oversampled_trainX, oversampled_trainY = smt.fit_sample(scaled_trainX, scaled_trainY)


oversampled_trainX = pd.DataFrame(oversampled_trainX, columns=scaled_trainX.columns.values)
oversampled_trainY = pd.DataFrame(oversampled_trainY, columns=pd.DataFrame(scaled_trainY).columns.values)
oversampled_train = pd.concat([oversampled_trainY, oversampled_trainX], axis=1)
oversampled_train.head()


oversampled_train.target.value_counts()


# O resultado acima mostra que o DataFrame agora está balanceado e podemos prosseguir na análise.


# ### Plotando um HeatMap após o Balaceamento


# Verificando as correlações no Dataset de Treino após o Balanceamento
oversampled_train_corr = oversampled_train.corr()
oversampled_train_corr


# Identificando as 10 variáveis que mais estão positivamente correlacionadas com o target
oversampled_train_most_pos_corr = oversampled_train_corr.nlargest(11, 'target')['target']
# Identificando as 10 variáveis que mais estão negativamente correlacionadas com o target
oversampled_train_most_neg_corr = oversampled_train_corr.nsmallest(10, 'target')['target']
# Concatenando as 20 variáveis
oversampled_train_most_corr = pd.concat([oversampled_train_most_pos_corr,oversampled_train_most_neg_corr])
oversampled_train_most_corr


# Nota: o resultado acima mostra que houve uma pequena melhora nas correlações após realizar o balanceamento.


# ### Feature Engineering (F.E)


# for x in oversampled_train.var_0.unique():
#     temp = oversampled_train[oversampled_train.var_0==x]
# #    print(x,temp['price'].mean())

# # Criação de um DataFrame para o tamanho médio das casas para cada ZIP CODE
# # Aproveitando o tamanho médio das casas com mesmo ZIP CODE
# df_zip = df.groupby(['zip']).agg({'size_house':'mean'})
# df_zip.shape

# # Fazendo uma merge dos dois DataFrames para capturar o tamanho médio das casas por ZIP CODE em pandas
# df = pd.merge(df, df_zip, how='inner', left_on='zip', right_index=True)

# data['Destination'].unique()
# data['Destination'] = np.where(data['Destination']=='Delhi','New Delhi', data['Destination'])

# df_train['var_0'].unique()
# df_train['var_0_c'] = np.where(df_train['var_0']=='Delhi','New Delhi', data['Destination'])

# df_train['var_0_c'] = [df_train['var_0'].value_counts().head(20)

# pd.Series(df_train['var_0'].unique()).value_counts()


# ## Rodando os modelos de Machine Learning


# Rodando uma primeira regressão do tipo Logística como Benchmark


# Definindo as variáveis explicativas e a variável target para o dataframe original
X = df_train.drop(['ID_code','target'], axis=1)
y = df_train.target
print(X.shape, y.shape)


# Rodando uma regressão logística com statsmodel para os dados originais para efeito de comparação
lr = sm.Logit(y, X)        # instanciando o modelo
result = lr.fit()          # ajustando o modelo
print(result.summary2())   # imprimindo o resumo dos resultados


# Definindo as variáveis explicativas e a variável target para o dataframe balanceado e reduzido via feature selection
X = oversampled_trainX
y = oversampled_trainY
print(X.shape, y.shape)


# Rodando a regressão logística via statsmodel
lr = sm.Logit(y, X)        # instanciando o modelo
result = lr.fit()          # ajustando o modelo
print(result.summary2())   # imprimindo o resumo dos resultados


# **Conclusão:** o resultado para os dados balanceados e reduzidos via feature selection produziu um R quadrado mais baixo, porém pode ajudar a não overfitar o modelo.


#  ### Separando os dados do Dataset de Treino via "train_test_split"


# Separando em dados de treino e teste - 80/20
X_train, X_test, y_train, y_test = train_test_split(oversampled_trainX, oversampled_trainY, test_size=0.2, random_state=42, stratify=oversampled_trainY)
y_train = pd.Series.ravel(y_train)


# Rodando uma regressão logística com o pacote sklearn
lr = LogisticRegression(solver='liblinear')   # instanciando o modelo
%time lr.fit(X_train, y_train)                # ajustando o modelo
y_pred = lr.predict(X_test)                   # calculando os preditos
lr.score(X_test, y_test)                      # obtendo o score do modelo


# Calculando a Matriz de Confusão
confusion_matrix(y_test, y_pred)


# Imprimindo as métricas 
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))


# Plotando a curva ROC
y_pred_proba = lr.predict_proba(X_test)[::,1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)

plt.plot(fpr,tpr,label="Logistic Regression, auc=%0.2f" % auc)
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc=4)
plt.show()


# Como um primeiro resultado, o valor da área AUC da curva ROC foi de 0.77.


# **Fazendo predições com outros modelos**


# Rodando outros modelos e medindo seus resultados, incluindo a curva ROC, a qual poderá nos ajudar a escolher qual o melhor modelo para a predição.


import time

# Definindo uma lista com todos os modelos
classifiers = [
    GaussianNB(),
    DecisionTreeClassifier(),
    RandomForestClassifier()]

# Rotina para instanciar, predizer e medir os resultados de todos os modelos
for clf in classifiers:
    start = time.time()
    # instanciando o modelo
    clf.fit(X_train, y_train)
    # armazenando o nome do modelo na variável name
    name = clf.__class__.__name__
    # imprimindo o nome do modelo
    print("="*30)
    print(name)
    # imprimindo os resultados do modelo
    print('****Results****')
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    
    # Plotando a curva ROC
    y_pred_proba = clf.predict_proba(X_test)[::,1]
    fpr, tpr, _ = roc_curve(y_test,  y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr,tpr,label=name+", auc="+str(auc))
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    plt.legend(loc=4)
    
    end = time.time()
    print('****Elapsed time to run:', end - start,'****')


# Observações:
# 
# O algoritmo **Gaussian Naive Bayes** teve um bom resultado, pois suas premissas foram satisfeitas:
# 
# - O requisito dos preditores é serem independentes. Vimos que as features possuem fraca correlação entre si no pair plot analisado com as 10 mais e 10 menos correlacionadas à variável target.
# - Quando os preditores assumem um valor contínuo e não são discretos, assumimos que esses valores são amostrados a partir de uma distribuição gaussiana.
# 
# Nota: **Bernoulli Naive Bayes** não faz sentido nesse caso, pois os preditores não são variáveis booleanas.
# 
# Os classificadores **Decision Tree** e **Random Forest** precisariam ter os parâmetros otimizados, pois pode ser que estejam se ajustando demais ao modelo não sendo assim generalistas.


# **Conclusão:**
# 
# Considerando que a avaliação para este desafio do Santander se baseia na AUC, os melhores resultados obtidos foram com os modelos:
# 
# - **Gaussian Naive Bayes**
# - **Random Forest Classifier**
# 
# "Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target."
# 
# Ref.: https://www.kaggle.com/c/santander-customer-transaction-prediction/overview/evaluation


# ## Avaliação do Modelo


# ### Realizando Cross-validation
# 
# Utilizando cross_val_score para validação e fine tune do modelo RandomForest.
# 
# **cross_val_score** -> uses stratifield kfold by default


# %time scores_GNB = cross_val_score(GaussianNB(), X_train, y_train, cv=5)
# scores_GNB


# %time scores_LR = cross_val_score(LogisticRegression(solver='liblinear'), X_train, y_train, cv=3)
# scores_LR


# %time scores_DT = cross_val_score(DecisionTreeClassifier(max_depth=10), X_train, y_train, cv=3)
# scores_DT


# %time scores_DT = cross_val_score(DecisionTreeClassifier(max_depth=20), X_train, y_train, cv=3)
# scores_DT


# %time scores_DT = cross_val_score(DecisionTreeClassifier(max_depth=40), X_train, y_train, cv=3)
# scores_DT


# %time scores_RF = cross_val_score(RandomForestClassifier(n_estimators=10), X_train, y_train, cv=3)
# scores_RF


# %time scores_RF = cross_val_score(RandomForestClassifier(n_estimators=20), X_train, y_train, cv=3)
# scores_RF


# %time scores_RF = cross_val_score(RandomForestClassifier(n_estimators=40), X_train, y_train, cv=3)
# scores_RF


# - Não houve variação dos valores aumentando o folder (cv), o que é um bom resultado
# - DecisionTreeClassifier -> produz melhor resultado aumentando a profundidade
# - Random Forest classifier -> produz melhor resultado aumentando a quantidade de trees


# ### Rodando os modelos para o dataset de Test


# Separando em dados de treino e teste - 80/20
X_train, X_test, y_train, y_test = train_test_split(oversampled_trainX, oversampled_trainY, test_size=0.2, random_state=42, stratify=oversampled_trainY)
y_train = pd.Series.ravel(y_train)


# Rodando a Regressão Logística


# instanciando o modelo
lr = LogisticRegression(solver='liblinear')
# ajustando o modelo
%time lr.fit(X_train, y_train)
# calculando os preditos para os dados de teste
y_pred_lr = lr.predict(scaled_testX)


#  Rodando o algoritmo de Gaussian Naive Bayes


# instanciando o modelo
gnb = GaussianNB()
# ajustando o modelo com os dados de treino
%time gnb.fit(X_train, y_train)
# calculando os preditos para os dados de teste
y_pred_gnb = gnb.predict(scaled_testX)


# Rodando o classificador Decision Tree


# instanciando o modelo
clf_tree = DecisionTreeClassifier()
# ajustando o modelo com os dados de treino
%time clf_tree.fit(X_train, y_train)
# calculando os preditos para os dados de teste
y_pred_tree = clf_tree.predict(scaled_testX)


# Rodando o classificador Random Forest


# instanciando o modelo
clf_rf = RandomForestClassifier(n_estimators=40)
# ajustando o modelo com os dados de treino
%time clf_rf.fit(X_train, y_train)
# calculando os preditos para os dados de teste
y_pred_rf = clf_rf.predict(scaled_testX)


# ### Criando um arquivo de submissão no KAGGLE


submission = pd.read_csv('../input/santander-customer-transaction-prediction/sample_submission.csv')
submission['target'] = y_pred_lr[0:200000]
submission.to_csv('submission_lr.csv', index=False)


submission = pd.read_csv('../input/santander-customer-transaction-prediction/sample_submission.csv')
submission['target'] = y_pred_gnb[0:200000]
submission.to_csv('submission_gnb.csv', index=False)


submission = pd.read_csv('../input/santander-customer-transaction-prediction/sample_submission.csv')
submission['target'] = y_pred_tree[0:200000]
submission.to_csv('submission_tree.csv', index=False)


submission = pd.read_csv('../input/santander-customer-transaction-prediction/sample_submission.csv')
submission['target'] = y_pred_rf[0:200000]
submission.to_csv('submission_rf.csv', index=False)


# ### Resultados do Kaggle



