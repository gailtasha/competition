# #Estudo de caso do Kaggle - Banco Santander
# #Uma abordagem com Redes Neurais construído no Keras
# 
# **Henrique Dias Marques - Analista Master - Petrobras**
# <br/>
# **Em meio a Pandemia de 2020**
# 
# 
# Nesta competição hospedada no Kaggle, o Banco Santander nos propôs um problema muito comum em qualquer atividade comercial, cujo propósito é avaliar os potenciais clientes nas ofertas de produtos e serviços. Especificamente o banco apresenta 200 características diferentes para cada cliente as quais serão consideradas para averiguar se os mesmos irão no futuro realizar determinada transação com o banco. Sempre visando dar atenção aos potenciais clientes, a estratégia do banco foi procurar a comunidade de Cientistas da Dados do mundo para estabelecer o melhor modelo que respondesse esta questão.
# 
# A base de dados é bem extensa pois possui 200 mil exemplos de clientes para a construção do modelo na fase de treinamento e apresenta a mesma quantidade para realizarmos os testes nos modelos. O problema é de classificação binária [0, 1] com aprendizagem supervisionada, onde o Banco apresenta o resultado do que aconteceu com 200 mil clientes.
# 
# Um aspecto interessante deste caso é que as 200 características dos clientes, apresentadas pelo Banco Santander, não possui a informação de seu metadado. Não sabemos o que significam as 200 características! Isto mostra o quão interessante são os algorítmos de Machine Learning, capazes de lidar e aprender com números que expressam informações de uma natureza só conhecida pelo seu resultado final: [1] - o cliente realizou a transação no futuro e [0] - o cliente desistiu da transação.


# #Carregando as bibliotecas
# 
# Vamos iniciar com a carga de bibliotecas necessárias para o desenvolvimento do modelo. À medida que o código for sendo explicado você entenderá a razão das bibliotecas aqui importadas.


import numpy as np 
import pandas as pd 
import tensorflow as tf
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from tensorflow.keras.initializers import TruncatedNormal, RandomUniform, RandomNormal
from keras.constraints import unit_norm, max_norm

from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, precision_score, \
            recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline


# #Carga de dados
# 
# O código abaixo realiza a carga dos dados em dataframes - Pandas - para uso no treinamento e predição. Cria também variáveis que serão utilizadas na construção da solução. O X_test_index se faz necessário porque na criação do arquivo de submissão ao Kaggle ele será usado como índice no arquivo.
# 
# Para exercitar este exemplo, substitua o caminho da pasta para sua condição específica.


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


print ('\nRealizando carga de dados....\n')
X_train = pd.read_csv("/kaggle/input/santander-customer-transaction-prediction/train.csv", index_col='ID_code')
X_test = pd.read_csv("/kaggle/input/santander-customer-transaction-prediction/test.csv", index_col='ID_code')
print ('\nCarga de dados concluída....\n')


print ('\nPreparando os conjuntos de treinamento e teste....\n')
y_train = X_train.target
X_train.drop(['target'], axis=1, inplace=True)
X_test_index = X_test.index


X_train.head()


X_test.head()


# <br />
# É importante verificarmos se há dados nulos ou ausentes nestes conjuntos. Os códigos abaixo avaliam isto e conclui que não há. Também logo a seguir vou "plotar" alguns gráficos que representam algumas das 200 características dos dados de treinamento e que depois você também poderá verificar nos dados de teste. É fácil notar que todas elas possuem uma distribuição normal. 
# 
# <br />


sum(X_train.columns.isnull())


sum(X_test.columns.isnull())


sns.set()
plt.figure(figsize=(30,3))
pos = 0
for col in X_train.columns.values[0:9]:
        pos+=1
        plt.subplot(190+pos)
        plt.hist(X_train[col], density=True, bins=60)
        plt.title(col)
        plt.ylabel('Probability')
        plt.xlabel('Data')
print ('\nGráficos das primeiras colunas...\n')
plt.show()


# #Feature Engineering
# <br/>
# Nesta sessão vou colocar um insight que tive baseado na ideia de que dados com desvios padrões pequenos possuirão pouca interferência nos resultados, portanto será melhor termos estes dados como valores ordinais. então a ideia será discretizá-los e transformá-los em colunas one-hot. O std_threshold é o desvio padrão máximo que aceitaremos em uma catacterística. Portanto todas as colunas com desvio padrão menor que std_threshold serão convertidos em valores discretos de acordo com o número de bins. 
# 
# Os valores discretos são transformados em características "dummies" no padrão one-hot(veja parâmetro 'discretizer')
# 


std_threshold = 8
bins = 6


print ('\nEstabelecendo as metricas para criação e features...\n')

    
#n_unicos_train = X_train.nunique()
#n_unicos_test = X_test.nunique()
    
std_train = X_train.std(axis=0)
std_test = X_test.std(axis=0)
    
col_train_interested = np.where(std_train >= std_threshold)
col_test_interested = np.where(std_test >= std_threshold)
    
col_train_for_bins = np.where(std_train < std_threshold)
col_test_for_bins = np.where(std_test < std_threshold)


print ('\nDiscricionando colunas com valores pequenos de Desvio Padrão...\n')
     
X_train_e = X_train[std_train.index[(col_train_for_bins)]]
X_test_e = X_test[std_test.index[(col_test_for_bins)]]
    
discretizer = preprocessing.KBinsDiscretizer(n_bins=bins, encode='onehot', strategy='uniform')
    
discretizer.fit(X_train_e)
sparse_matrix = discretizer.transform(X_train_e)
train_onehot = sparse_matrix.todense()

discretizer.fit(X_test_e)
sparse_matrix = discretizer.transform(X_test_e)
test_onehot = sparse_matrix.todense()


print ('\nEliminando colunas com std < std_threshold ...\n')
    
X_train = X_train[std_train.index[(col_train_interested)]]
X_test = X_test[std_test.index[(col_test_interested)]]


X_train.head()


# #Normalização.
# A normalização é um item essencial - para a performance da execução do treinamento - quando trabalhamos com Redes Neurais e também com outros algorítmos de Machine Learning que utilizam funções sigmoidais ou regressões logísticas. Ela também visa permitir que determinadas características dos dados não se sobreponham às outras.
# 
# Para facilitar nosso trabalho vamos utilizar bibliotecas do SciKit Learn que realizam tais operações com facilidades. No código abaixo utlizamos o método StandardScaler do módulo preprocessing para este fim. É importante notar que a saída destas funções retornam um conjunto de arrays (numpy arrays) e não mais dataframes.
# 


# Número de características com valores de desvio padrão maiores que std_threshold
X_train.shape


scaler =  preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


X_train


# Adicionando características que foram discretizadas. Já estão no formato one-hot 


print ('\nConcatenando as matrizes...\n')   
X_train = np.concatenate((X_train, train_onehot), axis=1)
X_test = np.concatenate((X_test, test_onehot), axis=1)


X_train.shape


# #Aplicando o primeiro modelo com "cross-validation"
# Neste exercício inicial vamos aplicar uma estrutura de rede que estabeleci após sucessivos testes para definição dos melhores hyerparâmetros da Rede Neural. Utilizei no modelo as bibliotecas do Keras - uma poderosa ferramenta para construção de Redes Neurais que executa instruções sobre o TensorFlow. O seu modelo sequencial é mais simples de codificação além do fácil entendimento das diversas camadas que foram escolhidas, portanto, optei por esta abordagem. Para o entedimento deste código pressuponho que o leitor já possui toda a base teórica de construção de Redes Neurais. O otimizador escolhido foi o Adam, depois que realizei sucessivos testes com outros otimizadores, dentre eles o SGD e RMSprop.
# 
# A solução neste momento também será realizada com o uso de validação cruzada - parâmetro 'validation_split'. Este processo de divisão de dados de treinamento é essencial para avaliarmos a performance do modelo. 
# 
# O modelo utiliza duas técnicas para regularização que é o "dropout" e o "kernel_constraint", pois verificamos nos sucessivos testes como facilmente nos defrontamos com "overfitting" durante a fase de treinamento. Verifique que no parâmetro 'metrics' inserimos nossas função 'Roc_auc' e 'average_precision' citadas no inicio deste artigo. Com estas funções é possível availarmos o desempenho do processo de aprendizado ao longo da execução do código. Os resultados ficam armazenados no callback - history.


input_len = X_train.shape[1]


    
model = Sequential([
#Dense(input_len, input_shape=(input_len,)),
Dense(256, input_shape=(input_len,)),
Activation('relu'),
Dropout(0.5),
Dense(128, kernel_initializer='random_normal', activation = 'relu',  kernel_constraint=unit_norm()),
Dropout(0.5),
Dense(64, kernel_initializer='random_normal', activation = 'relu',  kernel_constraint=unit_norm()),
Dropout(0.3),
Dense(32, kernel_initializer='random_normal', activation = 'relu',  kernel_constraint=unit_norm()),
Dropout(0.1),
Dense(1),
Activation('sigmoid'),
])
    
    
opt = optimizers.Adam(learning_rate=0.00005)    
#opt = optimizers.Adam(learning_rate=0.001)
    
        
model.compile(optimizer=opt,
              loss= 'binary_crossentropy',
              metrics=['accuracy'])
    
print(model.summary())


history = model.fit(X_train, y_train, \
                epochs= 120,\
                batch_size=1024,\
                validation_split=0.1,\
                )
           


# Os gráficos a seguir nos dão uma melhor avaliação do que ocorreu durante o treinamento. Temos as visões de acurácia e do 'loss' para as duas sessões dos dados: treinamento e validação.


sns.set()
plt.figure(figsize=(15, 7))

plt.subplot(141)
plt.plot(history.history['accuracy'], label='ACC')
plt.plot(history.history['val_accuracy'], label='Val_ACC')
plt.xlabel('Epochs')
plt.ylabel('Acurácia')
plt.legend()


plt.subplot(142)
plt.plot(history.history['loss'], label='LOSS')
plt.plot(history.history['val_loss'], label='Val_LOSS')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# A partir dos gráficos acima podemos concluir algumas questões:
# 
# * Embora o modelo continua crescente na sua performance nos sucessivos 'epochs' o 
# mesmo não acontece nos dados de validação.
# * O 'loss' deixa de cair em torno de 30 a 40 epochs
# As métricas nos dados de validação demonstram queda crescente após 10 epochs.
# 
# Fica claro que mesmo adotando um conjunto de ações para evitar o 'overfitting' o modelo claramente não consegue generalizar quando submetido aos dados de validação. As melhores abordagens para solução deste problema publicado em diversas documentações de Machine Learning sugerem que devemos trabalhar em duas vertentes:
# 
# * Acrescentar mais dados de treinamento ao modelo.
# * Repensar a estrutura da rede
# 
# Antes de aplicarmos algumas das soluções acima, vamos aplicar uma técnica muito comum e fácil de realizar - o 'early stopping'. Pelos gráficos vericamos que o modelo tem boa performance até 40 epochs. No código abaixo realizaremos novamente o treinamento e submeteremos o arquivo ao Kaggle para confrontar o resultado. Desta vez sem a validação cruzada, claro!
# 


input_len = X_train.shape[1]


model = Sequential([
#Dense(input_len, input_shape=(input_len,)),
Dense(256, input_shape=(input_len,)),
Activation('relu'),
Dropout(0.5),
Dense(128, kernel_initializer='random_normal', activation = 'relu',  kernel_constraint=unit_norm()),
Dropout(0.5),
Dense(64, kernel_initializer='random_normal', activation = 'relu',  kernel_constraint=unit_norm()),
Dropout(0.3),
Dense(32, kernel_initializer='random_normal', activation = 'relu',  kernel_constraint=unit_norm()),
Dropout(0.1),
Dense(1),
Activation('sigmoid'),
])
    
    
opt = optimizers.Adam(learning_rate=0.00005)    
#opt = optimizers.Adam(learning_rate=0.001)
    
        
model.compile(optimizer=opt,
             loss= 'binary_crossentropy',
              metrics=['accuracy'])
    
print(model.summary())


history = model.fit(X_train, y_train, \
                epochs= 40,\
                batch_size=1024,\
                #validation_split=0.1,\
                #class_weight = compute_class_weight('balanced', y_train.sum(), len(y_train))
                )


# # Envio do arquivo ao kaggle.
# Vamos agora aplicar o modelo treinado na base de teste e avaliar como será o seu resultado. O código abaixo envia as predições do modelo de acordo com o formado estabelecido nas competições.
# 


X_test.shape


predictions = model.predict(X_test)
   
pred = predictions.reshape((200000,))
output = pd.DataFrame({'ID_code': X_test_index, 'target': pred})
output.to_csv('ANN_Santander_GColab_fet_Eng_TPU_3.csv', index=False)
print("O arquivo para envio ao Kaggle foi salvo com sucesso!")


predictions[0:10]


# # Conclusão
# A aplicação de uma Rede Neural para a solução proposta pelo Banco Santander apresenta uma boa resposta para os dados originais como foram fornecidos. A performance ainda está distante das melhores resultados apresentados na competição que se deu no ano passado, mas que utilizaram outras abordagens de Machine Learning - como o LightGBM que é bastante usado como algoritmo venceder pelas equipes que participam do Kaggle - e um extenso trabalho de preparação dos dados. Vale salientar que neste exemplo não adotei nenhuma destas abordagem, e portanto, poderemos melhorar ainda mais esta performance com a utilização de Feature Engineearing ou mesmo modificando a estrutura da Rede Neural.
# 
# Na próxima publicação sobre esta competição irei mostrar como um pouco de Feature Engeeneearing sobre os dados de Treinamento e Testes será possível obter um ganho na performance.

