# <h1>NoteBook - Santander Competition</h1>
# 
# L'objectif de ce Notebook est de pouvoir prédire le comportement des clients de la banque espagnole Santander. Dans ce challenge, nous devons identifer quels client réalisera une transaction particulière dans le future.<br></br><br></br>
# Dans ce problème, nous avons deux résultats possibles : le client réalise la transaction ou non. Nous sommes donc dans un problème de classification binaire. 


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, mean_squared_error
from sklearn import metrics
from sklearn.feature_selection import SelectFromModel

import lightgbm as lgb

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os

# Any results you write to the current directory are saved as output.


#On se met dans le répertoire où se trouve nos jeux de données
if os.path.exists('../input'): 
    os.chdir('../input')
else: pass


data_train = 'train.csv'
data_test = 'test.csv'
X = pd.read_csv(data_train, index_col=None)
Y = X['target']

X.count()


# On peut voir avec la fonction **count** qu'il n'y a pas de missing values dans le jeu de données et il n'y a donc pas d'imputation à réaliser. Nous allons maintenant observer la corrélation entre les différentes variables du jeu de données.
#  


def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize =(100, 100))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    _ = sns.heatmap(
        df.corr(), 
        cmap = colormap,
        square=True, 
        cbar_kws={'shrink':.9 }, 
        ax=ax,
        annot=True, 
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12 }
    )
    
    plt.title('Pearson Correlation of Features', y=1.05, size=15)

correlation_heatmap(X)


# On observe à travers cette matrice des corrélations qu'il n'y a pas de liens forts entre les différentes variables du jeu de données. Nous allons maintenant observer la proportion de transactions de chacune des classes. 


cible = pd.value_counts(X["target"], sort=True).sort_index()
print(cible)

cible.plot(kind='bar', figsize=(10,7))
plt.title("Histogramme de transaction")
plt.xlabel("Type")
plt.ylabel("Nombre")
plt.show()


# On peut observer sur le graphique ci-dessus, qu'on a deux classes existantes et qu'il y a un déséquilibre entre les 0 et les 1. En effet, environ 90% des observations appartiennent à la classe 0 et 10% à la classe restantes.<br></br>
# Cela risque d'être un problème pour la création de modèles, nos modèles risqueront de toujours prédire la classe dominante, ici la classe 0. Pour palier à ce problème, nous allons faire appel à une stratégie d'échantillonage : **L'undersampling** 
# 
# <h2> Undersampling </h2>
# En analyse des données, cette méthode consiste au rééquilibrage de la distribution des classes dans le jeu de données. Pour ce faire, elle enlève des observation issues de la classe majoritaire (ici la classe 0). <br></br><br></br>
# En effet, cette éthode à pour objectif de rendre les algorithmes de Machine Learning que nous implémenterons plus intelligent et créer des modèles de classification plus pertinent. 


num_1 = len(X[X['target'] == 1])
non_num_1_indices = X[X.target == 0].index
random_indices = np.random.choice(non_num_1_indices,num_1, replace=False)
num_1_indices = X[X.target == 1].index
under_sample_indices = np.concatenate([num_1_indices,random_indices])
under_sample = X.loc[under_sample_indices]

y_under = under_sample['target']


cible = pd.value_counts(under_sample["target"], sort=True).sort_index()
print(cible)
under_sample.drop(['target', 'ID_code'], axis='columns', inplace=True)


# Désormais, nous avons autant d'observations dans la classe 1 et 0. Nous allons donc pouvoir commencer à tester quelques algorithmes de Machine Learning. <br></br> <br></br>
# Avant de passer au première algorithme, donnons une courte définition de ce qu'est le **Machine Learning** : Il peut être vu comme étant le champ d'étude qui vise à donner la capacité à une machine d'apprendre sans être explicitement programmé.
# <h2> La régression logistique</h2>
# 
# La régression logit est couramment utilisée pour estimer la probabilité quu'une observation appartienne à une classe particulière. Si la probabilité estimé est supérieure à 50%, alors notre modèle prédit que l'observation appartient à la classe 1, sinon il prédit qu'elle appartient à l'autre classe. On a donc affaire à un classificateur binaire, ce qui est compatible avec notre problème de transaction. 


# On commence par partitionner notrer jeu de données pour tester l'algorithme.
#Pour ce faire, on segmente notre jeu de données en deux parties : une partie entraînement et une de test.

X_train, X_test, y_train, y_test = train_test_split(under_sample, y_under, test_size=0.3, random_state=0, stratify=y_under)

clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_train, y_train)
predictions = clf.predict(X_test)
clf.predict_proba(X_test)
clf.score(X_test, y_test)

cnf_matrix=confusion_matrix(y_test,predictions)
print(" Le rappel de ce modèle est :",cnf_matrix[1,1]/(cnf_matrix[1,1]+cnf_matrix[1,0]))


fig= plt.figure(figsize=(6,3))# pour plot le graphe
print("TP",cnf_matrix[1,1,]) # nombre de transactions "1" prédit comme étant "1"
print("TN",cnf_matrix[0,0]) # nombre de transactions "0" prédit comme étant "0"
print("FP",cnf_matrix[0,1]) # nombre de transactions "0" prédit comme étant "1"
print("FN",cnf_matrix[1,0]) # nombre de transactions "1" prédit comme étant "0"
sns.heatmap(cnf_matrix,cmap="coolwarm_r",annot=True,linewidths=0.5)
plt.title("Confusion_matrix")
plt.xlabel("Predicted_class")
plt.ylabel("Real class")
plt.show()
print("\n--------------------------Classification Report------------------------------------")
print(classification_report(y_test,predictions))


# On observe que le modèle possède un rappel de 0.76, on a donc une proportion de positifs prédit parmis tous les positifs de la population. Au niveau de la précision (qui permet de mesurer la proportion de positifs de la population parmis tous les positifs prédits), on obtient 0.76 également. 
# <h2>Forêts aléatoires ou Random Forest</h2>
# Une forêt aléatoire est un ensemble de plusieurs arbres de décison construit aléatoirement qui s'exécutent en parallèle, avant de les moyenner. 


clf = RandomForestClassifier(n_estimators=200, max_depth=None, criterion='entropy', random_state=0, n_jobs=-1, min_samples_leaf=15)
clf.fit(X_train, y_train)
clf.feature_importances_
predictions1 = clf.predict(X_test)
clf.predict_proba(X_test)
clf.score(X_test, y_test)

cnf_matrix=confusion_matrix(y_test,predictions1)
print(" Le rappel de ce modèle est :",cnf_matrix[1,1]/(cnf_matrix[1,1]+cnf_matrix[1,0]))


fig= plt.figure(figsize=(6,3))# to plot le graphe
print("TP",cnf_matrix[1,1,]) # nombre de transactions "1" prédit comme étant "1"
print("TN",cnf_matrix[0,0]) # nombre de transactions "0" prédit comme étant "0"
print("FP",cnf_matrix[0,1]) # nombre de transactions "0" prédit comme étant "1"
print("FN",cnf_matrix[1,0]) # nombre de transactions "1" prédit comme étant "0"
sns.heatmap(cnf_matrix,cmap="coolwarm_r",annot=True,linewidths=0.5)
plt.title("Confusion_matrix")
plt.xlabel("Predicted_class")
plt.ylabel("Real class")
plt.show()
print("\n--------------------------Classification Report------------------------------------")
print(classification_report(y_test,predictions1))


# On observe que ce modèle se débrouille mieux que celui de la régression logistique pour classer les différentes observations. En effet, on peut voir un rappel d'environ 0.77 (soit un gain de 0.02 sur le modèle précédent) et une précision de 0.76. 
# <h2>Boosting : Light GBM </h2>
# Les méthodes de Boosting correspondent à une méthode de classification (et de régression) non linéaire très performante. Cette méthode agit par itérations successives, la connaissance d'un classifieur, dit faible, est rajouté à un classifieur final, que l'on appelle classifieur fort. <br></br>
# On utillise plusieurs paramètres qui permettent d'ajuster le modèle, tel que le taux d'apprentissage, le nombre de feuilles ou la profondeur maximale de l'arbre.  


def auc(m, train, test): 
    return (metrics.roc_auc_score(y_train,m.predict_proba(train)[:,1]),
                            metrics.roc_auc_score(y_test,m.predict_proba(test)[:,1]))

params = {
    'application': 'binary',
    'objective': 'binary',
    'metric': 'auc',
    'is_unbalance': 'false',
    'boosting': 'gbdt',
    'learning_rate': 0.030020874987649975,
    'max_depth':8,
    'num_leaves': 20,
    'min_data_in_leaf':879,
    'feature_fraction': 0.28058785906270434,
    'subsample': 0.9865508992786004
}
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)

print('Start training...')
gbm = lgb.train(params,
                train_data,
                num_boost_round=10000,
                valid_sets=test_data,
                early_stopping_rounds=500) # Arret si 500 iterations sans gain de performance

print('Start predicting...')
pred1s = gbm.predict(X_test, num_iteration=gbm.best_iteration)
print(pred1s)
print('LGBM scoring :', mean_squared_error(y_test, pred1s))

cnf_matrix=confusion_matrix(y_test,pred1s)
print(" Le rappel de ce modèle est :",cnf_matrix[1,1]/(cnf_matrix[1,1]+cnf_matrix[1,0]))


fig = plt.figure(figsize=(6,3))# to plot the graph
print("TP",cnf_matrix[1,1,]) # no of fraud transaction which are predicted fraud
print("TN",cnf_matrix[0,0]) # no. of normal transaction which are predited normal
print("FP",cnf_matrix[0,1]) # no of normal transaction which are predicted fraud
print("FN",cnf_matrix[1,0]) # no of fraud Transaction which are predicted normal
sns.heatmap(cnf_matrix,cmap="coolwarm_r",annot=True,linewidths=0.5)
plt.title("Confusion_matrix")
plt.xlabel("Predicted_class")
plt.ylabel("Real class")
plt.show()
print("\n---------------------------Classification Report------------------------------------")
print(classification_report(y_test,preds1))


# Cette fois-ci, on fait appelle à l'AUC (Area Under The Curve) pour scorer notre modèle. Plus grande est l'AUC, meilleur est le modèle.


importance = pd.DataFrame({'gain': gbm.feature_importance(importance_type='gain'),
                           'feature': gbm.feature_name()})
importance.sort_values(by='gain', ascending=False, inplace=True)
plt.figure(figsize=(10, 40))
ax = sns.barplot(x='gain', y='feature', data=importance)


# Sur le graphique ci-dessus, on peut observer le gain d'information qu'apporte au modèle chacune des variables. Cela nous permettra de choisir les variables qui fond du bruit et que l'on pourrait potentiellement enlever pour améliorer notre modèle par la suite.<br></br>
# Pour ce faire, nous allons créer une petite fonction seuil de gain, qui va nous permettre de récupérer la liste de svariable ayant un gain supérieur 


i_seuil = 600
l_new_set = []
valeurs = importance[['feature', 'gain']].values
for i in  range(valeurs.shape[0]):
    if valeurs[i][1] > i_seuil:
        l_new_set.append(valeurs[i][0])
l_new_set


# On obtient donc la liste des variables apportant un gain supérieur à 500 au modèle light GBM. Nous allons désormais réaliser un test avec les variables selectionnées.   


df_new = under_sample[l_new_set]

X_train, X_test, y_train, y_test = train_test_split(df_new, y_under, test_size=0.25, random_state=0, stratify=y_under)


def auc(m, train, test): 
    return (metrics.roc_auc_score(y_train,m.predict_proba(train)[:,1]),
                            metrics.roc_auc_score(y_test,m.predict_proba(test)[:,1]))

params = {
    'application': 'binary',
    'objective': 'binary',
    'metric': 'auc',
    'is_unbalance': 'false',
    'boosting': 'gbdt',
    'learning_rate': 0.03754469445963884,
    'max_depth':5,
    'num_leaves': 2,
    'min_data_in_leaf':453,
    'feature_fraction': 0.536120208317914,
    'subsample': 0.2452056321344909
}
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)

print('Start training...')
gbm = lgb.train(params,
                train_data,
                num_boost_round=10000,
                valid_sets=test_data,
                early_stopping_rounds=500) # Arret si 500 iterations sans gain de performance

print('Start predicting...')
pred1s = gbm.predict(X_test, num_iteration=gbm.best_iteration)
print(pred1s)
print('LGBM scoring :', mean_squared_error(y_test, pred1s))


# <h2>Optimisation des hyperparamètres de Light GBM </h2>
# 
# Les hyperparamètres sont des paramètres ajustables que l'on choisit pour entraîner notre modèle qui régit le processus d'entraînement lui-même. L'optimisation de ces hyperparamètres revient à résoudre le problème de choisir un set d'hyperparamètres optimaux pour l'algorithme de Machines Learning que l'on souhaite utiliser (dans notre cas : Light GBM). En plus de pouvoir aider à trouver les hyperparamètres optimaux, il permet aussi un grand gain de temps pour le Data Scientist. Il existe plusieurs approches de recherche du set optimal : Le Grid Search, le Random Search ou bien le Bayesian Optimization. <br></br>
# Dans notre cas, nous allons utiliser l'approche du Grid Search, qui est un mode de recherche exhaustif à travers d'intervals spécifié manuellemebt sur l'espace des Hyperparamètres de l'algorithme d'apprentissage. 


import skopt

N_ROWS=200000
TRAIN_PATH = 'train.csv'
STATIC_PARAMS = {'boosting': 'gbdt',
                'objective':'binary',
                'metric': 'auc',
                'num_threads': 8,
                }
N_CALLS = 2
NUM_BOOST_ROUND = 10000
EARLY_STOPPING_ROUNDS = 1000

def train_evaluate(X, y, params):
    X_train, X_valid, y_train, y_valid = train_test_split(df_new, y_under, test_size=0.3, random_state=0, stratify=y_under)

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

    model = lgb.train(params, train_data,
                      num_boost_round=NUM_BOOST_ROUND,
                      early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                      valid_sets=[valid_data], 
                      valid_names=['valid'])
    
    score = model.best_score['valid']['auc']
    return score


space = [skopt.space.Real(0.01, 0.5, name='learning_rate', prior='log-uniform'),
         skopt.space.Integer(1, 30, name='max_depth'),
         skopt.space.Integer(1, 100, name='num_leaves'),
         skopt.space.Integer(10, 1000, name='min_data_in_leaf'),
         skopt.space.Real(0.1, 1.0, name='feature_fraction', prior='uniform'),
         skopt.space.Real(0.1, 1.0, name='subsample', prior='uniform'),
         ]

data = pd.read_csv(TRAIN_PATH, nrows=N_ROWS)
    
X = data.drop(['ID_code', 'target'], axis=1)
y = data['target']

@skopt.utils.use_named_args(space)
def objective(**params):
    all_params = {**params, **STATIC_PARAMS}
    return -1.0 * train_evaluate(X, y, all_params)

results = skopt.dummy_minimize(objective, space, n_calls=N_CALLS)

print('Best Validation AUC: {}'.format(-1.0 * results.fun))
print('Best Params: {}'.format(results.x))



# Avec l'aide de cette fonction, on obtient les paramètres pour le score d'AUC le plus grand. la fonction *dummy_minimize* va, à l'aide des fonctions *objective* et *space*, chercher la valeur AUC la plus grande qui correspond au modèle le plus performant. Il indiquera juste en-dessous la valeur des hyperparamètres qui permettent d'obtenir une telle performance.  
# <h2>Soumission des résultats : 0.89204</h2>


test_data = pd.read_csv(data_test, index_col=None)
test_data.drop(['ID_code'], axis='columns', inplace=True)
from xlsxwriter.workbook import Workbook


preds = gbm.predict(test_data, num_iteration=gbm.best_iteration)
print(preds)

sub = pd.read_csv('sample_submission.csv')
sub['target'] = preds
    
sub.to_csv('sample_submission.csv')

