# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Importation des différentes librairies
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline


# Chargement des données d'entraînement et de test en utilisant le chemin vers les différents fichiers
train = pd.read_csv('../input/santander-customer-transaction-prediction/train.csv')
test = pd.read_csv('../input/santander-customer-transaction-prediction/test.csv')
sample_submission=pd.read_csv("../input/santander-customer-transaction-prediction/sample_submission.csv")


# On extrait la variable cible et on enlève pour des raisons de simplicités la variable "ID"
Y_train = train.target              
train.drop(['ID_code'], axis = 1, inplace = True) 
test.drop(['ID_code'], axis = 1, inplace = True)
train.drop(['target'], axis = 1, inplace = True) 
train = train.iloc[:, 1:].values.astype('float64')
test = test.iloc[:, 1:].values.astype('float64')


from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier


# On peut récupérer le modèle directement qu'on a sauvegardé :) 
import pickle
with open('../input/lgbm-model-saved/LGBM.pkl', 'rb') as input_file:
    model_recupere = pickle.load(input_file)
    input_file.close()


model_recupere


# On peut récupérer le modèle directement qu'on a sauvegardé :) 
import pickle
with open('../input/lighgbm/lgb.pkl', 'rb') as input_file:
    model_recupere2 = pickle.load(input_file)
    input_file.close()


model_recupere2.


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(train,Y_train, random_state =1234 , test_size = 0.33) 


model_recupere2.fit(X_train, y_train)

fpr, tpr, _ = roc_curve(y_test, model_recupere2.predict_proba(X_test)[:, 1])    
fprs.append(fpr)
tprs.append(tpr)
aucs.append(auc(fpr, tpr))


plt.figure(figsize=(9, 7))
plt.plot([0, 1], [0, 1], 'k--')

for fpr, tpr, auc, name in zip(fprs, tprs, aucs, names):
    plt.plot(fpr, tpr, label=name + ' (AUC=%.2f)' % auc, lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('Receiver operating characteristic', fontsize=18)
plt.legend(loc="lower right", fontsize=14)


from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from catboost import CatBoostClassifier
class Ensemble(object):
    def __init__(self, n_splits, stacker, base_models):
        self.n_splits = n_splits
        self.stacker = stacker
        self.base_models = base_models
        self.y_pred = np.empty([1,1])

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        folds = list(StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=2016).split(X, y))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):

            S_test_i = np.zeros((T.shape[0], self.n_splits))

            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
#                y_holdout = y[test_idx]

                print ("Fit %s fold %d" % (str(clf).split('(')[0], j+1))
                clf.fit(X_train, y_train)
                cross_score = cross_val_score(clf, X_train, y_train, cv=3, scoring='roc_auc')
                print("    cross_score: %.5f" % (cross_score.mean()))
                y_pred = clf.predict_proba(X_holdout)[:,1]                

                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict_proba(T)[:,1]
            S_test[:, i] = S_test_i.mean(axis=1)

        results = cross_val_score(self.stacker, S_train, y, cv=3, scoring='roc_auc')
        print("Stacker score: %.5f" % (results.mean()))

        self.stacker.fit(S_train, y)
        self.y_pred = self.stacker.predict_proba(S_test)[:,1]
        return self.y_pred


# LightGBM params

lgb_params2 = {}
lgb_params2['n_estimators'] = 280
lgb_params2['learning_rate'] = 0.1
lgb_params2['colsample_bytree'] = 0.3   
lgb_params2['subsample'] = 0.7
lgb_params2['subsample_freq'] = 2
lgb_params2['num_leaves'] = 16
lgb_params2['random_state'] = 99

param1 = {
    'num_leaves': 18,
     'max_bin': 63,
     'min_data_in_leaf': 5,
     'learning_rate': 0.010614430970330217,
     'min_sum_hessian_in_leaf': 0.0093586657313989123,
     'feature_fraction': 0.056701788569420042,
     'lambda_l1': 0.060222413158420585,
     'lambda_l2': 4.6580550589317573,
     'min_gain_to_split': 0.29588543202055562,
     'max_depth': 49,
     'save_binary': True,
     'seed': 1337,
     'feature_fraction_seed': 1337,
     'bagging_seed': 1337,
     'drop_seed': 1337,
     'data_random_seed': 1337,
     'objective': 'binary',
     'boosting_type': 'gbdt',
     'verbose': 1,
     'metric': 'auc',
     'is_unbalance': True,
     'boost_from_average': False
}
    
# paramètres optimaux qu'on avait obtenu du 1er modèle que j'avais pas sauvegardé mdrr !!! 
gb_params = {}
gb_params['max_depth'] = 6
gb_params['max_features'] = 4
gb_params['min_samples_leaf'] = 4
gb_params['min_samples_split'] = 8
gb_params['n_estimators'] = 265


# Modelès qu'on va utiliser

lgb_model1 = LGBMClassifier(**param1)
lgb_model2 = LGBMClassifier(**lgb_params2)
lgb_model3 = model_recupere.best_estimator_
gb_model = GradientBoostingClassifier(**gb_params)
cb_model = CatBoostClassifier(iterations=1000, loss_function='Logloss')
log_model = LogisticRegression()


stack = Ensemble(n_splits=3,
        stacker = log_model,
        base_models = (lgb_model2,lgb_model3,gb_model,cb_model,log_model
                      ))    


y_pred = stack.fit_predict(train, Y_train, test)


# On sauvegarde le modèle de méthodes d'ensemble
import pickle
with open('ensemble_learning2.pkl', 'wb') as output:
    pickle.dump(stack, output, pickle.HIGHEST_PROTOCOL)
    output.close()


# On ouvre le modèle récupéré !
import pickle
with open('ensemble_learning2.pkl', 'rb') as input_file:
    stack = pickle.load(input_file)
    input_file.close()


clf.fit(X_train, y_train)

    fpr, tpr, _ = roc_curve(y_test, clf.predict_proba(X_test)[:, 1])    
    fprs.append(fpr)
    tprs.append(tpr)
    aucs.append(auc(fpr, tpr))


# c'est complètement bidon ce que je fais puisque là je fais une validation
# avec les y_pred que j'ai prédit avec les données de test et le Y_train du modèle d'apprentissage
# ce qui en fait n'a aucun sens d'où mon résultat médiocre mdrrr !!! j'ai paniqué pour rien Lol :) 

from sklearn.metrics import roc_curve, auc
fpr, tpr, _ = roc_curve(Y_train, stack.y_pred)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label="LR (AUC=%.6f)" % roc_auc, lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('Receiver operating characteristic', fontsize=18)
plt.legend(loc="lower right", fontsize=16)


from sklearn.metrics import precision_recall_curve, average_precision_score
precision, recall, _ = precision_recall_curve(Y_train, stack.y_pred)

plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
average_precision = average_precision_score(Y_train, stack.y_pred)
plt.title('2-class Precision-Recall curve: AP={0:0.6f}'.format(
          average_precision))


y_pred


stack.y_pred


sub = pd.DataFrame()
sub['ID_code'] = sample_submission.ID_code
sub['target'] = y_pred
sub.to_csv('stacked_2.csv', index=False)


sub



