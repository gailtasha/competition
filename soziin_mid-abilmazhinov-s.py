# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

import os
import gc
import sys
import time
import shutil
import feather
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, roc_auc_score
import seaborn as sns

import pandas as pd
import numpy as np
from scipy.stats import skew
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
import xgboost as xgb

import warnings
warnings.simplefilter('ignore')

from scipy import stats
from scipy.stats import norm, skew #for some statistics

import os

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


test = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv', delimiter=',')
test.head()


train = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv', delimiter=',')
train.head()


train.describe()


train.corr()


# В первую очередь посмотрим на баланс между классами. Так как мы ещё не знаем как с этим бороться, то мы оставим это так, однако в дальнейшем с этим надо будет работать


sns.countplot(train['target'], palette='Set3')


# В первую очередь давайте посмотрим на распределения для нашего бинарного класса. Визуализируем это


def plot_feature_distribution(df1, df2, label1, label2, features):
    i = 0
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(10,10,figsize=(18,22))

    for feature in features:
        i += 1
        plt.subplot(10,10,i)
        sns.distplot(df1[feature], hist=False,label=label1)
        sns.distplot(df2[feature], hist=False,label=label2)
        plt.xlabel(feature, fontsize=9)
        locs, labels = plt.xticks()
        plt.tick_params(axis='x', which='major', labelsize=6, pad=-6)
        plt.tick_params(axis='y', which='major', labelsize=6)
    plt.show();


# Распределения средних и квадратичных отклонений


plt.figure(figsize=(16,6))
features = train_df.columns.values[2:202]
plt.title("Distribution of mean values per row in the train and test set")
sns.distplot(train_df[features].mean(axis=1),color="green", kde=True,bins=120, label='train')
sns.distplot(test_df[features].mean(axis=1),color="blue", kde=True,bins=120, label='test')
plt.legend()
plt.show()


# Распределения средних по столбцам между тестом и тренировочными данными


plt.figure(figsize=(16,6))
plt.title("Distribution of mean values per column in the train and test set")
sns.distplot(train_df[features].mean(axis=0),color="magenta",kde=True,bins=120, label='train')
sns.distplot(test_df[features].mean(axis=0),color="darkblue", kde=True,bins=120, label='test')
plt.legend()
plt.show()


# Распределения квадратичных отклонений по столбцам между тестом и тренировочными


plt.figure(figsize=(16,6))
plt.title("Distribution of std values per row in the train and test set")
sns.distplot(train_df[features].std(axis=1),color="black", kde=True,bins=120, label='train')
sns.distplot(test_df[features].std(axis=1),color="red", kde=True,bins=120, label='test')
plt.legend();plt.show()


# Теперь займемся исскуством feature engineering, попробуем и сократить количество признаков, и одновременно пародим новые более эффективные признаки. Первым делом выведем самые коррелированные признаки, на то что нам стоит обратить внимание, а затем на то, что не должно привлекать наше внимание


correlations = train_df[features].corr().abs().unstack().sort_values(kind="quicksort").reset_index()
correlations = correlations[correlations['level_0'] != correlations['level_1']]
correlations.head(10)
correlations.tail(10)


correlations.head(10)


# Корреляция как видим между ними малая


%%time
features = train_df.columns.values[2:202]
unique_max_train = []
unique_max_test = []
for feature in features:
    values = train_df[feature].value_counts()
    unique_max_train.append([feature, values.max(), values.idxmax()])
    values = test_df[feature].value_counts()
    unique_max_test.append([feature, values.max(), values.idxmax()])


np.transpose((pd.DataFrame(unique_max_train, columns=['Feature', 'Max duplicates', 'Value'])).\
            sort_values(by = 'Max duplicates', ascending=False).head(15))


%%time
idx = features = train_df.columns.values[2:202]
for df in [test_df, train_df]:
    df['sum'] = df[idx].sum(axis=1)  
    df['min'] = df[idx].min(axis=1)
    df['max'] = df[idx].max(axis=1)
    df['mean'] = df[idx].mean(axis=1)
    df['std'] = df[idx].std(axis=1)
    df['skew'] = df[idx].skew(axis=1)
    df['kurt'] = df[idx].kurtosis(axis=1)
    df['med'] = df[idx].median(axis=1)


# Давайте проверим наши новые фичи


train_df[train_df.columns[202:]].head()


def plot_new_feature_distribution(df1, df2, label1, label2, features):
    i = 0
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(2,4,figsize=(18,8))

    for feature in features:
        i += 1
        plt.subplot(2,4,i)
        sns.kdeplot(df1[feature], bw=0.5,label=label1)
        sns.kdeplot(df2[feature], bw=0.5,label=label2)
        plt.xlabel(feature, fontsize=11)
        locs, labels = plt.xticks()
        plt.tick_params(axis='x', which='major', labelsize=8)
        plt.tick_params(axis='y', which='major', labelsize=8)
    plt.show();


# Теперь поглядим на распределения наших фич группируя по классу


t0 = train_df.loc[train_df['target'] == 0]
t1 = train_df.loc[train_df['target'] == 1]
features = train_df.columns.values[202:]
plot_new_feature_distribution(t0, t1, 'target: 0', 'target: 1', features)


# Добавим наши новые фичи


features = [c for c in train_df.columns if c not in ['ID_code', 'target']]
for feature in features:
    train_df['r2_'+feature] = np.round(train_df[feature], 2)
    test_df['r2_'+feature] = np.round(test_df[feature], 2)
    train_df['r1_'+feature] = np.round(train_df[feature], 1)
    test_df['r1_'+feature] = np.round(test_df[feature], 1)

print('Train and test columns: {} {}'.format(len(train_df.columns), len(test_df.columns)))


X, y = train.iloc[:,2:], train.iloc[:,1]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state = 123, stratify = y)


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg = LogisticRegression()
logreg.fit(X_train, y_train)


logreg=LogisticRegression()
logreg.fit(X_train,y_train)


from sklearn.metrics import confusion_matrix
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

train_pred = logreg.predict(X_train)
print('RMSLE : {:.4f}'.format(rmsle(y_train, train_pred)))


# Naive Bayes


from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()


gnb.fit(X_train, y_train)


y_predit_svc = gnb.predict(X_test)


from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix

accuracy_score(y_test, y_predit_svc)


confusion_matrix(y_test,y_predit_svc)


print(classification_report(y_test, y_predit_svc))


y_preds_res = gnb.predict(X)


accuracy_score(y, y_preds_res)


print(classification_report(y, y_preds_res))


submission_nb = pd.DataFrame({
    "ID_code": test["ID_code"],
    "target": y_preds_res
})
submission_nb.to_csv('naive_baise_submission.csv', index=False)


Random Tree


from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

clf = tree.DecisionTreeClassifier(max_depth=10)
clf = clf.fit(X, y)


y_preds = clf.predict(X)


accuracy_score(y, y_preds)


confusion_matrix(y, y_preds)


print(classification_report(y, y_preds))


submission_tree = pd.DataFrame({
    "ID_code": test["ID_code"],
    "target": y_preds
})
submission_tree.to_csv('rand_tree_submission.csv', index=False)


# Random forest


from sklearn.ensemble import RandomForestClassifier


clf = RandomForestClassifier(max_depth=10, random_state=0)


clf.fit(X_train, 
        y_train)


y_pred = clf.predict(X_test)


accuracy_score(y_test, y_pred)


print(classification_report(y_test, y_pred))


# XgBoost


train_id = train['ID_code']
y_train = train['target']
X_train = train.drop(['ID_code', 'target'], axis=1, inplace = False)

test_id = test['ID_code']
X_test = test.drop('ID_code', axis=1, inplace = False)


model_xgb = xgb.XGBRegressor(n_estimators=5, max_depth=4, learning_rate=0.5) 
model_xgb.fit(X_train, y_train)

