# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


train = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')
test = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')
submission = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/sample_submission.csv')


train.shape, test.shape


# Ok, we have 200000 rows and 202 features on train dataset, and same rows and 201 features on test. I see that both of train and test have same amount of rows, which 50 % to 50%.


train.head()


train.target.dtype


# I am interested on target value, also let's check for missing data.


print("NULL values in train: ", train.isnull().sum().sum())
print("NULL values in test: ", test.isnull().sum().sum())


train.describe()


test.describe()


# As a result, we can make few observations: 
# 1. Standard deviation for both train and test relatively large.
# 2. mean, min, max, std for train and test is close.


# Let's check distribution of target value.


sns.countplot(train['target'])


print("There are {}% target values with 1".format(100 * train['target'].value_counts()[1]/train.shape[0]))


# The data is unbalanced in target value.


# Distribution of mean and std


# Let's check the distribution of the mean values for train and test.


plt.figure(figsize=(16,6))
features = train.columns.values[2:202]
plt.title("Distribution of mean values per row in the train and test set")
sns.distplot(train[features].mean(axis=1),color="green", kde=True,bins=120, label='train')
sns.distplot(test[features].mean(axis=1),color="blue", kde=True,bins=120, label='test')
plt.legend()
plt.show()


# Let's check the distribution of the mean values per columns in the train and test set.


plt.figure(figsize=(16,6))
plt.title("Distribution of mean values per column in the train and test set")
sns.distplot(train[features].mean(axis=0),color="magenta",kde=True,bins=120, label='train')
sns.distplot(test[features].mean(axis=0),color="darkblue", kde=True,bins=120, label='test')
plt.legend()
plt.show()


# Let's show the distribution of standard deviation of values per row for train and test datasets.


plt.figure(figsize=(16,6))
plt.title("Distribution of std values per row in the train and test set")
sns.distplot(train[features].std(axis=1),color="black", kde=True,bins=120, label='train')
sns.distplot(test[features].std(axis=1),color="red", kde=True,bins=120, label='test')
plt.legend()
plt.show()


# Let's check the distribution of the standard deviation of values per columns in the train and test datasets.
# 


plt.figure(figsize=(16,6))
plt.title("Distribution of std values per column in the train and test set")
sns.distplot(train[features].std(axis=0),color="blue",kde=True,bins=120, label='train')
sns.distplot(test[features].std(axis=0),color="green", kde=True,bins=120, label='test')
plt.legend(); plt.show()


# Let's check now the distribution of the mean value per row in the train dataset, grouped by value of target.
# 


t0 = train.loc[train['target'] == 0]
t1 = train.loc[train['target'] == 1]
plt.figure(figsize=(16,6))
plt.title("Distribution of mean values per row in the train set by target")
sns.distplot(t0[features].mean(axis=1),color="red", kde=True,bins=120, label='target = 0')
sns.distplot(t1[features].mean(axis=1),color="blue", kde=True,bins=120, label='target = 1')
plt.legend()
plt.show()


# Let's check now the distribution of the mean value per column in the train dataset, grouped by value of target.
# 


plt.figure(figsize=(16,6))
plt.title("Distribution of mean values per column in the train set")
sns.distplot(t0[features].mean(axis=0),color="green", kde=True,bins=120, label='target = 0')
sns.distplot(t1[features].mean(axis=0),color="darkblue", kde=True,bins=120, label='target = 1')
plt.legend()
plt.show()


# Observations:
# 1. By rows mean and std change slightly small, than by columns.
# 2. Data is normally distributed.


# Correlation


# The least correlated features


correlations = train[features].corr().abs().unstack().sort_values(kind="quicksort").reset_index()
correlations = correlations[correlations['level_0'] != correlations['level_1']]
correlations.head(10)


# The most correlated features.


correlations.tail(10)


# The correlation between the features is very small.


# Models


target = train['target']
train_value = train.drop(columns = ['target', 'ID_code'])
test_value = test.drop(columns = ['ID_code'])


X_train, X_test, y_train,  y_test = train_test_split(train_value, target, test_size=0.5, random_state=34)


print('Train:',X_train.shape)
print('Test:',X_test.shape)
print('Train:',y_train.shape)
print('Test:',y_test.shape)


lr = LogisticRegression(class_weight='balanced')
lr.fit(X_train, y_train)


lr_pred = lr.predict_proba(X_test)[:,1]


lr_pred


fpr, tpr, thresholds = roc_curve(y_test, lr_pred, pos_label=1)
print('AUC:')
print(auc(fpr, tpr))


# Before to start we standardize our features.


# from sklearn import preprocessing
# scaler = preprocessing.StandardScaler()
# scaler.fit(X)
# X = scaler.transform(X)
# X_test = scaler.transform(X_test)


# Split dataset to train and dev set.




# Logistic Regression












from sklearn.metrics import auc, roc_curve
fpr, tpr, thresholds = roc_curve(y_test, lr_pred, pos_label=1)
print("AUC: ")
print(auc(fpr, tpr))


lr_pred_test = lr.predict_proba(test_inp)[:,1]
submit = test[['ID_code']]
submit['target'] = lr_pred_test
submit.head()


submit.to_csv('lr.csv', index = False)


# RandomForest


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100, class_weight='balanced')


rfc.fit(X_train, y_train)


rfc_pred = rfc.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, rfc_pred, pos_label=1)
print("AUC: ")
print(auc(fpr, tpr))


# Decision Tree


from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(class_weight='balanced',max_depth=5)


tree.fit(X_train, y_train)


tree_pred = tree.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, tree_pred, pos_label=1)
print("AUC: ")
print(auc(fpr, tpr))






# GaussianNB


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train,y_train)


nb_pred


nb_pred = nb.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, nb_pred, pos_label=1)
print("AUC: ")
print(auc(fpr, tpr))
nb_pred


nb_pred_test = nb.predict_proba(test_inp)[:,1]
submit = test[['ID_code']]
submit['target'] = nb_pred_test
submit.head()


submit.to_csv('NB.csv', index = False)


# XGBoost


from sklearn.utils.testing import ignore_warnings
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from xgboost import XGBClassifier


xgb = XGBClassifier(max_depth=8,random_state=0)
xgb.fit(X_train, y_train)


xgb_pred = xgb.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, xgb_pred, pos_label=1)
print("AUC: ")
print(auc(fpr, tpr))







