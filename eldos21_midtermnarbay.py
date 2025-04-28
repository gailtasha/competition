# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


train = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')
test = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')


train.shape


test.shape


train.info()


train.describe()


train.isnull().sum()


# Target column distribution


sns.countplot(train['target'], palette='Set3')


from sklearn.neighbors import KNeighborsClassifier


X_tr_full = train.iloc[:, 2:].values
y_tr_full = train.target.values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_tr_full,y_tr_full, test_size = 0.2, random_state = 123, stratify = y_tr_full)


X_train.shape, y_train.shape


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(max_iter=300, solver='sag')


logreg.fit(X_train, y_train)


y_pred_log = logreg.predict(X_test)


print('log reg score: {:.2f}'.format(logreg.score(X_test, y_test)))


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_log))


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred_log)
print(confusion_matrix)


# results are normal, train on full set


logreg.fit(X_tr_full, y_tr_full)


X_test_f = test.iloc[:, 1:].values


y_pred_logf = logreg.predict(X_test_f)


sub_df = pd.DataFrame({"ID_code":test["ID_code"].values})
sub_df["target"] = y_pred_logf
sub_df.to_csv("sub_log_narbay.csv", index=False)


# # Gaussian


from sklearn.naive_bayes import GaussianNB


gnb = GaussianNB()
gnb.fit(X_tr_full, y_tr_full)


y_pred_gnb = gnb.predict(X_test_f)


sub_df = pd.DataFrame({"ID_code":test["ID_code"].values})
sub_df["target"] = y_pred_gnb
sub_df.to_csv("sub_gaus_narbay.csv", index=False)


# # XgBoost


import xgboost as xgb


xg_cl = xgb.XGBClassifier(objective = 'binary:logistic', max_depth=7, n_estimators = 100,  seed=123, class_weight= 'balanced' )


xg_cl.fit(X_tr_full, y_tr_full)


y_pred_xgb = xg_cl.predict(X_test_f)


sub_df = pd.DataFrame({"ID_code":test["ID_code"].values})
sub_df["target"] = y_pred_xgb
sub_df.to_csv("sub_xgboost2_narbay.csv", index=False)


# # KNN


knn = KNeighborsClassifier(n_neighbors = 3,leaf_size=15)


knn.fit(X_train,y_train)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 123, stratify = y)


y_preds_knn = knn.predict(X_test)

