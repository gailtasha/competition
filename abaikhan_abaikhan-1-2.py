# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


df=pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')
test_df = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')
df.head(5)


test = test_df.iloc[:, 1:].values


df.info()


    X = df.iloc[:, 2:].values
    y = df.iloc[:, 1].values
    y


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.200, random_state = 1,stratify =y)


from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix


# **Logistic Regression******


from sklearn.linear_model import LogisticRegression
from sklearn import metrics


logreg = LogisticRegression()
logreg.fit(X_train, y_train)


y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


y_pred = logreg.predict(test)


sub = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/sample_submission.csv')
sub['target'] = y_pred
sub.to_csv('logreg1.csv', index=False)


# LOGREG finished


# Naive Bayes


from sklearn.naive_bayes import GaussianNB
from sklearn import metrics


gnb = GaussianNB()
gnb.fit(X_train, y_train)


y_pred = gnb.predict(X_test)


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))


print("Accuracy:",metrics.classification_report(y_test, y_pred))


y_pred = gnb.predict(test)


sub = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/sample_submission.csv')
sub['target'] = y_pred
sub.to_csv('gaussian.csv', index=False)


from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split


bnb = BernoulliNB(binarize=0.0)
bnb.fit(X_train, y_train)


 bnb.score(X_test, y_test)


y_pred = bnb.predict(X_test)


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))


print("Accuracy:",metrics.classification_report(y_test, y_pred))


y_pred = bnb.predict(test)


sub = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/sample_submission.csv')
sub['target'] = y_pred
sub.to_csv('Bernollin.csv', index=False)


# DTrees and RF


import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.tree import DecisionTreeClassifier


clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


y_pred = clf.predict(test)


sub = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/sample_submission.csv')
sub['target'] = y_pred
sub.to_csv('DTrees.csv', index=False)


# XGBOOST


import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import xgboost as xgb
import pandas as pd


xg_cl = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=1,
              learning_rate=0.1, max_delta_step=0, max_depth=8,
              min_child_weight=1, missing=None, n_estimators=100, n_jobs=1,
              nthread=None, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=1)


xg_cl.fit(X_train, y_train)


y_pred = xg_cl.predict(X_test)


accuracy = float(np.sum(y_pred==y_test))/y_test.shape[0]
print("accuracy: %f" % (accuracy))


y_pred = xg_cl.predict(test)


sub = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/sample_submission.csv')
sub['target'] = y_pred
sub.to_csv('xgboost1.csv', index=False)

