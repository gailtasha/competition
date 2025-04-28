# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


dataset = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')
testset = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')


dataset.sample(10)


dataset.info()


dataset.isna().sum()


dataset.nunique()


dataset.dropna(inplace=True)


X = dataset.iloc[:, 2:].values
y = np.floor(dataset.iloc[:, 1].values)


X[1]


y[1]


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, random_state = 80,stratify =y)


from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix


target_names = ['0 chance', '1 chance']


print(classification_report(y_test, y_preds, target_names=target_names))


from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# ![](http://)Logistic Regression


logreg = LogisticRegression(max_iter=100000)
logreg.fit(X_train, y_train)


# test = testset.iloc[:, 1:].values
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
result = pd.DataFrame({"target" : np.array(y_pred).T})
result['target'].value_counts()


print(classification_report(y_test, y_pred, target_names=target_names))


X_train, X_test, y_train, y_test = train_test_split(X,y,train_size = 199998, random_state = 80,stratify =y)


logreg = LogisticRegression(max_iter=100000)
logreg.fit(X_train, y_train)


test = testset.iloc[:, 1:].values
y_pred = logreg.predict(test)


sub = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/sample_submission.csv')
sub['target'] = y_pred
sub.to_csv('logregFi.csv', index=False)




# > Naive Bayes
# > 


from sklearn.naive_bayes import GaussianNB


gnb = GaussianNB()


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, random_state = 80,stratify =y)


gnb.fit(X_train, y_train)


y_pred = gnb.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


print(classification_report(y_test, y_pred, target_names=target_names))


y_pred = gnb.predict(X_test)
print('Accuracy of gnb classifier on test set: {:.2f}'.format(gnb.score(X_test, y_test)))
result = pd.DataFrame({"target" : np.array(y_pred).T})
result['target'].value_counts()


X_train, X_test, y_train, y_test = train_test_split(X,y,train_size = 199998, random_state = 80,stratify =y)


test = testset.iloc[:, 1:].values

y_pred = gnb.predict(test)



sub = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/sample_submission.csv')
sub['target'] = y_pred
sub.to_csv('gaussianF.csv', index=False)


# BNB


from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, random_state = 80,stratify =y)


bnb = BernoulliNB(binarize=0.0)


 bnb.fit(X_train, y_train)


 bnb.score(X_test, y_test)


y_pred = bnb.predict(X_test)


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))



print('Accuracy of bnb classifier on test set: {:.2f}'.format(bnb.score(X_test, y_test)))
result = pd.DataFrame({"target" : np.array(y_pred).T})
result['target'].value_counts()


X_train, X_test, y_train, y_test = train_test_split(X,y,train_size = 199998, random_state = 80,stratify =y)


 bnb.fit(X_train, y_train)


y_pred = bnb.predict(test)


result = pd.DataFrame({"target" : np.array(y_pred).T})
# train_ds['target'].value_counts()
result['target'].value_counts()


sub = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/sample_submission.csv')
sub['target'] = y_pred
sub.to_csv('bernulliF.csv', index=False)


# DTress and RF


from sklearn.tree import DecisionTreeClassifier


from sklearn import metrics 


from sklearn.tree import export_graphviz 


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, random_state = 80,stratify =y)


clf = DecisionTreeClassifier()


clf = clf.fit(X_train,y_train)


y_pred = clf.predict(X_test)


print('Accuracy of clf classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))
result = pd.DataFrame({"target" : np.array(y_pred).T})
result['target'].value_counts()


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


X_train, X_test, y_train, y_test = train_test_split(X,y,train_size = 199998, random_state = 80,stratify =y)


clf = clf.fit(X_train,y_train)


y_pred = clf.predict(test)


result = pd.DataFrame({"target" : np.array(y_pred).T})
# train_ds['target'].value_counts()
result['target'].value_counts()


sub = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/sample_submission.csv')
sub['target'] = y_pred
sub.to_csv('decissionTreeFi.csv', index=False)


# XGboost


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


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, random_state = 10,stratify =y)


xg_cl.fit(X_train, y_train)


y_pred = xg_cl.predict(X_test)


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))



print('Accuracy of XGBOOST classifier on test set: {:.2f}'.format(xg_cl.score(X_test, y_test)))
result = pd.DataFrame({"target" : np.array(y_pred).T})
result['target'].value_counts()


print(classification_report(y_test, y_pred, target_names=target_names))


dataset_dmatrix = xgb.DMatrix(data = X,label = y)
dataset_dmatrix


params = {"objective":"reg:logistic", "max_depth":3}
params


# Perform cross-validation: cv_results
cv_results = xgb.cv(dtrain = dataset_dmatrix, params = params, num_boost_round = 5, nfold = 3, metrics = "rmse", as_pandas = True, seed = 123)


print(cv_results)


print(1-cv_results["test-rmse-mean"].tail(1))


cv_results = xgb.cv(dtrain = dataset_dmatrix, params = params, num_boost_round = 5, nfold = 3, metrics = "auc", as_pandas = True, seed = 123)


print(cv_results["test-auc-mean"].tail(1))


X_train, X_test, y_train, y_test = train_test_split(X,y,train_size = 199998, random_state = 80,stratify =y)


xg_cl.fit(X_train, y_train)


y_pred = xg_cl.predict(test)


result = pd.DataFrame({"target" : np.array(y_pred).T})
# train_ds['target'].value_counts()
result['target'].value_counts()


sub = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/sample_submission.csv')
sub['target'] = y_pred
sub.to_csv('XGBOOSTv2.csv', index=False)

