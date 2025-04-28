import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


test = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')


dataset = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')


dataset.head()


dataset.info()


dataset.isna().sum()


dataset.nunique()


dataset.dropna(inplace=True)


X = dataset.iloc[:, 2: -1].values
y = dataset.iloc[:, 1].values


X[1]


 y[1]


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state = 1,stratify =y)


from sklearn.linear_model import LogisticRegression
from sklearn import metrics


logreg = LogisticRegression()


logreg.fit(X_train, y_train)


X_test_dataset =test.iloc[:,2:]


y_pred = logreg.predict(X_test_dataset)


print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


y_pred


submission_rfc = pd.DataFrame({
        "ID_code": test["ID_code"],
        "target": y_pred
    })


submission_rfc.to_csv('submission_rfc.csv', index=False)


from sklearn.naive_bayes import GaussianNB


gnb = GaussianNB()


gnb.fit(X_train, y_train)


y_pred = gnb.predict(X_test)


from sklearn import metrics


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


from sklearn.metrics import classification_report, confusion_matrix


print(confusion_matrix(y_test, y_pred))


print("Accuracy:",metrics.classification_report(y_test, y_pred))


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


from sklearn.naive_bayes import MultinomialNB


from sklearn.model_selection import train_test_split


mnb = MultinomialNB(alpha=0.01)


from sklearn.preprocessing import Normalizer


normalizer = Normalizer(norm='l2', copy=True)


X_train = Normalizer(copy=False).fit_transform(X_train)


X_train


from sklearn.preprocessing import LabelEncoder


from sklearn.preprocessing import OneHotEncoder


yNB_pred = gnb.predict(X_test_dataset)


submission_rfc = pd.DataFrame({
        "ID_code": test["ID_code"],
        "target": yNB_pred
    })


submission_rfc.to_csv('NB_NazarbekovaB.csv', index=False)


import os


os.environ['KMP_DUPLICATE_LIB_OK']='True'


import xgboost as xgb
import pandas as pd


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state = 1,stratify =y)


xg_cl = xgb.XGBClassifier(objective = 'binary:logistic', n_estimators = 10, seed=123 )


xg_cl.fit(X_train, y_train)


y_pred = xg_cl.predict(X_test_dataset)


submission_rfc = pd.DataFrame({
        "ID_code": test["ID_code"],
        "target": y_pred
    })


submission_rfc.to_csv('XG_NazarbekovaB.csv', index=False)


from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.tree import export_graphviz


clf = DecisionTreeClassifier()


clf = clf.fit(X_train,y_train)


y_pred = clf.predict(X_test)


y_pred

