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


test = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')
test


dataset = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')
dataset


dataset.sample(10)


dataset.info()


dataset.isna().sum()


dataset.nunique()


dataset.dropna(inplace=True)


X = dataset.iloc[:, 2:-1].values
y = dataset.iloc[:, 1].values


y[1]


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state = 1,stratify =y)


from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


# KNN


knn = KNeighborsClassifier(51)


X_test[1:2]


knn.fit(X_train, y_train)


y_preds = knn.predict(X_test)


yKnn_preds = knn.predict(X_test_dataset)


sumbission_rfc = pd.DataFrame({
    "ID_code": test["ID_code"],
    "target": yKnn_preds
})
sumbission_rfc.to_csv('KossayA_knn.csv', index=False)


# <H2><B>Logistic Regression</B></H2>


from sklearn.linear_model import LogisticRegression
from sklearn import metrics


logreg = LogisticRegression()
logreg.fit(X_train, y_train)


X_test_dataset = test.iloc[:,2:]


y_pred = logreg.predict(X_test_dataset)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


y_pred


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


sumbission_rfc = pd.DataFrame({
    "ID_code": test["ID_code"],
    "target": y_pred
})
sumbission_rfc.to_csv('KossayA_logreg.csv', index=False)


# <h2><b> Naive Bayes</b></h2>


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


yGNB_pred = gnb.predict(X_test_dataset)
yBNB_pred = bnb.predict(X_test_dataset)


sumbission_rfc = pd.DataFrame({
    "ID_code": test["ID_code"],
    "target": yGNB_pred
})
sumbission_rfc.to_csv('KossayA_GNB.csv', index=False)


sumbission_rfc = pd.DataFrame({
    "ID_code": test["ID_code"],
    "target": yBNB_pred
})
sumbission_rfc.to_csv('KossayA_BNB.csv', index=False)


# <h2><b> DTress and RF</b></h2>


from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics 
from sklearn.tree import export_graphviz 


clf = DecisionTreeClassifier()


clf = clf.fit(X_train,y_train)


y_pred = clf.predict(X_test)


y_pred


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


yDtres_pred = clf.predict(X_test_dataset)


sumbission_rfc = pd.DataFrame({
    "ID_code": test["ID_code"],
    "target": yDtres_pred
})
sumbission_rfc.to_csv('KossayA_Dtress.csv', index=False)


clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)


clf = clf.fit(X_train,y_train)


y_pred = clf.predict(X_test)


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


yEntropy_pred = clf.predict(X_test_dataset)


sumbission_rfc = pd.DataFrame({
    "ID_code": test["ID_code"],
    "target": yEntropy_pred
})
sumbission_rfc.to_csv('KossayA_DtressEntropy.csv', index=False)


from sklearn.ensemble import RandomForestRegressor


regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)


y_pred


from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


from sklearn.ensemble import RandomForestClassifier


clf=RandomForestClassifier(n_estimators=100)


clf.fit(X_train,y_train)


y_pred=clf.predict(X_test)


from sklearn import metrics

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


yRandomForest_pred = clf.predict(X_test_dataset)


sumbission_rfc = pd.DataFrame({
    "ID_code": test["ID_code"],
    "target": yRandomForest_pred
})
sumbission_rfc.to_csv('KossayA_DtressRandomForest.csv', index=False)


# > <h2><b>XGboost</b></h2>


import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


import xgboost as xgb
import pandas as pd


xg_cl = xgb.XGBClassifier(objective = 'binary:logistic', n_estimators = 10, seed=123 )


xg_cl.fit(X_train, y_train)


y_pred = xg_cl.predict(X_test)
y_pred


X_test


X_test_dataset = test.iloc[:,2:]


X_test


test['ID_code']


test_dataset = test


test_dataset = test[:200000]
test_dataset['ID_code']


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state = 2,stratify =y)


X_test_datatset = test.iloc[:,2:]
X_test_datatset


xRow = X_test_dataset['var_1']
xRow


X_test


len(X_test)


y_pred = xg_cl.predict(X)
y_pred


len(y_pred)


sumbission_rfc = pd.DataFrame({
    "ID_code": test_dataset["ID_code"],
    "target": y_pred
})
sumbission_rfc.to_csv('XGBOOST_KossayA.csv', index=False)


import numpy as np
accuracy = float(np.sum(y_pred==y_test))/y_test.shape[0]
print("accuracy: %f" % (accuracy))


dataset_dmatrix = xgb.DMatrix(data = X,label = y)
dataset_dmatrix


params = {"objective":"reg:logistic", "max_depth":3}
params


cv_results = xgb.cv(dtrain = dataset_dmatrix, params = params, num_boost_round = 5, nfold = 3, metrics = "rmse", as_pandas = True, seed = 123)


print(cv_results)


print(1-cv_results["test-rmse-mean"].tail(1))


print(cv_results)


print(cv_results["test-auc-mean"].tail(1))


# *<h2><b>SVM</b></h2>*



