# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn import svm, datasets,metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, auc, confusion_matrix, classification_report
  


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


import pandas as pd
sample_submission = pd.read_csv("../input/santander-customer-transaction-prediction/sample_submission.csv")
test = pd.read_csv("../input/santander-customer-transaction-prediction/test.csv")
train = pd.read_csv("../input/santander-customer-transaction-prediction/train.csv")




test.head()


train.head()


test.nunique()


 pd.DataFrame.drop(train,columns=['ID_code'],axis=1,inplace=True)


X = train.iloc[:,1:]
y = train.iloc[:,0]




X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


print('Train share:',X_train.shape, '\nTest shape:',X_test.shape)


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


# Logistic Regression


reg=LogisticRegression()
reg.fit(X,y)


y_pred = reg.predict(X_test)


# Accuracy is equl 90%


accuracy_score(y_test,y_pred)


print(classification_report(y_test,y_pred))


cm=confusion_matrix(y_test, y_pred)
cm_sum = np.sum(cm, axis=1, keepdims=True)
cm_perc = cm / cm_sum.astype(float) * 100
annot = np.empty_like(cm).astype(str)
nrows, ncols = cm.shape
for i in range(nrows):
    for j in range(ncols):
        c = cm[i, j]
        p = cm_perc[i, j]
        if i == j:
            s = cm_sum[i]
            annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
        elif c == 0:
            annot[i, j] = ''
        else:
            annot[i, j] = '%.1f%%\n%d' % (p, c)

cm = pd.DataFrame(cm, index=np.unique(y_test), columns=np.unique(y_test))
cm.index.name = 'Actual'
cm.columns.name = 'Predicted'

fig, ax = plt.subplots(figsize=[5,2])

sns.heatmap(cm, cmap= "YlGnBu", annot= annot, fmt='', ax=ax)


# SVM


X_train_res,X_test_res,y_train_res,y_test_res=train_test_split(X_test,y_test,test_size=0.2,random_state=42)


# poly = SVC(kernel='poly', gamma='auto')
# poly.fit(X,y)
# X_train_res,X_test_res,y_train_res,y_test_res=train_test_split(X_res,y_res,test_size=0.2,random_state=0,stratify=y_res)
# poly.fit(X_train_res,y_train_res)
# y_pred = poly.predict(X_test)
# accuracy_score(y_pred, y_test)
# def make_meshgrid(x, y, h=.02):
#     x_min, x_max = x.min() - 1, x.max() + 1
#     y_min, y_max = y.min() - 1, y.max() + 1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#     return xx, yy

# def plot_contours(ax, clf, xx, yy, **params):
#     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
#     out = ax.contourf(xx, yy, Z, **params)
#     return out

# model = svm.SVC(kernel='poly').fit(X,y)


# fig, ax = plt.subplots()

# title = ('Decision surface of POLY SVC ')

# X0, X1 = X[:, 0], X[:, 1]
# xx, yy = make_meshgrid(X0, X1)

# plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
# ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
# ax.set_ylabel('y label here')
# ax.set_xlabel('x label here')
# ax.set_xticks(())
# ax.set_yticks(())
# ax.set_title(title)
# ax.legend()
# plt.show()


# Naive Bayes


naive  = GaussianNB()
naive.fit(X_train,y_train)


y_pred_res = naive.predict(X_test)


cm=confusion_matrix(y_test, y_pred_res)
cm_sum = np.sum(cm, axis=1, keepdims=True)
cm_perc = cm / cm_sum.astype(float) * 100
annot = np.empty_like(cm).astype(str)
nrows, ncols = cm.shape
for i in range(nrows):
    for j in range(ncols):
        c = cm[i, j]
        p = cm_perc[i, j]
        if i == j:
            s = cm_sum[i]
            annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
        elif c == 0:
            annot[i, j] = ''
        else:
            annot[i, j] = '%.1f%%\n%d' % (p, c)

cm = pd.DataFrame(cm, index=np.unique(y_test), columns=np.unique(y_test))
cm.index.name = 'Actual'
cm.columns.name = 'Predicted'

fig, ax = plt.subplots(figsize=[5,2])

sns.heatmap(cm, cmap= "YlGnBu", annot= annot, fmt='', ax=ax)


print(classification_report(y_test,y_pred_res))


# Random Tree


est = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
est.fit(X_train, y_train)


y_pred = est.predict(X)


print(accuracy_score(y,y_pred))


cm=confusion_matrix(y, y_pred)
cm_sum = np.sum(cm, axis=1, keepdims=True)
cm_perc = cm / cm_sum.astype(float) * 100
annot = np.empty_like(cm).astype(str)
nrows, ncols = cm.shape
for i in range(nrows):
    for j in range(ncols):
        c = cm[i, j]
        p = cm_perc[i, j]
        if i == j:
            s = cm_sum[i]
            annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
        elif c == 0:
            annot[i, j] = ''
        else:
            annot[i, j] = '%.1f%%\n%d' % (p, c)

cm = pd.DataFrame(cm, index=np.unique(y_test), columns=np.unique(y_test))
cm.index.name = 'Actual'
cm.columns.name = 'Predicted'

fig, ax = plt.subplots(figsize=[5,2])

sns.heatmap(cm, cmap= "YlGnBu", annot= annot, fmt='', ax=ax)


print(classification_report(y, y_pred))


# clf=RandomForestClassifier(n_estimators=100)
# clf.fit(X_train,y_train)


nb_pred_test = nb.predict_proba(test_inp)[:,1]
submit = test[['ID_code']]
submit['target'] = nb_pred_test
submit.head()


nb_pred = naive.predict_proba(X_test)[:, 1]
performance(y_test, nb_pred)


def performance(Y_test, logist_pred):
    logist_pred_var = [0 if i < 0.5 else 1 for i in logist_pred]
    fpr, tpr, thresholds = roc_curve(Y_test, logist_pred, pos_label=1)
    print('AUC:')
    print(auc(fpr, tpr))



