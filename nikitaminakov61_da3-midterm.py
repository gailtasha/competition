# # Machine Learning - 2😍😍
# 
# ## Minakov Nikita / CSSE1707-DA3/ 2020
# ## Midterm 


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


# Importing datasets


import pandas as pd
sample_submission = pd.read_csv("../input/santander-customer-transaction-prediction/sample_submission.csv")
test = pd.read_csv("../input/santander-customer-transaction-prediction/test.csv")
train = pd.read_csv("../input/santander-customer-transaction-prediction/train.csv")


train.head(10)


test.head(10)


# Distribution Analysis


train.info()


train.describe()


train.corr()


sns.countplot(x='target', data=train)


sns.distplot(train.var_0) 
sns.distplot(train.var_10) 
sns.distplot(train.var_20) 
sns.distplot(train.var_30) 
sns.distplot(train.var_40) 
sns.distplot(train.var_50) 


fig, ax = plt.subplots(ncols=3, figsize=(20, 4))

sns.distplot(train['var_0'], ax=ax[0], color='red')
sns.distplot(train['var_1'], ax=ax[1], color='green')
sns.distplot(train['var_3'], ax=ax[2], color='blue')

plt.show()


# The distributions "look like" gaussian, so a possible preprocessing can be just the normalization of those features.


print(train.shape, test.shape)


print (train.isna().sum())
print (train.isnull().sum())


# Is it possible to notice that the dataset is clearly imbalanced. Without the proper adjustements some of machine learning models could be biases toward the classification of 0 respect to 1.
# In addiction to that, a proper consideration has to be done on the choice of evaluation metrics since the accuracy is not completely reliable. 




X, y = train.iloc[:,2:], train.iloc[:,1]


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state = 123, stratify = y)


# ### KNN


# To many time lost for compiling, so i commebted KNN


from sklearn.neighbors import KNeighborsClassifier


#knn = KNeighborsClassifier(7)


#knn.fit(X_train, y_train)


#X_test1 = np.nan_to_num(X_test)


#y_preds = knn.predict(X_test1)


from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix

#accuracy_score(y_test, y_preds)


#confusion_matrix(y_test,y_preds)


#print(classification_report(y_test, y_preds))


# ## Linear regression


# Above there is a recap on Bayesian Linear regression related formulas.
# The method names are as much as possible coherent with their meaning in the context of the implementation of the formulas.


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


logreg=LogisticRegression()
logreg.fit(X_train,y_train)


from sklearn.metrics import confusion_matrix
y_pred = logreg.predict(X_test)
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


# Using Root Mean Squared Logarithmic Error (RMSLE) evaluation function


def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

train_pred = logreg.predict(X_train)
print('RMSLE : {:.4f}'.format(rmsle(y_train, train_pred)))


# ## SVM


from sklearn.svm import SVC
#svclassifier.fit(X_train, y_train)


#y_pred = svclassifier.predict(X_test)


from sklearn.metrics import classification_report, confusion_matrix
#print(confusion_matrix(y_test, y_pred))
#print(classification_report(y_test, y_pred))


#svc_model = SVC()


#svc_model.fit(X_train, y_train)




# ## Naive Bayes


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


confusion_matrix(y, y_preds_res)


print(classification_report(y, y_preds_res))


submission_nb = pd.DataFrame({
    "ID_code": test["ID_code"],
    "target": y_preds_res
})
submission_nb.to_csv('naive_baise_submission.csv', index=False)


# ## Random tree


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


# ## Random forest


# Random forest algorithm creates decision trees on data samples and then gets the prediction from each of them and finally selects the best solution by means of voting.


np.random.seed(13)


y_train = train['target']
x_train = train.drop(columns=['ID_code', 'target'])

y_train = y_train.astype('int8')
x_train = x_train.astype('float16')


from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from imblearn.ensemble import *
clf = BalancedRandomForestClassifier(n_estimators=500, 
                                     criterion='entropy', 
                                     n_jobs=-1)


clf.fit(x_train, y_train)


y_pred = clf.predict(x_train)


print('Score:', clf.score(x_train, y_train))


print('B Score:', metrics.balanced_accuracy_score(y_train, y_pred))



print('AUC Score:', metrics.roc_auc_score(y_train, y_pred))


confusion_matrix(y_train, y_pred)


print(classification_report(y_train, y_pred))


id_code = test.pop('ID_code')
test = test.astype('float16')
targets = clf.predict(test)
output = pd.DataFrame({'ID_code': id_code.values, 'target': targets})
output.to_csv('forest_submission.csv', index=False)


# We got 1.00 on '0''s precision and '1''s recall. Precision on '1''s is very little for good predictions.


# ## XgBoost


import pandas as pd
sample_submission = pd.read_csv("../input/santander-customer-transaction-prediction/sample_submission.csv")
test = pd.read_csv("../input/santander-customer-transaction-prediction/test.csv")
train = pd.read_csv("../input/santander-customer-transaction-prediction/train.csv")


# import data again, beacause of their editting in previous steps


train_id = train['ID_code']
y_train = train['target']
X_train = train.drop(['ID_code', 'target'], axis=1, inplace = False)

test_id = test['ID_code']
X_test = test.drop('ID_code', axis=1, inplace = False)


model_xgb = xgb.XGBRegressor(n_estimators=5, max_depth=4, learning_rate=0.5) 
model_xgb.fit(X_train, y_train)


# Using Root Mean Squared Logarithmic Error (RMSLE) evaluation function


def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

train_pred = model_xgb.predict(X_train)
print('RMSLE : {:.4f}'.format(rmsle(y_train, train_pred)))


xgb_preds = model_xgb.predict(X_test)
solution = pd.DataFrame({"ID_code":test_id, "target":xgb_preds})
solution.to_csv("xgb_submission.csv", index = False)


solution.head(100)


# During this milestone, I focused on Xgboost and Random forest, and tried to get a good result when working with these algorithms. In the case of Forest, I managed to achieve a 100% hit for some parameters with precisison and recall, but at the same time, the other parammeters sank significantly. But despite this, it turned out to achieve a fairly good result. Knn is poorly optimized and takes a lot of time to compile, which is why I had to disable it. The Xgboost method seemed to me the most optimal (perhaps because I liked it the most), although it was not possible to achieve the maximum score when using it.

