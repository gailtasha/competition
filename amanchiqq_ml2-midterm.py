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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline


submission = pd.read_csv('../input/santander-customer-transaction-prediction/sample_submission.csv')
train = pd.read_csv('../input/santander-customer-transaction-prediction/train.csv')
test = pd.read_csv('../input/santander-customer-transaction-prediction/test.csv')


submission.info()


train.info()


test.info()


test.columns


test.describe()


test.head(11)


test.corr()


print('Test: ', test.shape)
print('Train: ', train.shape)


import seaborn as sns
sns.countplot(train.target)


X, y = train.iloc[:,2:], train.iloc[:,1]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state = 123, stratify = y)


# # Logistic Regression


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


logreg=LogisticRegression()
logreg.fit(X_train,y_train)


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


from sklearn.metrics import mean_squared_error

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

train_pred = logreg.predict(X_train)
print('RMSLE : {:.4f}'.format(rmsle(y_train, train_pred)))


# # SVM


train_samp = train.sample(1000)
train_samp.head()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, train['target'], test_size = 0.20)


# # Naive Buyer


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)


gnb = GaussianNB()


y_pred = gnb.fit(X_train, y_train).predict(X_test)


print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))


from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report
print(classification_report(y_test, y_pred, target_names=['Very Good','Ideal']))


accuracy_score(y_test, y_pred)


confusion_matrix(y_test, y_pred)


precision_score(y_test, y_pred)


recall_score(y_test, y_pred)


# # Decision tree


from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(class_weight='balanced',max_depth=4)


tree.fit(X_train, y_train)


from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report
print(classification_report(y_test, y_pred, target_names=['1','0']))


def performance(Y_test, logist_pred):
    logist_pred_var = [0 if i < 0.5 else 1 for i in logist_pred]
    fpr, tpr, thresholds = roc_curve(Y_test, logist_pred, pos_label=1)
    print('AUC:')
    print(auc(fpr, tpr))


from sklearn.metrics import roc_curve, roc_auc_score, auc, confusion_matrix, classification_report


tree_pred = tree.predict_proba(X_test)[:, 1]
performance(y_test, tree_pred)


precision_score(y_test, y_pred)


recall_score(y_test, y_pred)


roc_value = roc_auc_score(y_test, y_pred)
roc_value


# # Random Forest


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, 
                               bootstrap = True,
                               max_features = 'sqrt')


y_pred = model.fit(X_train, y_train).predict(X_test)


precision_score(y_test, y_pred)


recall_score(y_test, y_pred)


from sklearn.metrics import roc_auc_score

roc_value = roc_auc_score(y_test, y_pred)
roc_value


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, y_pred)
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.figure()
plt.plot(fpr, tpr, label='Random forest')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()


# # XGboost


from sklearn.utils.testing import ignore_warnings
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from xgboost import XGBClassifier


xgb = XGBClassifier(max_depth=8,random_state=0)


xgb.fit(X_train, y_train)


xgb_pred = xgb.predict_proba(X_test)[:, 1]
performance(y_test, xgb_pred)



