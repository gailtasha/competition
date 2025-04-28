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


tests = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')
trains = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')
subms = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/sample_submission.csv')


import matplotlib.pyplot as plt
import seaborn as sns


trains


plt.figure(figsize=(7, 5))
plt.style.use('ggplot')
plt.hist(trains.target)
plt.show()


desc = trains.describe()
desc


means = list(desc[desc.target == 0.100490].values)[0][1:]
means = list(means)
means.index(max(means))


desc.var_120


max_mean_col = trains.var_120
plt.figure(figsize=(7, 5))
plt.hist(max_mean_col)
plt.show()


# Histogram of outliner of maximum


plt.figure(figsize=(7, 5))
plt.hist(max_mean_col, cumulative=True, density=True, bins=25)
plt.show()


# CDF of maximum


from scipy.stats import norm


plt.plot(max_mean_col, norm.pdf(max_mean_col))
plt.show()


# PDF of maximum


means.index(min(means))


min_mean_col = trains.var_120
plt.figure(figsize=(7, 5))
plt.hist(min_mean_col)
plt.show()


plt.figure(figsize=(7, 5))
plt.hist(min_mean_col, cumulative=True, density=True, bins=25)
plt.show()


plt.plot(min_mean_col, norm.pdf(min_mean_col))
plt.show()


np.corrcoef(max_mean_col, min_mean_col)


plt.plot(max_mean_col, min_mean_col)
plt.show()


# Correlation of maximum mean and minimum mean value


x = trains.drop(columns='var_120')
x = x.drop(columns=['target', 'ID_code'])


x


y = trains.target


from sklearn.model_selection import train_test_split


X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# <h1>Log Reg</h1>


from sklearn.linear_model import LogisticRegression


X_test


tests


log_reg=LogisticRegression()
log_reg.fit(X_train,Y_train)
y_pred=log_reg.predict(X_test)


import sklearn
sklearn.metrics.accuracy_score(Y_test,y_pred)


from sklearn.metrics import classification_report, confusion_matrix


confusion_matrix(y,log_reg.predict(x))


print(classification_report(y,log_reg.predict(x)))


import matplotlib.pyplot as plt
import seaborn as sn


cm=confusion_matrix(Y_test,y_pred)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sn.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")


# <h5>AUC ROC of LOG REG


from sklearn.metrics import roc_auc_score


roc_auc_score(Y_test,y_pred)


pred = log_reg.predict(tests.drop(columns=['ID_code','var_120']))


submit_log_reg = pd.DataFrame({'ID_code': tests['ID_code'], 'target': pred})
submit_log_reg.to_csv('submit_log_reg.csv', index=False)


# <h1>SVM</h1>


# SVM loads too long


# <h1>Naive Bayes


from sklearn.naive_bayes import GaussianNB
from sklearn import metrics


gnb = GaussianNB()


y_pred = gnb.fit(X_train, Y_train).predict(X_test)


metrics.accuracy_score(y,gnb.predict(x))


confusion_matrix(y,gnb.predict(x))


print(classification_report(y,gnb.predict(x)))


# <h5>AUC ROC for NB 


roc_auc_score(Y_test, y_pred)


pred = gnb.predict(tests.drop(columns=['ID_code','var_120']))


submit_nb = pd.DataFrame({'ID_code': tests['ID_code'], 'target': pred})
submit_nb.to_csv('submit_nb.csv', index=False)


# # Decision Tree and Random Forest


# <h3>DT


from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


dt_class = DecisionTreeClassifier(criterion = "entropy", max_depth=5, min_samples_split = 10 )


dt_class.fit(X_train, Y_train)


print(classification_report(y,dt_class.predict(x)))


# <h5>AUC ROC for DT


pred = dt_class.predict(tests.drop(columns=['ID_code','var_120']))


submit_dt = pd.DataFrame({'ID_code': tests['ID_code'], 'target': pred})
submit_dt.to_csv('submit_dt.csv', index=False)


roc_auc_score(y, dt_class.predict(x))


# <h3> RF


rf_class = RandomForestClassifier()


rf_class.fit(X_train, Y_train)


from sklearn.metrics import accuracy_score


prediction = rf_class.predict(X_test)
accuracy_score(Y_test,prediction)


print(classification_report(y,rf_class.predict(x)))


pred = rf_class.predict(tests.drop(columns=['ID_code','var_120']))


submit_rf = pd.DataFrame({'ID_code': tests['ID_code'], 'target': pred})
submit_rf.to_csv('submit_rf.csv', index=False)


# <h5>AUC ROC for RF


roc_auc_score(Y_test, rf_class.predict(y))


# # XGBOOST


import xgboost as xx


x_cls = xx.XGBClassifier()


x_cls = x_cls.fit(X_train, Y_train)


y_pred = x_cls.predict(X_test)


pred = x_cls.predict(tests.drop(columns=['ID_code','var_120']))


submit_xg = pd.DataFrame({'ID_code': tests['ID_code'], 'target': pred})
submit_xg.to_csv('submit_xg.csv', index=False)


# <h5>AUC ROC for XGBOOST


roc_auc_score(Y_test, y_pred)


# In conclusion I want to say that there are very low scores like in range of 0.5-0.7. I removed the column 'var_120' because it was the outliner. I couldn't make SVC, Random Forest and XGBOOST, because they runs too long, and I didn't have enough time. But I write codes of predictions, I think if you have enough time you can run and check it. I'm sorry.

