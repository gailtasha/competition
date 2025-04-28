# # DA-4 Midterm
# ## Author: Brankov Ravil CSSE-1707DA-3
# This work is concerned on prediction of completing transaction by customer. We should use several method of classification to reach the maximum ROG AUC score. AUC score is measure of prediction the FP/TP/TN/FN cases, which are made after fitting the model on test part of dataset. Let's go deeper. 


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


# The key libraries is SciKit Learn which contain main classification tools.


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, auc, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
%matplotlib inline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# The dataset is already splitted


train = pd.read_csv("../input/santander-customer-transaction-prediction/train.csv")
test = pd.read_csv("../input/santander-customer-transaction-prediction/test.csv")


# # EDA
# Now, we must provide the analysis of our data, for better understanding the classification process.


train.head(10)


test.head(10)


# As we can see, all columns are anonymous, so it will be hard to preprocess it.


train.shape, test.shape


train.isna().sum()


# With 200000 records we have no one NaN value. Good!


sns.countplot(x=target,data=train, palette='hls')
plt.show()


train = train.drop_duplicates()
train.count()


# So, we have no duplicate data. Correlations we can see below


plt.figure(figsize=(20,10))
c= train.corr()
sns.heatmap(c,cmap="BrBG",annot=True)
c


# # Logistic Regression
# Logistic regression using the linear function to divide the classes


target = train['target']
df_train = train.drop(columns = ['target', 'ID_code'])
df_test = test.drop(columns = ['ID_code'])


X_train, X_test, y_train,  y_test = train_test_split(df_train, target,test_size=0.2, random_state=142)


X_train.shape, X_test.shape, y_train.shape,y_test.shape


logs = LogisticRegression(class_weight='balanced')
logs.fit(X_train, y_train)


from sklearn import metrics
fpr, tpr, _ = metrics.roc_curve(y_test, logs.predict_proba(X_test)[:,1])
auc = metrics.auc(fpr, tpr)
auc


logs_pred_test = logs.predict_proba(df_test)[:,1]
submit = test[['ID_code']]
submit['target'] = logs_pred_test
submit.head()


submit.to_csv('log_reg.csv', index = False)


# # Decision Tree
# Decision tree is based on binary tree and set the class through evaluating parameters and assigning the 0 or 1 for the answer


tree = DecisionTreeClassifier(class_weight='balanced',max_depth=5)


tree.fit(X_train, y_train)


y_pred = tree.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
auc = metrics.auc(fpr, tpr)
auc


# # Random Forest
# Random forest is complete models separately like decision tree manner with random parameter and combine predictions for giving the average or weighted output


model = RandomForestClassifier(n_estimators=100, class_weight='balanced')


model.fit(X_train, y_train)


y_pred= model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
auc = metrics.auc(fpr, tpr)
auc


# # Naive Bayes


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)


y_preds_gnb = gnb.predict(X_test)
fpr, tpr, _ = metrics.roc_curve(y_test, y_preds_gnb)
auc = metrics.auc(fpr, tpr)
auc


test_data = pd.read_csv("../input/santander-customer-transaction-prediction/test.csv")


X_test_data = test_data.iloc[:,1:202]


y_preds_test_data_gnb = gnb.predict(X_test_data)


my_submission_gnb = pd.DataFrame({'ID_code': test_data.ID_code, 'target': y_preds_test_data_gnb})
my_submission_gnb.to_csv('submission_gnb.csv', index=False)


# # XGBoost


from sklearn.utils.testing import ignore_warnings
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from xgboost import XGBClassifier
xgb = XGBClassifier(max_depth=8,random_state=0)


xgb.fit(X_train, y_train)


y_pred = xgb.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
auc = metrics.auc(fpr, tpr)
auc



