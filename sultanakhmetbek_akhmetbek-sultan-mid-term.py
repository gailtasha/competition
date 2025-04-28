# # MIDTERM


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


train_df = pd.read_csv("/kaggle/input/santander-customer-transaction-prediction/train.csv")
test_df = pd.read_csv("/kaggle/input/santander-customer-transaction-prediction/test.csv")


train_df.head()


print(train_df.shape, test_df.shape)


print(train_df.isna().sum().sum(), test_df.isna().sum().sum())


# Both test_df and train_df haven't got any Nan's assigned


# ### Train set description:


train_df.describe()


# ### Test set description:


test_df.describe()


# **According to two cells above, I can say that: **
# 1. standard deviation of train and test columns is large
# 2. values of mean, std, min, max for both data look similar
# 3. also, mean values are flactuated between a large range of values for each column.


sns.set(style="darkgrid")
sns.countplot(train_df['target'])


train_df["target"].value_counts()


# And the values for target column variables, as we see, distinguish in amount
# 
# So we need to implement *data augmentation* on this dataset, in order to avoid imbalanced classification
# 
# By data augmentation I mean: to artificially expand the size of a training dataset, 
# 
# to which we referred in classes as SMOTE (or Synthetic Minority Oversampling Technique)


sns.distplot(train_df[train_df.columns[2:]].mean(), kde=False)


# In above you may notice a histogram
# 
# I plotted distribution of all column means, so to say, in order to observe visually the dataset
# 
# Looking at this plot, I can say, it's pattern is bimodal


train_df.info()


# Overall, we have all float/int values in column, except one column which is ID_code
# 
# So I do not have to do any datattype changes


# I want to evaluate correlation for each column with target values, just to see how much do I need them in the dataset
# 
# So we'll see which ones are irreplaceble (and which one are not)


column_corr = train_df.corr()['target']


column_corr


print(column_corr.sort_values().tail(11))


# I showed last 11 elements, because it includes target values too (of course, target correlates well with itself)
# 
# And so, we have 10 positive parameters that correlate relatively well with target parameter 


print(column_corr.sort_values().head(10))


# These are the parameters that correlate also well with target, but correlate negatively 
# 
# I mean, they are also important for prediction


X = train_df.iloc[:,2:202]
y = train_df.iloc[:,1]


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.fit_transform(X)


# ## SMOTE


from imblearn.over_sampling import SMOTE


# ## LogReg


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(random_state=0,)
logreg.fit(X_train, y_train)


from sklearn.metrics import confusion_matrix

y_pred = logreg.predict(X_test)
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


from sklearn.metrics import roc_auc_score

roc_auc_score(y_test, y_pred)


logreg.score(X_test, y_test)


y_pred = logreg.predict(test_df.drop(columns = ['ID_code']))


from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

proba = logreg.predict_proba(X_test)[:, 1]
score = roc_auc_score(y_test, proba)
fpr, tpr, _  = roc_curve(y_test, proba)

plt.figure()
plt.plot(fpr, tpr, color='c', label=f"ROC curve (auc = {score})")
plt.plot([0, 1], [0, 1], color='m', linestyle='--')
plt.title("Results")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.show()


submission_logreg = pd.DataFrame({ "ID_code": test_df["ID_code"], "target": y_pred })
submission_logreg.to_csv('submission_logreg.csv', index=False)


# ## SVM


# Unfortunately, I didn't have enough strengths of will to wait until SVC fits data, so I just left it
# 


# from sklearn.svm import SVC, LinearSVC


# SVC = SVC()
# LSVC = LinearSVC()


# SVC.fit(X_train,y_train)
# SVC_predict = SVC.predict(X_test)
# print("SVC Accuracy :", accuracy_score(y_test, SVC_predict))


# print(classification_report(y_test, LSVC_predict))


# roc_auc_score(y_test, LSVC_predict)


# ## NB


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


model = GaussianNB()
model.fit(X_train, y_train)
predicted= model.predict(X_test)
print("NBGaussian Accuracy :", accuracy_score(y_test, predicted))


roc_auc_score(y_test, predicted)


proba = model.predict_proba(X_test)[:, 1]
score = roc_auc_score(y_test, proba)
fpr, tpr, _  = roc_curve(y_test, proba)

plt.figure()
plt.plot(fpr, tpr, color='c', label=f"ROC curve (auc = {score})")
plt.plot([0, 1], [0, 1], color='m', linestyle='--')
plt.title("Results")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.show()


# ## DT and RF


from sklearn import tree
Tree = tree.DecisionTreeClassifier()
Tree = Tree.fit(X_train,y_train)


predicted= Tree.predict(X_test)
print("Decision Tree Accuracy :", accuracy_score(y_test, predicted))


roc_auc_score(y_test, predicted)


proba = Tree.predict_proba(X_test)[:, 1]
score = roc_auc_score(y_test, proba)
fpr, tpr, _  = roc_curve(y_test, proba)

plt.figure()
plt.plot(fpr, tpr, color='c', label=f"ROC curve (auc = {score})")
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.title("Results")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.show()


from sklearn.ensemble import RandomForestClassifier
Forest = RandomForestClassifier(n_estimators = 100)
Forest = Forest.fit(X_train,y_train)


predicted= Forest.predict(X_test)
print("Random Forest Accuracy :", accuracy_score(y_test, predicted))


roc_auc_score(y_test, predicted)


proba = Forest.predict_proba(X_test)[:, 1]
score = roc_auc_score(y_test, proba)
fpr, tpr, _  = roc_curve(y_test, proba)

plt.figure()
plt.plot(fpr, tpr, color='c', label=f"ROC curve (auc = {score})")
plt.plot([0, 1], [0, 1], color='m', linestyle='--')
plt.title("Results")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.show()


# ## XGBoost


import xgboost as xgb
XGB_model = xgb.XGBClassifier()

XGB_model = XGB_model.fit(X_train, y_train)

predicted= XGB_model.predict(X_test)

print("XGBoost Accuracy :", accuracy_score(y_test, predicted))



roc_auc_score(y_test, predicted)


proba = XGB_model.predict_proba(X_test)[:, 1]
score = roc_auc_score(y_test, proba)
fpr, tpr, _  = roc_curve(y_test, proba)

plt.figure()
plt.plot(fpr, tpr, color='c', label=f"ROC curve (auc = {score})")
plt.plot([0, 1], [0, 1], color='m', linestyle='--')
plt.title("Results")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.show()


# To sum up, we saw, that the model's auc_roc_score of each model showed not well performance and also I know the reason why.
# 
# I did not implement SMOTE on training set, that's why the predicted list perceived incorrect outputs.
# 
# As I mentioned above, I didn't have enough time to wait for SMOTE class cell to run 
# 
# *(because it takes forever for kaggle notebook even to open )*
# 
# 
# So if I would apply training set on SMOTE model, I believe the results would be much greater. But it was not the only reason.
# 
# The second point to mention, a lot of parameters of model do affect the prediction. 
# 
# And as you have noticed, I didn't set any parameter, because (of course slow processing of the notebook) I wanted to see the pure model behavior.

