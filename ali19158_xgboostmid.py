# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import gc
import os
import logging
import datetime
import warnings
import lightgbm as lgb
from tqdm import tqdm_notebook
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


dstest = pd.read_csv("/kaggle/input/santander-customer-transaction-prediction/test.csv")
dstrain = pd.read_csv("/kaggle/input/santander-customer-transaction-prediction/train.csv")


dstest


#dstestw = dstest.drop(['ID_code'], axis=1).astype('float32')
#dstrainw = dstrain.drop(['ID_code', 'target'], axis=1).astype('float32')


dstest.shape


dstest.info()
dstrain.info()


dstest.describe()


sns.countplot(dstrain['target'], palette='Set3')


#X = dstrain.values
#y = dstrain.target.astype('uint8').values


# def plot_feature_distribution(df1, df2, label1, label2, features):
#     i = 0
#     sns.set_style('whitegrid')
#     plt.figure()
#     fig, ax = plt.subplots(10,10,figsize=(18,22))

#     for feature in features:
#         i += 1
#         plt.subplot(10,10,i)
#         sns.distplot(df1[feature], hist=False,label=label1)
#         sns.distplot(df2[feature], hist=False,label=label2)
#         plt.xlabel(feature, fontsize=9)
#         locs, labels = plt.xticks()
#         plt.tick_params(axis='x', which='major', labelsize=6, pad=-6)
#         plt.tick_params(axis='y', which='major', labelsize=6)
#     plt.show();


# t0 = dstrain.loc[dstrain['target'] == 0]
# t1 = dstrain.loc[dstrain['target'] == 1]
# features = dstrain.columns.values[2:102]
# plot_feature_distribution(t0, t1, '0', '1', features)


plt.figure(figsize=(16,6))
features = dstrain.columns.values[2:202]
plt.title("Distribution of mean values per row in the train and test set")
sns.distplot(dstrain[features].mean(axis=1),color="green", kde=True,bins=120, label='train')
sns.distplot(dstest[features].mean(axis=1),color="blue", kde=True,bins=120, label='test')
plt.legend()
plt.show()


# scale numerical features
from sklearn.preprocessing import StandardScaler


st_sc = StandardScaler()
from sklearn.model_selection import train_test_split


# split the inputs and outputs
X_train = dstrain.iloc[:, dstrain.columns != 'target']
y_train = dstrain.iloc[:, 1].values
X_test = dstest.iloc[:,dstest.columns != 'ID_code'].values
X_train = X_train.iloc[:,X_train.columns != 'ID_code'].values

# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, stratify = y, random_state = 0)


# encode categoricals
from sklearn.preprocessing import LabelEncoder


le = LabelEncoder()


import xgboost as xgb


xg_cl = xgb.XGBClassifier(
    objective = 'binary:logistic',
    n_estimators = 1000, seed=123,
    learning_rate=0.25, max_depth=2,
    colsample_bytree=0.35, subsample=0.82,
    min_child_weight=53, gamma = 9.9,tree_method='gpu_hist'
                          )


 xg_cl.fit(X_train, y_train)


print(xg_cl.feature_importances_)


import matplotlib.pyplot as plt
plt.bar(range(len(xg_cl.feature_importances_)), xg_cl.feature_importances_)
plt.show()


from xgboost import plot_importance
plot_importance(xg_cl)
plt.show()


y_pred_xg=xg_cl.predict(X_test)


from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel
from numpy import sort
from sklearn.metrics import accuracy_score
predictions = [round(value) for value in y_pred_xg]
accuracy = accuracy_score(y_train, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
# Fit model using each importance as a threshold
thresholds = sort(xg_cl.feature_importances_)
for thresh in thresholds:
	# select features using threshold
	selection = SelectFromModel(xg_cl, threshold=thresh, prefit=True)
	select_X_train = selection.transform(X_train)
	# train model
	selection_model = XGBClassifier()
	selection_model.fit(select_X_train, y_train)
	# eval model
	select_X_test = selection.transform(X_test)
	y_pred_xg = selection_model.predict(select_X_test)
	predictions = [round(value) for value in y_pred_xg]
	accuracy = accuracy_score(y_train, predictions)
	print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))


# dataset_xg = pd.concat((dstest.ID_code, pd.Series(y_pred_xg).rename('target')), axis=1)
# dataset_xg.target.value_counts()


# dataset_xg.to_csv('xgbost_mid.csv', index=False)


# from sklearn.neighbors import KNeighborsClassifier
# knn = KNeighborsClassifier()


# knn.fit(X_train, y_train)


#  y_preds=knn.predict(X_test)


# dataset_knn = pd.concat((dstest.ID_code, pd.Series(y_preds).rename('target')), axis=1)
# dataset_knn.target.value_counts()


#dataset_knn.to_csv('knn_mid.csv', index=False)


from sklearn import svm
clf = svm.SVC()




 clf.fit(X_train, y_train)


clf_preds = clf.predict(X_test)


dataset_clf = pd.concat((dstest.ID_code, pd.Series(clf_preds).rename('target')), axis=1)
dataset_clf.target.value_counts()


dataset_clf.to_csv('clf_mid.csv', index=False)



