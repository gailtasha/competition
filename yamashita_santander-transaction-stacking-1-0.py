# In this Module I have stacked the Validation and Submission outputs using KFold Cross Validation technique and Stratified K-Fold Cross validatiom technique. Referring to the my previous kernel
# 
# **Stratified K Folds on Santander**
# https://www.kaggle.com/roydatascience/eda-pca-simple-lgbm-santander-transactions
# 
# **K Folds on Santander**
# https://www.kaggle.com/roydatascience/fork-of-eda-pca-simple-lgbm-kfold
# 
# The attempt is to improve the accuracy using Baysian Ridge Stacking approach


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedKFold
import os
print(os.listdir("../input/"))


#Import the Validation output and submissions

oof = pd.read_csv("../input/santander-outputs/Validation_Kfold.csv")['0']
oof_2 = pd.read_csv("../input/santander-outputs/Validation_fold.csv")['0']

predictions = pd.read_csv("../input/santander-outputs/submission_kfold.csv")["target"]
predictions_2 = pd.read_csv("../input/santander-outputs/submission_fold.csv")["target"]


train = pd.read_csv('../input/santander-customer-transaction-prediction/train.csv')
features = [c for c in train.columns if c not in ['ID_code', 'target']]


target = train['target']
train = train.drop(["ID_code", "target"], axis=1)


train_stack = np.vstack([oof,oof_2]).transpose()
test_stack = np.vstack([predictions, predictions_2]).transpose()

folds_stack = RepeatedKFold(n_splits=5, n_repeats=2, random_state=15)
oof_stack = np.zeros(train_stack.shape[0])
predictions_3 = np.zeros(test_stack.shape[0])

for fold_, (trn_idx, val_idx) in enumerate(folds_stack.split(train_stack,target)):
    print("fold {}".format(fold_))
    trn_data, trn_y = train_stack[trn_idx], target.iloc[trn_idx].values
    val_data, val_y = train_stack[val_idx], target.iloc[val_idx].values
    
    clf_3 = RandomForestClassifier(max_depth=5, n_estimators=100)
    clf_3.fit(trn_data, trn_y)
    
    oof_stack[val_idx] = clf_3.predict_proba(val_data)[:, 1]
    predictions_3 += clf_3.predict_proba(test_stack)[:, 1] / 10


predictions_3[:33]


sample_submission = pd.read_csv('../input/santander-customer-transaction-prediction/sample_submission.csv')
sample_submission['target'] = predictions_3
sample_submission.to_csv('submission_ashish.csv', index=False)



