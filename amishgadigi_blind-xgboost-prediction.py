# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


import xgboost as xgb
import matplotlib
from sklearn.metrics import roc_auc_score


dftrain = pd.read_csv('../input/train.csv')
dftrain.head()


dftest = pd.read_csv('../input/test.csv')
dftest.head()


train_cols = dftrain.columns
train_cols = train_cols.drop(['ID_code','target'])
train_cols


    dtrain = xgb.DMatrix(dftrain[train_cols], label = dftrain['target'])
    dtest = xgb.DMatrix(dftest[train_cols])


param = {'max_depth':10, 'eta':0.5, 'objective':'binary:logistic','eval_metric':'auc','gamma':100,'subsample':0.95,'lambda':1000,'alpha':1000, 'min_child_weight':100,
         'max_delta_step':0,'tree_method':'hist','max_bin':1024,'max_leaves':1000,'grow_policy':'lossguide','feature_selector':'greedy','top_k':0}


xgbmodel = xgb.train(param, dtrain)
preds = xgbmodel.predict(dtest)


dfprediction = dftest[['ID_code']]


dfprediction['target']=preds


dfprediction.head()


dfprediction.to_csv("submission.csv", index=False)

