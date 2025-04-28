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


import os
import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import lightgbm as lgb


NFOLDS = 5
RANDOM_STATE = 42

#script_name = os.path.basename(__file__).split('.')[0]
#MODEL_NAME = "{0}__folds{1}".format(script_name, NFOLDS)

#print("Model: {}".format(MODEL_NAME))

print("Reading training data")
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

y = train.target.values
train_ids = train.ID_code.values
train = train.drop(['ID_code', 'target'], axis=1)
feature_list = train.columns

test_ids = test.ID_code.values
test = test[feature_list]

X = train.values.astype(float)
X_test = test.values.astype(float)

clfs = []
folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=RANDOM_STATE)
oof_preds = np.zeros((len(train), 1))
test_preds = np.zeros((len(test), 1))


from sklearn.ensemble import RandomForestClassifier


param = {
    'bagging_freq': 5,          
    'bagging_fraction': 0.331,   'boost_from_average':'false',   
    'boost': 'gbdt',             'feature_fraction': 0.0405,     
    'learning_rate': 0.0083,
    'max_depth': -1,             'metric':'auc',                
    'min_data_in_leaf': 80,     
    'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 13,            
    'num_threads': 8,            'tree_learner': 'serial',   
    'objective': 'binary',       'verbosity': 1
}



for fold_, (trn_, val_) in enumerate(folds.split(y, y)):
    print("Current Fold: {}".format(fold_))
    trn_x, trn_y = X[trn_, :], y[trn_]
    val_x, val_y = X[val_, :], y[val_]

    
    trn_data = lgb.Dataset(trn_x, label=trn_y)
    val_data = lgb.Dataset(val_x, label=val_y)
    clf = lgb.train(param, trn_data, 1000000, valid_sets = [trn_data, val_data], verbose_eval=500, early_stopping_rounds = 4000)

    val_pred = clf.predict(val_x)
    test_fold_pred = clf.predict(X_test)

    print("AUC = {}".format(metrics.roc_auc_score(val_y, val_pred)))
    oof_preds[val_, :] = val_pred.reshape((-1, 1))
    test_preds += test_fold_pred.reshape((-1, 1))

test_preds /= NFOLDS


roc_score = metrics.roc_auc_score(y, oof_preds.ravel())
print("Overall AUC = {}".format(roc_score))

print("Saving OOF predictions")
oof_preds = pd.DataFrame(np.column_stack((train_ids, oof_preds.ravel())), columns=['ID_code', 'target'])
#oof_preds.to_csv('../kfolds/{}__{}.csv'.format(MODEL_NAME, str(roc_score)), index=False)

print("Saving code to reproduce")
#shutil.copyfile(os.path.basename(__file__), '../model_source/{}__{}.py'.format(MODEL_NAME, str(roc_score)))

print("Saving submission file")
sample = pd.read_csv('../input/sample_submission.csv')
sample.target = test_preds.astype(float)
sample.ID_code = test_ids
sample.to_csv('../model_predictions/submission__{}.csv'.format(str(roc_score)), index=False)











