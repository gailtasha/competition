# # Accelerating XGboost with GPU
# 
# This kernel uses the Xgboost models, running on CPU and GPU. With the GPU acceleration, we gain a ~8.5x performance improvement on an NVIDIA K80 card compared to the 2-core virtual CPU available in the Kaggle VM (1h 8min 46s vs. 8min 20s).
# 
# The gain on a NVIDIA 1080ti card compared to an Intel i7 6900K 16-core CPU is ~6.6x.
# 
# To turn GPU support on in Kaggle, in notebook settings, set the **GPU beta** option to "GPU on".
# 
# ## Notebook  Content
# 1. [Loading the data](#0) <br>    
# 1. [Training the model on CPU](#1)
# 1. [Training the model on GPU](#2)
# 1. [Submission](#3)


# <a id="0"></a>
# ## 1. Loading the data


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import gc
import xgboost as xgb

pd.set_option('display.max_columns', 200)


train_df = pd.read_csv('../input/train.csv', engine='python')
test_df = pd.read_csv('../input/test.csv', engine='python')


# <a id="1"></a> 
# ## 2. Training the model on CPU


import subprocess
print((subprocess.check_output("lscpu", shell=True).strip()).decode())


MAX_TREE_DEPTH = 8
TREE_METHOD = 'hist'
ITERATIONS = 1000
SUBSAMPLE = 0.6
REGULARIZATION = 0.1
GAMMA = 0.3
POS_WEIGHT = 1
EARLY_STOP = 10

params = {'tree_method': TREE_METHOD, 'max_depth': MAX_TREE_DEPTH, 'alpha': REGULARIZATION,
          'gamma': GAMMA, 'subsample': SUBSAMPLE, 'scale_pos_weight': POS_WEIGHT, 'learning_rate': 0.05, 
          'silent': 1, 'objective':'binary:logistic', 'eval_metric': 'auc', 'silent':True, 
          'verbose_eval': False}


%%time
nfold = 5
skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=2019)

oof = np.zeros(len(train_df))
predictions = np.zeros(len(test_df))

target = 'target'
predictors = train_df.columns.values.tolist()[2:]

i = 1
for train_index, valid_index in skf.split(train_df, train_df.target.values):
    print("\nFold {}".format(i))
    xg_train = xgb.DMatrix(train_df.iloc[train_index][predictors].values,
                           train_df.iloc[train_index][target].values,                           
                           )
    xg_valid = xgb.DMatrix(train_df.iloc[valid_index][predictors].values,
                           train_df.iloc[valid_index][target].values,                           
                           )   

    
    clf = xgb.train(params, xg_train, ITERATIONS, evals=[(xg_train, "train"), (xg_valid, "eval")],
                early_stopping_rounds=EARLY_STOP, verbose_eval=False)
    oof[valid_index] = clf.predict(xgb.DMatrix(train_df.iloc[valid_index][predictors].values)) 
    
    predictions += clf.predict(xgb.DMatrix(test_df[predictors].values)) / nfold
    i = i + 1

print("\n\nCV AUC: {:<0.2f}".format(metrics.roc_auc_score(train_df.target.values, oof)))


# <a id="2"></a>
# ## 3. Training the model on GPU


!nvidia-smi


# We now train the model with a K80 GPU available in Kaggle. Xgboost provides out of the box support for single GPU training. On a local workstation, a GPU-ready xgboost docker image can be obtained from https://hub.docker.com/r/rapidsai/rapidsai/.
# 
# All we need to change is to set: `TREE_METHOD = 'gpu_hist'`


MAX_TREE_DEPTH = 8
TREE_METHOD = 'gpu_hist'
ITERATIONS = 1000
SUBSAMPLE = 0.6
REGULARIZATION = 0.1
GAMMA = 0.3
POS_WEIGHT = 1
EARLY_STOP = 10

params = {'tree_method': TREE_METHOD, 'max_depth': MAX_TREE_DEPTH, 'alpha': REGULARIZATION,
          'gamma': GAMMA, 'subsample': SUBSAMPLE, 'scale_pos_weight': POS_WEIGHT, 'learning_rate': 0.05, 
          'silent': 1, 'objective':'binary:logistic', 'eval_metric': 'auc',
          'n_gpus': 1}


%%time
nfold = 5
skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=2019)

oof = np.zeros(len(train_df))
predictions = np.zeros(len(test_df))

target = 'target'
predictors = train_df.columns.values.tolist()[2:]

i = 1
for train_index, valid_index in skf.split(train_df, train_df.target.values):
    print("\nFold {}".format(i))
    xg_train = xgb.DMatrix(train_df.iloc[train_index][predictors].values,
                           train_df.iloc[train_index][target].values,                           
                           )
    xg_valid = xgb.DMatrix(train_df.iloc[valid_index][predictors].values,
                           train_df.iloc[valid_index][target].values,                           
                           )   

    
    clf = xgb.train(params, xg_train, ITERATIONS, evals=[(xg_train, "train"), (xg_valid, "eval")],
                early_stopping_rounds=EARLY_STOP, verbose_eval=False)
    oof[valid_index] = clf.predict(xgb.DMatrix(train_df.iloc[valid_index][predictors].values)) 
    
    predictions += clf.predict(xgb.DMatrix(test_df[predictors].values)) / nfold
    i = i + 1

print("\n\nCV AUC: {:<0.2f}".format(metrics.roc_auc_score(train_df.target.values, oof)))


# <a id="3"></a>
# ## 4. Submission


sub_df = pd.DataFrame({"ID_code": test_df.ID_code.values})
sub_df["target"] = predictions
sub_df[:10]


sub_df.to_csv("xgboost_gpu.csv", index=False)



