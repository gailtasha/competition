# # Accelerating hyper-parameter searching with GPU
# 
# This kernel perform a random hyper-parameter seach using the Xgboost models. Since running time on CPU is prohibitively long, we accelerate the search with the K80 GPU available in Kaggle to achieve a 6x speed-up over the CPU.
# 
# To turn GPU support on in Kaggle, in notebook settings, set the "GPU beta" option to "GPU on". Xgboost provides out-of-the-box support for single GPU training. On a local workstation, a GPU-ready xgboost docker image can be obtained from https://hub.docker.com/r/rapidsai/rapidsai/.
# 
# ## Notebook  Content
# 1. [Loading the data](#0) 
# 1. [Hyper-parameter search with CPU](#1)
# 1. [Hyper-parameter search with GPU](#2)
# 1. [Submission](#3)
# 


# <a id="0"></a> 
# ## 1. Loading the data


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

pd.set_option('display.max_columns', 200)


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


train_df = pd.read_csv('../input/train.csv', engine='python')
test_df = pd.read_csv('../input/test.csv', engine='python')

#Experimenting with a small subset
train_df = train_df[1:10000]


# <a id="1"></a>
# ## 2. Hyper-parameter search with CPU


import subprocess
print((subprocess.check_output("lscpu", shell=True).strip()).decode())


# A parameter grid for XGBoost
params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.02, 0.05]    
        }


# We carry out a quick search on the CPU with 3-fold cross valiation and 1 random parameter combo. This takes ~60m for a single parameter combination. So in order to carry out a random search over 100 combinations, the estimated time will be ~100h.


%%time 
folds = 3
param_comb = 1

target = 'target'
predictors = train_df.columns.values.tolist()[2:]

X = train_df[predictors]
Y = train_df[target]

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

xgb = XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic', nthread=1)

random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='roc_auc', n_jobs=4, cv=skf.split(X,Y), verbose=3, random_state=1001)

# Here we go
start_time = timer(None) # timing starts from this point for "start_time" variable
random_search.fit(X, Y)
timer(start_time) # timing ends here for "start_time" variable


# <a id="2"></a>
# ## 3. Hyper-parameter search with GPU


# We will accelerate hyper-parameter search with the K80 GPU available in Kaggle.


!nvidia-smi


# The only change we need to make is to set `TREE_METHOD = 'gpu_hist'` when initializing Xgboost.


%%time 
folds = 3
param_comb = 1

target = 'target'
predictors = train_df.columns.values.tolist()[2:]

X = train_df[predictors]
Y = train_df[target]

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

xgb = XGBClassifier(learning_rate=0.02, n_estimators=1000, objective='binary:logistic',
                    silent=True, nthread=6, tree_method='gpu_hist', eval_metric='auc')

random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='roc_auc', n_jobs=4, cv=skf.split(X,Y), verbose=3, random_state=1001 )

# Here we go
start_time = timer(None) # timing starts from this point for "start_time" variable
random_search.fit(X, Y)
timer(start_time) # timing ends here for "start_time" variable


# The GPU acceleration provide a 8x speedup. Now we can afford to perform a more thorough search with 20 random configurations.


%%time 

folds = 3
param_comb = 20

target = 'target'
predictors = train_df.columns.values.tolist()[2:]

X = train_df[predictors]
Y = train_df[target]

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

xgb = XGBClassifier(learning_rate=0.02, n_estimators=1000, objective='binary:logistic',
                    silent=True, nthread=6, tree_method='gpu_hist', eval_metric='auc')

random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='roc_auc', n_jobs=4, cv=skf.split(X,Y), verbose=3, random_state=1001 )

# Here we go
start_time = timer(None) # timing starts from this point for "start_time" variable
random_search.fit(X, Y)
timer(start_time) # timing ends here for "start_time" variable


# <a id="2"></a>
# ## 3. Submission


y_test = random_search.predict_proba(test_df[predictors])
y_test.shape


sub_df = pd.DataFrame({"ID_code": test_df.ID_code.values, "target": y_test[:,1]})
sub_df[:10]


sub_df.to_csv("xgboost_gpu_randomsearch.csv", index=False)



