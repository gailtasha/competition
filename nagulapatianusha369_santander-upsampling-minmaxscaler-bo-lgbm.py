# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from bayes_opt import BayesianOptimization
from hyperopt import hp
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import lightgbm as lgbm

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


def reduce_mem_usage(df, verbose=True):
    numerics = ['int8','int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


trainnotreduced = pd.read_csv("../input/train.csv")


    trainnotreduced.info()


train = reduce_mem_usage(pd.read_csv("../input/train.csv"))
test = reduce_mem_usage(pd.read_csv("../input/test.csv"))


train.info()


train_majority = train[train.target==0]
train_minority = train[train.target==1]
 


train_majority.info()


from sklearn.utils import resample
# Upsample minority class
train_minority_upsampled = resample(train_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=179902,    # to match majority class
                                 random_state=123) # reproducible results
 
# Combine majority class with upsampled minority class
train_upsampled = pd.concat([train_majority, train_minority_upsampled])
 
# Display new class counts
train_upsampled.target.value_counts()


plt.figure();


plt.style.use('ggplot')
columns = list(train_upsampled)[1:-2]
train_upsampled[columns].hist(stacked=False, bins=100, figsize=(12,70), layout=(50,4));


X = train.drop(['ID_code','target'],axis = 1)
y = train['target']


from sklearn.preprocessing import MinMaxScaler


mms = MinMaxScaler()


X_mms = mms.fit_transform(X)


X_mms = pd.DataFrame(X_mms,columns= X.columns)


X_mms.describe()


# X_mms.var_68


from hyperopt import STATUS_OK
from hyperopt.pyll.base import scope

N_FOLDS = 10


train_set = lgbm.Dataset(X_mms,y)
def objective(params, n_folds = N_FOLDS):
    """Objective function for Gradient Boosting Machine Hyperparameter Tuning"""
    
    # Perform n_fold cross validation with hyperparameters
    # Use early stopping and evalute based on ROC AUC
    cv_results = lgbm.cv(params, train_set, nfold = n_folds, num_boost_round = 5000, early_stopping_rounds=1000
                        ,metric = 'auc', seed = 50)
  
    # Extract the best score
    best_score = max(cv_results['auc-mean'])
    
    # Loss must be minimized
    loss = 1 - best_score
    
    # Dictionary with information for evaluation
    return {'loss': loss, 'params': params, 'status': STATUS_OK}


# Define the search space
space = {
  
    
    'num_leaves': scope.int(hp.quniform('num_leaves',10, 90, 3)),
    'learning_rate': hp.quniform('learning_rate',0, 3,0.1),
    'subsample_for_bin': scope.int( hp.quniform('subsample_for_bin', 20000, 400000, 20000)),
    'min_child_samples': scope.int(hp.quniform('min_child_samples',20,500,5)),
    'reg_alpha': hp.uniform('reg_alpha', 0.0, 5.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 5.0),
    'colsample_bytree': hp.quniform('colsample_by_tree', 0.0, 1.0,0.01)
}


from hyperopt import tpe
# Algorithm
tpe_algorithm = tpe.suggest


from hyperopt import Trials
# Trials object to track progress
bayes_trials = Trials()


from hyperopt import fmin

MAX_EVALS = 10

# Optimize
best = fmin(fn = objective, space = space, algo = tpe.suggest,max_evals = MAX_EVALS, trials = bayes_trials)


best


best = {'colsample_by_tree': 0.9400000000000001,
 'learning_rate': 0.1,
 'min_child_samples': 335,
 'num_leaves': 27,
 'reg_alpha': 2.4376113452795822,
 'reg_lambda': 4.942768696457326,
 'subsample_for_bin': 240000}


lgbtrain = lgbm.train(best,train_set=train_set,num_boost_round=5000)


testdata = pd.read_csv("../input/test.csv")


test = testdata.drop(['ID_code'],axis=1)




testMMs = mms.fit_transform(test)


Test_mms = pd.DataFrame(testMMs,columns= test.columns)


Test_mms.describe()


testY = lgbtrain.predict(Test_mms)


submission = pd.DataFrame({
        "ID_code": testdata["ID_code"],
        "target": testY
    })
submission.to_csv('submission.csv', index=False)



