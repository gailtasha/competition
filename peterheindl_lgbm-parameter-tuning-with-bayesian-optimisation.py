# **Aim of this Kernel:**
# 
# In this Kernel I show a minimal example how to implement Bayesian Optimisation (BO) with a Light Gradient Boosting Machine (LGBM). However, this example can easily be applied to other algorithms as well.
# 
# **Introduction:**
# 
# Tuning of hyperparameters of boosting algorithms can be daunting. Apart of random search or grid search, Bayesian optimisation (BO) offers a more structured approach to parameter tuning. 
# 
# Bayesian optimization aims at optimising a “black box function” (here the boosting algorithms) without knowing much about the function. Optimisation is not based on derivatives or the like. Instead, previous outcomes are used to try to find a set of parameter which improve the objective (this is why it is called Bayesian). This post is very helpful to get an idea of what is going on with Bayesian optimisation: [http://krasserm.github.io/2018/03/21/bayesian-optimization/](http://)
# 
# **Getting Started:**
# 
# All we need for a start is the “bayesian-optimization” package (e.g. use *pip install bayesian-optimization*). The package is well documented and you can find a very helpful introduction here: [https://github.com/fmfn/BayesianOptimization/blob/master/examples/basic-tour.ipynb](http://).
# 
# **All you need to do is to...**
# 
# *a) Define a “black box function” to be optimised
# b) Pass some parameter over which optimisation is done
# c) Start the BO routine*
# 
# **Let's load and prepare the data first:**


import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
import random, json
from bayes_opt import BayesianOptimization

# Load data for training 
mydata = pd.read_csv('../input/train.csv', sep=',')
mydata = mydata.drop('ID_code', 1)
# Load prediction data
preddata = pd.read_csv('../input/test.csv', sep=',')
predids = preddata[['ID_code']] 
iddf = preddata[['ID_code']] 
preddata = preddata.drop('ID_code', 1)

# Test train split
df_train, df_test = train_test_split(mydata, test_size=0.3, random_state=76)
# Same random state to make sure rows merge
y_train = df_train['target']
y_test = df_test['target']
X_train = df_train.drop('target', 1)
X_test = df_test.drop('target', 1)

# Scale data
scaler = preprocessing.StandardScaler()
scaled_df = scaler.fit_transform(X_train)
X_train = pd.DataFrame(scaled_df)
scaled_df = scaler.fit_transform(X_test)
X_test = pd.DataFrame(scaled_df)
scaled_df = scaler.fit_transform(preddata)
preddata = pd.DataFrame(scaled_df)

# Create dataset for lightgbm input
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)


# **Application to Light GBM:**
# 
# It is obvious that our LGBM is the “black box function”. So all we need to do is to define a function which contains the LGBM model parameter and the LGBM command itself (*lgbmfunc*). I use LGBM cross-validation here for obvious reasons.
# 
# Next we define the parameter bounds over which BO is implemented (*pbounds*). The BO routine searches within the bounds of these values for an optimal parameter set. Please note that BO uses and returns float here. If you need to return integer values to your function, which is true for some of the LGBM parameter, you need to specify *int()*.
# 
# Next we declare the optimiser function and start it, passing the *rounds* argument, which specifies how often the BO routine iterates to find best parameters.
# 
# *Note: In a real-world application you would use a much higher number of boosting rounds and a higher value for early stopping.*


def lgbmfunc(min_split_gain, min_child_weight):
    params = {
        'boost_from_average' : False,
        'objective' :'binary',
        'learning_rate' : 0.002,
        'num_leaves' : 24, 
        'feature_fraction': 0.07, 
        'bagging_fraction': 0.2, 
        'bagging_freq': 3, 
        'max_bin' : 255, #default 255
        'scale_pos_weight' : 1,  #default = 1 (used only in binary application) weight of labels with positive class
        'boosting_type' : 'gbdt',
        'metric': 'auc',
        'num_threads' : 4,
        'tree_learner': 'serial', 
        'boost_from_average':'false',
        'min_child_samples': 3
    }
    params['min_split_gain'] = min_split_gain
    params['min_child_weight'] = min_child_weight

    # LGBM CV
    cv_results = lgb.cv(params, lgb_train, num_boost_round=500, nfold=5, early_stopping_rounds=50, metrics='auc')
    return max(cv_results['auc-mean'])

pbounds = { 
        'min_split_gain': (0.2, 0.3), 
        'min_child_weight': (0.1, 0.3),
    }

optimizer = BayesianOptimization(
    f=lgbmfunc,
    pbounds=pbounds,
    verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1,
)

optimizer.maximize(
    init_points=2,
    n_iter=10,
)


# **Results:**
# 
# After the routine has finished, we can inspect the results obtained in each iteration. However, what is even more interesting is the optimal set of parameter which can be called by the .max command. It will look like this:
# 
# *best result
# {'target': 0.8638091350110189, 'params': {'min_child_weight': 0.10002287496346898, 'min_split_gain': 0.23023325726318397}}*
# 
# **Next Steps:**
# 
# While the code presented here is only a toy application, it is very easy to expand the code so to make it a useful application for hyperparameter tuning. All you need to do is to replace the “fixed” parameter in params with a variable equivalent so that you can pass the values to the BO routine. 
# 
# **Happy coding!**


print("list of results")
for i, res in enumerate(optimizer.res):
    print("Iteration {}: \n\t{}".format(i, res))

print("best result")
print(optimizer.max)

