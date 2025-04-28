# **1. Introduction**
# 
# Finding good sets of parameter for algorithms such as xgboost or lightgbm can be daunting. In essence there are three options to tune parameter:
# 
# **a) Random search:** Randomly set parameters, obtain results (e.g. AUC), pick the parameter set which delivers the best result after x iterations
# 
# **b) Grid search: **Define a range of parameter values, iterate stepwise over the range for each parameter, obtain results (e.g. AUC), pick the parameter set which delivers the best result after all iterations
# 
# **c) Bayesian optimization:** Define a acquisition function which represents the probability of improvement at step x and stepwise iterate to improve the parameter set. Unlike the former two approaches, Bayesian optimization aims at using information from previous rounds to update the parameter set. In a sense there is “reinforcement” with respect to learning on the choice of parameters. 
# For further reading, see: [http://krasserm.github.io/2018/03/21/bayesian-optimization/](http://)
# 
# In what follows I present a simple random parameter search routine for lightgbm. I use data from the Kaggle Santander competition ([https://www.kaggle.com/c/santander-customer-transaction-prediction](http://)).
# 
# **2. Concept**
# 
# **i)** Set a threshold for the evaluation metric (AUC in this case)
# 
# **ii) **Randomly choose a set of parameter 
# 
# **iii)** Obtain AUC by cross-validation 
# 
# **iv)** If CV AUC is greater or equal to threshold AUC: use parameter set to obtain predictions, else continue
# 
# **3. Kernel**
# 
# Let’s start and first load some packages:


import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
import random, json, os


# Here I set some parameter. The first (*minauc*) is the minimal required AUC to make a prediction. The remaining parameter are the maximal bossting rounds (*maxrounds*), the number of random parameter draws for cross-validation (*cvrounds*), the early stopping rule (*estop*) and the CV-folds (*fol*).
# 
# **Note:** The values are set rather low here, so that the routine can finish in a reasonable period of time. To obtain good results you may use higher values.


# Min. cross validated AUC reqired to obtain/save model predictions
minauc = 0.85
# Max. rounds (of lgb model)
# This value can be high because there is early stopping
maxrounds = 25000
# Number of parameter trys by cross validation
# (these are not the "folds")
cvrounds = 2
# Early stopping rounds
estop = 1000
# CV folds
fol = 5


# Load the data:


# Load data for training 
mydata = pd.read_csv('../input/train.csv', sep=',')
mydata = mydata.drop('ID_code', 1)
# Load prediction data
preddata = pd.read_csv('../input/test.csv', sep=',')
predids = preddata[['ID_code']] 
iddf = preddata[['ID_code']] 
preddata = preddata.drop('ID_code', 1)


# Preprocessing:


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


# Next I start a loop in which a set of parameters is chosen randomly in each step. The current set of patameters is stored in *params* and is passed on to the *lgb.cv* model, from which the AUC is obtained (*curauc*). 
# 
# If the CV AUC is sufficiently large, predictions are obtained and stored. Also the current set of parameter is stored to the hard drive as a txt file. 
# 
# Finally, for each round in which predictions are obtained, predictions are averaged over all rounds (for which CV AUC >= min. AUC), and the predictions are saved as csv file, which later can be used for submission.


for i in range(0,cvrounds):
    params = {
        'boost_from_average' : False,
        'objective' :'binary',
        'learning_rate' : random.uniform(0.002, 0.003),
        'num_leaves' : random.randint(20, 26), 
        'feature_fraction': random.uniform(0.05, 0.16), 
        'bagging_fraction': random.uniform(0.2, 0.4), 
        'bagging_freq': random.randint(3, 5), 
        'max_bin' : random.randint(250, 260), 
        'scale_pos_weight' : random.randint(1, 3),  
        'boosting_type' : 'gbdt',
        'metric': 'auc',
        'num_threads' : 4,
        'tree_learner': 'serial', #neu
        'boost_from_average':'false',#neu
        'min_split_gain': random.uniform(0.15, 0.3),
        'min_child_weight': random.uniform(0.01, 0.2),
        'min_child_samples': random.randint(3, 6)
    }
    print(params)
    # Cross validation of parameter
    cv_results = lgb.cv(params, lgb_train, num_boost_round=maxrounds, nfold=fol, early_stopping_rounds=estop, metrics='auc')

    cvlist = cv_results['auc-mean']
    curauc = max(cvlist)
    curaucpos = cvlist.index(max(cvlist))
    curparams = params
    print("Current CV-AUC is %s (steps: %s)" %(curauc,curaucpos))

    if curauc >= minauc:
        # Train model with current parameters
        gbm = lgb.train(curparams, lgb_train, num_boost_round=maxrounds, valid_sets=lgb_eval, early_stopping_rounds=estop)
        # Predict (to get the AUC on test)
        y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
        auc = roc_auc_score(y_test, y_pred)
        print("Round: %s, Test AUC: %s" %(i, round(auc, 5)))

        # Get AUC as string to lable output file
        aucstring = str(auc)
        aucstring = aucstring.replace(".", "")
        # Write params to HDD
        params = {'params': params}
        with open("params"+str(aucstring)+".txt", 'w') as file:
            file.write(json.dumps(params)) 
        
        # Predict on submission data
        y_pred = gbm.predict(preddata, num_iteration=gbm.best_iteration)
        y_pred = y_pred.tolist()
        
        # Append dataframe with current predictions
        preddf = pd.DataFrame({'pred':y_pred})
        iddf = pd.concat([iddf, preddf], axis=1)
                
        # Updated submission file with new predictions (averaged)
        submission = pd.DataFrame({'ID_code': iddf['ID_code']})
        iddf_temp = iddf.drop('ID_code', 1)
        submission['target'] = iddf_temp.mean(axis=1)
        # Submission file contains average propensity scores
        submission.to_csv("submission.csv", sep=',', index=False)

