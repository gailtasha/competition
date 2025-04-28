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


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


train=pd.read_csv("../input/train.csv",index_col="ID_code")
test=pd.read_csv("../input/test.csv",index_col="ID_code")
target=train.target
train=train.drop("target",axis=1)


train.head()


from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
from sklearn.model_selection import KFold
import time


import lightgbm as lgb


#  params={
#         "metric":"auc",
#         "num_threads":4,
#         "object":"binary",
#         "boosting":"gbdt",
#         "num_iterations":10000,
#         "learning_rate":0.1,
#         "num_leaves":32,
#         "max_depth":5,
#         "early_stopping_round":100,
#         "lambda_l2":0.1,
#         "min_data_in_leaf":200,
#         "bagging_fraction":0.9,
#         "feature_fraction":0.9
#     }
# mean val auc:0.885


#  params={
#         "metric":"auc",
#         "num_threads":4,
#         "object":"binary",
#         "boosting":"gbdt",
#         "num_iterations":10000,
#         "learning_rate":0.1,
#         "num_leaves":32,
#         "max_depth":2,# <-----detpth ==2
#         "early_stopping_round":100,
#         "lambda_l2":1,
#         "min_data_in_leaf":200,
#         "bagging_fraction":0.9,
#         "feature_fraction":0.9
#     }
# test_preds=train_model(params,samples=100000)
# mean val auc:0.888966825071865


#  params={
#         "metric":"auc",
#         "num_threads":4,
#         "object":"binary",
#         "boosting":"gbdt",
#         "num_iterations":10000,
#         "learning_rate":0.1,
#         "num_leaves":32,
#         "max_depth":2,  #  large depth does not help.......
#         "early_stopping_round":100,
#         "lambda_l2":1,
#         "min_data_in_leaf":200,
#         "bagging_fraction":0.9,
#         "feature_fraction":0.9
#     }
# test_preds=train_model(params,samples=200000)
# mean val auc:0.894   oh,very good!



def train_model(params,train=train,target=target,test=test,samples=-1):
    # lgb params
    kford=KFold(n_splits=5,random_state=2,shuffle=True)
    start_time=time.time()
    aucs=[]

    test_preds=[]
    # for early stopping
    # it takes a long time if using all the samples.
    if samples<=-1:
        samples=train.shape[0]
    else:
        samples=min(train.shape[0],samples)
    print("##################################################################")
    print("########## start fit model ###################")
    print("fit on {} samples".format(samples))
    for ford,(train_idx,val_idx) in enumerate(kford.split(train[:samples],target[:samples])):
        print("####################################")
        print("############ford:",ford)
        sample_x=train.iloc[train_idx].values
        sample_y=target.iloc[train_idx].values

        sample_val_x=train.iloc[val_idx].values
        sample_val_y=target.iloc[val_idx].values
        
        train_dataset=lgb.Dataset(data=sample_x,label=sample_y)
        val_dataset=lgb.Dataset(data=sample_val_x,label=sample_val_y)
        ford_time=time.time()
        #
        clf=lgb.train(params,train_dataset,valid_sets=[train_dataset,val_dataset],verbose_eval=50)
        #
        print("epoch cost time {:1}s".format(time.time()-ford_time))
        y_pred_prob=clf.predict(sample_x,clf.best_iteration)
        y_val_pred_prob=clf.predict(sample_val_x,clf.best_iteration)

        train_auc=metrics.roc_auc_score(sample_y,y_pred_prob)
        val_auc=metrics.roc_auc_score(sample_val_y,y_val_pred_prob)
        print("train auc:{:4},val auc:{:4}".format(train_auc,val_auc))
        aucs.append([train_auc,val_auc])
        test_preds.append(clf.predict(test,clf.best_iteration))

    end_time=time.time()
    val_aucs=[auc[1] for auc in aucs]
    print("using {} samples,total time:{:1}s,mean val auc:{:4}".format(samples,end_time-start_time,np.mean(val_aucs)))
    test_preds=pd.DataFrame(test_preds).T
    test_preds.index=test.index
    return test_preds


 params={
        "metric":"auc",
        "num_threads":8,
        "object":"binary",
        "boosting":"gbdt",
        "num_iterations":10000,
        "learning_rate":0.03,
        "num_leaves":32,
        "max_depth":2,  #  large depth does not help.......
        "early_stopping_round":100,
        "lambda_l2":1,
        "min_data_in_leaf":2000,# for small this does take effect (I think)
        "bagging_fraction":0.9,
        "feature_fraction":0.9
    }
test_preds=train_model(params,samples=-1)


submission=pd.DataFrame(test_preds.mean(axis=1),columns=["target"])
submission.head()
submission.to_csv("submission.csv")


l

