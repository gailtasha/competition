# ![](https://github.com/fmfn/BayesianOptimization/blob/master/examples/func.png?raw=true)


# # **Introduction**
# <br>
# Hi guys! <br>
# Have you guys experienced some frustrating moments when you tuned hyperparameters with Grid Search and Random Search?  <br>
# Well...I have! I waited for two hours or more to run codes for both methods and ended up losing my focus by watching Youtube videos...
# <br>
# ![](https://media.giphy.com/media/qjF9Akev3QPNC/giphy.gif)
# <br>
# So, I thought that there must be a faster way to tune hyperparameters and did some research about the new method of tuning, which is called Baysian Optimization.<br>
# If you have done Kaggle for a while or are an expert in this field, you probably have used or heard the Baysian Optimization. <br>
# This kernel will give you a good idea of Baysian Optimization and the simple implementation of Baysian Optimization with BayesianOptimization, which I found that it is faster than other Baysian Optimization functions in Python (ex. Scikit-Optimize and Hyperopt) for my model.  <br> 
# 
# **The goal is to identify which customers will make a specific transaction in the future and maximize the evaluation function (AUC)**
# <br> <br>
# 
# # **Special Thanks to:** <br> 
# 
# I recommend you to see  [Fayzur's kernel](https://www.kaggle.com/fayzur/lgb-bayesian-parameters-finding-rank-average) and [sz8416's kernel](https://www.kaggle.com/sz8416/simple-bayesian-optimization-for-lightgbm) if you are interested in seeing what other people have done. Their kernels were very helpful to understand about the Baysian optimization process in Python. 


# # **Ready, Set, Go!** <br>
# 
# Before starting the kernel, I guarantee that tuning hyperparameter process will not take more than 10 minutes. And the whole process (loading dataset, tuning the hyperparameter, and training LightGBM) will not take more than 20 minutes.  **Time is important!!** <br>
# ![](https://media.giphy.com/media/3oz8xKaR836UJOYeOc/giphy.gif)


# <br>
# # ** CONTENTS**
# 
# 1. [What Is Bayesian Optimization and Why Do We Care?](#1)
# 2. [Loading Library and Dataset](#2)
# 3. [Bayesian Optimization with LightGBM](#3)
# 4. [Training LightGBM](#4)
# 5. [Understanding the Model Better](#5)


# <a id="1"></a> <br>
# 
# # **What Is Bayesian Optimization and Why Do We Care?**


# **Bayesian Optimization** is a probabilistic model based approach for finding the minimum of any function that returns a real-value metric. [(source)](https://towardsdatascience.com/an-introductory-example-of-bayesian-optimization-in-python-with-hyperopt-aae40fff4ff0)<br> It is very effective with real-world applications in high-dimensional parameter-tuning for complex machine learning algorithms. Bayesian optimization utilizes the Bayesian technique of setting a prior over the objective function and
# combining it with evidence to get a posterior function. I attached one graph that demonstrates Bayes’ theorem below. <br> <br>
# 
# <img src="https://www.analyticsvidhya.com/wp-content/uploads/2016/06/12-768x475.jpg"  alt="Drawing" style="width: 600px;"/>
# <br> <br> 
#  The prior belief is our belief in parameters before modeling process. The posterior belief is our belief in our parameters after observing the evidence.
# <br> Another way to present the Bayes’ theorem is: 
# 
# <img src="https://www.maths.ox.ac.uk/system/files/attachments/Bayes_0.png"  alt="Drawing" style="width: 600px;"/> <br> 
# 
# For continuous functions, Bayesian optimization typically works by assuming the unknown function was sampled from a Gaussian process and maintains a posterior distribution for this function as observations are made, which means that we need to give range of values of hyperparameters (ex. learning rate range from 0.1 to 1).  So, in our case, the Gaussian process gives us a prior distribution on functions. Gaussian process approach is a non-parametric approach, in that it finds a distribution over the possible functions 
# f(x) that are consistent with the observed data. Gaussian processes have proven to be useful surrogate models for computer experiments and good
# practices have been established in this context for sensitivity analysis, calibration and prediction While these strategies are not considered in the context of optimization, they can be useful to researchers in machine learning who wish to understand better the sensitivity of their models to various hyperparameters. [(source)](http://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.pdf)
# 
# <br>
# **Well..why do we care about it?** <br> 
# According to the [study](http://proceedings.mlr.press/v28/bergstra13.pdf), hyperparameter tuning by Bayesian Optimization of machine learnin models is more efficient than Grid Search and Random Search. Bayesian Optimization has better overall performance on the test data and takes less time for optimization. Also, we do not need to set a certain values of parameters like we do in Random Search and Grid Search. For Bayesian Optimization tuning, we just give a range of a hyperparameter. 
# 


# <a id="2"></a> <br>
# # **Loading Library and Dataset**


#basic tools 
import os
import numpy as np
import pandas as pd
import warnings

#tuning hyperparameters
from bayes_opt import BayesianOptimization
from skopt  import BayesSearchCV 

#graph, plots
import matplotlib.pyplot as plt
import seaborn as sns

#building models
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
import time
import sys

#metrics 
from sklearn.metrics import roc_auc_score, roc_curve
import shap
warnings.simplefilter(action='ignore', category=FutureWarning)


# By Changing the data type of each column, I reduced memory usages by 75%. By taking the minimum and the maximum of each column, the function assigns which numeric data type is optimal for the column and change the data type. If you want to know more about how it works, I suggest you to read [Eryk's article](https://towardsdatascience.com/make-working-with-large-dataframes-easier-at-least-for-your-memory-6f52b5f4b5c4)! 


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


%%time
train= reduce_mem_usage(pd.read_csv("../input/train.csv"))
test= reduce_mem_usage(pd.read_csv("../input/test.csv"))
print("Shape of train set: ",train.shape)
print("Shape of test set: ",test.shape)


# <a id="3"></a> <br>
# # **Bayesian Optimization with LightGBM**
# <br>
# Now I am going to prepare data for modeling and a Baysian Optimization function. You can put more parameters (ex. lambda_l1 and lambda_l2) into the function.  


y=train['target']
X=train.drop(['ID_code','target'],axis=1)


%%time

def bayes_parameter_opt_lgb(X, y, init_round=15, opt_round=25, n_folds=3, random_seed=6,n_estimators=10000, output_process=False):
    # prepare data
    train_data = lgb.Dataset(data=X, label=y, free_raw_data=False)
    # parameters
    def lgb_eval(learning_rate,num_leaves, feature_fraction, bagging_fraction, max_depth, max_bin, min_data_in_leaf,min_sum_hessian_in_leaf,subsample):
        params = {'application':'binary', 'metric':'auc'}
        params['learning_rate'] = max(min(learning_rate, 1), 0)
        params["num_leaves"] = int(round(num_leaves))
        params['feature_fraction'] = max(min(feature_fraction, 1), 0)
        params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
        params['max_depth'] = int(round(max_depth))
        params['max_bin'] = int(round(max_depth))
        params['min_data_in_leaf'] = int(round(min_data_in_leaf))
        params['min_sum_hessian_in_leaf'] = min_sum_hessian_in_leaf
        params['subsample'] = max(min(subsample, 1), 0)
        
        cv_result = lgb.cv(params, train_data, nfold=n_folds, seed=random_seed, stratified=True, verbose_eval =200, metrics=['auc'])
        return max(cv_result['auc-mean'])
     
    lgbBO = BayesianOptimization(lgb_eval, {'learning_rate': (0.01, 1.0),
                                            'num_leaves': (24, 80),
                                            'feature_fraction': (0.1, 0.9),
                                            'bagging_fraction': (0.8, 1),
                                            'max_depth': (5, 30),
                                            'max_bin':(20,90),
                                            'min_data_in_leaf': (20, 80),
                                            'min_sum_hessian_in_leaf':(0,100),
                                           'subsample': (0.01, 1.0)}, random_state=200)

    
    #n_iter: How many steps of bayesian optimization you want to perform. The more steps the more likely to find a good maximum you are.
    #init_points: How many steps of random exploration you want to perform. Random exploration can help by diversifying the exploration space.
    
    lgbBO.maximize(init_points=init_round, n_iter=opt_round)
    
    model_auc=[]
    for model in range(len( lgbBO.res)):
        model_auc.append(lgbBO.res[model]['target'])
    
    # return best parameters
    return lgbBO.res[pd.Series(model_auc).idxmax()]['target'],lgbBO.res[pd.Series(model_auc).idxmax()]['params']

opt_params = bayes_parameter_opt_lgb(X, y, init_round=5, opt_round=10, n_folds=3, random_seed=6,n_estimators=10000)


# Here is my optimal parameter for LightGBM. 


opt_params[1]["num_leaves"] = int(round(opt_params[1]["num_leaves"]))
opt_params[1]['max_depth'] = int(round(opt_params[1]['max_depth']))
opt_params[1]['min_data_in_leaf'] = int(round(opt_params[1]['min_data_in_leaf']))
opt_params[1]['max_bin'] = int(round(opt_params[1]['max_bin']))
opt_params[1]['objective']='binary'
opt_params[1]['metric']='auc'
opt_params[1]['is_unbalance']=True
opt_params[1]['boost_from_average']=False
opt_params=opt_params[1]
opt_params


# <a id="4"></a> <br>
# # **Training LightGBM** <br> <br>
# 
# Based on the parameter from the previous step, I am going to train LightGBM. 


%%time 

target=train['target']
features= [c for c in train.columns if c not in ['target','ID_code']]


folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=31416)
oof = np.zeros(len(train))
predictions = np.zeros(len(test))
feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
    print("Fold {}".format(fold_))
    trn_data = lgb.Dataset(train.iloc[trn_idx][features], label=target.iloc[trn_idx])
    val_data = lgb.Dataset(train.iloc[val_idx][features], label=target.iloc[val_idx])

    num_round = 15000
    clf = lgb.train(opt_params, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=500, early_stopping_rounds = 250)
    oof[val_idx] = clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    predictions += clf.predict(test[features], num_iteration=clf.best_iteration) / folds.n_splits

print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))


cols = (feature_importance_df[["Feature", "importance"]]
        .groupby("Feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:20].index)
best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]

plt.figure(figsize=(20,28))
sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance",ascending=False))
plt.title('Features importance (averaged/folds)')
plt.tight_layout()
plt.savefig('Feature_Importance.png')


# <a id="5"></a> <br>
# # **Understanding the Model Better**
# <br>
# 
# To get an overview of which features are most important for a model, we can plot the SHAP values of every feature for every sample. The plot below sorts features by the sum of SHAP value magnitudes over all samples, and uses SHAP values to show the distribution of the impacts each feature has on the model output. [(source)](https://github.com/slundberg/shap) The color represents the feature value (red high, blue low). This reveals for example that a high var_139 lowers the probability of being a customer who will make a specific transaction in the future. 


explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X)

shap.summary_plot(shap_values, X)


# We can also plot a tree from the model and see each tree! 


#tree visualization
graph = lgb.create_tree_digraph(clf, tree_index=3, name='Tree3' )
graph.graph_attr.update(size="110,110")
graph


# # **Work in Progress..** <br><br>
# 
# Although I feel confident that I can implement Bayesian Optimization with LightGBM by Python and understand what my Python code does, I do not feel so comfortable with math behind it...😥  I will keep working on researching more about math behind Bayesian Optimization and share with you! <br><br>
# Here are some academic papers about Bayesian Optimization just in case you are interested in:<br>
# 1) http://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.pdf<br>
# 2) https://arxiv.org/pdf/1012.2599v1.pdf
# 
# <br>
# <center>**I hope you guys enjoyed my kernel and do not forget to upvote if you think that it's helpful!**</center> <br>
# 
# ![](https://media.giphy.com/media/osjgQPWRx3cac/giphy.gif)



