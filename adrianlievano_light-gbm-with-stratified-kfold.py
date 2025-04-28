# ## Background Information ##


# Financial institutions, like Santander, help people and businesses prosper by providing tools and services to assess their personal financial health and to identify additional ways to help customers reach their monetary goals. In the United States, it is estimated that 40% of Americans cannot cover a $400 emergency expense1. As a result, it is imperative that financial institutions learn consumer habits to adopt new technologies to better serve their financial needs. 


# ## Problem Statement ##
# Santander, a financial institution, is trying to predict the next transaction a given customer is trying to complete based on historical banking information. This is a binary classification problem where the input data contains 299 unnamed normally-distributed feature variables. The solution to this problem will be evaluated on a provided test data set by Santander. 


# ## Solution Statement ##


# The provided train.csv file contains 200,000 unique rows corresponding to customer data. Given the large dataset, and the need to complete binary classification, there are many solutions to this problem: mine will involve using a deep neural network, after preprocessing the inputs by normalizing and scaling features, to classify the two target variables. After the model is trained and validated on a subset of the data from the train.csv file, I will run my trained model on the provided test set from Santander and measure the accuracy of each prediction. 


# ## Load Data ##


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split


import os
print(os.listdir("../input"))

# Pretty display for notebooks
%matplotlib inline

# Load the Santander dataset
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
submission_data = pd.read_csv('../input/sample_submission.csv')


# ## Data Exploration & Visualizations ##


#Size of training data
train_data.shape


train_data['target'].head(5)


# ### Calculate Dataset Statistics ###


train_data.describe()


# Immediate Key Takeaways: The mean, standard deviation, and maximum values of the features vary widely; if we choose to implement a black-box algorithm, like neural networks, a key step will be data preprocessing, which will involve feature scaling, potentially outlier-detection and removal, and definitely normalization of these variables.


# ### Separate Training and Validation Datasets ###


X_train, X_val, y_train, y_val = train_test_split(feature_train_data, train_data['target'], test_size = 0.20, random_state = 25)


# From the dataset statistics, we can see that the mean and standard deviatation of each var feature significantly varies. If we apply supervised learning algorithms without any feature scaling or preprocessing, the algorithm will bias to certain features over others when learning the relationship between the input features and the target classification of the customer.


# ## Implement LGBM Algorithm ##


from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb


df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')


random_state = 42

lgb_params = {
    "objective" : "binary",
    "metric" : "auc",
    "boosting": 'gbdt',
    "max_depth" : -1,
    "num_leaves" : 13,
    "learning_rate" : 0.01,
    "bagging_freq": 5,
    "bagging_fraction" : 0.4,
    "feature_fraction" : 0.05,
    "min_data_in_leaf": 80,
    "min_sum_heassian_in_leaf": 10,
    "tree_learner": "serial",
    "boost_from_average": "false",
    #"lambda_l1" : 5,
    #"lambda_l2" : 5,
    "bagging_seed" : random_state,
    "verbosity" : 1,
    "seed": random_state
}


df_train.head()


skf = StratifiedKFold(n_splits = 5, shuffle=True, random_state=random_state)
skf.get_n_splits(X_train, y_train)


features = [col for col in df_train.columns if col not in ['target', 'ID_code']]
X_test = df_test[features].values
feature_importance_df = pd.DataFrame()
predictions = df_test[['ID_code']]

for fold, (trn_idx, val_idx) in enumerate(skf.split(df_train, df_train['target'])):
    print("FOLD: ", fold, "TRAIN:", train_index, "TEST:", test_index)
    X_train, y_train = df_train.iloc[trn_idx][features], df_train.iloc[trn_idx]['target']
    X_valid, y_valid = df_train.iloc[val_idx][features], df_train.iloc[val_idx]['target']
    
    N = 5
    p_valid = 0
    yp = 0
    
    for i in range(N):
        
        trn_data = lgb.Dataset(X_train, label = y_train)
        val_data = lgb.Dataset(X_valid, label = y_valid)
        
        lgb_clf = lgb.train(lgb_params,
                   trn_data,
                   100000,
                   valid_sets = [trn_data, val_data],
                    verbose_eval = 5000,
                    early_stopping_rounds = 3000)
        
        p_valid += lgb_clf.predict(X_valid)
        yp += lgb_clf.predict(X_test)
    
    
    #Get importance of the fold when predicting test set
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = lgb_clf.feature_importance()
    fold_importance_df["fold"] = fold + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    predictions['fold{}'.format(fold+1)] = yp/N


X_train.head()


# Stratified KFold is a variation of KFold that returns stratified folds. The folds are made by preserving the percentage of samples for each class. Especially for this problem, where we have an unbalanced binary classification issue, we need to keep this in mind.


predictions['target'] = np.mean(predictions[[col for col in predictions.columns if col not in ['ID_code', 'target']]].values, axis=1)
predictions.to_csv('lgb_all_predictions.csv', index=None)
sub_df = pd.DataFrame({"ID_code":df_test["ID_code"].values})
sub_df["target"] = predictions['target']
sub_df.to_csv("lgb_submission.csv", index=False)
oof.to_csv('lgb_oof.csv', index=False)


dfa = pd.DataFrame(np.random.randn(5, 4),
                    columns=list('ABCD'),
                   index=pd.date_range('20130101', periods=5))


dfa


dfa.columns


dfa.iloc[0:2][[col for col in dfa.columns if col not in ['B', 'C']] ]


col for col in df_train.columns if col not in ['target', 'ID_code']







































