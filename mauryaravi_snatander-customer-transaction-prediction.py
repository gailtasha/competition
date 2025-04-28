#    <h1><b><i><u>Santander Exploratory Data Analysis</u></i></b></h1>
#    <hr>
#    ![Santander](https://imagesvc.meredithcorp.io/v3/mm/image?url=https%3A%2F%2Fmoneydotcomvip.files.wordpress.com%2F2016%2F07%2Fgettyimages-96378039.jpg&w=800&c=sc&poi=face&q=85)
#    <p>*Banco Santander, S.A., doing business as Santander Group, is a Spanish multinational commercial bank and financial services company founded and based in Santander, Spain. In addition to hubs in Madrid and Barcelona, Santander maintains a presence in all global financial centres as the largest Spanish banking institution in the world. Although known for its European banking operations, it has extended operations across North and South America, and more recently in continental Asia.*</p>
#    <p>*Many subsidiaries, such as Abbey National, have been rebranded under the Santander name. The company is a component of the Euro Stoxx 50 stock market index. In May 2016, Santander was ranked as 37th in the Forbes Global 2000 list of the world's biggest public companies. Santander is Spain’s largest bank.*</p>
# 
#    <p>*As of 2017, Santander is the 5th largest bank in Europe with approximately US D 1.4 trillion in total assets-under-management (AUM).Traded on the Euro Stoxx 50 stock market index, the bank has a total market capitalization of $69.9 billion.*</p>


# ***Load Packages***


import os
import datetime
import warnings
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')
mpl.style.use('seaborn-notebook')
sns.set_style('whitegrid')


print(os.listdir('../input'))


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train.shape, test.shape


# Training data has one extra column than Test data which indicates the target variable of dataset.
# 
# Let's take a look at the train and test data


train.head()


test.head()


# Train Contains:-
# *     **ID_code** as id variable (*String*)
# *     **target** as target variable (*Numeric*)
# *     **var_i** where i = (0,199) feature variables (*Numeric*)
# 
# Test Contains:-
# *     **ID_code** as id variable (*String*)
# *     **var_i** where i = (0,199) feature variables (*Numeric*)
# 
# Let's Check for Missing data


def missing(data):
    total = data.isnull().sum()
    percent = (data.isnull().sum()/data.isnull().count()*100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    return(np.transpose(tt))


missing(train)


# NO missing values in Training Data, So that's good.


missing(test)


# Also, NO missing values in Tests Data, So that's good.
# <hr>


# Numerical Analysis for Train and Test Data


train.describe()


test.describe()


# Few points to observe:-
# *     Mean, Std, Min, Max are almost close to each other for train and test data
# *     Std is quite large for both datasets
# *     Range for features is distributed largely
# 
# Let's make scaterr plot function to plot scatter plots for datasets


def scatter_plt(data1, data2, feat):
    i=0
    plt.figure()
    fig, ax = plt.subplots(4,4,figsize=(13,13))
    
    for feature in feat:
        i +=1
        plt.subplot(4,4,i)
        plt.scatter(data1[feature], data2[feature], marker='*')
        plt.xlabel(feature, fontsize=9)
    plt.show()


feat = ['var_0', 'var_1','var_2','var_3', 'var_4', 'var_5', 'var_6', 'var_7', 'var_8', 'var_9', 'var_10','var_11','var_12', 
        'var_13', 'var_14', 'var_15']
scatter_plt(train[::15], test[::15], feat)


sns.countplot(train['target'])


print(' {} % of 1 are there in Training \n {} % of 0 are there in Training'.format(((train['target'].value_counts()[1]/train.shape[0])*100),((train['target'].value_counts()[0]/train.shape[0])*100)))


# Data is highly Unbalanced w.r.t. target variable


def distribution_plt(data1, data2, label1, label2, feat):
    i = 0
    plt.figure()
    fig, ax = plt.subplots(10,10,figsize=(18,22))

    for feature in feat:
        i += 1
        plt.subplot(10,10,i)
        sns.kdeplot(data1[feature], bw=0.5,label=label1)
        sns.kdeplot(data2[feature], bw=0.5,label=label2)
        plt.xlabel(feature, fontsize=9)
        locs, labels = plt.xticks()
        plt.tick_params(axis='x', which='major', labelsize=6, pad=-6)
        plt.tick_params(axis='y', which='major', labelsize=6)
    plt.show();


feat = train.columns.values[2:102]
distribution_plt(train, test, 'train', 'test', feat)
feat = train.columns.values[102:202]
distribution_plt(train, test, 'train', 'test', feat)


# Train and Test data is distributed in balance to numeric values. 


features = train.columns.values[2:202]
correlations = train[features].corr().abs().unstack().sort_values(kind="quicksort").reset_index()
correlations = correlations[correlations['level_0'] != correlations['level_1']]


correlations.head(10)


correlations.tail(10)


features = train.columns.values[2:202]
unique_max_train = []
unique_max_test = []
for feature in features:
    values = train[feature].value_counts()
    unique_max_train.append([feature, values.max(), values.idxmax()])
    values = test[feature].value_counts()
    unique_max_test.append([feature, values.max(), values.idxmax()])


np.transpose((pd.DataFrame(unique_max_train, columns=['Feature', 'Max duplicates', 'Value'])).sort_values(by = 'Max duplicates', ascending=False).head(15))


np.transpose((pd.DataFrame(unique_max_test, columns=['Feature', 'Max duplicates', 'Value'])).sort_values(by = 'Max duplicates', ascending=False).head(15))


# ***Feature Engineering***


idx = features = train.columns.values[2:202]
for df in [test, train]:
    df['sum'] = df[idx].sum(axis=1)  
    df['min'] = df[idx].min(axis=1)
    df['max'] = df[idx].max(axis=1)
    df['mean'] = df[idx].mean(axis=1)
    df['std'] = df[idx].std(axis=1)
    df['skew'] = df[idx].skew(axis=1)
    df['kurt'] = df[idx].kurtosis(axis=1)
    df['med'] = df[idx].median(axis=1)


train[train.columns[202:]].head()


test[test.columns[201:]].head()


features = [c for c in train.columns if c not in ['ID_code', 'target']]
for feature in features:
    train['r2_'+feature] = np.round(train[feature], 2)
    test['r2_'+feature] = np.round(test[feature], 2)
    train['r1_'+feature] = np.round(train[feature], 1)
    test['r1_'+feature] = np.round(test[feature], 1)


features = [c for c in train.columns if c not in ['ID_code', 'target']]
target = train['target']
param = {
    'bagging_freq': 5,
    'bagging_fraction': 0.4,
    'boost_from_average':'false',
    'boost': 'gbdt',
    'feature_fraction': 0.05,
    'learning_rate': 0.01,
    'max_depth': -1,  
    'metric':'auc',
    'min_data_in_leaf': 80,
    'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 13,
    'num_threads': 8,
    'tree_learner': 'serial',
    'objective': 'binary', 
    'verbosity': 1
}


folds = StratifiedKFold(n_splits=10, shuffle=False, random_state=44000)
oof = np.zeros(len(train))
predictions = np.zeros(len(test))
feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
    print("Fold {}".format(fold_))
    trn_data = lgb.Dataset(train.iloc[trn_idx][features], label=target.iloc[trn_idx])
    val_data = lgb.Dataset(train.iloc[val_idx][features], label=target.iloc[val_idx])

    num_round = 1000000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 3000)
    oof[val_idx] = clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    predictions += clf.predict(test[features], num_iteration=clf.best_iteration) / folds.n_splits

print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))


sub = pd.DataFrame({"ID_code":test["ID_code"].values})
sub["target"] = predictions
sub.to_csv("submission.csv", index=False)


# > Refernce:-
# *     [Santander EDA and Predictions](https://www.kaggle.com/gpreda/santander-eda-and-prediction/notebook)

