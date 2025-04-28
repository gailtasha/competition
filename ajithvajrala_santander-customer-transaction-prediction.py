# **Project Aim**
# 
# * The aim of this project is to identify which customers will make a specific transaction in the future, irrespective of the amount of money transacted. 
# 
# **Data**
# * The anonymized dataset containing numeric feature variables, the binary target column, and a string ID_code column.


import gc
import os
import logging
import datetime
import warnings
import lightgbm
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
warnings.filterwarnings('ignore')


train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")


train_df.shape, test_df.shape


train_df.head()


x = train_df.drop(['ID_code', 'target'], axis=1)
y = train_df['target']
x_test = test_df.drop(['ID_code'], axis =1)


x.shape, x_test.shape, y.shape


mean_train = x.mean()
mean_test = x_test.mean()
plt.figure(figsize=(15,10))
plt.plot(mean_train, color='blue')
plt.plot(mean_test, color = 'pink')
plt.show();


pd.DataFrame([mean_test, mean_train])


std_train = x.std()
std_test = x_test.std()
plt.figure(figsize=(15,10))
plt.plot(std_train, color='blue')
plt.plot(std_test, color = 'pink')
plt.show();


pd.DataFrame([std_test, std_train])


#Mean Distribution

plt.figure(figsize=(16,6))
features = x.columns
plt.title("Distribution of mean values per row in the train and test set")
sns.distplot(x[features].mean(axis=1),color="green", kde=True,bins=120, label='train')
sns.distplot(x_test[features].mean(axis=1),color="blue", kde=True,bins=120, label='test')
plt.legend()
plt.show();


plt.figure(figsize=(16,6))
plt.title("Distribution of mean values per column in the train and test set")
sns.distplot(x[features].mean(axis=0),color="magenta",kde=True,bins=120, label='train')
sns.distplot(x_test[features].mean(axis=0),color="darkblue", kde=True,bins=120, label='test')
plt.legend()
plt.show()


plt.figure(figsize=(16,6))
plt.title("Distribution of std values per row in the train and test set")
sns.distplot(x[features].std(axis=1),color="black", kde=True,bins=120, label='train')
sns.distplot(x_test[features].std(axis=1),color="red", kde=True,bins=120, label='test')
plt.legend();
plt.show();


plt.figure(figsize=(16,6))
plt.title("Distribution of std values per column in the train and test set")
sns.distplot(x[features].std(axis=0),color="blue",kde=True,bins=120, label='train')
sns.distplot(x_test[features].std(axis=0),color="green", kde=True,bins=120, label='test')
plt.legend(); plt.show()


%%time
correlations = x[features].corr().abs().unstack().sort_values(kind="quicksort").reset_index()
correlations = correlations[correlations['level_0'] != correlations['level_1']]


correlations.tail(10)


gc.collect()


x_test = (x_test - mean_test) + (mean_train)
x_test = x_test / (std_train)
x = x / (std_train)


#features


for df in [x, x_test]:
        for feat in features:
            df['r2_'+feat] = np.round(df[feat], 2)
            df['r2_'+feat] = np.round(df[feat], 2)
            
        df['sum'] = df[features].sum(axis=1)  
        df['min'] = df[features].min(axis=1)
        df['max'] = df[features].max(axis=1)
        df['mean'] = df[features].mean(axis=1)
        df['std'] = df[features].std(axis=1)
        df['skew'] = df[features].skew(axis=1)
        df['kurt'] = df[features].kurtosis(axis=1)
        df['med'] = df[features].median(axis=1)
print('Train and test shape:',x.shape, x_test.shape)


model = ExtraTreesClassifier()
model.fit(x,y)


impotrant_featues = {}
for i, j in enumerate(model.feature_importances_):
    impotrant_featues[i] = j


variables = [k for k in sorted(impotrant_featues, key=impotrant_featues.get, reverse=True)][:50]


imp_features = []
for var in variables:
    if var <200:
        imp_features.append("var_"+str(var))


train_df[imp_features].shape


def dist(a,b):
    return distance.euclidean(a, b)

for i in imp_features:
    
    mean_1 = train_df[i][train_df['target']==1].mean()
    mean_0 = train_df[i][train_df['target']==0].mean()
    
    mean_1_col =  i + "_mean_1_dist"
    mean_0_col =  i + "_mean_0_dist"
    
    x[mean_1_col] = x.apply(lambda p: dist(p[i], mean_1), axis=1)
    x[mean_0_col] = x.apply(lambda p: dist(p[i], mean_0), axis=1)
    
    x_test[mean_1_col] = x_test.apply(lambda x: dist(x[i], mean_1), axis=1)
    x_test[mean_0_col] = x_test.apply(lambda x: dist(x[i], mean_0), axis=1)
    
    print("done ",i)


x.head()


 param = {
        'bagging_freq': 5,
        'bagging_fraction': 0.38,
        'boost_from_average':'false',
        'boost': 'gbdt',
        'feature_fraction': 0.045,
        'learning_rate': 0.0095,
        'max_depth': -1,  
        'metric':'auc',
        'min_data_in_leaf': 80,
        'min_sum_hessian_in_leaf': 10.0,
        'num_leaves': 13,
        'num_threads': 8,
        'tree_learner': 'serial',
        'objective': 'binary', 
        'verbosity': 1,
         'lambda_l1': 4.972,
        'lambda_l2': 2.276,
         'is_unbalance': True
    }


param = {'max_depth': 3, 
         'num_leaves': 6, 
         'min_child_samples': 200, 
         'scale_pos_weight': 1.0, 
         'subsample': 0.6, 
         'colsample_bytree': 0.6, 
         'metric': 'auc', 
         'nthread': 8, 
         'boosting_type': 'gbdt', 
         'objective': 'binary', 
         'learning_rate': 0.15, 
         'max_bin': 100, 
         'min_child_weight': 0, 
         'min_split_gain': 0, 
         'subsample_freq': 1,
         'is_unbalance': True,
        'bagging_freq': 5,
        'bagging_fraction': 0.38,
        'boost_from_average':'false',
          'feature_fraction': 0.045,
        'learning_rate': 0.0095,
         'min_data_in_leaf': 80,
        'min_sum_hessian_in_leaf': 10.0,
         'tree_learner': 'serial'}


param = {
    'bagging_freq': 5,
    'bagging_fraction': 0.335,
    'boost_from_average':'false',
    'boost': 'gbdt',
    'feature_fraction': 0.041,
    'learning_rate': 0.0083,
    'max_depth': -1,
    'metric':'auc',
    'min_data_in_leaf': 80,
    'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 13,
    'num_threads': 8,
    'tree_learner': 'serial',
    'objective': 'binary', 
    'verbosity': -1
}
gc.collect()


nfold = 5 #5
k =StratifiedKFold(n_splits=nfold, shuffle=True, random_state=49)
#k = RepeatedStratifiedKFold(n_splits = nfold, n_repeats=5, random_state=49)
oof = np.zeros(len(y))
predictions = np.zeros(len(x_test))

i =1

for train_idx, val_idx in k.split(x, y.values):
    print("\n fold {}".format(i))
    
    light_train = lightgbm.Dataset(x.iloc[train_idx].values,
                                  label = y.iloc[train_idx].values,
                                  free_raw_data = False)
    light_val = lightgbm.Dataset(x.iloc[val_idx].values,
                                label = y.iloc[val_idx].values,
                                free_raw_data = False)
    
    clf = lightgbm.train(param, light_train, 9000, valid_sets=[light_val], verbose_eval=200, early_stopping_rounds=100)
    
    oof[val_idx] = clf.predict(x.iloc[val_idx].values, num_iteration = clf.best_iteration)
    
    predictions += clf.predict(x_test.values, num_iteration= clf.best_iteration) /nfold
    
    i+=1


print("CV AUC: {:<0.2f}".format(metrics.roc_auc_score(y.values, oof)))


sub_df = pd.DataFrame()
sub_df['ID_code'] = test_df['ID_code']
sub_df['target'] = predictions
sub_df.to_csv("sub1.csv", index = False)



