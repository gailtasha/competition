# 1. 1. ## Kaggle competition  Santander customer transaction using LGBM
# 
# It's a fairly simple neural network with enhancements targeted for later. 
# Steps:
# 1. Load data
# 2. Add features
# 3. Model training using 5-fold StratifiedKfol


!rm -r /opt/conda/lib/python3.6/site-packages/lightgbm
!git clone --recursive https://github.com/Microsoft/LightGBM


!apt-get install -y -qq libboost-all-dev


%%bash
cd LightGBM
rm -r build
mkdir build
cd build
cmake -DUSE_GPU=1 -DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=/usr/local/cuda/include/ ..
make -j$(nproc)


!cd LightGBM/python-package/;python3 setup.py install --precompile


!mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd
!rm -r LightGBM


!nvidia-smi


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix,f1_score, precision_recall_curve, average_precision_score, roc_auc_score, confusion_matrix

import matplotlib.pyplot as plt
%matplotlib inline
import gc

pd.set_option('display.max_columns', 1000)
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns

plt.style.use('ggplot')
import os
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)


%%time
df_train_raw = pd.read_csv('../input/train.csv')
df_test_raw = pd.read_csv('../input/test.csv')


%%time
df_train = df_train_raw.copy()
df_test = df_test_raw.copy()


train_cols = [col for col in df_train.columns if col not in ['ID_code', 'target']]
y_train = df_train['target']


df_train.shape


interactions= {'var_81':['var_53','var_139','var_12','var_76'],
               'var_12':['var_139','var_26','var_22', 'var_53','var_110','var_13'],
               'var_139':['var_146','var_26','var_53', 'var_6', 'var_118'],
               'var_53':['var_110','var_6'],
              'var_26':['var_110','var_109','var_12'],
              'var_118':['var_156'],
              'var_9':['var_89'],
              'var_22':['var_28','var_99','var_26'],
              'var_166':['var_110'],
              'var_146':['var_40','var_0'],
              'var_80':['var_12']}


%%time
for col in train_cols:
        df_train[col+'_2'] = df_train[col] * df_train[col]
        df_train[col+'_3'] = df_train[col] * df_train[col]* df_train[col]
#         df_train[col+'_4'] = df_train[col] * df_train[col]* df_train[col]* df_train[col]
        df_test[col+'_2'] = df_test[col] * df_test[col]
        df_test[col+'_3'] = df_test[col] * df_test[col]* df_test[col]


%%time
for df in [df_train, df_test]:
    df['sum'] = df[train_cols].sum(axis=1)  
    df['min'] = df[train_cols].min(axis=1)
    df['max'] = df[train_cols].max(axis=1)
    df['mean'] = df[train_cols].mean(axis=1)
    df['std'] = df[train_cols].std(axis=1)
    df['skew'] = df[train_cols].skew(axis=1)
    df['kurt'] = df[train_cols].kurtosis(axis=1)
    df['med'] = df[train_cols].median(axis=1)


%%time
for key in interactions:
    for value in interactions[key]:
        df_train[key+'_'+value+'_mul'] = df_train[key]*df_train[value]
        df_train[key+'_'+value+'_div'] = df_train[key]/df_train[value]
        df_train[key+'_'+value+'_sum'] = df_train[key] + df_train[value]
        df_train[key+'_'+value+'_sub'] = df_train[key] - df_train[value]
        
        df_test[key+'_'+value+'_mul'] = df_test[key]*df_test[value]
        df_test[key+'_'+value+'_div'] = df_test[key]/df_test[value]
        df_test[key+'_'+value+'_sum'] = df_test[key] + df_test[value]
        df_test[key+'_'+value+'_sub'] = df_test[key] - df_test[value]


df_train['num_zero_rows'] = (df_train_raw[train_cols] == 0).astype(int).sum(axis=1)
df_test['num_zero_rows'] = (df_test_raw[train_cols] == 0).astype(int).sum(axis=1)


df_train.head()


all_columns = [col for col in df_train.columns if col not in ['ID_code', 'target']]


# ### Start LGBM


params = {
        'num_leaves': 13,
        'max_bin': 63,
        'min_data_in_leaf': 80,
        'learning_rate': 0.0081,
        'min_sum_hessian_in_leaf': 10.0,
        'bagging_fraction': 0.331, 
        'bagging_freq': 5, 
        'max_depth': -1,
        'save_binary': True,
        'feature_fraction': 0.041,
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'verbose': 1,
        'metric': 'auc',
#         'is_unbalance': True,
        'boost_from_average': False,
        'device': 'gpu',
        'gpu_platform_id':0,
        'gpu_device_id': 0,
        'seed':44000
    }

num_round = 20000


%%time

folds = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 44000)
oof = np.zeros(len(df_train))

predictions = np.zeros(len(df_test))
feature_import_df = pd.DataFrame()

for n_fold, (train_idx, val_idx) in enumerate(folds.split(df_train, y_train)):
    print("fold number =", n_fold+1)
    train_data = lgb.Dataset(df_train.iloc[train_idx][all_columns], label = y_train.iloc[train_idx])
    val_y = y_train.iloc[val_idx]
    val_data = lgb.Dataset(df_train.iloc[val_idx][all_columns], label = val_y)
    
    
    watchlist = [train_data,val_data]
    clf = lgb.train(params, train_data, num_boost_round = num_round,
                   valid_sets = watchlist, verbose_eval = 4000,
                   early_stopping_rounds=3000)
    
    oof[val_idx] = clf.predict(df_train.iloc[val_idx][all_columns], num_iteration=clf.best_iteration)
    
    fold_import_df = pd.DataFrame()
    fold_import_df['Feature'] = all_columns
    fold_import_df["importance"] = clf.feature_importance()
    fold_import_df['fold'] = n_fold +1
    feature_import_df = pd.concat([feature_import_df,fold_import_df], axis = 0)
    
    predictions += clf.predict(df_test[all_columns])/folds.n_splits
    
    print("\tFold AUC Score: {}\tf1_score: {}\n".format(roc_auc_score(val_y,oof[val_idx]),
                                                       f1_score(val_y,np.round(oof[val_idx]))))
    gc.collect()
          
print("\n CV AUC Score and std", roc_auc_score(y_train, oof),np.std(oof))
print("CV F1 Score", f1_score(y_train, np.round(oof)))


# ### Checking best iteration


# ### Preparing submission file


sub = pd.DataFrame({'ID_code': df_test.ID_code.values,
                   'target': predictions})
sub.to_csv('lgbm_0401_kernelgpu.csv', index = False)



