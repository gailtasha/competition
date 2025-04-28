# ## Our goal
# 
# In this competition we are asked to predict if a customer will make a transaction or not regardless of the amount of money transacted. Hence our goal is to solve a binary classification problem. In the data description you can see that the features given are numeric and anonymized. Furthermore the data seems to be artificial as they state that "the data has the same structure as our real data". 
# 
# ### Table of contents
# 
# 1. [Loading packages](#load) (complete)
# 2. [Sneak a peek at the data](#data) (complete)
# 2. [What can we say about the target?](#target) (complete)
# 3. [Can we find relationships between features?](#correlation) (complete)
# 4. [Baseline submissions](#baselines)
# 5. [Feature engineering](#engineering)


# ## Loading packages <a class="anchor" id="load"></a>


# data manipulation
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set()

# sklearn models & tools
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer

# ignore warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import os
print(os.listdir("../input"))


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
submission = pd.read_csv("../input/sample_submission.csv")


# ## Sneak a peek at the data <a class="anchor" id="data"></a>


# ### Train


train.shape


# Ok, 200.000 rows and 202 features. 


X = train.drop(['ID_code','target'],axis=1)
y = train['target']

X_test = test.drop('ID_code',axis=1)


X['i_am_train'] = 1
X_test['i_am_train'] = 0


full = pd.concat([X,X_test],axis=0)


your_feature='var_68'
your_threshold=5.036
fig, ax = plt.subplots(1,1,figsize=(20,5))
plt.hist(full.loc[:, your_feature].values, bins=500);
ax.axvline(your_threshold, c="red")


full['var68_bin_index'] = np.digitize(full['var_68'],plt.hist(full.loc[:, your_feature].values, bins=500)[1])


# [242,259,226,275,210,194,291,178,307,162,146,130,113,81,97,323,339,355,371,388,404]


full['var_0' + "_bin"] = np.where(full.loc[:, 'var_0'] <= 12.1, 1, 0)
full['var_1' + "_bin"] = np.where(full.loc[:, 'var_1'] <= 7.3, 1, 0)
full['var_2' + "_bin"] = np.where(full.loc[:, 'var_2'] <= 12, 1, 0)
full['var_4' + "_bin"] = np.where(full.loc[:, 'var_4'] <= 14.1, 1, 0)
full['var_6' + "_bin"] = np.where(full.loc[:, 'var_6'] <= 5.6, 1, 0)
full['var_7' + "_bin"] = np.where(full.loc[:, 'var_7'] <= 10.55, 1, 0)
full['var_9' + "_bin"] = np.where(full.loc[:, 'var_9'] <= 8, 1, 0)
full['var_12' + "_bin"] = np.where(full.loc[:, 'var_12'] <= 14, 1, 0)
full['var_12_weirdos'] = 0
full.loc[np.round(full['var_12'],3).isin([13.554,13.555]), 'var_12_weirdos'] = 1
full['var_13' + "_bin"] = np.where(full.loc[:, 'var_13'] <= 9.7, 1, 0)
full['var_16' + "_bin"] = np.where(full.loc[:, 'var_16'] <= 10.6, 1, 0)
full['var_26' + "_bin"] = np.where(full.loc[:, 'var_26'] <= 8, 1, 0)
full['var_27' + "_bin"] = np.where(full.loc[:, 'var_27'] <= 2.1, 1, 0)
full['var_28' + "_bin"] = np.where(full.loc[:, 'var_28'] <= 5.55, 1, 0)
full['var_29' + "_bin"] = np.where(full.loc[:, 'var_29'] <= 11, 1, 0)
full['var_35' + "_bin"] = np.where(full.loc[:, 'var_35'] <= 14, 1, 0)
full['var_37' + "_bin"] = np.where(full.loc[:, 'var_37'] <= 11, 1, 0)
full['var_40' + "_bin"] = np.where(full.loc[:, 'var_40'] <= 10.8, 1, 0)
full['var_41' + "_bin"] = np.where(full.loc[:, 'var_41'] <= 9.2, 1, 0)
full['var_43' + "_bin"] = np.where(full.loc[:, 'var_43'] <= 10.95, 1, 0)
full['var_53' + "_bin"] = np.where(full.loc[:, 'var_53'] <= 7.74, 1, 0)
full['var_55' + "_bin"] = np.where(full.loc[:, 'var_55'] <= 10.1, 1, 0)
full['var_59' + "_bin"] = np.where(full.loc[:, 'var_59'] <= 7.37, 1, 0)
full['var_60' + "_bin"] = np.where(full.loc[:, 'var_60'] <= 12, 1, 0)
full['var_68_highcounts'] = 0
full.loc[full['var68_bin_index'].isin([242,259,226,275,210,194,291,178,307,162,146,130,113,81,97,323,339,355,371,388,404]), 'var_68_highcounts'] = 1
full['var_73' + "_bin"] = np.where(full.loc[:, 'var_73'] <= 6, 1, 0)
full['var_75' + "_bin"] = np.where(full.loc[:, 'var_75'] <= 3.3, 1, 0)
full['var_80' + "_bin"] = np.where(full.loc[:, 'var_80'] <= 3.15, 1, 0)
full['var_81' + "_bin"] = np.where(full.loc[:, 'var_81'] <= 9.5, 1, 0)
full['var_86' + "_bin"] = np.where(full.loc[:, 'var_86'] <= 2, 1, 0)
full['var_88' + "_bin"] = np.where(full.loc[:, 'var_88'] <= 4, 1, 0)
full['var_89' + "_bin"] = np.where(full.loc[:, 'var_89'] <= 12.2, 1, 0)
full['var_92' + "_bin"] = np.where(full.loc[:, 'var_92'] <= 4.7, 1, 0)
full['var_95' + "_bin"] = np.where(full.loc[:, 'var_95'] <= 1.2, 1, 0)
full['var_98' + "_bin"] = np.where(full.loc[:, 'var_98'] <= 0.3, 1, 0)
full['var_99' + "_bin"] = np.where(full.loc[:, 'var_99'] <= 3.6, 1, 0)
full['var_101' + "_bin"] = np.where(full.loc[:, 'var_101'] <= 2.5, 1, 0)
full['var_108' + "_bin"] = np.where(full.loc[:, 'var_108'] <= 14.2155, 1, 0)
full['var_115' + "_bin"] = np.where(full.loc[:, 'var_115'] <= 2.4, 1, 0)
full['var_123' + "_bin"] = np.where(full.loc[:, 'var_123'] <= 5.65, 1, 0)
full['var_126' + "_bin"] = np.where(full.loc[:, 'var_126'] <= 11.55, 1, 0)
full['var_129' + "_bin"] = np.where(full.loc[:, 'var_129'] <= 7.1, 1, 0)
full['var_131' + "_bin"] = np.where(full.loc[:, 'var_131'] <= 0.75, 1, 0)
full['var_134' + "_bin"] = np.where(full.loc[:, 'var_134'] <= 8.5, 1, 0)
full['var_135' + "_bin"] = np.where(full.loc[:, 'var_135'] <= 11, 1, 0)
full['var_139' + "_bin"] = np.where(full.loc[:, 'var_139'] <= 4, 1, 0)
full['var_141' + "_bin"] = np.where(full.loc[:, 'var_141'] <= 5, 1, 0)
full['var_145' + "_bin"] = np.where(full.loc[:, 'var_145'] <= 12.8, 1, 0)
full['var_150' + "_bin"] = np.where(full.loc[:, 'var_150'] <= 12.4, 1, 0)
full['var_151' + "_bin"] = np.where(full.loc[:, 'var_151'] <= 9, 1, 0)
full['var_153' + "_bin"] = np.where(full.loc[:, 'var_153'] <= 12.7, 1, 0)
full['var_157' + "_bin"] = np.where(full.loc[:, 'var_157'] <= 8, 1, 0)
full['var_158' + "_bin"] = np.where(full.loc[:, 'var_158'] <= 2, 1, 0)
full['var_163' + "_bin"] = np.where(full.loc[:, 'var_163'] <= 12.5, 1, 0)
full['var_164' + "_bin"] = np.where(full.loc[:, 'var_164'] <= 7.5, 1, 0)
full['var_166' + "_bin"] = np.where(full.loc[:, 'var_166'] <= 2.15, 1, 0)
full['var_168' + "_bin"] = np.where(full.loc[:, 'var_168'] <= 6.6, 1, 0)
full['var_173' + "_bin"] = np.where(full.loc[:, 'var_173'] <= 10, 1, 0)
full['var_175' + "_bin"] = np.where(full.loc[:, 'var_175'] <= 11.6, 1, 0)
full['var_176' + "_bin"] = np.where(full.loc[:, 'var_176'] <= 10.8, 1, 0)
full['var_177' + "_bin"] = np.where(full.loc[:, 'var_177'] <= 7.3, 1, 0)
full['var_180' + "_bin"] = np.where(full.loc[:, 'var_180'] <= 9, 1, 0)
full['var_181' + "_bin"] = np.where(full.loc[:, 'var_181'] <= 10.76, 1, 0)
full['var_186' + "_bin"] = np.where(full.loc[:, 'var_186'] <= 4, 1, 0)
full['var_187' + "_bin"] = np.where(full.loc[:, 'var_187'] <= 13, 1, 0)
full['var_188' + "_bin"] = np.where(full.loc[:, 'var_188'] <= 8, 1, 0)
full['var_191' + "_bin"] = np.where(full.loc[:, 'var_191'] <= 8.2, 1, 0)
full['var_194' + "_bin"] = np.where(full.loc[:, 'var_194'] <= 11.5, 1, 0)
full['var_195' + "_bin"] = np.where(full.loc[:, 'var_195'] <= 3, 1, 0)
full['var_196' + "_bin"] = np.where(full.loc[:, 'var_196'] <= 13.5, 1, 0)


full.head()


# Split back into train and test
X = full.loc[full['i_am_train']==1]
X_test = full.loc[full['i_am_train']==0]

del full


del X['i_am_train'], del X_test['i_am_train']


X.shape


X_test.shape


# ### New feature importances


import lightgbm as lgb
import time


params = {'num_leaves': 13,
         'min_data_in_leaf': 80,
         'objective': 'binary',
         'max_depth': -1,
         'learning_rate': 0.01,
         'boosting': 'gbdt',
         'bagging_freq': 5,
         'bagging_fraction': 0.4,
         'feature_fraction': 0.627,
         'bagging_seed': 1337,
         #'reg_alpha': 1.728910519108444,
         #'reg_lambda': 4.9847051755586085,
         'random_state': 1337,
         'metric': 'auc',
         'verbosity': -1,
         #'subsample': 0.81,
         #'min_gain_to_split': 0.01077313523861969,
         #'min_child_weight': 19.428902804238373,
          'min_sum_hessian_in_leaf': 10,
         'num_threads': 4}


folds = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

prediction = np.zeros(len(X_test))
oofs = np.zeros(len(X))
for fold_n, (train_index, valid_index) in enumerate(folds.split(X,y)):
    print('Fold', fold_n, 'started at', time.ctime())
    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
    
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid)
        
    model = lgb.train(params,train_data,num_boost_round=99999,
                    valid_sets = [train_data, valid_data],verbose_eval=300,early_stopping_rounds = 1000)
            
    oofs[valid_index] = model.predict(X_valid, num_iteration = model.best_iteration)
    prediction += model.predict(X_test, num_iteration=model.best_iteration)/5


sub = pd.DataFrame({"ID_code": test.ID_code.values})
sub["target"] = prediction
sub.to_csv('submission_original_bin.csv', index=False)


oof = pd.DataFrame({"ID_code": train.ID_code.values})
oof["target"] = oofs
oof.to_csv("oofs_original_bin.csv", index=False)

