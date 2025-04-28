# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
import lightgbm as lgb
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
import time

SEED = 50

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


dt_train = pd.read_csv("../input/train.csv")
print("Number of Training Examples = {}" .format(dt_train.shape))


dt_test = pd.read_csv("../input/test.csv")
print("Number of Test Examples = {}" .format(dt_test.shape))


#Here we print in the columns on our TRAIN and TEST Datasets
print(dt_train.columns)
print(dt_test.columns)


#We use the .info to check informations about our dataset and then the .head to get 5 details from our train data  
print(dt_train.info())
dt_train.head()


#We use the .info to check informations about our dataset and then the .head to get 5 details from our test data 
print(dt_test.info())
dt_test.head()


#we check for numeric missing values
dt_train.name = "Training Dataset"
dt_test.name = "Test Dataset"
for dt in [dt_train, dt_test]:
    print('{} have {} missing values.'.format(dt.name, int(dt.isnull().values.any())))


#We checked to see how many percent of class 1 and 0 we have in our dataset (using the Target column)
ones = dt_train['target'].value_counts()[1]
zeros = dt_train['target'].value_counts()[0]
ones_per = ones / dt_train.shape[0] * 100
zeros_per = zeros / dt_train.shape[0] * 100

print('{} out of {} rows are Class 1 and it is the {:.2f}% of the dataset.'.format(ones, dt_train.shape[0], ones_per))
print('{} out of {} rows are Class 0 and it is the {:.2f}% of the dataset.'.format(zeros, dt_train.shape[0], zeros_per))


#we plot a graph to show or visual the percentage of each class showing who has the largest and the lowest
plt.figure(figsize=(10, 8))
sns.countplot(dt_train['target'])

plt.xlabel('Target')
plt.xticks((0, 1), ['Class 0 ({0:.2f}%)'.format(zeros_per), 'Class 1 ({0:.2f}%)'.format(ones_per)])
plt.ylabel('Count')
plt.title('Training Set Target Distribution')

plt.show()


#we try to see how many uniques values are there in each column on our train and test datasets
dt_train_unique = dt_train.agg(['nunique']).transpose().sort_values(by='nunique')
dt_test_unique = dt_test.agg(['nunique']).transpose().sort_values(by='nunique')
dt_uniques = dt_train_unique.drop('target').reset_index().merge(dt_test_unique.reset_index(), how='right', right_index=True, left_index=True)
print(dt_uniques.head())


#since both the x and y index are both showing same columns above, we had to delete one for easy reading
dt_uniques.drop(columns=['index_y'], inplace=True)
dt_uniques.columns = ['Feature', 'Training Set Unique Count', 'Test Set Unique Count']
print(dt_uniques.head())


#we plot a joint graph that shows that the train set has more unique values than the test set
sns.jointplot(x='Training Set Unique Count', y='Test Set Unique Count', data=dt_uniques, size=10, space=10, ratio=15, kind="scatter")
sns.set(style="darkgrid")


#features = [c for e in dt_train.columns if c not in ['ID_code', 'target']
cols=["target","ID_code"]
X = dt_train.drop(cols, axis=1)
y = dt_train.target


params = {'objective': "binary",
          'boost': "gbdt",
          'boost_from_average': "false",
          'num_threads': 8,
          'learning rate': 0.001,
          'num_leaves': 4,
          'max_depth': 8,
          'tree_learner': "serial",
          'feature_fraction': 0.05,
          'bagging_fraq': 1.0,
          'bagging_fraction': 0.4,
          'min_data_in_leaf': 80,
          'verbosity': 1,
          'metric': 'auc',
          'sigmoid':1.0,
          'lambda_l1':0.01,
          'snapshot_freq':1
}


folds = StratifiedKFold(n_splits=5, shuffle=False, random_state=10)
length_train = np.zeros(len(dt_train))
dt_predictions = np.zeros(len(dt_test))
important_dt = pd.DataFrame()


for fold_n, (train_index, valid_index) in enumerate (folds.split(X, y)):
    print("folds {}," "fold_n {}," "started at {}".format(folds, fold_n, time.ctime()))
    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
    
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid)
    
    lgb_model = lgb.train(params, train_data, num_boost_round=2000,
                         valid_sets = [train_data, valid_data], verbose_eval=1000,
                         early_stopping_rounds = 2500, )
    
    length_train[valid_index] = lgb_model.predict(X_valid, num_iteration=lgb_model.best_iteration) 
    

    
    


dt_tes = dt_test.drop(['ID_code'], axis=1)
print(dt_tes.head())


    dt_predictions += lgb_model.predict(dt_tes, num_iteration = lgb_model.best_iteration)/folds.n_splits
    
print("CV Score: {:<8.5f}".format(roc_auc_score(y, length_train)))


dt_predictio = pd.DataFrame(dt_predictions)


submit = pd.DataFrame({"ID_code": dt_test["ID_code"].values})
submit["target"] = dt_predictio
submit.to_csv("submit.csv", index=False)


new = pd.read_csv("submit.csv")


new.info()

