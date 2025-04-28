# Copied from Santander EDA and Prediction, by Gabriel Preda


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
import os
import logging
import datetime
import warnings
import seaborn as sns
import lightgbm as lgb
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings('ignore')


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


print(dirname)


%%time
train_df = pd.read_csv(os.path.join(dirname, 'train.csv'))
test_df = pd.read_csv(os.path.join(dirname, 'test.csv'))


train_df.shape, test_df.shape


train_df.head()


test_df.head() 


def missing_data(data):
    total = data.isnull().sum()
    percent = (total/data.isnull().count())*100
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types=[]
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    return np.transpose(tt)


%%time
missing_data(train_df)


%%time
missing_data(test_df)


%%time
train_df.describe()


%%time
test_df.describe()


def plot_feature_scatter(df1, df2, features):
    sns.set_style('whitegrid')
#     plt.figure()
    fig, ax = plt.subplots(4,4,figsize=(14,14))
    
    i=0
    
#     for feature in features:
#         i+=1
#         plt.subplot(4,4,i)
#         plt.scatter(df1[feature], df2[feature], marker='+')
#         plt.xlabel(feature, fontsize=9)
    
    for feature in features:
        axis = ax.reshape(-1)[i]
        axis.scatter(df1[feature], df2[feature], marker='+')
        axis.set_xlabel(feature, fontsize=9)
        i+=1
        
    plt.show()


features = ['var_0', 'var_1','var_2','var_3', 'var_4', 'var_5', 'var_6', 'var_7', 
           'var_8', 'var_9', 'var_10','var_11','var_12', 'var_13', 'var_14', 'var_15', 
           ]
plot_feature_scatter(train_df[::20], test_df[::20], features)


sns.countplot(train_df['target'], palette='Set3')


print("% of target values with 1:")
print(100*train_df['target'].value_counts()[1]/train_df.shape[0])


def plot_feature_distribution()

