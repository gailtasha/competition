# <h1><center><font size="6">Santander EDA</font></center></h1>
# 
# <h2><center><font size="4">Dataset used: Santander Customer Transaction Prediction</font></center></h2>
# 
# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/4a/Another_new_Santander_bank_-_geograph.org.uk_-_1710962.jpg/640px-Another_new_Santander_bank_-_geograph.org.uk_-_1710962.jpg" width="500"></img>
# 
# <br>
# 
# # <a id='0'>Content</a>
# 
# - <a href='#1'>All imports necessary</a>  
# - <a href='#2'>A bit of configuration</a>  
# - <a href='#3'>List files available</a>     
# - <a href='#4'>Auxiliary methods</a>
# - <a href='#5'>Read the data</a>
# - <a href='#6'>Look at data</a>  
#     - <a href='#61'>Train</a>
#     - <a href='#62'>Test</a>
# - <a href='#7'>Look at differences in statistics</a>
#     - <a href='#71'>Between Train & Test</a>
#     - <a href='#72'>Between Positive & Negative classes</a>
#     - <a href='#73'>Between Test & Negative class</a>
#     - <a href='#74'>Between Test & Positive class</a>
# - <a href='#11'>Calculate Mann-Whitney U Test for all features</a>
#     - <a href='#111'>Between Train & Test</a>
#     - <a href='#112'>Between Positive & Negative classes</a>
#     - <a href='#111'>Between Test & Negative class</a>
#     - <a href='#112'>Between Test & Positive class</a>


# # <a id='1'>All imports necessary</a>


import os

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook

import numpy as np

import warnings

import scipy


# # <a id='2'>A bit of configuration</a>


warnings.filterwarnings("ignore")

%matplotlib inline

pd.options.display.max_rows = 10000

pd.options.display.max_colwidth = 1000


# # <a id='3'>List files available</a>


print(os.listdir("../input"))


# # <a id='4'>Auxiliary methods</a>


def calculate_h0_rejected(df1, df2, alpha=0.05):
    features = ['var_{}'.format(feature_number) for feature_number in range(200)]
    p_values = np.array(
        [
            scipy.stats.mannwhitneyu(
                df1[feature],
                df2[feature]
            )[1] for feature in tqdm_notebook(features)
        ])
    h0_rejected_hypotheses = p_values < alpha
    return pd.DataFrame(
        {
            'p_values': p_values,
            'h0_rejected_hypotheses': h0_rejected_hypotheses
        },
        index=features
    )


def calculate_statistics_distributions(df1, df2, names):
    statistics = ['mean', 'std', 'min', '25%', '50%', '75%', 'max']
    
    df1_features_stats = pd.melt(
        df1.describe().T.reset_index(),
        id_vars=['index'],
        value_vars=statistics
    )
    df1_features_stats['data_part'] = names[0]

    df2_features_stats = pd.melt(
        df2.describe().T.reset_index(),
        id_vars=['index'],
        value_vars=statistics
    )
    df2_features_stats['data_part'] = names[1]
    
    df1_df2_features_stats = pd.concat(
        [
            df1_features_stats,
            df2_features_stats
        ],
        ignore_index=True
    )
    
    return df1_df2_features_stats


# # <a id='5'>Read the data</a>


train = pd.read_csv('../input/train.csv', index_col=0)


test = pd.read_csv('../input/test.csv', index_col=0)


# # <a id='6'>Look at data</a>


# ## <a id='61'>Train</a>


train[train.columns.difference(['target'])].describe().T


train.head().T


train.info(verbose=True, null_counts=True)


# ## <a id='62'>Test</a>


test.describe().T


test.head().T


test.info(verbose=True, null_counts=True)


# # <a id='7'>Look at differences in statistics</a>


# ## <a id='71'>Between Train & Test</a>


train_test_features_stats = calculate_statistics_distributions(
    train[train.columns.difference(['target'])],
    test,
    ['train', 'test']
)


train_test_features_stats.head()


plt.figure(figsize=(20, 10))
sns.boxplot(x='variable', y='value', hue='data_part', data=train_test_features_stats)
plt.show()


plt.figure(figsize=(20, 10))
sns.boxplot(x='data_part', y='value', hue='variable', data=train_test_features_stats)
plt.show()


# ## <a id='72'>Between Positive & Negative classes</a>


pos_neg_features_stats = calculate_statistics_distributions(
    train[train.target == 1][train.columns.difference(['target'])],
    train[train.target == 0][train.columns.difference(['target'])],
    ['positive', 'negative']
)


pos_neg_features_stats.head()


plt.figure(figsize=(20, 10))
sns.boxplot(x='variable', y='value', hue='data_part', data=pos_neg_features_stats)
plt.show()


plt.figure(figsize=(20, 10))
sns.boxplot(x='data_part', y='value', hue='variable', data=pos_neg_features_stats)
plt.show()


# ## <a id='73'>Between Test & Negative class</a>


neg_test_features_stats = calculate_statistics_distributions(
    train[train.target == 0][train.columns.difference(['target'])],
    test,
    ['negative', 'test']
)


neg_test_features_stats.head()


plt.figure(figsize=(20, 10))
sns.boxplot(x='variable', y='value', hue='data_part', data=neg_test_features_stats)
plt.show()


plt.figure(figsize=(20, 10))
sns.boxplot(x='data_part', y='value', hue='variable', data=neg_test_features_stats)
plt.show()


# ## <a id='74'>Between Test & Positive class</a>


pos_test_features_stats = calculate_statistics_distributions(
    train[train.target == 1][train.columns.difference(['target'])],
    test,
    ['positive', 'test']
)


pos_test_features_stats.head()


plt.figure(figsize=(20, 10))
sns.boxplot(x='variable', y='value', hue='data_part', data=pos_test_features_stats)
plt.show()


plt.figure(figsize=(20, 10))
sns.boxplot(x='data_part', y='value', hue='variable', data=pos_test_features_stats)
plt.show()


# # <a id='8'>Calculate Mann-Whitney U Test for all features</a>


alpha = 0.001


# ## <a id='81'>Between Train & Test</a>


train_test_H0_rejected = calculate_h0_rejected(train, test, alpha)


plt.figure(figsize=(20, 10))
sns.countplot(x=train_test_H0_rejected.h0_rejected_hypotheses)
plt.show()


plt.figure(figsize=(20, 10))
sns.boxplot(x=train_test_H0_rejected.p_values)
plt.grid(True)
plt.show()


# ## <a id='82'>Between Positive & Negative classes</a>


neg_pos_H0_rejected = calculate_h0_rejected(train[train.target == 0], train[train.target == 1], alpha)


plt.figure(figsize=(20, 10))
sns.countplot(x=neg_pos_H0_rejected.h0_rejected_hypotheses)
plt.show()


plt.figure(figsize=(20, 10))
sns.boxplot(x=neg_pos_H0_rejected.p_values)
plt.grid(True)
plt.show()


# ## <a id='83'>Between Test & Negative class</a>


neg_test_H0_rejected = calculate_h0_rejected(train[train.target == 0], test, alpha)


plt.figure(figsize=(20, 10))
sns.countplot(x=neg_test_H0_rejected.h0_rejected_hypotheses)
plt.show()


plt.figure(figsize=(20, 10))
sns.boxplot(x=neg_test_H0_rejected.p_values)
plt.grid(True)
plt.show()


# ## <a id='84'>Between Test & Positive class</a>


pos_test_H0_rejected = calculate_h0_rejected(train[train.target == 1], test, alpha)


plt.figure(figsize=(20, 10))
sns.countplot(x=pos_test_H0_rejected.h0_rejected_hypotheses)
plt.show()


plt.figure(figsize=(20, 10))
sns.boxplot(x=pos_test_H0_rejected.p_values)
plt.grid(True)
plt.show()


# # THE NOTEBOOK IS NOT FINISHED YET...

