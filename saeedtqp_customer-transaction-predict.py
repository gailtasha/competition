import gc
import os
import logging
import datetime
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import lightgbm as lgb
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score , roc_curve
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings('ignore')


PATH = '../input/santander-customer-transaction-prediction/'


# Load the train and test data files.

%time
train_df = pd.read_csv(PATH +'train.csv')
test_df = pd.read_csv(PATH + 'test.csv')


# Data Exploration

train_df.shape , test_df.shape


# Glimpse train and test dataset

train_df.head()


test_df.head()


def missing_data(data):
    
    total = data.isnull().sum()
    percent = (data.isnull().sum() / data.isnull().count() * 100 )
    tt = pd.concat([total , percent] , axis = 1 , keys = ['Total','Percent'])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt['types'] = types
    return np.transpose(tt)


%time
missing_data(train_df)


# Check Test dataset

%time
missing_data(test_df)


%time
train_df.describe()


%time
test_df.describe()


# standard deviation is relatively large for both train and test variable data;
# min, max, mean, sdt values for train and test data looks quite close;
# mean values are distributed over a large range.


# ScatterPlot for train and test Dataset

def plot_feature_scatter(df1 , df2 ,features):
    i = 0
    sns.set_style('whitegrid')
    plt.figure()
    fig , ax = plt.subplots(4,4,figsize = (14,14))

    for feature in features:
        i += 1
        plt.subplot(4,4,i)
        plt.scatter(df1[feature], df2[feature],marker = '+')
        plt.xlabel(feature , fontsize = 9)
    plt.show();


# Show just 5% of the data

features = ['var_0','var_1','var_2','var_3','var_4','var_5','var_6','var_7',
           'var_8','var_9','var_10','var_11','var_12','var_13','var_14','var_15']

plot_feature_scatter(train_df[::20],test_df[::20],features)


# Check The distribution of target value

sns.countplot(train_df['target'],palette = 'Set3')


print('There are {}% target values with 1'.format(100 * train_df['target'].value_counts()[1] / train_df.shape[0]))


# **The data is unbalanced with respect with target value.**


def plot_feature_distribution(df1, df2, label1, label2, features):
    i = 0
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(10,10,figsize=(18,22))

    for feature in features:
        i += 1
        plt.subplot(10,10,i)
        sns.distplot(df1[feature], hist=False,label=label1)
        sns.distplot(df2[feature], hist=False,label=label2)
        plt.xlabel(feature, fontsize=9)
        locs, labels = plt.xticks()
        plt.tick_params(axis='x', which='major', labelsize=6, pad=-6)
        plt.tick_params(axis='y', which='major', labelsize=6)
    plt.show();


t0 = train_df.loc[train_df['target'] == 0]
t1 = train_df.loc[train_df['target'] == 1]
features = train_df.columns.values[2:102]
plot_feature_distribution(t0 , t1 , '0','1',features)


# next

features  = train_df.columns.values[102:202]
plot_feature_distribution(t0 , t1 , '0','1',features)


#Distribution of data 

features = train_df.columns.values[2:102]
plot_feature_distribution(train_df , test_df , 'train','test',features)


# Next

features = train_df.columns.values[102:202]
plot_feature_distribution(train_df , test_df , 'train','test',features)


# **The train and test seems to be well ballanced with respect with distribution of the numeric variables.**


# Check the distribution of the mean values per row in the train and test set.
# axis = 1 = row
plt.figure(figsize = (16,6))
features = train_df.columns.values[2:202]
plt.title('Distribution of mean values per row in the train and test set')
sns.distplot(train_df[features].mean(axis = 1),color = 'green',kde = True , bins = 120 , label = 'train')
sns.distplot(test_df[features].mean(axis = 1),color = 'blue',kde = True,bins = 120 , label = 'test')
plt.legend()
plt.show()


# Check the distribution of the mean values per columns in the train and test set.
# axis = 0  = columns

plt.figure(figsize =(16,6))
plt.title('Distriution of mean values per column in the train and test set')
sns.distplot(train_df[features].mean(axis = 0),color = 'magenta',kde = True,bins = 120 , label = 'train')
sns.distplot(test_df[features].mean(axis = 0),color ='darkblue',kde = True , bins = 120 , label = 'test')
plt.legend()
plt.show();


# Distribution of standard deviation of values per row for train and test dataset

plt.figure(figsize = (16,6))
plt.title('Distribution of std values per row in the train and test set')
sns.distplot(train_df[features].std(axis = 1),color = 'black',kde = True,bins = 120 , label = 'train')
sns.distplot(test_df[features].std(axis = 1),color = 'red', kde = True  , bins =120 , label = 'test')
plt.legend()
plt.show();


# Distribution of the standard deviation of values per columns in the train and test dataset.

plt.figure(figsize = (16,6))
plt.title('Distribution of std values per column in the train and test set')
sns.distplot(train_df[features].std(axis = 0),color = 'blue',kde = True ,bins = 120, label = 'train')
sns.distplot(test_df[features].std(axis = 0 ),color = 'green',kde = True , bins = 120 , label = 'test')
plt.legend()
plt.show()


# Check the distribution of the mean value per row in the train dataset , grouped by value of target

t0 = train_df.loc[train_df['target'] == 0]
t1 = train_df.loc[train_df['target'] == 1]
plt.figure(figsize = (16,6))
plt.title('Distribution of mean values per row in the train set')
sns.distplot(t0[features].mean(axis = 1),color = 'red',kde = True , bins = 120 , label = 'target = 0')
sns.distplot(t1[features].mean(axis = 1),color = 'blue',kde = True,bins = 120 , label = 'target = 1')
plt.legend()
plt.show();


# Check the distribution of the mean value per column , grouped by value of target.

plt.figure(figsize = (16,6))
plt.title('distribution of mean values per column in the train set')
sns.distplot(t0[features].mean(axis = 0),color = 'green',kde = True , bins = 120,label = 'target = 0')
sns.distplot(t1[features].mean(axis = 0),color = 'blue',kde = True , bins = 120 , label = 'target = 1')
plt.legend()
plt.show();


# Check the distribution of min per row in the train and test set

plt.figure(figsize = (16,6))
features = train_df.columns.values[2:202]
plt.title('Distribution of min values per row in the train and test set')
sns.distplot(train_df[features].min(axis = 1),color = 'red',kde = True , bins = 120 , label = 'train')
sns.distplot(test_df[features].min(axis = 1),color = 'orange',kde = True , bins = 120 , label = 'test')
plt.legend()
plt.show();


# check the distribtuion of min per column in the train and test set

plt.figure(figsize =(16,6))
features = train_df.columns.values[2:202]
plt.title('Distribution of min values per column in the train and test set')
sns.distplot(train_df[features].min(axis = 0),color = 'magenta',kde = True , bins = 120, label = 'train')
sns.distplot(test_df[features].min(axis = 0),color = 'darkblue',kde = True , bins =120,label = 'test')
plt.legend()
plt.show()


# Check the distribution of max values per rows for train and test set
plt.figure(figsize = (16,6))
features = train_df.columns.values[2:202]
plt.title('Distribution of max values per row in the train and test set')
sns.distplot(train_df[features].max(axis = 1),color = 'brown',kde = True , bins =120 , label = 'train')
sns.distplot(test_df[features].max(axis = 1),color = 'yellow',kde = True , bins =120 , label = 'test')
plt.legend()
plt.show()


# Check max distribution on columns for train and test set

plt.figure(figsize = (16,6))
features = train_df.columns.values[2:202]
plt.title('Distribution of max values per column in the train and test set')
sns.distplot(train_df[features].max(axis = 0), color = 'blue',kde = True , bins = 120 , label = 'train')
sns.distplot(test_df[features].max(axis = 0),color = 'red',kde = True , bins =120 , label = 'test')
plt.legend()
plt.show()


# Check distribution of min values per row in tran set, separated on the values of target(0 and 1)

t0 = train_df.loc[train_df['target'] == 0]
t1 = train_df.loc[train_df['target'] == 1]
plt.figure(figsize = (16,6))
plt.title('Distribution of min values per row in the train set')
sns.distplot(t0[features].min(axis = 1),color = 'orange',kde = True, bins =120 , label ='train = 0')
sns.distplot(t1[features].min(axis = 1),color = 'darkblue',kde = True , bins =120 , label = 'train = 1')
plt.legend()
plt.show();


# Check distribution of min values per row in train set

plt.figure(figsize = (16,6))
plt.title('Distribution of min values per column in the train set')
sns.distplot(t0[features].min(axis = 0),color = 'red',kde = True , bins =120 , label = 'target =0')
sns.distplot(t1[features].min(axis = 0),color = 'blue',kde = True , bins = 120 , label = 'target = 1')
plt.legend()
plt.show();


# Check distribution of max values per rown in the train set

plt.figure(figsize = (16,6))
plt.title('Distribution of max values per row in the train set')
sns.distplot(t0[features].max(axis = 1),color = 'gold',kde = True, bins = 120, label = 'target = 0')
sns.distplot(t1[features].max(axis = 1),color = 'darkblue',kde = True , bins = 120 ,label = 'target = 1')
plt.legend()
plt.show();


# Check distribution of max values per columns in the train set

plt.figure(figsize = (16,6))
plt.title('Distribution of max values per column in the train set')
sns.distplot(t0[features].max(axis = 0),color = 'red',kde = True , bins = 120 , label = 'target = 0')
sns.distplot(t1[features].max(axis = 0),color = 'blue',kde = True , bins =120 , label = 'target  = 1')
plt.legend()
plt.show();


# Check distribution of skewness calculated per rows in train and test sets.


# **It is the degree of distortion from the symmetrical bell curve or the normal distribution. It measures the lack of symmetry in data distribution.
# It differentiates extreme values in one versus the other tail. A symmetrical distribution will have a skewness of 0.
# There are two types of Skewness: Positive and Negative**


# ![](https://miro.medium.com/max/600/1*nj-Ch3AUFmkd0JUSOW_bTQ.jpeg)


plt.figure(figsize = (16,6))
plt.title('Distribution of skew per row in the train and test set')
sns.distplot(train_df[features].skew(axis = 1),color = 'red',kde = True , bins = 120,label = 'train')
sns.distplot(train_df[features].skew(axis = 1),color = 'orange',kde = True , bins = 120 ,label = 'test')
plt.legend()
plt.show()


# See first the distribution of skewness calculated per columns

plt.figure(figsize = (16,6))
plt.title('Distribution of skew per column in the train and test set')
sns.distplot(train_df[features].skew(axis = 0),color = 'magenta',kde = True , bins = 120 , label = 'train')
sns.distplot(test_df[features].skew(axis = 0), color = 'darkblue',kde = True , bins = 120 , label = 'test')
plt.legend()
plt.show()


# Check distribution of kurtosis values per rows and columns


# **Kurtosis
# Kurtosis is all about the tails of the distribution — not the peakedness or flatness. It is used to describe the extreme values in one versus the other tail. It is actually the measure of outliers present in the distribution.**


# ![](http://miro.medium.com/max/371/1*Nqu07THa7APRTOF7kaVr5Q.jpeg)


plt.figure(figsize = (16,6))
plt.title('Distribution of kurtosis per row in the train and test set')
sns.distplot(train_df[features].kurtosis(axis = 1),color = 'darkblue',kde = True , bins = 120, label = 'train')
sns.distplot(test_df[features].kurtosis(axis  = 1),color = 'yellow',kde = True , bins = 120 , label = 'test')
plt.legend()
plt.show()


# Check the distribution of kurtosis calculated per columns in train and test sets

plt.figure(figsize=(16,6))
plt.title("Distribution of kurtosis per column in the train and test set")
sns.distplot(train_df[features].kurtosis(axis=0),color="magenta", kde=True,bins=120, label='train')
sns.distplot(test_df[features].kurtosis(axis=0),color="green", kde=True,bins=120, label='test')
plt.legend()
plt.show()


t0 = train_df.loc[train_df['target'] == 0]
t1 = train_df.loc[train_df['target'] == 1]
plt.figure(figsize=(16,6))
plt.title("Distribution of skew values per row in the train set")
sns.distplot(t0[features].skew(axis=1),color="red", kde=True,bins=120, label='target = 0')
sns.distplot(t1[features].skew(axis=1),color="blue", kde=True,bins=120, label='target = 1')
plt.legend(); plt.show()


t0 = train_df.loc[train_df['target'] == 0]
t1 = train_df.loc[train_df['target'] == 1]
plt.figure(figsize=(16,6))
plt.title("Distribution of skew values per column in the train set")
sns.distplot(t0[features].skew(axis=0),color="red", kde=True,bins=120, label='target = 0')
sns.distplot(t1[features].skew(axis=0),color="blue", kde=True,bins=120, label='target = 1')
plt.legend(); plt.show()


t0 = train_df.loc[train_df['target'] == 0]
t1 = train_df.loc[train_df['target'] == 1]
plt.figure(figsize=(16,6))
plt.title("Distribution of kurtosis values per row in the train set")
sns.distplot(t0[features].kurtosis(axis=1),color="red", kde=True,bins=120, label='target = 0')
sns.distplot(t1[features].kurtosis(axis=1),color="blue", kde=True,bins=120, label='target = 1')
plt.legend(); plt.show()


t0 = train_df.loc[train_df['target'] == 0]
t1 = train_df.loc[train_df['target'] == 1]
plt.figure(figsize=(16,6))
plt.title("Distribution of kurtosis values per column in the train set")
sns.distplot(t0[features].kurtosis(axis=0),color="red", kde=True,bins=120, label='target = 0')
sns.distplot(t1[features].kurtosis(axis=0),color="blue", kde=True,bins=120, label='target = 1')
plt.legend(); plt.show()


%%time
correlations = train_df[features].corr().abs().unstack().sort_values(kind="quicksort").reset_index()
correlations = correlations[correlations['level_0'] != correlations['level_1']]
correlations.head(10)


correlations.tail(10)


correlations.head(10)


%%time
features = train_df.columns.values[2:202]
unique_max_train = []
unique_max_test = []
for feature in features:
    values = train_df[feature].value_counts()
    unique_max_train.append([feature, values.max(), values.idxmax()])
    values = test_df[feature].value_counts()
    unique_max_test.append([feature, values.max(), values.idxmax()])


# top 15 max of duplicate values per train set
np.transpose((pd.DataFrame(unique_max_test, columns=['Feature', 'Max duplicates', 'Value'])).\
            sort_values(by = 'Max duplicates', ascending=False).head(15))


# Feature Engineering

%time

idx = features = train_df.columns.values[2:202]
for df in [test_df , train_df]:
    df['sum'] = df[idx].sum(axis = 1)
    df['min'] = df[idx].min(axis = 1)
    df['max'] = df[idx].max(axis = 1)
    df['mean'] = df[idx].mean(axis = 1)
    df['std'] = df[idx].std(axis = 1)
    df['skew'] = df[idx].skew(axis = 1)
    df['kurt'] = df[idx].kurtosis(axis = 1)
    df['med'] = df[idx].median(axis = 1)


# new created features

train_df[train_df.columns[202:]].head()


test_df[test_df.columns[201:]].head()


def plot_new_feature_distribtuion(df1,df2 , label1,label2 , features):
    
    i = 0
    sns.set_style('whitegrid')
    plt.figure()
    fig , ax = plt.subplots(2,4,figsize = (18,8))
    
    for feature in features:
        i += 1
        plt.subplot(2,4,i)
        sns.kdeplot(df1[feature],bw = 0.5 , label = label1)
        sns.kdeplot(df2[feature],bw = 0.5 , label = label2)
        
        plt.xlabel(feature,fontsize = 11)
        locs , labels = plt.xticks()
        plt.tick_params(axis = 'x',which = 'major',labelsize = 8)
        plt.tick_params(axis = 'y',which = 'major',labelsize = 8)
        


# Check the distribution of these new engineered features.
t0 = train_df.loc[train_df['target'] == 0]
t1 = train_df.loc[train_df['target'] == 1]
features = train_df.columns.values[202:]
plot_new_feature_distribtuion(t0 , t1 , 'target:0','target:1',features)


# show the distribution of new features value

features = train_df.columns.values[202:]
plot_new_feature_distribtuion(train_df , test_df,'train','test',features)


# Rounded features

features = [c for c in train_df.columns if c not in ['ID_code','target']]
for feature in features:
    train_df['r2_' + feature] = np.round(train_df[feature],2)
    test_df['r2_' + feature] = np.round(test_df[feature],2)
    train_df['r1_' + feature] = np.round(train_df[feature],2)
    test_df['r1' + feature] = np.round(train_df[feature],2)


# How many features we have now.

print('Train and test columns: {}{}'.format(len(train_df.columns), len(test_df.columns)))


# Model
# We drop the ID and target to form the features list

features = [c for c in train_df.columns if c not in['ID_code','target']]
target = train_df['target']


# Define the hyperparametrs for the model

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


# run model

folds = StratifiedKFold(n_splits = 10 , shuffle = False , random_state = 44000)
oof = np.zeros(len(train_df))
predictions = np.zeros(len(test_df))
feature_importance_df = pd.DataFrame()

for fold_ , (trn_idx , val_idx) in enumerate(folds.split(train_df.values,target.values)):
    print('Fold {}'.format(fold_))
    trn_data = lgb.Dataset(train_df.iloc[trn_idx][features],label = target.iloc[trn_idx])
    val_data = lgb.Dataset(train_df.iloc[val_idx][features],label = target.iloc[val_idx])
    
    num_round = 1000000
    clf = lgb.train(param , trn_data , num_round , valid_sets = [trn_data , val_data],verbose_eval = 1000,early_stopping_rounds= 3000)
    oof[val_idx] = clf.predict(train_df.iloc[val_idx][features],num_iteration = clf.best_iteration)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df['Feature'] = features
    fold_importance_df['importance'] = clf.feature_importance()
    fold_importance_df['fold'] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df,fold_importance_df],axis = 0)
    
    predictions += clf.predict(test_df[features],num_iteration = clf.best_iteration) / folds.n_splits
    
    print('Cv score:{:<8.5f}'.format(roc_auc_score(target , oof)))
    


# Check The feature importance

cols = (feature_importance_df[['feature','importance']].groupby('Feature').mean().sort_values(by = 'importance',ascending = False)[:150].index)
best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]

plt.figure(figsize = (14,28))
sns.barplot(x = 'importance' , y = 'Feature',data = best_features.sort_values(by = 'importance',ascending = False))
plt.title('Features importance (averaged / folds)')
plt.tight_layout()
plt.savefig('Fi.png')


# Submission

sub_df = pd.DataFrame({'ID_code':test_df['ID_code'].values})
sub_df['target'] = predictions
sub_df.to_csv('submission.csv',index = False)

