# Hi there guys. <br />
# I'm a **person who have jumped into kaggle three month ago**. During studying many enlighting expert's kernels, I've felt kind of **embarrassed feeling** about using hyperparameters of major algorithms such as Xgboost and LightGBM. You guys could reply my opinion like this, **"Why you blame your fault to them?"** 
# ### But, I definitely have **HUGE THANKS TO THEM!!** **Thanks to SUPER BRILLIANT EXPERTS OF KAGGLE** <br />
# 
# The reason why I make this kernel is that some people use **"lightgbm.train"** and the others use **"lightgbm.LGBMClassifier"** for their model. When I see the differences of them, It makes me insane!! Because the **parameters btw two kinds of kernel as I said above seem pretty different!!** <br />
# 
# So, In this kernel, I'll discover <br />
# **1. the true meaning of them and aliases of hyperparameters by looking official document of lightgbm.** <br />
# **2. parameter tuning by referring two website where I'll comment below notebooks** <br />
# 
# ### I hope that **two kinds of people** to see this kernel,
# 
# 1. **One is for people who have felt simliar feeling like me.** For them, I'll describe as detail as I can what I've learned and I'd like to share magnificant post for explaining what the Gradient Boosting and the Xgboost are!!<br />
# 
# The posts are below!!
# * Gradient Boost
# >  https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/
# * Xgboost
# > https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
# 
# 
# 2. **I hope THE OTHERS are enlighting experts** who could give comment about what I understands through this kernel, I would really happy if you guys comment this kernel !! and Could you guys give me a great post about LightGBM parameter or parmeter tuning?? cuz I already have few posts about GBM and XGBoost but I don't have about LightGBM!! (I know generally it seems same one but I think there is regularization in LightGBM) 
# 
# There is Korean comments for my studying for each sentences by Gabriel Preda's explaination. But I didn't only copy and paste this code. I've changed some code for my own!! 
# 
# # I'm staying to tune hyperparameters and I will frequently update this Kernel frequently!!.


# # Reference
# 
# Gabriel Preda's santander-eda-and-prediction
# > https://www.kaggle.com/gpreda/santander-eda-and-prediction
# this kernel uses lightgbm.train for prediction
# 
# Will Koehrsen's A Complete Introduction and Walkthrough [Costa Rican Houshold] 
# > https://www.kaggle.com/willkoehrsen/a-complete-introduction-and-walkthrough
# this kernel uses LGBMClassifier for prediction
# 
# Rudolph's Porto: xgb+lgb kfold LB 0.282 [Porto Seguro’s Safe Driver Prediction]
# > https://www.kaggle.com/rshally/porto-xgb-lgb-kfold-lb-0-282
# this kernel uses lightgbm.train for prediction


# # <a id='1'>Introduction</a>  
# 
# In this challenge, Santander invites Kagglers to help them identify which customers will make a specific transaction in the future, irrespective of the amount of money transacted. The data provided for this competition has the same structure as the real data they have available to solve this problem.  
# 
# 이번 컴피티션에서는 어느 소비자들이 훗날에 현금을 인출할 것인지를 구분하는 것이 목표입니다. 이번 대회의 데이터는 실제 데이터와 같은 구조로 제공되어있습니다.
# 
# The data is anonimyzed, each row containing 200 numerical values identified just with a number.  
# 
# 데이터는 무기명으로 되어있고 각각의 row는 200개의 서로 다른 컬럼을 가지고 있습니다.
# 
# In the following we will explore the data, prepare it for a model, train a model and predict the target value for the test set, then prepare a submission.
# 
# 다음에서 우리는 데이터를 살펴보고, 모델링 준비를하고, 모델을 훈련시키고 타겟 값을 테스트셋에서 예측하고 제출까지 해솝시다.


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
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings('ignore')


# ## Load data   
# 
# Let's check what data files are available.
# 
# 우리가 사용가능한 데이터 파일들을 알아 봅시다.


IS_LOCAL = False
if(IS_LOCAL):
    PATH="../input/Santander/"
else:
    PATH="../input/"
os.listdir(PATH)


# Let's load the train and test data files.


%%time
train_df = pd.read_csv(PATH+"train.csv")
test_df = pd.read_csv(PATH+"test.csv")


# # <a id='3'>Data exploration</a>  
# 
# ## <a id='31'>Check the data</a>  
# 
# Let's check the train and test set.
# 
# 훈련셋과 테스트셋을 확인해봅시다.


train_df.shape, test_df.shape


# Both train and test data have 200,000 entries and 202, respectivelly 201 columns. 
# 
# 훈련셋과 테스트셋 모두 200,000개의 행을가지고 각각 202, 201 개의 컬럼수를 가지고 있습니다.
# 
# Let's glimpse train and test dataset.
# 
# 간단하게 두 세트를 살펴볼까요.


train_df.head()


test_df.head()


# Train contains:  
# 
# * **ID_code** (string);  
# * **target**;  
# * **200** numerical variables, named from **var_0** to **var_199**;
# 
# 훈련세트는. ID, Target 그리고 200개의 숫자값들이 있습니다.
# 
# Test contains:  
# 
# * **ID_code** (string);  
# * **200** numerical variables, named from **var_0** to **var_199**;
# 
# 테스트 셋에는 타겟값을 제외한 것들이 있습니다.
# 
# Let's check if there are any missing data. We will also chech(*k) the type of data.
# 
# 손실값들에 대해서 한번 살펴볼까요> 그리고 데이터들의 타입에 대해서도 알아봅시다.
# 
# We check first train.
# 
# 먼저 훈련세트입니다.


def missing_data(data):
    total = data.isnull().sum()
    percent = (data.isnull().sum()/data.isnull().count()*100)
    tt = pd.concat([total,percent],axis=1,keys=['Total','Percent'])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    return (np.transpose(tt))


%%time
missing_data(train_df)


%%time
missing_data(test_df)


%%time
train_df.describe()


%%time
test_df.describe()


# We can make few observations here:   
# 우리가 찾은 것들은 아래와 같습니다.
# 
# * standard deviation is relatively large for both train and test variable data;
# 훈련 데이터와 테스트 데이터 모두 표준편차가 크다는 것
# * min, max, mean, sdt values for train and test data looks quite close;
# 최소,최대,평균,표준편차 값이 훈련과 테스트셋에서 밀접해 보인다는 것
# * mean values are distributed over a large range.
# 평균값의 변동이 크다는 것
# 
# The number of values in train and test set is the same. Let's plot the scatter plot for train and test set for few of the features.
# 훈련과 테스트 셋에서의 값의 수는 동일하다. 그렇다면 몇몇 특징들에 대해서 산포도를 그려봅시다.


def plot_feature_scatter(df1,df2,features):
    i = 0
    sns.set_style('whitegrid')
    plt.figure()
    fig,ax=plt.subplots(4,4,figsize=[14,14])
    
    for feature in features:
        i+=1
        plt.subplot(4,4,i)
        plt.scatter(df1[feature],df2[feature],marker='+')
        plt.xlabel(feature,fontsize=9)
    plt.show()


features = ['var_0', 'var_1','var_2','var_3', 'var_4', 'var_5', 'var_6', 'var_7', 
           'var_8', 'var_9', 'var_10','var_11','var_12', 'var_13', 'var_14', 'var_15', 
           ]
plot_feature_scatter(train_df[::20],test_df[::20], features)


sns.countplot(train_df['target'])


print("There are {}% target values with 1".format(100 * train_df["target"].value_counts()[1]/train_df.shape[0]))


# 
# ## <a id='32'>Density plots of features</a>  
# 
# Let's show now the density plot of variables in train dataset. 
# 
# We represent with different colors the distribution for values with **target** value **0** and **1**.


def plot_feature_distribution(df1,df2,label1,label2,features):
    i = 0
    sns.set_style('whitegrid')
    plt.figure()
    fig,ax = plt.subplots(10,10,figsize=[18,22])
    
    for feature in features:
        i += 1
        plt.subplot(10,10,i)
        sns.kdeplot(df1[feature],bw=0.5,label=label1)
        sns.kdeplot(df2[feature],bw=0.5,label=label2)
        plt.xlabel(feature,fontsize=9)
        locs, labels = plt.xticks()
        plt.tick_params(axis='x',which='major',labelsize=6,pad=-6)
        plt.tick_params(axis='y',which='major',labelsize=6)
    plt.show()


t0 = train_df.loc[train_df['target']==0]
t1 = train_df.loc[train_df['target']==1]
features = train_df.columns.values[2:102]
plot_feature_distribution(t0,t1,'0','1',features)


features = train_df.columns.values[102:202]
plot_feature_distribution(t0, t1, '0', '1', features)


# We can observe that there is a considerable number of features with significant different distribution for the two target values.  
# For example, **var_0**, **var_1**, **var_2**, **var_5**, **var_9**, **var_13**, **var_106**, **var_109**, **var_139** and many others.
# 
# 우리는 두 개의 타겟값에 따라서 상당이 다른 분포를 가지고 있는 상당한 수의 특징들을 살펴볼 수 있습니다.
# 예를 들면, **var_0**, **var_1**, **var_2**, **var_5**, **var_9**, **var_13**, **var_106**, **var_109**, **var_139** 와 다른 것들 말입니다.
# 
# Also some features, like **var_2**, **var_13**, **var_26**, **var_55**, **var_175**, **var_184**, **var_196** shows a distribution that resambles to a bivariate distribution.
# 
# 그리고 몇몇 특징들, **var_2**, **var_13**, **var_26**, **var_55**, **var_175**, **var_184**, **var_196**, 은 이변량분포와 닮은 분포를 보여줍니다.
# 
# We will take this into consideration in the future for the selection of the features for our prediction model.  
# 
# 우리는 이것들을 우리의 예측모델에 feature selection시에 고려하는 참고자료로 사용할 것입니다.
# 
# Le't s now look to the distribution of the same features in parallel in train and test datasets. 
# 
# 그렇다면 이제는 훈련셋과 테스트셋을 평행적으로 같이 보겠습니다.
# 
# The first 100 values are displayed in the following cell. Press <font color='red'>**Output**</font> to display the plots.
# 
# 첫 번째 100개의 값들은 아래의 그림과 같이 생겼습니다.


features = train_df.columns.values[2:102]
plot_feature_distribution(train_df, test_df, 'train', 'test', features)


features = train_df.columns.values[102:202]
plot_feature_distribution(train_df, test_df, 'train', 'test', features)


# The train and test seems to be well ballanced with respect with distribution of the numeric variables.  
# 
# 훈련셋과 테스트셋은 numeric 변수들의 분포들이 잘 균형을 갖추고 있는 듯 합니다.
# 
# ## <a id='33'>Distribution of mean and std</a>  
# 평균과 표준편차의 분포
# 
# Let's check the distribution of the mean values per row in the train and test set.
# 
# 행별로 훈련과 테스트셋의 평균 값의 분포를 알아봅시다.


plt.figure(figsize=[6,6])
features = train_df.columns.values[2:202]
plt.title("Distribution of mean values per row in the train and test set")
sns.distplot(train_df[features].mean(axis=1),color="green",kde=True,bins=120,label='train')
sns.distplot(test_df[features].mean(axis=1),color="blue",kde=True,bins=120,label='test')
plt.legend()
plt.show()


plt.figure(figsize=(16,6))
plt.title("Distribution of mean values per column in the train and test set")
sns.distplot(train_df[features].mean(axis=0),color="magenta",kde=True,bins=120, label='train')
sns.distplot(test_df[features].mean(axis=0),color="darkblue", kde=True,bins=120, label='test')
plt.legend()
plt.show()


plt.figure(figsize=(16,6))
plt.title("Distribution of std values per row in the train and test set")
sns.distplot(train_df[features].std(axis=1),color="black", kde=True,bins=120, label='train')
sns.distplot(test_df[features].std(axis=1),color="red", kde=True,bins=120, label='test')
plt.legend();plt.show()


plt.figure(figsize=(16,6))
plt.title("Distribution of std values per column in the train and test set")
sns.distplot(train_df[features].std(axis=0),color="blue",kde=True,bins=120, label='train')
sns.distplot(test_df[features].std(axis=0),color="green", kde=True,bins=120, label='test')
plt.legend(); plt.show()


t0 = train_df.loc[train_df['target'] == 0]
t1 = train_df.loc[train_df['target'] == 1]
plt.figure(figsize=(16,6))
plt.title("Distribution of mean values per row in the train set")
sns.distplot(t0[features].mean(axis=1),color="red", kde=True,bins=120, label='target = 0')
sns.distplot(t1[features].mean(axis=1),color="blue", kde=True,bins=120, label='target = 1')
plt.legend(); plt.show()


plt.figure(figsize=(16,6))
plt.title("Distribution of mean values per column in the train set")
sns.distplot(t0[features].mean(axis=0),color="green", kde=True,bins=120, label='target = 0')
sns.distplot(t1[features].mean(axis=0),color="darkblue", kde=True,bins=120, label='target = 1')
plt.legend(); plt.show()


# ## <a id='34'>Features correlation</a>  컬럼간 상관관계
# 
# We calculate now the correlations between the features in train set.  
# The following table shows the first 10 the least correlated features.
# 
# 우리는 훈련세트에 컬럼간에 상관관계를 계산해보려고합니다. 
# 아래의 테이블은 처음 10개의 상관관계 특징들을 보여줍니다.
# 
# 
# > Reference from about guidelines about correlations <br />https://www.kaggle.com/willkoehrsen/a-complete-introduction-and-walkthrough
# 
# #### The general guidelines for correlation values are below, but these will change depending on who you ask (source for these)
# 
# * 00-.19 “very weak” <br />
# * 20-.39 “weak” <br />
# * 40-.59 “moderate” <br />
# * 60-.79 “strong” <br />
# * 80-1.0 “very strong” <br />
# 
# What these correlations show is that there are some weak relationships that hopefully our model will be able to use to learn a mapping from the features to the Target.
# 
# In that Kernel, he droped one of the columns what have high corrleation between them above 0.95
# So, I'd like to drop them also here.
# But we don't have any columns what I told above. So I don't delete anything about 200 coulmns


#I think this code is better than orgin code

correlations = train_df[features].corr().where(np.triu(np.ones(train_df[features].corr().shape),k=1).astype(np.bool))
correlations_df = correlations.abs().unstack().dropna().sort_values().reset_index()
correlations_df.shape


correlations_df.head(5)


correlations_df.tail(5)


[col for col in correlations.columns if any(abs(correlations[col])>0.95)]


# The correlation between the features is very small. 
# 
# ## <a id='35'>Duplicate values</a>  중복값 처리
# 
# Let's now check how many duplicate values exists per columns.
# 
# 컬럼당 얼마나 중복된 값들이 있는지 확인 해보자


%%time
features = train_df.columns.values[2:202]
unique_max_train = []
unique_max_test = []
for feature in features:
    values = train_df[feature].value_counts()
    unique_max_train.append([feature,values.max(),values.idxmax()])

    values = test_df[feature].value_counts()
    unique_max_test.append([feature,values.max(),values.idxmax()])


np.transpose(pd.DataFrame(unique_max_train,columns=['Feature','Max duplicates','Values']).sort_values(by='Max duplicates',ascending=False).head(15))


np.transpose(pd.DataFrame(unique_max_test,columns=['Feature','Max duplicates','Values']).sort_values(by='Max duplicates',ascending=False).head(15))


# Same columns in train and test set have the same or very close number of duplicates of same or very close values. This is an interesting pattern that we might be able to use in the future.
# 
# 훈련세트와 테스트세트에서 같은 컬럼들이 같거나 가까운 양의 중복값을 가지며 이 중복값의 값 또한 같거나 비슷했다. 이는 나중에 사용하기에도 흥미로운 패턴이다.


# # <a id='4'>Feature engineering</a>  
# 
# This section is under construction.  
# 
# Let's calculate for starting few aggregated values for the existing features.


%%time

i = 1
for df in [test_df, train_df]:
    idx = df.columns.values[i:i+200]
    df['sum'] = df[idx].sum(axis=1)  
    df['min'] = df[idx].min(axis=1)
    df['max'] = df[idx].max(axis=1)
    df['mean'] = df[idx].mean(axis=1)
    df['std'] = df[idx].std(axis=1)
    df['skew'] = df[idx].skew(axis=1)
    df['kurt'] = df[idx].kurtosis(axis=1)
    df['med'] = df[idx].median(axis=1)
    df['range'] = df['max']-df['min']
    i = i + 1


train_df[train_df.columns[202:]].head()


test_df[test_df.columns[201:]].head()


features = train_df.columns.values[2:]
correlations = train_df[features].corr().where(np.triu(np.ones(train_df[features].corr().shape),k=1).astype(np.bool))
correlations_df = correlations.abs().stack().reset_index().rename(columns={0:'corr'}).sort_values(by='corr')
correlations_df.shape


correlations_df.head()


correlations_df.tail()


drop_cols = [col for col in correlations.columns if any(abs(correlations[col])>0.95)]


# sum has perfect correaltion wth mean. So, I'd like to delete sum instead of mean.


print("Shape of train_df: {}, test_df: {}".format(train_df.shape,test_df.shape))


train_df = train_df.drop(columns=drop_cols)
test_df = test_df.drop(columns=drop_cols)


print("Shape of train_df: {}, test_df: {}".format(train_df.shape,test_df.shape))


def plot_new_feature_distribution(df1,df2,label1,label2,features):
    i = 0
    sns.set_style('whitegrid')
    fig, ax = plt.subplots(2,4,figsize=[18,8])
    
    for feature in features:
        i += 1
        plt.subplot(2,4,i)
        sns.kdeplot(df1[feature],bw=0.5,label=label1)
        sns.kdeplot(df2[feature],bw=0.5,label=label2)
        plt.xlabel(feature,fontsize=11)
        locs, lables = plt.xticks()
        plt.tick_params(axis="x",which="major",labelsize=8)
        plt.tick_params(axis="y",which="major",labelsize=8)
    plt.show()


t0 = train_df.loc[train_df['target'] == 0]
t1 = train_df.loc[train_df['target'] == 1]
features = train_df.columns.values[202:]
plot_new_feature_distribution(t0, t1, 'target: 0', 'target: 1', features)


features = train_df.columns.values[202:]
plot_new_feature_distribution(train_df, test_df, 'train', 'test', features)


print('Train and test columns: {} {}'.format(len(train_df.columns), len(test_df.columns)))


# # Feature Selection
# 
# **In here, I'd like to select features via SFM and REFCV but I couldn't. Because this data set is so huge as you guys know!! So this I'll try later...


train = train_df.drop(columns=['ID_code','target'])
train_label = train_df['target']
test = test_df.drop(columns='ID_code')
test_ids = test_df['ID_code']


# ## SFM


# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import make_scorer,roc_auc_score
# from sklearn.model_selection import cross_val_score

# scorer = make_scorer(roc_auc_score,greater_is_better=True)

# rf = RandomForestClassifier(random_state=12,n_estimators=15000,n_jobs=-1)

# rf.fit(train,train_label)

# # cv_score = cross_val_score(rf,train,train_label,cv=5,scoring=scorer)
# # print(f"5 fold cv score is {cv_score.mean()}")

# indices = np.argsort(rf.feature_importances_)[::-1]
# feature_names = train.columns
# importances = rf.feature_importances_

# df = pd.DataFrame(columns=['feature','importance'])
# df['feature'] = feature_names
# df['importance'] = importances


# df.sort_values(by='importance',ascending=False).tail()


# ## REFCV


# from sklearn.feature_selection import RFECV

# estimator = RandomForestClassifier(random_state=12,n_estimators=15000,n_jobs=-1)

# selector = RFECV(estimator,step=1,cv=5,scoring=scorer,n_jobs=-1)

# selector.fit(train,train_label)


# # <a id='5'>Model</a>  
# 
# From the train columns list, we drop the ID and target to form the features list.


# ## INFOMATION ABOUT PARAMS
# 
# ### The params what used at Gabriel's code
# 
#     params = {
#         'num_leaves': 6,
#         'max_bin': 63,
#         'min_data_in_leaf': 45,
#         'learning_rate': 0.01,
#         'min_sum_hessian_in_leaf': 0.000446,
#         'bagging_fraction': 0.55, 
#         'bagging_freq': 5, 
#         'max_depth': 14,
#         'save_binary': True,
#         'seed': 31452,
#         'feature_fraction_seed': 31415,
#         'feature_fraction': 0.51,
#         'bagging_seed': 31415,
#         'drop_seed': 31415,
#         'data_random_seed': 31415,
#         'objective': 'binary',
#         'boosting_type': 'gbdt',
#         'verbose': 1,
#         'metric': 'auc',
#         'is_unbalance': True,
#         'boost_from_average': False,
#     }
# 
# ### The params when I make lightgbm.LGBMClassifier.get_params()
# 
#     params = {   
#       'boosting_type': 'gbdt', 
#       'class_weight': None,
#       'colsample_bytree': 1.0,
#       'importance_type': 'split',
#       'learning_rate': 0.1,
#       'max_depth': -1,
#       'min_child_samples': 20,
#       'min_child_weight': 0.001,
#       'min_split_gain': 0.0,
#       'n_estimators': 100,
#       'n_jobs': -1,
#       'num_leaves': 31,
#       'objective': None,
#      'random_state': None,
#      'reg_alpha': 0.0,
#      'reg_lambda': 0.0,
#      'silent': True,
#      'subsample': 1.0,
#      'subsample_for_bin': 200000,
#      'subsample_freq': 0
#         }
# 
# > Reference from <br />
# https://lightgbm.readthedocs.io/en/latest/Python-API.html <br />
# https://lightgbm.readthedocs.io/en/latest/Parameters.html
# 
# * 'boosting_type': 'gbdt' <br />
#  **alias with boosting** (Default:gbdt, options gbdt,gbrt,rf,random_forest,dart,goss)
#  
# * 'class_weight': None <br />
# (default=None) – Weights associated with classes in the form {class_label: weight}. **Use this parameter only for multi-class classification task**; **for binary classification task** you may use **is_unbalance or scale_pos_weight parameters.** The ‘balanced’ mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y)). If None, all classes are supposed to have weight one. Note, that these weights will be multiplied with sample_weight (passed through the fit method) if sample_weight is specified.  
#  
# * 'colsample_bytree': 1.0 <br />
# (Default=1.0 / constraints 0.0 < value <= 1.0) **alias: feature_fraction**[simliar with max_features of GBM]  <br />    
#  lightgbm will randomly select iteration if feature_fraction smaller than 1.0.
#  e.g if I set it 0.8 lightgbm will select 80% of features before training each tree.
#  can be used to speed up training.
#  can be used to deal with over-fitting.
# 
# * 'importance_type': 'split' (default="split") <br />
# How the importance is calculated. If “split”, result contains numbers of times the feature is used in a model. If “gain”, result contains total gains of splits which use the feature. <br />
# => sort of method how to gain feature_importance
# 
# * 'learning_rate': 0.1 <br />
#  (Default=1.0 / constraints learning_rate > 0.0) **alias with with shrinkage_rate,eta**
#  
# * 'max_depth': -1 <br />
#   limit the max depth for tree model. This is used to deal with overfitting when data is small
#  
# * 'min_child_samples': 20 <br />
#   (Default = 20 / constraints min_data_in_leaf >= 0) **alias with min_data_in_leaf,min_data-per_leaf,min_data,min_child_samples** 
# 
# * 'min_child_weight': 0.001 <br />
#  (Default = 1e-3 // Default min_sum_hessian_in_leaf >= 0.0) <br />
#  **alias with min_sum_hessian_in_leaf,min_sum_hessian_per_leaf,min_sum_hessian,min_hessian,min_child_weight **<br />
# 
# * 'min_split_gain': 0.0  **only in lightgbm.LGBMClassifier()[not in lightgbm.train()]<br />
#  (Default =0.0 / constraints: min_gain_to_splot >= 0.0) **alias with min_gain_to_split,min_split_gain** <br />
#  the minimal gain to perform split  <br />
#  
# * 'n_estimators': 100 <br />
#  (Default = 100 / constraints n_estimator >= 0) **alias with num_iteration,n_iter,num_tree,num_trees,num_round,num_rounds,num_boost_round,n_estimators** <br />
#  number of boosting iterations
# 
# * 'n_jobs': -1 <br />
# (Default = 0) **alias with num_thread,nthread,nthreads,n_jobs**
# 
# * 'num_leaves': 31 <br />
#  (Default = 31 / constraints: num_leaves > 1) **aliases: num_leaf, max_leaves, max_leaf** <br />  
#  max number of leaves in one tree
#     
# * 'objective': None <br />
# (Default = regression / options: regression, regression_l1, huber, fair, poisson, quantile, mape, gammma, tweedie, binary, multiclass, multiclassova, xentropy, xentlambda, lambdarank)
#  **aliases: objective_type, app, application**
#     
# * 'random_state': None <br />
# (Default = None) **aliases: random_seed, random_state** <br />
# this seed is used to generate other seeds, e.g. data_random_seed, feature_fraction_seed, etc. <br />
# by default, this seed is unused in favor of default values of other seeds <br />
# this seed has lower priority in comparison with other seeds, which means that it will be overridden, if you set other seeds explicitly <br />
# 
# * 'reg_alpha': 0.0 <br />
# (Default = 0.0 / constraints: lambda_l1 >= 0.0) **aliases: reg_alpha** <br /> 
# L1 regularization <br />
# 
# * 'reg_lambda': 0.0 <br />
# (Default = 0.0 /  constraints: lambda_l2 >= 0.0) **aliases: reg_lambda, lambda** <br />
# L2 regularization <br />
# 
# * 'silent': True **only in lightgbm.LGBMClassifier()(not in lightgbm.train())**<br />
# silent (bool, optional (default=False)) – Whether to print messages during construction
# 
# * 'subsample': 1.0 <br />
# (Default = 1.0 / constraints: 0.0 < bagging_fraction <= 1.0 ) **aliases: sub_row, subsample, bagging** <br /> 
# like feature_fraction, but this will randomly select part of data without resampling <br /> 
# can be used to speed up training <br /> 
# can be used to deal with over-fitting <br /> 
# Note: to enable bagging, bagging_freq should be set to a non zero value as well <br /> 
# 
# * 'subsample_for_bin': 200000 <br />
# (Default = 200000 / constraints: bin_construct_sample_cnt > 0)  **aliases: subsample_for_bin** <br /> 
# number of data that sampled to construct histogram bins <br />
# setting this to larger value will give better training result, but will increase data loading time <br />
# set this to larger value if data is very sparse <br />
# 
# * 'subsample_freq': 0
# (Default = 0) **aliases: subsample_freq, frequency for bagging** <br />
# 0 means disable bagging; k means perform bagging at every k iteration <br />
# Note: to enable bagging, bagging_fraction should be set to value smaller than 1.0 as wel <br />l
# 
# * 'reg_alpha': 0.0
# (default = 0.0) **aliases: reg_alpha**<br /> 
# constraints: lambda_l1 >= 0.0 //  L1 regularization<br />
# 
# * 'reg_lambda': 0.0
# (default = 0.0) **aliases: reg_lambda, lambda** <br />
# constraints: lambda_l2 >= 0.0 // L2 regularization


# ## So We could get some results about comparing two API "lightgbm.train()" and "lightgbm.LGBMClassifier"
# 
# ### common params btw two APIs
# 
# * boosting_type': 'gbdt' ==  'boosting_type': 'gbdt' 
# * 'feature_fraction' == 'colsample_bytree'
# * 'is_unbalance': True == 'class_weight': None 
# * 'learning_rate' == 'learning_rate' 
# * 'max_depth' == 'max_depth'
# * 'min_data_in_leaf' == 'min_child_samples'
# * 'min_sum_hessian_in_leaf' == 'min_child_weight'    
# * num_round == 'n_estimators'
# * 'num_leaves' ==  'num_leaves'
# * 'objective' == 'objective'
# * 'seed' == 'random_state'
# * 'subsample' == 'bagging_fraction'
# * 'subsample_freq' == 'baggin_freq'
# * 'subsample_for_bin' == 'bin_construct_sample_cnt' [**Gabriel didn't tuning it**]
# 
# ### only in lightgbm.LGBMClassifier()
# 
# * 'importance_type'
# * 'min_split_gain'
# * 'silent'
# * 'class_weight'
# * 'reg_alpha'
# * 'reg_lambda'
# **(But, I don't know when I should tune about 'reg_xx' If someone knows it plz comment at this kernel)**


import re
import string

def del_punct(one_list):
    
    return_list = []    
    regex = re.compile('['+re.escape("'")+']')
    
    for element in one_list:
        return_list.append(regex.sub(" ",element).strip())
    
    return return_list


def distinguish_str(value_list):
    
    output = []
    
    regex = re.compile('[0-9]')
    
    for i,element in enumerate(value_list):
        if regex.search(element):
            output.append(float(element))
        else:
            output.append(element)
    
    return output


def model_gbm(train,train_label,test,test_ids,nfolds=5,hyperparameters=None):
    
    feature_names = list(train.columns)
    
    valid_scores = np.zeros(len(train))
    predictions = np.zeros(len(test))
    
    feature_importance_df = pd.DataFrame()
    
    max_iters_df = pd.DataFrame(columns=["folds","iters"])
    
    iters = []
    folds = []
    
    if hyperparameters:
        params = hyperparameters
        
#         If you guys get hyperparams below dataframe by hyperopt, the dictionary will be string type!! 
#         So You should change to dict following commented area. 
#         But, As I mentioned below, I'll put my hyperparams what tested at colab environment already!!
        
#         keys = []
#         values = []
        
#         integer_elements = ['subsample_freq','max_depth','num_leaves','subsample_for_bin','min_child_samples','n_estimators']
        
#         for element in params[1:-1].split(","):
#             keys.append(element.split(":")[0])
#             values.append(element.split(":")[1])
            
#         keys = del_punct(keys)
#         values = distinguish_str(del_punct(values)) 
        
#         params = dict(zip(keys,values))

#         for element in integer_elements:
#             params[element] = int(params[element])

        del(params['n_estimators'])
        
        params['boost_from_average'] = True
        params['seed'] = 31452
        params['feature_fraction_seed'] = 31415
        params['bagging_seed'] = 31415
        params['drop_seed'] = 31415
        params['data_random_seed'] =31415
        params['metric'] = 'auc'
    
    #The hyperparams where I got from Gabriel's code
    else:
        params = {
        'num_leaves': 6,
        'max_bin': 63,
        'min_data_in_leaf': 45,
        'learning_rate': 0.01,
        'min_sum_hessian_in_leaf': 0.000446,#min_child_weight
        'bagging_fraction': 0.55, 
        'bagging_freq': 5, 
        'max_depth': 14,
        'save_binary': True,
        'seed': 31452,
        'feature_fraction_seed': 31415, 
        'feature_fraction': 0.51, #colsample_by_tree => 매 트리 생성시 가져오는 피쳐의 개수
        'bagging_seed': 31415, #배깅을 사용한다면 쓰는 시드
        'drop_seed': 31415,
        'data_random_seed': 31415,
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'verbose': 1,
        'metric': 'auc',
        'is_unbalance': True,
        'boost_from_average': False
    }
    
    strfkold = StratifiedKFold(n_splits=nfolds,shuffle=True,random_state=12)
    
    for i,(train_indices,valid_indices) in enumerate(strfkold.split(train.values,train_label.values)):
        
        print("{} fold processing".format(i+1)+"#"*20)
        
        d_train = lgb.Dataset(train.values[train_indices,:],label = train_label[train_indices])
        d_valid = lgb.Dataset(train.values[valid_indices,:],label = train_label[valid_indices])
        
        n_rounds = 15000
        
        lgb_model = lgb.train(params,d_train,num_boost_round=n_rounds,valid_sets=[d_train,d_valid],valid_names=['train','valid'],verbose_eval=1000,early_stopping_rounds=250)
        
        valid_scores[valid_indices] = lgb_model.predict(train.values[valid_indices,:],num_iteration=lgb_model.best_iteration)
        
        fold_importance_df = pd.DataFrame(columns=["Feature","importance","fold"])
        fold_importance_df["Feature"] = feature_names
        fold_importance_df["importance"] = lgb_model.feature_importance()
        fold_importance_df["fold"] = i + 1
        
        feature_importance_df = pd.concat([feature_importance_df,fold_importance_df],axis=0)
        
        folds.append(i+1)
        iters.append(lgb_model.best_iteration)
        
        predictions += lgb_model.predict(test.values,num_iteration=lgb_model.best_iteration)/nfolds    
        
        display("valid_set score is %f and best_iteration is %d of %d fold"%(roc_auc_score(train_label[valid_indices],valid_scores[valid_indices]),lgb_model.best_iteration,i+1))
        
    max_iters_df["folds"] = folds
    max_iters_df["iters"] = iters
    
    display("CV score of valid_set for %d fold is %f and maximum of best_iteration is %d of %d fold"%(nfolds,roc_auc_score(train_label,valid_scores),max_iters_df['iters'].max(),max_iters_df['iters'].idxmax()+1))
    
    return valid_scores,predictions,feature_importance_df


# ## Hyperparameter Tunning using Hyperopt
# 
# ### The thing what I can do from below kernel is tunning parameters what we saw above through hyperopt!!
# > https://www.kaggle.com/willkoehrsen/a-complete-introduction-and-walkthrough
# 
# #### In this phase we need to comply following 4 phases
# 1. making objective function
# 2. defining space for parameters
# 3. choosing algorithm for hyperopt
# 4. using all of them through fmin of hyperopt
# 
# ### I'd like to complie all of precess using hyperopt but you guys know this process is pretty time-consuming!!!
# **So I'll post my hyperparameters via this process and finally I put in the gbm_model for making predictions!!**


from hyperopt import hp,tpe,Trials,fmin, STATUS_OK
from hyperopt.pyll.stochastic import sample


import csv
import ast
from timeit import default_timer as timer


# ### Making user metric for objective function


def lgb_roc_auc(labels,predictions):
#     print(predictions)
#     predictions = predictions.reshape(len(np.unique(labels)),-1).argmax(axis=0)
    
    metric_value = roc_auc_score(labels,predictions)
    
    return 'roc_auc',metric_value,True


# ### Objective Function
# 
# P.S) I do this process **briefly**, cuz this is **time-consumming process** as I mentioned before <br />
# So, I recommend to set like this if you do yourself in own environment <br />
# 
# * n_estimators => 15000
# * early_stopping_rounds => 250
# * verbose => 1000


def objective(hyperparameters, nfold=5):
    
    global ITERATION
    ITERATION += 1
    
    for parameter_name in ['max_depth','num_leaves','subsample_for_bin','min_child_samples','subsample_freq']:
        hyperparameters[parameter_name] = int(hyperparameters[parameter_name])
        
    strkfold = StratifiedKFold(n_splits=nfold,shuffle=True)
        
    features = np.array(train)
    labels = np.array(train_label).reshape((-1))
        
    valid_scores = []
    best_estimators = []
    run_times = []
    
    model = lgb.LGBMClassifier(**hyperparameters,n_jobs=-1,metric='None',n_estimators=1000)
        
    for i, (train_indices,valid_indices) in enumerate(strkfold.split(features,labels)):

        print("#"*20,"%d fold of %d itertaion"%(i+1,ITERATION))
        
        X_train,X_valid = features[train_indices],features[valid_indices]
        y_train,y_valid = labels[train_indices], labels[valid_indices]
            
        start = timer()
        #250 / 1000    
        model.fit(X_train,y_train,early_stopping_rounds=50,
                eval_metric=lgb_roc_auc,eval_set=[(X_train,y_train),(X_valid,y_valid)],
                eval_names=['train','valid'],verbose=200)
            
        end = timer()
            
        valid_scores.append(model.best_score_['valid']['roc_auc'])
            
        best_estimators.append(model.best_iteration_)
            
        run_times.append(end-start)
            
    score = np.mean(valid_scores)
    score_std = np.std(valid_scores)
    loss = 1-score
        
    run_time = np.mean(run_times)
    run_time_std = np.std(run_times)
        
    estimators = int(np.mean(best_estimators))
    hyperparameters['n_estimators'] = estimators
        
    of_connection = open(OUT_FILE,'a')
    writer = csv.writer(of_connection)
    writer.writerow([loss,hyperparameters,ITERATION,run_time,score,score_std])
    of_connection.close()
    
    display(f'Iteration: {ITERATION}, Score: {round(score, 4)}.')
    
    if ITERATION % PROGRESS == 0:
        display(f'Iteration: {ITERATION}, Current Score: {round(score, 4)}.')
    
    return {'loss': loss, 'hyperparameters': hyperparameters, 'iteration': ITERATION,
            'time': run_time, 'time_std': run_time_std, 'status': STATUS_OK, 
            'score': score, 'score_std': score_std}


# ### Defining Space for Hyperparameters


space = {
    'boosting_type':'gbdt',
    'objective':'binary',
    'is_unbalance':True,
    'subsample': hp.uniform('gbdt_subsample',0.5,1),
    'subsample_freq':hp.quniform('gbdt_subsample_freq',1,10,1),
    'max_depth': hp.quniform('max_depth',5,20,3),
    'num_leaves': hp.quniform('num_leaves',20,60,10),
    'learning_rate':hp.loguniform('learning_rate',np.log(0.025),np.log(0.25)),
    'subsample_for_bin':hp.quniform('subsample_for_bin',2000,100000,2000),
    'min_child_samples': hp.quniform('min_child_samples',5,80,5),
    'colsample_bytree': hp.uniform('colsample_by_tree', 0.5, 1.0),
    'min_child_weight':hp.uniform('min_child_weight',0.01,0.000001)
}


# ### Make a sample via space what we defined


sample(space)


# ### Selecting algorithm
# This algorithm is called by Tree Parzen Estimators. but I don't know how it works.. **So I'll keep trying to understanding!!! Or if you guys have a good site for TPE plz comment below!!**


algo = tpe.suggest


# ### For recording our result of hyperopt


# Record results
trials = Trials()

# Create a file and open a connection
OUT_FILE = 'optimization.csv'
of_connection = open(OUT_FILE, 'w')
writer = csv.writer(of_connection)

MAX_EVALS = 10
PROGRESS = 10
ITERATION = 0

# Write column names
headers = ['loss', 'hyperparameters', 'iteration', 'runtime', 'score', 'std']
writer.writerow(headers)
of_connection.close()


# ### Final phase of hyperopt


import datetime

print("beginning time is {}".format(datetime.datetime.now()))
display("Running Optimization for {} Trials.".format(MAX_EVALS))

# Run optimization
best = fmin(fn = objective, space = space, algo = tpe.suggest, trials = trials,max_evals = MAX_EVALS)

print("end time is {}".format(datetime.datetime.now()))


import json

with open('trials.json','w') as f:
    f.write(json.dumps(str(trials)))


results = pd.read_csv(OUT_FILE).sort_values('loss', ascending = True).reset_index()
results.head()


# ### making the plot using hyperopt


plt.figure(figsize=[8,6])
sns.regplot('iteration','score',data=results);
plt.title('OPT Scores')
plt.xticks(list(range(1,results.iteration.max()+1,3)))


hyperparameters = results.hyperparameters[0]


# ## I could get the params below like that by changing params few times using colab, You guys shoud do that not getting someone's params!! 
# If Kaggle's session was long, I'll use hyperparameters what I got from hyporpot above. But you know, it's not! So, I'll use my own parameters to be gotten through hyperopt in colab environment!!


hyperparameters = {
    'boosting_type': 'gbdt',
    'colsample_bytree': 0.7812943473676428,
    'is_unbalance': True,
    'learning_rate': 0.012732207618246335,
    'max_bin': 200,
    'max_depth': 14,
    'min_child_samples': 70,
    'min_child_weight': 0.0010242091278688855,
    'num_leaves': 10,
    'objective': 'binary',
    'subsample': 0.8026192939361728,
    'subsample_for_bin': 72000,
    'subsample_freq': 7,
    'n_estimators': 6589}


# ## make a model using our own hyperparameters!!


val_scores, predictions, gbm_fi= model_gbm(train,train_label,test,test_ids,hyperparameters=hyperparameters)


submission = pd.read_csv('../input/sample_submission.csv')
submission['target'] = predictions
submission.to_csv("submission.csv",index=False)

