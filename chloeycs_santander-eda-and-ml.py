
import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
%matplotlib inline
plt.style.use('seaborn')

from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge,ElasticNet, SGDRegressor, LogisticRegression
from sklearn.model_selection import train_test_split
from scipy import sparse
from scipy.stats import norm, skew

from sklearn.decomposition import TruncatedSVD, PCA, LatentDirichletAllocation, NMF
from sklearn.manifold import TSNE

import copy

import os
import time
import warnings
import gc
import os
import pickle
from six.moves import urllib
import warnings
warnings.filterwarnings('ignore')


# # Load data


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
full_df = pd.concat((train_df, test_df))

print("Train data:", train_df.info())
print("Test data:", test_df.info())


# ## Define variables that are useful for later use



num_vars = []

cat_vars = []
for var, dtype in full_df.dtypes.items():
    if 'float' in str(dtype) or 'int' in str(dtype):
        num_vars.append(var)
    if 'object' in str(dtype):
        cat_vars.append(var)
        
id_var = 'ID_code'
cat_vars.remove(id_var)
target_var = 'target'
num_vars.remove(target_var)

print ("There are %d numerical features: %s" 
       % (len(num_vars), num_vars))

print ("There are %d numerical features: %s" 
       % (len(cat_vars), cat_vars))


# # EDA


# ## Description of datasets:


train_df.describe()


test_df.describe()


# ## Target distribution


sns.countplot(train_df['target'])


# ## Missing values


def missing_data(data):
    total = data.isnull().sum()
    percent = (data.isnull().sum()/data.isnull().count()*100)  
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    return(np.transpose(tt))


missing_data(train_df)


missing_data(test_df)


# ## Unique values


train_unique_df = train_df[num_vars].nunique().reset_index().\
        rename(columns={'index':'feature',0:'unique'}).\
        sort_values('unique')
sns.barplot(x='feature', y='unique',color='blue',
    data=train_unique_df)


test_unique_df = test_df[num_vars].nunique().reset_index().\
        rename(columns={'index':'feature',0:'unique'}).\
        sort_values('unique')
sns.barplot(x='feature', y='unique',color='blue',
    data=test_unique_df)


# ## Feature correlation


corr_df = full_df[num_vars].corr()
corr_df


cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr_df, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


#  ## Normality tests


train_norm_df = train_df[num_vars].apply(lambda x:stats.normaltest(x)[1])

print("There are %d features normally distributed." % ((train_norm_df<0.05).sum())) 


print("Top 5 features with highest P value:")
train_norm_df.sort_values(ascending=False).head()


test_norm_df = test_df[num_vars].apply(lambda x:stats.normaltest(x)[1])

print("There are %d features normally distributed." %((test_norm_df<0.05).sum()))

print("Top 5 features with highest P value:")
test_norm_df.sort_values(ascending=False).head()



sns.distplot(train_df['var_146'])


# 
# ## Dimensionality Reduction


from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD


# ### TSNE


%%time

tsne = TSNE(n_components=1)
tsne1d = tsne.fit_transform(train_df[num_vars][:10000].values)
tsne1d_df = pd.DataFrame({'tsne_0':tsne1d.reshape(-1), 'target':train_df['target'][:10000].values})
sns.distplot(tsne1d_df.query('target==0')['tsne_0'], label='target:0')
sns.distplot(tsne1d_df.query('target==1')['tsne_0'], label='target:1')
plt.legend()


%%time

tsne = TSNE(n_components=2)
tsne2d = tsne.fit_transform(train_df[num_vars][:10000].values)
tsne2d_df = pd.DataFrame({'tsne_0':tsne2d[:,0],'tsne_1':tsne2d[:,1], 
                          'target':train_df['target'][:10000].values})
sns.lmplot(x='tsne_0', y='tsne_1', data=tsne2d_df, hue='target', fit_reg=False)
plt.legend()


# ### PCA


%%time

pca = PCA(n_components=2) 
pca2d = pca.fit_transform(train_df[num_vars][:10000].values)
print (pca.explained_variance_ratio_) 
print (pca.explained_variance_) 

pca2d_df = pd.DataFrame({'pca_0':pca2d[:,0],
                         'pca_1':pca2d[:,1], 
                          'target':train_df['target'][:10000].values})

sns.lmplot(x='pca_0', y='pca_1', data=pca2d_df, hue='target', fit_reg=False)


# ### TruncatedSVD


%%time

svd = TruncatedSVD(n_components=2)
svd2d = svd.fit_transform(train_df[num_vars][:10000].values)
svd2d_df = pd.DataFrame({'svd_0':svd2d[:,0],'svd_1':svd2d[:,1], 
                          'target':train_df['target'][:10000].values})
sns.lmplot(x='svd_0', y='svd_1', data=svd2d_df, hue='target', fit_reg=False)



# # Feature engineering


# ## Standardization


std_scaler = StandardScaler()

std_scaler.fit(full_df[num_vars].values) 
train_std_df = pd.DataFrame(std_scaler.transform(train_df[num_vars].values), columns=num_vars)
test_std_df = pd.DataFrame(std_scaler.transform(test_df[num_vars].values) , columns=num_vars)

train_std_df['target'] = train_df['target'].values


train_std_df[num_vars].describe()


# ### Cross validation with raw feature


from sklearn.model_selection import cross_val_score

train_x = train_df[num_vars].values
train_y = train_df['target'].values
test_x = test_df[num_vars].values


lr_cv_raw = cross_val_score(LogisticRegression(),   
                            train_x[:10000], train_y[:10000], 
                            scoring='roc_auc', 
                            cv=5,   
                            n_jobs=-1) 

print("Logistic regression CV score with raw features:", lr_cv_raw.mean())


# ### Cross validation with standardized features


%%time
train_x = train_std_df[num_vars].values
train_y = train_std_df['target'].values
test_x = test_std_df[num_vars].values

lr_cv_std = cross_val_score(LogisticRegression(), 
                            train_x[:10000], train_y[:10000], 
                            scoring='roc_auc',
                            cv=5, 
                            n_jobs=-1)

print("Logistic regression CV score with standardized features:", lr_cv_std.mean())


# #### Visualization 


%%time

pca = PCA(n_components=2)
pca2d = pca.fit_transform(train_std_df[num_vars][:10000].values)
pca2d_df = pd.DataFrame({'pca_0':pca2d[:,0],'pca_1':pca2d[:,1], 
                          'target':train_df['target'][:10000].values})
sns.lmplot(x='pca_0', y='pca_1', data=pca2d_df, hue='target', fit_reg=False)


%%time

pca = PCA(n_components=1)
pca1d = pca.fit_transform(train_std_df[num_vars][:10000].values)
pca1d_df = pd.DataFrame({'pca_0':pca1d.reshape(-1), 'target':train_df['target'][:10000].values})
sns.distplot(pca1d_df.query('target==0')['pca_0'], label='target:0')
sns.distplot(pca1d_df.query('target==1')['pca_0'], label='target:1')
plt.legend()


# ## L2 norm


train_std_df['norm_2'] = train_std_df[num_vars].apply(lambda x:np.linalg.norm(x), axis=1)
test_std_df['norm_2'] = test_std_df[num_vars].apply(lambda x:np.linalg.norm(x), axis=1)

sns.distplot(train_std_df.query('target==0')['norm_2'], label='target:0')
sns.distplot(train_std_df.query('target==1')['norm_2'], label='target:1')
plt.legend()


roc_auc_score(train_y, train_std_df['norm_2'])


# ## Feature importance


# ## Feature importance from Logistic Regression


train_x = train_std_df[num_vars].values
train_y = train_std_df['target'].values
test_x = test_std_df[num_vars].values

lr = LogisticRegression(solver='lbfgs')
lr.fit(train_x, train_y)


lr_feature_importance = pd.DataFrame({'feature':num_vars, 'lr_importance':lr.coef_.reshape(-1), 
                                      'abs_lr_importance': abs(lr.coef_.reshape(-1))})
                            
lr_feature_importance.sort_values('abs_lr_importance', ascending=False).head()


# ## Feature importance from LightGBM


lgb_clf = lgb.LGBMClassifier(n_jobs=-1)
lgb_clf.fit(train_x, train_y)


lgb_feature_importance = pd.DataFrame({'feature':num_vars, 
                                       'lgb_importance':lgb_clf.feature_importances_.reshape(-1)})
                                        
lgb_feature_importance.sort_values('lgb_importance', ascending=False).head()


# ### Combined feature importance


feature_importance = pd.merge(lr_feature_importance, lgb_feature_importance, on='feature')
feature_importance.head()


sns.distplot(train_std_df['var_53'])


 # Target 1
sns.distplot(train_std_df.query('target==1')['var_53'], label='target:0')


# Target 0
sns.distplot(train_std_df.query('target==0')['var_53'], label='target:0')


# var_53 against target
sns.distplot(train_std_df.query('target==0')['var_53'], label='target:0')
sns.distplot(train_std_df.query('target==1')['var_53'], label='target:1')
plt.legend()


# var_81 against target
sns.distplot(train_std_df.query('target==0')['var_81'], label='target:0')
sns.distplot(train_std_df.query('target==1')['var_81'], label='target:1')
plt.legend()


# # Training our first model


lgb_params = {
    "boost_from_average": "false",
    "objective" : "binary",
    "metric" : "auc",
    "boosting": 'gbdt',
    "max_depth" : -1,
    "num_leaves" : 15,
    "learning_rate" : 0.01,
    "bagging_freq": 1,
    "bagging_fraction" : 0.8,
    "feature_fraction" : 0.7,
    "verbosity" : 1,
    "seed": 42
}


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
oof = train_df[['ID_code', 'target']]
oof['predict'] = 0
predictions = test_df[['ID_code']]
val_aucs = []
feature_importance_df = pd.DataFrame()

X_test = test_std_df[num_vars]


for fold, (trn_idx, val_idx) in enumerate(skf.split(train_std_df, train_std_df['target'])):
    X_train, y_train = train_std_df.iloc[trn_idx][num_vars], train_std_df.iloc[trn_idx]['target']
    X_valid, y_valid = train_std_df.iloc[val_idx][num_vars], train_std_df.iloc[val_idx]['target']
     
    trn_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_valid, label=y_valid)
    
    evals_result = {}
    lgb_clf = lgb.train(lgb_params,
                        trn_data,
                        100000,
                        valid_sets = [trn_data, val_data],
                        early_stopping_rounds=3000,
                        verbose_eval = 1000,
                        evals_result=evals_result
                       )    
 
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = num_vars
    fold_importance_df["importance"] = lgb_clf.feature_importance()
    fold_importance_df["fold"] = fold + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0) 
    
    p_valid = lgb_clf.predict(X_valid)
    oof['predict'][val_idx] = p_valid
     
    val_score = roc_auc_score(y_valid, p_valid)
    val_aucs.append(val_score)
   
    predictions['fold{}'.format(fold+1)] = lgb_clf.predict(X_test)


mean_auc = np.mean(val_aucs)
std_auc = np.std(val_aucs)
all_auc = roc_auc_score(oof['target'], oof['predict'])
print("Mean auc: %.9f, std: %.9f. All auc: %.9f." % (mean_auc, std_auc, all_auc))


predictions['target'] = np.mean(predictions[[col for col in predictions.columns 
                                             if col not in ['ID_code', 'target']]].values, axis=1)

sub_df = pd.DataFrame({"ID_code":test_df["ID_code"].values})
sub_df["target"] = predictions['target']

print(predictions.head())
print(sub_df.head())

sub_df.to_csv("lgb_submission.csv", index=False)


# # Model tuning
# ## LightGBM
# ### Manual tuning


gc.collect() 
print ("starting...")

full_vars = num_vars
cat_vars = None

full_vars = num_vars
train_x = train_df[full_vars].values
train_y = train_df[target_var].values
test_x = test_df[full_vars].values

import copy
default_lgb_params = {}
default_lgb_params["learning_rate"] = 0.1 
default_lgb_params["metric"] = 'auc'
default_lgb_params["bagging_freq"] = 1
default_lgb_params["seed"] = 42
default_lgb_params["objective"] = "binary"
default_lgb_params["boost_from_average"] = "false"

params_lgb_space = {}
params_lgb_space['feature_fraction'] = np.arange(0.1, 1, 0.1)
params_lgb_space['num_leaves'] = [2, 4, 8, 16, 32]  
params_lgb_space['max_depth'] = [3 ,4 ,5 ,6, -1]
params_lgb_space['min_gain_to_split'] = [0, 0.1, 0.3, 1, 1.5, 2, 3]
params_lgb_space['bagging_fraction'] = np.arange(0.1, 1, 0.1)
params_lgb_space['min_sum_hessian_in_leaf'] = [1, 5, 10, 30, 100]
params_lgb_space['lambda_l1'] = [0, 0.01, 0.1, 1, 10, 100, 300]
params_lgb_space['lambda_l2'] = [0, 0.01, 0.1, 1, 10, 100, 300]

greater_is_better = True

best_lgb_params = copy.copy(default_lgb_params)


for p in params_lgb_space: 
    print ("\n Tuning parameter %s in %s" % (p, params_lgb_space[p]))
    params = best_lgb_params
    scores = []    
    for v in params_lgb_space[p]: 
        gc.collect()
        print ('\n    %s: %s' % (p, v), end="\n")
        params[p] = v
        
        cv_results = lgb.cv(params, 
                        lgb.Dataset(train_x, label=train_y), 
                        stratified=True,
                        shuffle=True,
                        nfold=5,
                        num_boost_round=100000,
                        early_stopping_rounds=100,
                        verbose_eval=0)
        
        best_lgb_score = max(cv_results['auc-mean'])
        print ('Score: %f ' % (best_lgb_score))
        scores.append([v, best_lgb_score])

    # best param value in the space
    best_param_value = sorted(scores, key=lambda x:x[1],reverse=greater_is_better)[0][0]
    best_param_score = sorted(scores, key=lambda x:x[1],reverse=greater_is_better)[0][1]
    best_lgb_params[p] = best_param_value
    print ("Best %s is %s with a score of %f" %(p, best_param_value, best_param_score))

    
best_param_value = sorted(scores, key=lambda x:x[1],reverse=greater_is_better)[0][0]
best_param_score = sorted(scores, key=lambda x:x[1],reverse=greater_is_better)[0][1]
best_lgb_params[p] = best_param_value
print ("Best %s is %s with a score of %f" %(p, best_param_value, best_param_score))
print ('\n Best manually tuned parameters:', best_lgb_params)   


# ## Retrain the model with manually tuned parameters


best_param_value = {'learning_rate': 0.1, 'metric': 'auc', 
                    'seed': 42, 'objective': 'binary', "boost_from_average": "false",
                    'feature_fraction': 0.1, 'bagging_freq': 1, 
                    'num_leaves': 2, 'max_depth': 3, 'min_gain_to_split': 0, 
                    'bagging_fraction': 0.4, 'min_sum_hessian_in_leaf': 30, 
                    'lambda_l2': 0.01, 'lambda_l1': 0.01}

best_param_value['learning_rate'] = 0.01


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
oof = train_df[['ID_code', 'target']]
oof['predict'] = 0
predictions = test_df[['ID_code']]
val_aucs = []
feature_importance_df = pd.DataFrame()

X_test = test_df[num_vars]


for fold, (trn_idx, val_idx) in enumerate(skf.split(train_df, train_df['target'])):
    X_train, y_train = train_df.iloc[trn_idx][num_vars], train_df.iloc[trn_idx]['target']
    X_valid, y_valid = train_df.iloc[val_idx][num_vars], train_df.iloc[val_idx]['target']
    
    trn_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_valid, label=y_valid)
    evals_result = {}
    
    lgb_clf = lgb.train(best_param_value,
                        trn_data,
                        100000,
                        valid_sets = [trn_data, val_data],
                        early_stopping_rounds=3000,
                        verbose_eval = 1000,
                        evals_result=evals_result
                       )
    
  
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = num_vars
    fold_importance_df["importance"] = lgb_clf.feature_importance()
    fold_importance_df["fold"] = fold + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    p_valid = lgb_clf.predict(X_valid)
    oof['predict'][val_idx] = p_valid
    val_score = roc_auc_score(y_valid, p_valid)
    val_aucs.append(val_score)
    
    predictions['fold{}'.format(fold+1)] = lgb_clf.predict(X_test)


mean_auc = np.mean(val_aucs)
std_auc = np.std(val_aucs)
all_auc = roc_auc_score(oof['target'], oof['predict'])
print("Mean auc: %.9f, std: %.9f. All auc: %.9f." % (mean_auc, std_auc, all_auc))


predictions['target'] = np.mean(predictions[[col for col in predictions.columns 
                                             if col not in ['ID_code', 'target']]].values, 
                                axis=1)
sub_df = pd.DataFrame({"ID_code":test_df["ID_code"].values})
sub_df["target"] = predictions['target']

sub_df.to_csv("lgb_sub_manual_tuned.csv", index=False)


# ### Automated tuning


train_x = train_df[num_vars].values
train_y = train_df['target'].values


from bayes_opt import BayesianOptimization


def lgb_evaluate(
    num_leaves,
    max_depth,
    min_sum_hessian_in_leaf,
    min_gain_to_split,
    feature_fraction,
    bagging_fraction,
    lambda_l2,
    lambda_l1
):
    params = dict()
    params['objective'] = 'binary'
    params['learning_rate'] = 0.1
    params['seed'] = 1234
    params['num_leaves'] = int(num_leaves)
    params['max_depth'] = int(max_depth)
    params['min_sum_hessian_in_leaf'] = int(min_sum_hessian_in_leaf)
    params['min_gain_to_split'] = min_gain_to_split
    params['feature_fraction'] = feature_fraction
    params['bagging_fraction'] = bagging_fraction
    params['bagging_freq'] = 1
    params['lambda_l2'] = lambda_l2
    params['lambda_l1'] = lambda_l1
    params["metric"] = 'auc'

    cv_results = lgb.cv(params,
                        lgb.Dataset(train_x, label=train_y),
                        stratified=True,
                        shuffle=True,
                        nfold=5,
                        num_boost_round=100000,
                        early_stopping_rounds=100,
                        verbose_eval=0)
    best_lgb_score = max(cv_results['auc-mean'])
    print ('Score: %f ' % (best_lgb_score))
    return best_lgb_score


lgb_BO = BayesianOptimization(lgb_evaluate,
                              {
                                  'num_leaves': (2,72),
                                  'max_depth': (-1, -1),
                                  'min_sum_hessian_in_leaf': (0, 100),
                                  'min_gain_to_split': (0, 100),
                                  'feature_fraction': (0.005, 0.1),
                                  'bagging_fraction': (0.3, 0.7),
                                  'lambda_l2': (0, 1),
                                  'lambda_l1': (0, 1)
                              }
                              )

lgb_BO.maximize(init_points=3, n_iter=7)


lgb_BO.max


lgb_BO.res


a = [{**x, **x.pop('params')} for x in xgb_BO.max]
xgb_BO_scores = pd.DataFrame(a)
xgb_BO_max = pd.DataFrame(xgb_BO.max).T


params= lgb_BO_max.iloc[1].to_dict()
#params = lgb_BO_scores.iloc[0].to_dict()
best_lgb_auto_params = dict()
best_lgb_auto_params['objective'] = 'binary'
best_lgb_auto_params["metric"] = 'auc'
best_lgb_auto_params['learning_rate'] = 0.01 # Smaller learning rate
best_lgb_auto_params['num_leaves'] = int(params['num_leaves'])    
best_lgb_auto_params['max_depth'] = int(params['max_depth'])    
best_lgb_auto_params['min_sum_hessian_in_leaf'] = params['min_sum_hessian_in_leaf']
best_lgb_auto_params['min_gain_to_split'] = params['min_gain_to_split']     
best_lgb_auto_params['feature_fraction'] = params['feature_fraction']
best_lgb_auto_params['bagging_fraction'] = params['bagging_fraction']
best_lgb_auto_params['bagging_freq'] = 1
best_lgb_auto_params['lambda_l2'] = params['lambda_l2']
best_lgb_auto_params['lambda_l1'] = params['lambda_l1']
best_lgb_auto_params['random_state'] = 4590
best_lgb_auto_params["n_jobs"] = 8


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
oof = train_df[['ID_code', 'target']]
oof['predict'] = 0
predictions = test_df[['ID_code']]
val_aucs = []
feature_importance_df = pd.DataFrame()

X_test = test_df[num_vars]



for fold, (trn_idx, val_idx) in enumerate(skf.split(train_df, train_df['target'])):
    X_train, y_train = train_df.iloc[trn_idx][num_vars], train_df.iloc[trn_idx]['target']
    X_valid, y_valid = train_df.iloc[val_idx][num_vars], train_df.iloc[val_idx]['target']
    
    trn_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_valid, label=y_valid)
    evals_result = {}
    lgb_clf = lgb.train(best_lgb_auto_params,
                        trn_data,
                        100000,
                        valid_sets = [trn_data, val_data],
                        early_stopping_rounds=3000,
                        verbose_eval = 1000,
                        evals_result=evals_result
                       )
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = num_vars
    fold_importance_df["importance"] = lgb_clf.feature_importance()
    fold_importance_df["fold"] = fold + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    p_valid = lgb_clf.predict(X_valid)
    oof['predict'][val_idx] = p_valid
    val_score = roc_auc_score(y_valid, p_valid)
    val_aucs.append(val_score)
    
    predictions['fold{}'.format(fold+1)] = lgb_clf.predict(X_test)


# # Stacking


def lgb_binary_stack(rgr_params, train_x, train_y, test_x, kfolds, stratified=False,  random_state=42,
                     early_stopping_rounds=0, missing=None, full_vars=None, cat_vars=None, y_dummy=None, verbose=False):
    if stratified:
        kf = StratifiedKFold(n_splits=kfolds, shuffle=True,
                             random_state=random_state)
        kf_ids = list(kf.split(train_x, y_dummy))
    else:
        kf = KFold(n_splits=kfolds, random_state=random_state)
        kf_ids = list(kf.split(train_y))

    train_blend_x = np.zeros((train_x.shape[0], len(rgr_params)))
    test_blend_x = np.zeros((test_x.shape[0], len(rgr_params)))
    blend_scores = np.zeros((kfolds, len(rgr_params)))
    if verbose:
        print("Start stacking.")
    for j, params in enumerate(rgr_params):
        if verbose:
            print("Stacking model", j+1, params)
        test_blend_x_j = np.zeros((test_x.shape[0]))
        for i, (train_ids, val_ids) in enumerate(kf_ids):
            start = time.time()
            if verbose:
                print("Model %d fold %d" % (j+1, i+1))
            train_x_fold = train_x[train_ids]
            train_y_fold = train_y[train_ids]
            val_x_fold = train_x[val_ids]
            val_y_fold = train_y[val_ids]
            if verbose:
                print(i, params)

            train_dataset = lgb.Dataset(train_x_fold,
                                        train_y_fold,
                                        feature_name=full_vars,
                                        categorical_feature=cat_vars
                                        )
            valid_dataset = lgb.Dataset(val_x_fold,
                                        val_y_fold,
                                        feature_name=full_vars,
                                        categorical_feature=cat_vars
                                        )

            if early_stopping_rounds == 0:
                num_boost_round = copy.deepcopy(params['num_boost_round'])
                model = lgb.train(params,
                                  train_dataset,
                                  num_boost_round=num_boost_round,
                                  valid_sets=[train_dataset, valid_dataset],
                                  valid_names=['train', 'valid'],
                                  verbose_eval=verbose
                                  )
                val_y_predict_fold = model.predict(val_x_fold)
                score = roc_auc_score(val_y_fold, val_y_predict_fold)
                if verbose:
                    print("Score for Model %d fold %d: %f " %
                          (j+1, i+1, score))
                blend_scores[i, j] = score
                train_blend_x[val_ids, j] = val_y_predict_fold
                test_blend_x_j = test_blend_x_j + model.predict(test_x)
                if verbose:
                    print("Model %d fold %d finished in %d seconds." %
                          (j+1, i+1, time.time()-start))
            else:
                model = lgb.train(params,
                                  train_dataset,
                                  valid_sets=[train_dataset, valid_dataset],
                                  valid_names=['train', 'valid'],
                                  num_boost_round=150000,
                                  early_stopping_rounds=early_stopping_rounds,
                                  verbose_eval=verbose
                                  )
                best_iteration = model.best_iteration
                if verbose:
                    print(model.best_score['valid']['auc'])
                val_y_predict_fold = model.predict(
                    val_x_fold)
                score = roc_auc_score(val_y_fold, val_y_predict_fold)
                if verbose:
                    print("Score for Model %d fold %d: %f " %
                          (j+1, i+1, score))
                blend_scores[i, j] = score
                train_blend_x[val_ids, j] = val_y_predict_fold
                test_blend_x_j = test_blend_x_j + \
                    model.predict(test_x)
                if verbose:
                    print("Model %d fold %d finished in %d seconds." %
                          (j+1, i+1, time.time()-start))

        test_blend_x[:, j] = test_blend_x_j/kfolds
        print("Score for model %d is %f" % (j+1, np.mean(blend_scores[:, j])))
    return train_blend_x, test_blend_x, blend_scores




def sk_binary_stack(models, train_x, train_y, test_x, kfolds, random_state=42, verbose_eval=1,
                    stratified=True):

    if stratified:
        kf = StratifiedKFold(n_splits=kfolds, shuffle=True,
                             random_state=random_state)
        kf_ids = list(kf.split(train_x, train_y))
    else:
        kf = KFold(n_splits=kfolds, random_state=random_state)
        kf_ids = kf.split(train_y)

    train_blend_x = np.zeros((train_x.shape[0], len(models)))
    test_blend_x = np.zeros((test_x.shape[0], len(models)))
    blend_scores = np.zeros((kfolds, len(models)))

    if verbose_eval > 0:
        print("Start stacking.")
    for j, model in enumerate(models):
        if verbose_eval > 0:
            print("Stacking model", j+1, model)
        test_blend_x_j = np.zeros((test_x.shape[0]))
        for i, (train_ids, val_ids) in enumerate(kf_ids):
            start = time.time()
            if verbose_eval > 0:
                print("Model %d fold %d" % (j+1, i+1))
            train_x_fold = train_x[train_ids, :]
            train_y_fold = train_y[train_ids]
            val_x_fold = train_x[val_ids, :]
            val_y_fold = train_y[val_ids]
            if verbose_eval > 0:
                print(i, model)

            model.fit(train_x_fold, train_y_fold)
            val_y_predict_fold = model.predict_proba(val_x_fold)[:, 1]
            score = roc_auc_score(val_y_fold, val_y_predict_fold)
            if verbose_eval > 0:
                print("Score for Model %d fold %d: %f " % (j+1, i+1, score))
            blend_scores[i, j] = score
            train_blend_x[val_ids, j] = val_y_predict_fold
            test_blend_x_j = test_blend_x_j + model.predict_proba(test_x)[:, 1]
            if verbose_eval > 0:
                print("Model %d fold %d finished in %d seconds." %
                      (j+1, i+1, time.time()-start))

        test_blend_x[:, j] = test_blend_x_j/kfolds
        if verbose_eval > 0:
            print("Score for model %d is %f" %
                  (j+1, np.mean(blend_scores[:, j])))
    return train_blend_x, test_blend_x, blend_scores


# ## Level 1 LightGBM


full_vars = num_vars
train_x = train_df[full_vars].values
train_y = train_df[target_var].values
test_x = test_df[full_vars].values


a = [{**x, **x.pop('params')} for x in lgb_BO.res]
lgb_BO_scores = pd.DataFrame(a)


lgb_stack_params = []
for i in range(3):
    params = lgb_BO_scores.iloc[i].to_dict()
    lgb_params = dict()
    lgb_params['objective'] = 'binary'
    lgb_params["metric"] = 'auc'
    lgb_params['learning_rate'] = 0.01 # Smaller learning rate
    lgb_params['num_leaves'] = int(params['num_leaves'])    
    lgb_params['max_depth'] = int(params['max_depth'])    
    lgb_params['min_sum_hessian_in_leaf'] = params['min_sum_hessian_in_leaf']
    lgb_params['min_gain_to_split'] = params['min_gain_to_split']     
    lgb_params['feature_fraction'] = params['feature_fraction']
    lgb_params['bagging_fraction'] = params['bagging_fraction']
    lgb_params['bagging_freq'] = 1
    lgb_params['lambda_l2'] = params['lambda_l2']
    lgb_params['lambda_l1'] = params['lambda_l1']
    lgb_params['random_state'] = 42
    lgb_params["n_jobs"] = 8
    lgb_stack_params.append(lgb_params)

print (lgb_stack_params)

train_stack_x_lgb, test_stack_x_lgb, blend_scores_lgb = \
        lgb_binary_stack(lgb_stack_params, 
                         train_x, train_y, test_x, 
                         5, 
                         early_stopping_rounds=200, 
                         stratified=True, 
                         random_state=4590,
                         full_vars=full_vars, 
                         cat_vars=None,
                         y_dummy=train_y )


# ## Level 2 - Logistic Regression


from sklearn.linear_model import LogisticRegression

train_stack_x_l1 = copy.copy(train_stack_x_lgb)
test_stack_x_l1 = copy.copy(test_stack_x_lgb)


l2_stack_models = [LogisticRegression()
                  ]
train_sk_stack_x_l2, test_sk_stack_x_l2, _ = \
        sk_binary_stack(l2_stack_models, 
                        train_stack_x_l1, train_y, test_stack_x_l1, 
                        5, 
                        #y_dummy=train_y, 
                        random_state=42,
                        stratified=True)

print('All AUC for level 2 Logistic Regression:', roc_auc_score( train_y, train_sk_stack_x_l2.mean(axis=1)))


## Create submission
sub_df = pd.DataFrame({"ID_code":test_df["ID_code"].values})
sub_df["target"] = test_sk_stack_x_l2.mean(axis=1)
sub_df.to_csv("sub_l1_lgb_l2_lr.csv", index=False)


# ## Level 2 - LightGBM


train_stack_x_l1 = np.hstack((train_x, train_stack_x_lgb))
test_stack_x_l1 = np.hstack((test_x, test_stack_x_lgb))

train_stack_x_lgb_l2, test_stack_x_lgb_l2, blend_scores_lgb = \
        lgb_binary_stack(lgb_stack_params, 
                         train_stack_x_l1, train_y, test_stack_x_l1, 
                         5, 
                         early_stopping_rounds=200, 
                         stratified=True, 
                         random_state=4590,
                         full_vars=full_vars, 
                         cat_vars=None,
                         y_dummy=train_y)


## Create submission
sub_df = pd.DataFrame({"ID_code":test_df["ID_code"].values})
sub_df["target"] = test_stack_x_lgb_l2.mean(axis=1)
sub_df.to_csv("sub_l1_lgb_l2_lgb.csv", index=False)

