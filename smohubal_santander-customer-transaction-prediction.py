import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, GridSearchCV, train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

import shap
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.ensemble.partial_dependence import partial_dependence
from sklearn.metrics import roc_auc_score

from sklearn.decomposition import PCA
from imblearn.over_sampling import SVMSMOTE
import warnings
warnings.filterwarnings('ignore')

import os
print(os.listdir("../input"))


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2 # Memory total(Ram)
    
    for col in df.columns:
        col_type = df[col].dtypes
        
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            
            # Int
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            
            # Float
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


train = reduce_mem_usage(pd.read_csv('../input/train.csv'))
test = reduce_mem_usage(pd.read_csv('../input/test.csv'))


# ## data wrangling


def missing_percent_plot(data):
    missing_col = list(data.isna().sum() != 0)
    
    try:
        if True not in missing_col:
            raise ValueError("There is no missing values.")

        data = data.loc[:,missing_col]

        missing_percent = (data.isna().sum()/data.shape[0]) * 100

        df = pd.DataFrame()
        df['perc_missing'] = missing_percent
        sns.barplot(x=df.perc_missing.index, y='perc_missing', data=df); plt.xticks(rotation=90)
    except:
        return print('There is no missing values...')
    return list(data.columns)


missing_percent_plot(train)


def target_symmetry(train):    
    target_0 = len(train.target[train.target == 0]) / len(train.target) * 100
    target_1 = len(train.target[train.target == 1]) / len(train.target) * 100
    print("Target : 0 exist {} percent in total target \n\
           Target : 1 exist {} percent in total target ".format(target_0,
                                                                target_1))
    return sns.countplot(x='target', hue='target', data = train)


target_symmetry(train)


train.head()


# ## Modeling


# ### Parameters selection


def grid_model(params, train, target, model):
    t_x, v_x, t_y, v_y = train_test_split(train, target, test_size  =.30, random_state =42)
    
    # xgbm = xgb.XGBClassifier()
    # xgbm = GridSearchCV(xgbm, params, cv=3) 
    model.fit(t_x, t_y)
    
    t_pred = model.predict_proba(t_x)[:,1] # For train score
    v_pred = model.predict_proba(v_x)[:,1] # For val score
    
    score = cross_val_score(model, train, target, cv=5)
    # train_score = roc_auc_score(t_y, t_pred)
    # val_score   = roc_auc_score(v_y, v_pred)

    # print('roc_auc_score(t_y, t_pred)', train_score)
    # print('roc_auc_score(v_y, v_pred)', val_score)
    
    return model, score, model.predict_proba(test)


# ### Score using k-fold


def shap_DP(df, feature):
    return shap.dependence_plot(feature, shap_values, df)

def importance_plot(df, model, max_plot=30):
    importances = pd.DataFrame({'features':list(df.columns), 'importance':model.feature_importances_})
    importances.sort_values(by = 'importance', ascending = False, inplace = True)
    sns.barplot(x = importances.features[:max_plot], y = importances.importance[:max_plot] , label = "Total", color = "b");plt.xticks(rotation = 45);plt.title('Feature importances')
    return importances

def feature_correlation_heatmap(importances_df, start=0, last=10):
    def heatmap(features):    
        sns.set(style="white")

        # Compute the correlation matrix
        corr = train.loc[:,features].corr()

        # Generate a mask for the upper triangle
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(11, 9))

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        return sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                           square=True, linewidths=.5, cbar_kws={"shrink": .5}, 
                           annot =True, annot_kws = {'size':9})

    n_to_n = list(importances_df.features)[start:last]
    n_to_n.append('target')
    return heatmap(n_to_n)


params = dict(max_depth=7, 
                  leaning_rate=0.1, 
                  n_estimators=70, 
                  objective='binary:hinge', 
                  n_jobs=-1, 
                  random_state=42, 
                  importance_type ='gain', 
                  reg_lambda=0.2, 
                  early_stopping_rounds=3)
xgbm = xgb.XGBClassifier(**params)


def score_n_predict(x, y, test, model, cv=5, verbose=False):
    """Calculate train, validation score(roc_auc score)"""
    
    k = KFold(n_splits=cv)
    t_score = 0
    v_score = 0
    
    for t_i, v_i in k.split(x):
        t_x, t_y = x.iloc[t_i].values, y.iloc[t_i].values
        v_x, v_y = x.iloc[v_i].values, y.iloc[v_i].values
        
        model.fit(t_x, t_y)
        
        # Predict and calculate score for train and val
        t_pred = model.predict_proba(t_x)[:,1] # For train score
        v_pred = model.predict_proba(v_x)[:,1] # For val score

        train_score = roc_auc_score(t_y, t_pred)
        val_score   = roc_auc_score(v_y, v_pred)
        
        if verbose == True:
            print('train score{} \n\
                   val score{}'.format(train_score,
                                       val_score))
        
        # Iteratively add score divided by k-fold
        t_score += train_score / k.get_n_splits()
        v_score += val_score   / k.get_n_splits()
        
    if verbose == True:
        print("{}-fold average accuracy".format(k.get_n_splits()))
        print('train score : {} \n\
                 val score : {}'.format(t_score,
                                        v_score))
    return model


# model_original= score_n_predict(train.iloc[:,2:], train.target, test.iloc[:,1:], xgbm, cv=3, verbose=True)


# - We can see there is trivial features approximatly 100 features cause using half features we can get similar result than to using orginal 200 features.


original_importances = importance_plot(train.iloc[:,2:], model_original, max_plot=30)


explainer = shap.TreeExplainer(model_original)
shap_values = explainer.shap_values(train.iloc[:, 2:])
shap.summary_plot(shap_values, train.iloc[:, 2:], max_display = 50)


# best 40 features 
model_2_a = score_n_predict(train[list(original_importances.features)[:40]], train.target, test[list(original_importances.features)[:40]], xgbm, cv=5, verbose=True)


for col in list(original_importances.features)[:40]:
    sns.distplot(train[train.target == 1].loc[:, col], bins='auto', label=1)
    sns.distplot(train[train.target == 0].loc[:, col], bins='auto', label=0); plt.legend();plt.show()
del model_2_a


# # partial dependencies & correlation


feature_correlation_heatmap(original_importances, start=0, last=10)
feature_correlation_heatmap(original_importances, start=10, last=20)
# feature_correlation_heatmap(original_importances, start=20, last=30)
# feature_correlation_heatmap(original_importances, start=30, last=40)
# feature_correlation_heatmap(original_importances, start=40, last=50)


for col in list(original_importances.features):
    shap_DP(train.iloc[:, 2:], col)


# worst 10 features
feature_correlation_heatmap(original_importances, start = 190, last = 200)


# worst 10 features
features_xgbm_worst = list(original_importances.features)[-10:]
for col in features_xgbm_worst:
    shap_DP(col)


# ## PCA & SMOTE
# - To control bias in data imbalance create extra data points using SVM-SMOTE.


train = reduce_mem_usage(pd.read_csv('../input/train.csv'))
test = reduce_mem_usage(pd.read_csv('../input/test.csv'))


def pca(train, test, n = 'mle'):
    pca_1 = PCA(n_components=n);pca_2 = PCA(n_components=n)
    return pd.DataFrame(pca_1.fit_transform(train)), pd.DataFrame(pca_2.fit_transform(test))


scaler = StandardScaler()
pca_train, pca_test = pca(scaler.fit_transform(train.iloc[:,2:]),
                          scaler.fit_transform(test.iloc[:,1:]), n=2)

sm = SVMSMOTE(random_state=42, n_jobs=-1, sampling_strategy = 'minority')
X_res, y_res = sm.fit_resample(pca_train, train.target)
pca_smote_train = pd.concat([pd.DataFrame(X_res), pd.DataFrame(y_res, columns=['target'])], axis=1)

# Usig permuted index, shuffle rows of training data to prevent ValueError caused by imbalance data(only one label included in training)
pca_smote_train = pca_smote_train.iloc[np.random.permutation(np.arange(len(pca_smote_train)))]
pca_sm_model= score_n_predict(pca_smote_train.drop(['target'], axis=1), pca_smote_train.target, pca_test, xgbm, cv=5, verbose=True)


del pca_sm_model, pca_smote_train, X_res, y_res, sm, 


# - Using SVM-SMOTE, we could increase accuracy roughly 20%.


# ## Feature engineering


train = reduce_mem_usage(pd.read_csv('../input/train.csv'))
test = reduce_mem_usage(pd.read_csv('../input/test.csv'))


train_test = pd.concat([train, test], axis=0).reset_index()
del train, test


# train_test.fillna(-1, inplace=True) # test_target(np.nan) into -1
poly = PolynomialFeatures(2, include_bias=False)
poly.fit(train_test.loc[:,features_40])
extra_features = pd.DataFrame(poly.transform(train_test.loc[:,features_40]))

extra_features.rename(columns=lambda x: str(x) + '_poly_feature', inplace=True)


train_test_ = pd.concat([train_test, extra_features], axis=1)
del train_test, extra_features

train = train_test_[~train_test_.target.isna()]
test = train_test_[train_test_.target.isna()].drop(['target'], axis=1)
del train_test_

pca_train, pca_test = pca(train.iloc[:,3:], test.iloc[:,2:], n=2)
pca_model= score_n_predict(pca_train, train.target, pca_test, xgbm, cv=5, verbose=True)
del pca_train, pca_test, pca_model, pca_y_pred


# - Polynomial feature engineering actually doesn't effect on accuracy.


# ## Modeling


train = reduce_mem_usage(pd.read_csv('../input/train.csv'))
test = reduce_mem_usage(pd.read_csv('../input/test.csv'))
def pre_processing(train, test, train_y, stantardization=False):
    
    # Standardization
    if stantardization == True:
        scaler = StandardScaler()
        train  = scaler.fit_transform(train)
        test   = scaler.fit_transform(test)
    
    # PCA & Over-sampling
    train, test = pca(train, test, n=2)
    sm = SVMSMOTE(random_state=42, n_jobs=-1, sampling_strategy = 'minority')
    x, y = sm.fit_resample(train, train_y)
     
    train = pd.concat([pd.DataFrame(y, columns=['target']), pd.DataFrame(x)], axis=1)
    
    # Usig permuted index, shuffle rows of training data to prevent ValueError caused by imbalance data(only one label included in training)
    train = train.iloc[np.random.permutation(np.arange(len(train)))]
    return train, test
train_id = train.ID_code
test_id  = test.ID_code
train_y  = train.target


train, test = pre_processing(train.iloc[:,2:], test.iloc[:,1:], train_y)


scaler = StandardScaler()
train_y = train.target
train  = pd.DataFrame(scaler.fit_transform(train.drop('target', axis=1)))
test   = pd.DataFrame(scaler.fit_transform(test))


params = {'early_stopping_rounds': 3,
 'importance_type': 'gain',
 'leaning_rate': 0.1,
 'max_depth': 15,
 'n_estimators': 1400,
 'n_jobs': -1,
 'objective': 'binary:hinge',
 'random_state': 42,
 'reg_lambda': 0.2}
xgbm = xgb.XGBClassifier(**params)
xgbm.fit(train, train_y)
xgbm_score, xgbm, xgbm_pred = GridSearchCV(xgbm, params)


xgbm_pred = xgbm.predict_proba(test)


params = dict(C=[0.5, 0.1],
              class_weight =['balanced', None], 
              random_state=[42], 
              solver = ['liblinear', 'newton-cg'], 
              n_jobs =[-1], 
              max_iter =[70, 100, 500, 1000])
LR = LogisticRegression()
LR = GridSearchCV(LR, params)
LR, LR_score, LR_pred = grid_model(params, train, train_y, LR)


params = dict(C=[0.5],
              class_weight =['balanced', None],
              kernel = ['linear'],
              random_state=[42],
              max_iter =[100], 
              probability=[True])
svm = SVC()
svm = GridSearchCV(svm, params)
svm, svm_score, svm_pred = grid_model(params, train, train_y, svm)


# ### Submission


def submit(y_pred, name):
    submission = pd.read_csv('../input/sample_submission.csv')
    submission.target = y_pred
    submission.target = submission.target.apply(lambda x: 0 if x < .5 else 1)
    submission.to_csv('submission'+ name + '.csv', index=False)


submit(xgbm_pred[:, 1], name='xgbm')


submit(((xgbm_pred[:, 1] + svm_pred[:, 1] + LR_pred[:, 1]) / 3), name='xgbm_svm_LR')

