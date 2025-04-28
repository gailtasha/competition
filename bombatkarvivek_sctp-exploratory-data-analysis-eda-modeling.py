# # SANTANDER CUSTOMER TRANSACTION PREDICTION
# 
# **BHAVESHKUMAR THAKER**


# ## INTRODUCTION
# 
# In this challenge, Santander invites Kagglers to help them identify which customers will make a specific transaction in the future, irrespective of the amount of money transacted. The data provided for this competition has the same structure as the real data they have available to solve this problem.
# 
# The data is anonimyzed, each row containing 200 numerical values identified just with a number.


# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#SANTANDER-CUSTOMER-TRANSACTION-PREDICTION" data-toc-modified-id="SANTANDER-CUSTOMER-TRANSACTION-PREDICTION-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>SANTANDER CUSTOMER TRANSACTION PREDICTION</a></span><ul class="toc-item"><li><span><a href="#INTRODUCTION" data-toc-modified-id="INTRODUCTION-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>INTRODUCTION</a></span></li></ul></li><li><span><a href="#LOAD-PACKAGES" data-toc-modified-id="LOAD-PACKAGES-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>LOAD PACKAGES</a></span></li><li><span><a href="#DEFINE-GENERIC-COMMON-METHODS" data-toc-modified-id="DEFINE-GENERIC-COMMON-METHODS-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>DEFINE GENERIC COMMON METHODS</a></span></li><li><span><a href="#LOAD-DATA-AND-PERFORM-ANALYSIS" data-toc-modified-id="LOAD-DATA-AND-PERFORM-ANALYSIS-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>LOAD DATA AND PERFORM ANALYSIS</a></span><ul class="toc-item"><li><span><a href="#ANALYSE-TRAIN-DATA" data-toc-modified-id="ANALYSE-TRAIN-DATA-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>ANALYSE TRAIN DATA</a></span></li><li><span><a href="#ANALYSE-TEST-DATA" data-toc-modified-id="ANALYSE-TEST-DATA-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>ANALYSE TEST DATA</a></span></li></ul></li><li><span><a href="#PERFORM-EXPLORATORY-DATA-ANALYSIS-(EDA)" data-toc-modified-id="PERFORM-EXPLORATORY-DATA-ANALYSIS-(EDA)-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>PERFORM EXPLORATORY DATA ANALYSIS (EDA)</a></span><ul class="toc-item"><li><span><a href="#UNDERSTANDING-DISTRIBUTION-OF-FEATURE(S)" data-toc-modified-id="UNDERSTANDING-DISTRIBUTION-OF-FEATURE(S)-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>UNDERSTANDING DISTRIBUTION OF FEATURE(S)</a></span></li><li><span><a href="#UNDERSTANDING-TARGET-VARIABLE" data-toc-modified-id="UNDERSTANDING-TARGET-VARIABLE-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>UNDERSTANDING TARGET VARIABLE</a></span></li></ul></li><li><span><a href="#FEATURE-ENGINEERING" data-toc-modified-id="FEATURE-ENGINEERING-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>FEATURE ENGINEERING</a></span><ul class="toc-item"><li><span><a href="#UP-SAMPLE-MINORITY-CLASS-(TARGET)-TO-BALANCE-DATA" data-toc-modified-id="UP-SAMPLE-MINORITY-CLASS-(TARGET)-TO-BALANCE-DATA-6.1"><span class="toc-item-num">6.1&nbsp;&nbsp;</span>UP-SAMPLE MINORITY CLASS (TARGET) TO BALANCE DATA</a></span></li><li><span><a href="#SEPERATE-DATA-AND-TARGET-FEATURES" data-toc-modified-id="SEPERATE-DATA-AND-TARGET-FEATURES-6.2"><span class="toc-item-num">6.2&nbsp;&nbsp;</span>SEPERATE DATA AND TARGET FEATURES</a></span></li><li><span><a href="#CORRELATION-MATRIX" data-toc-modified-id="CORRELATION-MATRIX-6.3"><span class="toc-item-num">6.3&nbsp;&nbsp;</span>CORRELATION MATRIX</a></span></li></ul></li><li><span><a href="#MODELING" data-toc-modified-id="MODELING-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>MODELING</a></span><ul class="toc-item"><li><span><a href="#BINOMIAL-CLASSIFICATION:-Using-LGBMClassifier-with-BayesSearchCV" data-toc-modified-id="BINOMIAL-CLASSIFICATION:-Using-LGBMClassifier-with-BayesSearchCV-7.1"><span class="toc-item-num">7.1&nbsp;&nbsp;</span>BINOMIAL CLASSIFICATION: Using <strong>LGBMClassifier</strong> with BayesSearchCV</a></span><ul class="toc-item"><li><span><a href="#PERFORM-CLASSIFICATION" data-toc-modified-id="PERFORM-CLASSIFICATION-7.1.1"><span class="toc-item-num">7.1.1&nbsp;&nbsp;</span>PERFORM CLASSIFICATION</a></span></li></ul></li><li><span><a href="#BINOMIAL-CLASSIFICATION:-Using-XGBClassifier-with-BayesSearchCV" data-toc-modified-id="BINOMIAL-CLASSIFICATION:-Using-XGBClassifier-with-BayesSearchCV-7.2"><span class="toc-item-num">7.2&nbsp;&nbsp;</span>BINOMIAL CLASSIFICATION: Using <strong>XGBClassifier</strong> with BayesSearchCV</a></span><ul class="toc-item"><li><span><a href="#PERFORM-CLASSIFICATION" data-toc-modified-id="PERFORM-CLASSIFICATION-7.2.1"><span class="toc-item-num">7.2.1&nbsp;&nbsp;</span>PERFORM CLASSIFICATION</a></span></li></ul></li></ul></li><li><span><a href="#COMPARE-MODELS" data-toc-modified-id="COMPARE-MODELS-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>COMPARE MODELS</a></span></li><li><span><a href="#PREDICTIONS-OF-TEST-DATA" data-toc-modified-id="PREDICTIONS-OF-TEST-DATA-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>PREDICTIONS OF TEST DATA</a></span></li></ul></div>


# # LOAD PACKAGES


import time
notebookstart = time.time()
notebookstart


import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


import platform
import sys
import importlib
import multiprocessing
import random


import numpy as np
import pandas as pd

random.seed(321)
np.random.seed(321)

pd.options.display.max_columns = 9999


import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors

%matplotlib inline

mpl.rc('figure', figsize=(15, 12))
plt.figure(figsize=(15, 12))
plt.rcParams['figure.facecolor'] = 'azure'
mpl.style.use('seaborn')
plt.style.use('seaborn')

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')


import seaborn as sns

sns.set(rc={'figure.figsize': (15, 12)})
sns.set(context='notebook', style='darkgrid', font='sans-serif',
        font_scale=1.1, rc={'figure.facecolor': 'azure',
        'axes.facecolor': 'azure', 'grid.color': 'steelblue'})
sns.color_palette(mcolors.TABLEAU_COLORS);


import missingno as msno


import scikitplot as skplt


import sklearn

from sklearn.model_selection import train_test_split, \
    cross_val_predict, cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit, \
    StratifiedKFold, GridSearchCV

from sklearn.preprocessing import StandardScaler, MinMaxScaler, \
    RobustScaler, MaxAbsScaler, Normalizer
from sklearn.preprocessing import LabelBinarizer, label_binarize

from sklearn.metrics import accuracy_score, confusion_matrix, \
    classification_report
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import average_precision_score, \
    precision_recall_fscore_support

from sklearn.utils import shuffle, resample
from sklearn.base import BaseEstimator, ClassifierMixin


from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier


import xgboost as xgb
from xgboost import XGBClassifier


import lightgbm as lgbm
from lightgbm import LGBMClassifier


import catboost
from catboost import CatBoostClassifier


import skopt
from skopt import BayesSearchCV


import imblearn
from imblearn.over_sampling import SMOTE, SMOTENC


print('Operating system version........', platform.platform())
print('Python version is............... %s.%s.%s' % sys.version_info[:3])
print('scikit-learn version is.........', sklearn.__version__)
print('pandas version is...............', pd.__version__)
print('numpy version is................', np.__version__)
print('matplotlib version is...........', mpl.__version__)
print('seaborn version is..............', sns.__version__)
print('scikit-plot version is..........', skplt.__version__)
print('missingno version is............', msno.__version__)
print('xgboost version is..............', xgb.__version__)
print('catboost version is.............', catboost.__version__)
print('lightgbm version is.............', lgbm.__version__)
print('scikit-optimize version is......', skopt.__version__)
print('imblearn version is.............', imblearn.__version__)


# # DEFINE GENERIC COMMON METHODS


def getDatasetInformation(csv_filepath, is_corr_required=True):
    """
    Read CSV (comma-separated) file into DataFrame
    
    Returns,
    - DataFrame
    - DataFrame's shape
    - DataFrame's data types
    - DataFrame's describe
    - DataFrame's sorted unique value count
    - DataFrame's missing or NULL value count
    - DataFrame's correlation between numerical columns
    """

    dataset_tmp = pd.read_csv(csv_filepath)

    dataset_tmp_shape = pd.DataFrame(list(dataset_tmp.shape),
            index=['No of Rows', 'No of Columns'], columns=['Total'])
    dataset_tmp_shape = dataset_tmp_shape.reset_index()

    dataset_tmp_dtypes = dataset_tmp.dtypes.reset_index()
    dataset_tmp_dtypes.columns = ['Column Names', 'Column Data Types']

    dataset_tmp_desc = pd.DataFrame(dataset_tmp.describe())
    dataset_tmp_desc = dataset_tmp_desc.transpose()

    dataset_tmp_unique = dataset_tmp.nunique().reset_index()
    dataset_tmp_unique.columns = ['Column Name', 'Unique Value(s) Count'
                                  ]

    dataset_tmp_missing = dataset_tmp.isnull().sum(axis=0).reset_index()
    dataset_tmp_missing.columns = ['Column Names',
                                   'NULL value count per Column']
    dataset_tmp_missing = \
        dataset_tmp_missing.sort_values(by='NULL value count per Column'
            , ascending=False)

    if is_corr_required:
        dataset_tmp_corr = dataset_tmp.corr(method='spearman')
    else:
        dataset_tmp_corr = pd.DataFrame()

    return [
        dataset_tmp,
        dataset_tmp_shape,
        dataset_tmp_dtypes,
        dataset_tmp_desc,
        dataset_tmp_unique,
        dataset_tmp_missing,
        dataset_tmp_corr,
        ]


def getHighlyCorrelatedColumns(dataset, NoOfCols=6):
    df_corr = dataset.corr()

    # set the correlations on the diagonal or lower triangle to zero,
    # so they will not be reported as the highest ones

    df_corr *= np.tri(k=-1, *df_corr.values.shape).T
    df_corr = df_corr.stack()
    df_corr = \
        df_corr.reindex(df_corr.abs().sort_values(ascending=False).index).reset_index()
    return df_corr.head(NoOfCols)


def createFeatureEngineeredColumns(dataset):
    dataset_tmp = pd.DataFrame()

    dataset_tmp['CountOfZeroValues'] = (dataset == 0).sum(axis=1)
    dataset_tmp['CountOfNonZeroValues'] = (dataset != 0).sum(axis=1)

    weight = ((dataset != 0).sum() / len(dataset)).values
    dataset_tmp['WeightedCount'] = (dataset * weight).sum(axis=1)

    dataset_tmp['SumOfValues'] = dataset.sum(axis=1)

    dataset_tmp['VarianceOfValues'] = dataset.var(axis=1)
    dataset_tmp['MedianOfValues'] = dataset.median(axis=1)
    dataset_tmp['MeanOfValues'] = dataset.mean(axis=1)
    dataset_tmp['StandardDeviationOfValues'] = dataset.std(axis=1)
    #dataset_tmp['ModeOfValues'] = dataset.mode(axis=1)
    dataset_tmp['SkewOfValues'] = dataset.skew(axis=1)
    dataset_tmp['KurtosisOfValues'] = dataset.kurtosis(axis=1)

    dataset_tmp['MaxOfValues'] = dataset.max(axis=1)
    dataset_tmp['MinOfValues'] = dataset.min(axis=1)
    dataset_tmp['DiffOfMinMaxOfValues'] = \
        np.subtract(dataset_tmp['MaxOfValues'],
                    dataset_tmp['MinOfValues'])

    dataset_tmp['QuantilePointFiveOfValues'] = dataset[dataset
            > 0].quantile(0.5, axis=1)

    dataset = pd.concat([dataset, dataset_tmp], axis=1)

    return dataset


def getZeroStdColumns(dataset):
    columnsWithZeroStd = dataset.columns[dataset.std() == 0].tolist()
    return columnsWithZeroStd


def getUniqueValueColumns(dataset, valueToCheck=0):
    columnsWithUniqueValue = dataset.columns[dataset.nunique()
            == valueToCheck].tolist()
    return columnsWithUniqueValue


def plotCategoricalVariableDistributionGraph(target_value, title='', xticksrotation=0):
    tmp_count = target_value.value_counts()
    
    fig=plt.figure()
    fig.suptitle(title, fontsize=18)
    
    ax1=fig.add_subplot(221)
    sns.pointplot(x=tmp_count.index, y=tmp_count, ax=ax1)
    ax1.set_title('Distribution Graph')
    plt.xticks(rotation=xticksrotation)
    
    ax2=fig.add_subplot(222)
    sns.barplot(x=tmp_count.index, y=tmp_count, ax=ax2)
    ax2.set_title('Distribution Graph - Bar')
    plt.xticks(rotation=xticksrotation)
    
    ax3=fig.add_subplot(212)
    ax3.pie(tmp_count, labels=tmp_count.index, autopct="%1.1f%%", shadow=True, startangle=195)
    ax3.axis('equal')
    ax3.set_title('Distribution Graph - Pie')
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.90)
    plt.show()


def plot_distplot(dataset):
    colors = mcolors.TABLEAU_COLORS

    dataset_fordist = dataset.select_dtypes([np.int, np.float])
    number_of_subplots = len(dataset_fordist.columns)
    number_of_columns = 3

    number_of_rows = number_of_subplots // number_of_columns
    number_of_rows += number_of_subplots % number_of_columns

    postion = range(1, number_of_subplots + 1)

    fig = plt.figure(1)
    for k in range(number_of_subplots):
        ax = fig.add_subplot(number_of_rows, number_of_columns,
                             postion[k])
        sns.distplot(dataset_fordist.iloc[:, k],
                     color=random.choice(list(colors.keys())), ax=ax)
    fig.tight_layout()
    plt.show()


def convertIntFloatToInt(dictObj):
    for (k, v) in dictObj.items():
        if float('Inf') == v:
            pass
        elif int(v) == v and isinstance(v, float):
            dictObj[k] = int(v)
    return dictObj


# # LOAD DATA AND PERFORM ANALYSIS


(
    dataset_sctp_train,
    df_train_shape,
    df_train_dtypes,
    df_train_describe,
    df_train_unique,
    df_train_missing,
    df_train_corr,
    ) = getDatasetInformation('../input/train.csv', False)

(
    dataset_sctp_test,
    df_test_shape,
    df_test_dtypes,
    df_test_describe,
    df_test_unique,
    df_test_missing,
    df_test_corr,
    ) = getDatasetInformation('../input/test.csv', False)


# ## ANALYSE TRAIN DATA


dataset_sctp_train.head()


df_train_shape


df_train_dtypes


df_train_describe


df_train_unique


df_train_missing


msno.matrix(dataset_sctp_train, color=(33 / 255, 102 / 255, 172 / 255));


# ## ANALYSE TEST DATA


dataset_sctp_test.head()


df_test_shape


df_test_dtypes


df_test_describe


df_test_unique


df_test_missing


msno.matrix(dataset_sctp_test, color=(33 / 255, 102 / 255, 172 / 255));


del(df_train_shape, df_train_dtypes, df_train_describe, df_train_unique, df_train_missing, df_train_corr)
del(df_test_shape, df_test_dtypes, df_test_describe, df_test_unique, df_test_missing, df_test_corr)


# # PERFORM EXPLORATORY DATA ANALYSIS (EDA)


# ## UNDERSTANDING DISTRIBUTION OF FEATURE(S)


plot_distplot(dataset_sctp_train.iloc[:, 2:29])


plot_distplot(dataset_sctp_train.iloc[:, 29:56])


plot_distplot(dataset_sctp_train.iloc[:, 56:83])


plot_distplot(dataset_sctp_train.iloc[:, 83:110])


plot_distplot(dataset_sctp_train.iloc[:, 110:137])


plot_distplot(dataset_sctp_train.iloc[:, 137:164])


plot_distplot(dataset_sctp_train.iloc[:, 164:191])


plot_distplot(dataset_sctp_train.iloc[:, 191:204])


# ## UNDERSTANDING TARGET VARIABLE


dataset_sctp_train.target.unique()


dataset_sctp_train.target.value_counts()


plotCategoricalVariableDistributionGraph(dataset_sctp_train.target, 'Target (feature) - Distribution', xticksrotation=90)


# # FEATURE ENGINEERING


# ## UP-SAMPLE MINORITY CLASS (TARGET) TO BALANCE DATA


dataset_sctp_train_majority = dataset_sctp_train[dataset_sctp_train.target==0]
dataset_sctp_train_minority = dataset_sctp_train[dataset_sctp_train.target==1]

dataset_sctp_train_minority_upsampled = resample(dataset_sctp_train_minority, replace=True, n_samples=100000,)

dataset_sctp_train_upsampled = pd.concat([dataset_sctp_train_majority, dataset_sctp_train_minority_upsampled])
dataset_sctp_train_upsampled.target.value_counts()


plotCategoricalVariableDistributionGraph(dataset_sctp_train_upsampled.target, 'Target (feature) - Distribution', xticksrotation=90)


# ## SEPERATE DATA AND TARGET FEATURES


#y=dataset_sctp_train['target']
#X=dataset_sctp_train.drop(['ID_code', 'target'], axis=1)

y=dataset_sctp_train_upsampled['target']
X=dataset_sctp_train_upsampled.drop(['ID_code', 'target'], axis=1)

dataset_sctp_test_ID_code = dataset_sctp_test['ID_code']
dataset_sctp_test.drop(['ID_code'], axis=1, inplace=True)

(X.shape, y.shape, dataset_sctp_test.shape)


# ## CORRELATION MATRIX


getHighlyCorrelatedColumns(X, 20)


X=createFeatureEngineeredColumns(X)


dataset_sctp_test=createFeatureEngineeredColumns(dataset_sctp_test)


(X.shape, dataset_sctp_test.shape)


columnsWithZeroStdToRemove = getZeroStdColumns(X)
print(f'Columns with Zero STD to drop from Train and Test dataset(s) are {columnsWithZeroStdToRemove}.')

X.drop(columnsWithZeroStdToRemove, axis=1, inplace=True)
dataset_sctp_test.drop(columnsWithZeroStdToRemove, axis=1, inplace=True)

(X.shape, dataset_sctp_test.shape)


X_columns_one_unique_value = getUniqueValueColumns(X, 1)
print(f'Columns with only 1 as value to drop from Train and Test datasets are {X_columns_one_unique_value}.')

X.drop(X_columns_one_unique_value, axis=1, inplace=True)
dataset_sctp_test.drop(X_columns_one_unique_value, axis=1, inplace=True)

(X.shape, dataset_sctp_test.shape)


X_columns_zero_unique_value = getUniqueValueColumns(X, 0)
print(f'Columns with only 0 as value to drop from Train and Test datasets are {X_columns_zero_unique_value}.')

X.drop(X_columns_zero_unique_value, axis=1, inplace=True)
dataset_sctp_test.drop(X_columns_zero_unique_value, axis=1, inplace=True)

(X.shape, dataset_sctp_test.shape)


# # MODELING


n_cpus_avaliable = multiprocessing.cpu_count()

print(f'We\'ve got {n_cpus_avaliable} cpus to work with.')


model_scores = pd.DataFrame(columns=['Classification_Type', 'Model_Name',
                            'Accuracy_Score'])


(X_train, X_test, y_train, y_test) = train_test_split(X, y,
        test_size=0.20)


# ## BINOMIAL CLASSIFICATION: Using **LGBMClassifier** with BayesSearchCV


# ### PERFORM CLASSIFICATION


dataset_sctp_train_lgbm = lgbm.Dataset(data=X, label=y)
dataset_sctp_test_lgbm = lgbm.Dataset(data=X_test, label=y_test)
watchlist = [dataset_sctp_train_lgbm, dataset_sctp_test_lgbm]

evaluation_results = {}

lgbmparams = {
    'boosting_type': 'gbdt',
    'class_weight': 'balanced',
    'colsample_bytree': 0.8220008732467731,
    'importance_type': 'split',
    'learning_rate': 0.6161285857013439,
    'max_depth': 33,
    'metric': 'binary_logloss',
    'min_child_samples': 9,
    'min_child_weight': 0.7823020109077508,
    'min_split_gain': 0.7211200712627109,
    'n_estimators': 295,
    'n_jobs': n_cpus_avaliable,
    'num_leaves': 41,
    'objective': 'binary',
    'reg_alpha': 0.0,
    'reg_lambda': 0.002053077897015527,
    'silent': False,
    'subsample': 0.6820394902275151,
    'subsample_for_bin': 200000,
    'subsample_freq': 8,
    }

lgbm_twoclass_model = lgbm.train(
    lgbmparams,
    train_set=dataset_sctp_train_lgbm,
    num_boost_round=5000,
    valid_sets=watchlist,
    early_stopping_rounds=120,
    evals_result=evaluation_results,
    verbose_eval=100,
    )


y_pred_proba = lgbm_twoclass_model.predict(X_test)
y_pred = [round(value) for value in y_pred_proba]

ac = accuracy_score(y_test, y_pred)
print('The accuracy score of the LightGBM (Two Class) model is: {}%'.format(ac * 100))
print('\n')

model_scores = model_scores.append({'Classification_Type': 'Binomial',
                                   'Model_Name': lgbm_twoclass_model.__class__.__name__ + ' - BayesSearchCV',
                                   'Accuracy_Score': '{:.4f}'.format(ac * 100)}, 
                                   ignore_index=True)

cr = classification_report(y_test, y_pred)
print(cr)
print('\n')

skplt.metrics.plot_confusion_matrix(y_test, y_pred, title='Binomial Classification (LightGBM Confusion Matrix)',
                                    x_tick_rotation=90, 
                                    cmap='Oranges',
                                   );


ax = lgbm.plot_metric(evaluation_results)
plt.show()


ax = lgbm.plot_importance(lgbm_twoclass_model, max_num_features=30, color=mcolors.TABLEAU_COLORS)
plt.show()


# ## BINOMIAL CLASSIFICATION: Using **XGBClassifier** with BayesSearchCV


# ### PERFORM CLASSIFICATION


dtrain = xgb.DMatrix(X, label=y)
dtest = xgb.DMatrix(X_test, label=y_test)

watchlist = [(dtrain, 'train'), (dtest, 'valid')]

xgbparams = {
    'base_score': 0.5,
    'booster': 'gbtree',
    'colsample_bylevel': 0.48973134715974026,
    'colsample_bytree': 0.46131821596707184,
    'device': 'cpu',
    'eval_metric': 'auc',
    'gamma': 8.0762512124703e-06,
    'learning_rate': 0.054722068464825926,
    'max_delta_step': 3,
    'max_depth': 29,
    'min_child_weight': 14,
    'missing': 'None',
    'n_estimators': 274,
    'n_jobs': n_cpus_avaliable,
    'objective': 'binary:logistic',
    'reg_alpha': 0,
    'reg_lambda': 2.1021696940800796,
    'scale_pos_weight': 43.8858626195784,
    'silent': True,
    'subsample': 0.8113392946402368,
    'tree_method': 'approx',
    }
xgb_twoclass_model = xgb.train(
    xgbparams,
    dtrain,
    5000,
    watchlist,
    verbose_eval=200,
    early_stopping_rounds=100,
    )


y_pred_proba = xgb_twoclass_model.predict(dtest)
y_pred = [round(value) for value in y_pred_proba]

ac = accuracy_score(y_test, y_pred)
print('The accuracy score of the XGBoost (Two Class) model is: {}%'.format(ac * 100))
print('\n')

model_scores = model_scores.append({'Classification_Type': 'Binomial',
                                   'Model_Name': xgb_twoclass_model.__class__.__name__ + ' - CV',
                                   'Accuracy_Score': '{:.4f}'.format(ac * 100)}, 
                                   ignore_index=True)

cr = classification_report(y_test, y_pred)
print(cr)
print('\n')

skplt.metrics.plot_confusion_matrix(y_test, y_pred, title='Binomial Classification (XGBoost Confusion Matrix)',
                                    x_tick_rotation=90, 
                                    cmap='Oranges',
                                   );


# # COMPARE MODELS


model_scores.Accuracy_Score = model_scores.Accuracy_Score.astype('float32')
model_scores


sns.barplot(x='Model_Name', y='Accuracy_Score', data=model_scores);
plt.xticks(rotation=90)
plt.show()


# # PREDICTIONS OF TEST DATA


sctp_predictions_lgbm = lgbm_twoclass_model.predict(dataset_sctp_test)
sctp_predictions_lgbm = [round(value) for value in sctp_predictions_lgbm]
sctp_predictions_lgbm = list(map(int, sctp_predictions_lgbm))
sctp_predictions_lgbm[:20]


dataset_submission = pd.DataFrame()
dataset_submission['ID_code'] = dataset_sctp_test_ID_code
dataset_submission['target'] = sctp_predictions_lgbm
dataset_submission.to_csv('lgbm_twoclass_model_submission.csv', index=False)


dataset_sctp_test_xgb = xgb.DMatrix(dataset_sctp_test)
sctp_predictions_xgb = xgb_twoclass_model.predict(dataset_sctp_test_xgb)
sctp_predictions_xgb = [round(value) for value in sctp_predictions_xgb]
sctp_predictions_xgb = list(map(int, sctp_predictions_xgb))
sctp_predictions_xgb[:20]


dataset_submission = pd.DataFrame()
dataset_submission['ID_code'] = dataset_sctp_test_ID_code
dataset_submission['target'] = sctp_predictions_xgb
dataset_submission.to_csv('xgb_twoclass_model_submission.csv', index=False)


print("Notebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))

