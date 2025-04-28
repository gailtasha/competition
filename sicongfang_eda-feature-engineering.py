# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **Step0 - Import Libraries, Load Data**


import warnings

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler


import matplotlib.pyplot as plt
rcParams = plt.rcParams.copy()
import seaborn as sns
plt.rcParams = rcParams
% matplotlib inline
plt.rcParams["figure.dpi"] = 100
np.set_printoptions(precision=3, suppress=True)
plt.style.use(['fivethirtyeight'])


# Load data to dataframe
raw_train = pd.read_csv("../input/train.csv")
raw_test = pd.read_csv("../input/test.csv")
raw_train.head(50)


raw_test.head(50)


# Create training and test data
X_train = raw_train.drop(columns=["ID_code","target"])
y_train = raw_train[["target"]] 
X_test = raw_test.drop(columns=["ID_code"])
y_test = raw_test[["ID_code"]]


X_train.head()


y_train.head(20)


# Check X training data dimensions
X_train.shape


X_train.describe()


# **Step1 - Exploration and Preparation**
# 
# In this step, I will perform the following actions:
# 
# * Check the imbalance in data 
# * Check missing values
# * Understand the data better using plots
# * Perform basic feature engineering which will be used accross all model sets.
# * Make some hypothesis using the plots and try to make some features representing them. Note that these features might/might not work because they are just hypothesis.


# **1.1 Check Imbalance**   
# The result below suggests high imbalance in training data. Sampling methods should be considered when developing models


# Check y training data distribution
sns.countplot(x="target", data=y_train)
print(y_train['target'].value_counts()/y_train.shape[0])
print('{} samples are positive'.format(np.sum(y_train['target'] == 1)))
print('{} samples are negative'.format(np.sum(y_train['target'] == 0)))


# **1.2 Check missing values**  
# We'll check nan values and zero values (may represent missing values). Also I'll add a column for number of unique values because that'll be interesting to know.


df1 = pd.concat([X_train.apply(lambda x: sum(x.isnull())).rename("num_missing"),
                 X_train.apply(lambda x: sum(x==0)).rename("num_zero"),
                 X_train.apply(lambda x: len(np.unique(x))).rename("num_unique")],axis=1).sort_values(by=['num_unique'])
df1


df2 = pd.concat([X_test.apply(lambda x: sum(x.isnull())).rename("num_missing"),
                 X_test.apply(lambda x: sum(x==0)).rename("num_zero"),
                 X_test.apply(lambda x: len(np.unique(x))).rename("num_unique")],axis=1).sort_values(by=['num_unique'])
df2


# No missing value comfirmed
np.sum(df1['num_missing']!=0)


# Let's check if zero values should be treated as Nan values. From the density and rug plots below, the number of zero value samples are closed to that of other values. Therefore, we should not replace them with any imputed values such as average or median.


sns.distplot(a=X_train['var_71'],rug=True)


sns.distplot(a=X_train['var_131'],rug=True)


# **1.3 Visualize Features:**  
# We'll make 2 plots to visualize these features:
# 
# * histogram
# * box-whiskers plot with the outcome
# * counts of top 10 most occuring unique values to see if there are any dominating ones


#create a function which makes the plot:
from matplotlib.ticker import FormatStrFormatter
def visualize_numeric(ax1, ax2, ax3, df, col, target):
    #plot histogram:
    df.hist(column=col,ax=ax1,bins=200)
    ax1.set_xlabel('Histogram')
    
    #plot box-whiskers:
    df.boxplot(column=col,by=target,ax=ax2)
    ax2.set_xlabel('Transactions')
    
    #plot top 10 counts:
    cnt = df[col].value_counts().sort_values(ascending=False)
    cnt.head(10).plot(kind='barh',ax=ax3)
    ax3.invert_yaxis()  # labels read top-to-bottom
#     ax3.yaxis.set_major_formatter(FormatStrFormatter('%.2f')) #somehow not working 
    ax3.set_xlabel('Count')


for col in list(df1.index[:20]):
    fig, axes = plt.subplots(1, 3,figsize=(10,3))
    ax11 = plt.subplot(1, 3, 1)
    ax21 = plt.subplot(1, 3, 2)
    ax31 = plt.subplot(1, 3, 3)
    fig.suptitle('Feature: %s'%col,fontsize=5)
    visualize_numeric(ax11,ax21,ax31,raw_train,col,'target')
    plt.tight_layout()


# Some interesting fact can be found in:   
# *var_68* :  periodic spikes can be observed. (Other kaggler said this column may be time. From the histgram, its periodicity supports this opnion as well).  
# *var_108* :  a peak can be found at 14.1999 and 14.2. Also, the top values are very close to each other, we may consider round them up to less decimals.  
# *var_12* : a peak can be found at 13.5545. it shows similar features as *var_108*  
#   
# General insights:  
# * Most of the columns follow normal distribution  
# * Since there are lot of outliers, using a robustscalar might make more sense
# * polynomial features are not very intuitive here, but can be tried later


# **1.4 Get PCA and t-SNE for visualization:**  
# PCA requires scaled data but we need not scale the data for both model sets. So we'll use a pipeline here. We saw earlier that data has outliers so we'll use the robust scaler. Its a safe bet because if the data is normal, it'll work similar to standard scaler.


#get vars except target:
x_vars = X_train.columns


from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

pca = make_pipeline(RobustScaler(), PCA(n_components=2))
train_pca = pca.fit_transform(X_train[x_vars])
plt.scatter(train_pca[:, 0], train_pca[:, 1], c=y_train['target'], alpha=.1)
plt.xlabel("first principal component")
plt.ylabel("second principal component")


# This looks like a highly non-linear relationship between the data and targets. Two classes are concentrated seperately.
# 
# We could try creating PCA as new features


from sklearn.pipeline import Pipeline
pca_50 = PCA(n_components=50)
# pca_50 = Pipeline(steps=[('sampler', RobustScaler()),('pca', pca_50)])
pca_50_train = pca_50.fit_transform(X_train[x_vars])
pca_50_test = pca_50.transform(X_test[x_vars])
print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca_50.explained_variance_ratio_)))


# from sklearn.manifold import TSNE
# from matplotlib.ticker import NullFormatter
# from time import time
# tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
# tsne_results = tsne.fit_transform(pca_result_50)

# print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))


# I tried to first calculate 50 principle components and applied t-SNE. Since t-SNE is very computationally expensive, we abandoned adding t-SNE component as new features.


# ax.set_title("Perplexity=%d" % perplexity)
# ax.scatter(Y[red, 0], Y[red, 1], c="r")
# ax.scatter(Y[green, 0], Y[green, 1], c="g")
# ax.xaxis.set_major_formatter(NullFormatter())
# ax.yaxis.set_major_formatter(NullFormatter())
# ax.axis('tight')


from sklearn.cluster import KMeans
def process_data(train_df, test_df):
#     logger.info('Features engineering - numeric data')
    idx = [c for c in train_df.columns if c not in ['ID_code', 'target']]
    for df in [test_df, train_df]:
        for feat in idx:
            df['r2_'+feat] = np.round(df[feat], 2)
            df['r2_'+feat] = np.round(df[feat], 2)
        df['sum'] = df[idx].sum(axis=1)  
        df['min'] = df[idx].min(axis=1)
        df['max'] = df[idx].max(axis=1)
        df['mean'] = df[idx].mean(axis=1)
        df['std'] = df[idx].std(axis=1)
        df['skew'] = df[idx].skew(axis=1)
        df['kurt'] = df[idx].kurtosis(axis=1)
        df['med'] = df[idx].median(axis=1)
        tmp=np.array(df[['var_0', 'var_2', 'var_26', 'var_76','var_81','var_139','var_191']])
        kms=KMeans(n_clusters=30)
        y=kms.fit_predict(tmp)
        df['category'] = y
        df['category'] = df['category'].astype('category')
    print('Train and test shape:',train_df.shape, test_df.shape)
    return train_df, test_df


# X_train1, X_test1 = process_data(raw_train, raw_test)
# X_train1.head(30)
# X_test1.head(30)


# X_train1[['var_1','r2_var_1']].iloc[0]


#Add PCA features:
X_train_01 = pd.concat([X_train, pd.DataFrame(train_pca,columns=['comp1_pca','comp2_pca'])],axis=1)
#Add t-SNE features:
# X_train = pd.concat([X_train, pd.DataFrame(tsne_results,columns=['comp1_tsne','comp2_tsne'])],axis=1)
#get PCA test components:
test_pca = pca.transform(X_test[x_vars])
X_test_01 = pd.concat([X_test, pd.DataFrame(test_pca,columns=['comp1_pca','comp2_pca'])],axis=1)
# #get t-SNE test components:
# test_pca_50 = pca_50.transform(X_test[x_vars])
# test_tsne = tsne.transform(test_pca_50)
# X_test = pd.concat([X_test, pd.DataFrame(test_pca,columns=['comp1_pca','comp2_pca'])],axis=1)
# X_test = pd.concat([X_test, pd.DataFrame(test_tsne,columns=['comp1_tsne','comp2_tsne'])],axis=1)
#check shape: (4 more columns added)
X_train_01.shape, X_test_01.shape


# **Step2 - ModelSet1
# In this step, we expect you to perform the following steps relevant to the models you choose for set1:**
# 
# feature engineering  
# validation  
# feature selection  
# final model selection  


#copy the data so that model specific feature engineering can be performed.
rscaler = RobustScaler()
X_train_02 = pd.DataFrame(rscaler.fit_transform(X_train))
X_test_02 = rscaler.transform(X_test)
X_train_02.describe()


# Check correlation of features
corr = X_train_01.corr()
for i in corr.columns:
    print('top correlation columns with {0} are : \n{1}'.format(i,list(zip(list(corr[i][(corr[i].abs() > 0.5) & (corr[i].abs() <1.0)].index),
                                                                         list(corr[i][(corr[i].abs() > 0.5) & (corr[i].abs() <1.0)].values)))))


# Visualize the correlation of the first 10 features


plt.figure(figsize=(10,10))
sns.heatmap(X_train_01.loc[:, 'var_1':'var_10'].corr(),
           annot=True, fmt=".4f")


#copy the data so that model specific feature engineering can be performed.
rscaler = RobustScaler()
X_train_02 = pd.DataFrame(rscaler.fit_transform(X_train))
X_test_02 = rscaler.transform(X_test)
X_train_02.describe()


# **Methods to check featurn importance**  
# I show four methods to find important feature below.
# * check the feature importance using f_classif and mutual_info_classif  
# * check the feature importance using random forest
# * check the feature importance using the absolute value of coefficients of logistic regression
# * check the feature importance using permutation-importance https://www.kaggle.com/dansbecker/permutation-importance


# 1. Lets check the feature importance using **f_classif and mutual_info_classif**



from sklearn.feature_selection import f_classif, mutual_info_classif

f_values, p_values = f_classif(X_train_01[x_vars], y_train['target'])
mi_scores = mutual_info_classif(X_train_01[x_vars], y_train['target'])


#plot them
fig = plt.figure(figsize=(40, 4))
plt.xticks(range(X_train.shape[1]), x_vars, rotation='vertical')
line_f, = plt.plot(f_values, 'o', c='r')
plt.ylabel("F value")
ax2 = plt.twinx()
line_s, = ax2.plot(mi_scores, 'o', alpha=.7)
ax2.set_ylabel("MI score")
plt.legend([line_s, line_f], ["Mutual info scores", "F values"], loc=(0, 1))


# 2. Lets check the feature importance using **logistic regression**


# **Generic Model Functions**  
# Before going into modeling, we'll just define generic functions which will can use to run model and perform cross-validation.  
# Note that SVC models dont give probabilities by default so we'll use the decision function for the purpose of AUC calculation. The parameter "prob_available" can be used to specify whether the model has "predict_proba" function or not.


from sklearn import model_selection, metrics
def modelfit(alg, dtrain, predictors, y_train, performCV=True, print_feature_importance=False, print_coef=False,
             prob_available=True,export_model=False,cv_folds=5,export_name=''):

    #Perform cross-validation:
    if performCV:
        cv_score = model_selection.cross_val_score(alg, dtrain[predictors], y_train, 
                                                   cv=cv_folds, scoring='roc_auc', n_jobs=-1)
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], y_train)
        
    #Predict train set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    if prob_available:
        dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    else:
        dtrain_predprob = alg.decision_function(dtrain[predictors])
    
    #Print model report:
    print("\nModel Report")
    print("Accuracy (Train) : %.4g" % metrics.accuracy_score(y_train.values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(y_train, dtrain_predprob))
    
    if performCV:
        print("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
        
    #Print Feature Importance:
    if print_feature_importance:
        plt.figure(figsize=(5, 10))
        feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values()
        feat_imp.plot(kind='barh', title='Feature Importances')
        plt.xlabel('Feature Importance Score')
        
    #Print Coefficients:
    if print_coef:
        plt.figure(figsize=(5, 10))
        coeff = pd.Series(alg.coef_[-1,:], predictors).sort_values()
        coeff.plot(kind='barh', title='Feature Importances')
        plt.xlabel('Coefficients')
        
    #export model:
    if export_model:
        dexport = dtest[['ID_code']]
        dexport['target'] = dtest_predprob
        dexport[['ID_code','target']].to_csv('final_data/%s.csv'%export_name,index=False)


from sklearn.linear_model import LogisticRegressionCV
log_reg = LogisticRegressionCV()
%%timeit -n1 -r1
#get the CV score now:
modelfit(log_reg,X_train_01,x_vars, y_train['target'], performCV=True, print_coef=True)


#get the best parameter:
log_reg.scores_


# 3. Lets check the feature importance using **random forest**


 # Feature importance using Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
params = {
    'max_leaf_nodes':list(range(10,161,10))
}

rf=RandomForestClassifier(n_estimators=100)
params = {
    'max_leaf_nodes':list(range(10,161,10))
}
# model_rf = Pipeline(steps=[('sample',RandomUnderSampler()), ('rf',RandomForestClassifier(n_estimators=150,
#                                                                       criterion='entropy',
#                                                                       max_depth=5,
#                                                                       min_samples_split=500,
#                                                                       min_samples_leaf=50,
#                                                                       random_state=0,
#                                                                       class_weight='balanced_subsample'))])
gs = GridSearchCV(rf,params,scoring='roc_auc', cv=5)
gs.fit(X_train, y_train['target'])




res = pd.DataFrame(gs.cv_results_)
res = res.pivot_table(index="param_max_leaf_nodes",
                      values=["mean_test_score", "mean_train_score"])
res

# score_rf = np.mean(cross_val_score(model_rf, X_train, y_train, scoring='roc_auc', cv=5))
# print('Average ROC AUC Score: {:.3f}'.format(score_rf))


res.plot()


gs.best_params_


modelfit(gs,X_train_01,x_vars, y_train['target'], performCV=True, print_feature_importance=False)


# 4. Lets check the feature importance using **Permutation Importance**


import eli5
from eli5.sklearn import PermutationImportance
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold
# Need this to display each ELI5 HTML report within the for loop.
from IPython.display import display
%matplotlib inline


SEED = 314
CV = KFold(n_splits=5)
FEATURES = X_train.columns.tolist()
TARGET_COL = "target"


for fold, (train_idx, valid_idx) in enumerate(CV.split(X_train, y_train[TARGET_COL])):
    clf = LGBMClassifier(random_state=SEED, n_threads=-1, 
                         eval_metric="auc", n_estimators=10000)
    clf.fit(X_train.loc[train_idx, FEATURES], 
            y_train.loc[train_idx, TARGET_COL], 
            eval_metric="auc",
            verbose=0,
            early_stopping_rounds=1000,
            eval_set=[(X_train.loc[valid_idx, FEATURES], 
                       y_train.loc[valid_idx, TARGET_COL])])
    permutation_importance = PermutationImportance(clf, random_state=SEED)
    permutation_importance.fit(X_train.loc[valid_idx, FEATURES], 
                               y_train.loc[valid_idx, TARGET_COL])
    print(f"Permutation importance for fold {fold}")
    display(eli5.show_weights(permutation_importance, feature_names = FEATURES))


import warnings

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler


# from sklearn.linear_model import LogisticRegression

# warnings.filterwarnings('ignore')
# param_grid = {'logisticregression__C': np.logspace(-1, 1, 7)}
# pipe = make_pipeline(RandomUnderSampler(), StandardScaler(), LogisticRegression(class_weight='balanced',
#                                                                                 random_state=0))
# model_01 = GridSearchCV(pipe, param_grid, cv=5)
# model_01.fit(X_train, y_train)

# score_01 = np.mean(cross_val_score(model_01, X_train, y_train, scoring='roc_auc', cv=7))
# print('Average ROC AUC Score: {:.3f}'.format(score_01))


# y_pred_01 = model_01.predict(X_test)
# #  print('   Test ROC AUC Score: {:.3f}'.format(roc_auc_score(y_test, y_pred)))
# y_test['target_01'] = y_pred_01
# y_test_01 = y_test[['ID_code','target_01']].copy()
# y_test_01.head()


# from xgboost import XGBClassifier

# model_02 = XGBClassifier(max_depth=2,
#                          learning_rate=1,
#                          min_child_weight = 1,
#                          subsample = 0.5,
#                          colsample_bytree = 0.1,
#                          scale_pos_weight = round(sum(y_train.target == 1)/len(y_train.target),2),
#                          #gamma=3,
#                          seed=0)
# model_02.fit(X_train, y_train.values)

# score_02 = np.mean(cross_val_score(model_02, X_train, y_train.values, scoring='roc_auc', cv=7))
# print('Average ROC AUC Score: {:.3f}'.format(score_02))


# y_pred_02 = model_02.predict(X_test)
# #  print('   Test ROC AUC Score: {:.3f}'.format(roc_auc_score(y_test, y_pred)))

# y_test['target'] = y_pred_02
# y_test_02 = y_test[['ID_code','target']].copy()
# y_test_02.head()
# y_test_02.to_csv('../input/sample_submission.csv', encoding='utf-8', index=False)


# from sklearn.ensemble import ExtraTreesClassifier

# model_08 = make_pipeline(RandomUnderSampler(), ExtraTreesClassifier(n_estimators=150,
#                                                                     criterion='entropy',
#                                                                     max_depth=8,
#                                                                     min_samples_split=300,
#                                                                     min_samples_leaf=15,
#                                                                     random_state=0,
#                                                                     class_weight='balanced_subsample'))
# model_08.fit(X_train, y_train)

# score_08 = np.mean(cross_val_score(model_08, X_train, y_train, scoring='roc_auc', cv=7))
# print('Average ROC AUC Score: {:.3f}'.format(score_08))

