# **Project title :- Santander customer transaction prediction using Python**


# **Problem statement :-**
# 
# In this challenge, we need to identify which customers will make a specific transaction in
# the future, irrespective of the amount of money transacted.


# **Contents:**
# 
#  1. Exploratory Data Analysis
#            * Loading dataset and libraries
#            * Data cleaning
#            * Typecasting the attributes
#            * Target classes count        
#            * Missing value analysis
#         2. Attributes Distributions and trends
#            * Distribution of train attributes
#            * Distribution of test attributes
#            * Mean distribution of attributes
#            * Standard deviation distribution of attributes
#            * Skewness distribution of attributes
#            * Kurtosis distribution of attributes      
#            * Outliers analysis
#         4. Correlation matrix 
#         5. Split the dataset into train and test dataset
#         7. Modelling the training dataset
#            * Logistic Regression Model
#            * SMOTE Model
#            * LightGBM Model
#         8. Cross Validation Prediction
#            * Logistic  Regression CV Prediction
#            * SMOTE CV Prediction
#            * LightGBM CV Prediction
#         9. Model performance on test dataset
#            * Logistic Regression Prediction
#            * SMOTE Prediction
#            * LightGBM Prediction
#         10. Model Evaluation Metrics
#            * Confusion Matrix
#            * ROC_AUC score
#         11. Choosing best model for predicting customer transaction


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,cross_val_predict,cross_val_score
from sklearn.metrics import roc_auc_score,confusion_matrix,make_scorer,classification_report,roc_curve,auc
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import ClusterCentroids,NearMiss, RandomUnderSampler
import lightgbm as lgb
import eli5
from eli5.sklearn import PermutationImportance
from sklearn import tree
import graphviz
from pdpbox import pdp, get_dataset, info_plots
import scikitplot as skplt
from scikitplot.metrics import plot_confusion_matrix,plot_precision_recall_curve


from scipy.stats import randint as sp_randint
import warnings
warnings.filterwarnings('ignore')

import os
print(os.listdir("../input"))

random_state=42
np.random.seed(random_state)


# **Importing the train dataset**


#importing the train dataset
train_df=pd.read_csv('../input/train.csv')
train_df.head()


# **Shape of the train dataset**


#Shape of the train dataset
train_df.shape


# Summary of the dataset


#Summary of the dataset
train_df.describe()


# **Target classes count**


%%time
#target classes count
target_class=train_df['target'].value_counts()
print('Count of target classes :\n',target_class)
#Percentage of target classes count
per_target_class=train_df['target'].value_counts()/len(train_df)*100
print('percentage of count of target classes :\n',per_target_class)

#Countplot and violin plot for target classes
fig,ax=plt.subplots(1,2,figsize=(20,5))
sns.countplot(train_df.target.values,ax=ax[0],palette='husl')
sns.violinplot(x=train_df.target.values,y=train_df.index.values,ax=ax[1],palette='husl')
sns.stripplot(x=train_df.target.values,y=train_df.index.values,jitter=True,color='black',linewidth=0.5,size=0.5,alpha=0.5,ax=ax[1],palette='husl')
ax[0].set_xlabel('Target')
ax[1].set_xlabel('Target')
ax[1].set_ylabel('Index')


# **Take aways:**                   
# * We have a unbalanced data,where 90% of the data is the number of customers those will not make a transaction and 10% of the data is those who will make a transaction.
# * Look at the violin plots seems that there are no relationship between the target with the index of the train dataframe.This is more dominated by the zero targets then for the ones.
# * Look at the jitter plots with violin plots. We can observed that targets looks uniformly distributed over the indexs of the dataframe.


# **Let us look distribution of train attributes**


%%time
def plot_train_attribute_distribution(t0,t1,label1,label2,train_attributes):
    i=0
    sns.set_style('whitegrid')
    
    fig=plt.figure()
    ax=plt.subplots(10,10,figsize=(22,18))
    
    for attribute in train_attributes:
        i+=1
        plt.subplot(10,10,i)
        sns.distplot(t0[attribute],hist=False,label=label1)
        sns.distplot(t1[attribute],hist=False,label=label2)
        plt.legend()
        plt.xlabel('Attribute',)
        sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
    plt.show()


# Let us see first 100 test attributes will be displayed in next cell.


%%time
t0=train_df[train_df.target.values==0]
t1=train_df[train_df.target.values==1]
train_attributes=train_df.columns.values[2:102]
plot_train_attribute_distribution(t0,t1,'0','1',train_attributes)


# Let us see next 100 test attributes will be displayed in next cell.


%%time
train_attributes=train_df.columns.values[102:203]
plot_train_attribute_distribution(t0,t1,'0','1',train_attributes)


# **Take aways:**
# * We can observed that their is a considerable number of features which are significantly have different distributions for two target variables. For example like var_0,var_1,var_9,var_198 var_180 etc.
# *  We can observed that their is a considerable number of features which are significantly have same distributions for two target variables. For example like var_3,var_7,var_10,var_171,var_185 etc.


# **Importing the test dataset**


#importing the test dataset
test_df=pd.read_csv('../input/test.csv')
test_df.head()


# **Shape of the test dataset**


#Shape of the test dataset
test_df.shape


# **Let us look distribution of test attributes**


%%time
def plot_test_attribute_distribution(test_attributes):
    i=0
    sns.set_style('whitegrid')
    
    fig=plt.figure()
    ax=plt.subplots(10,10,figsize=(22,18))
    
    for attribute in test_attributes:
        i+=1
        plt.subplot(10,10,i)
        sns.distplot(test_df[attribute],hist=False)
        plt.xlabel('Attribute',)
        sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
    plt.show()


# Let us see first 100 test attributes will be displayed in next cell.


%%time
test_attributes=test_df.columns.values[1:101]
plot_test_attribute_distribution(test_attributes)


# Let us see next 100 test attributes will be displayed in next cell.


%%time
test_attributes=test_df.columns.values[101:202]
plot_test_attribute_distribution(test_attributes)


# **Take aways:**
# * We can observed that their is a considerable number of features which are significantly have different distributions. 
#   For example like var_0,var_1,var_9,var_180 var_198 etc.
# * We can observed that their is a considerable number of features which are significantly have same distributions. 
#   For example like var_3,var_7,var_10,var_171,var_185,var_192 etc.


# **Let us look distribution of mean values per rows and columns in train and test dataset**


%%time
#Distribution of mean values per column in train and test dataset
plt.figure(figsize=(16,8))
#train attributes
train_attributes=train_df.columns.values[2:202]
#test attributes
test_attributes=test_df.columns.values[1:201]
#Distribution plot for mean values per column in train attributes
sns.distplot(train_df[train_attributes].mean(axis=0),color='blue',kde=True,bins=150,label='train')
#Distribution plot for mean values per column in test attributes
sns.distplot(test_df[test_attributes].mean(axis=0),color='green',kde=True,bins=150,label='test')
plt.title('Distribution of mean values per column in train and test dataset')
plt.legend()
plt.show()

#Distribution of mean values per row in train and test dataset
plt.figure(figsize=(16,8))
#Distribution plot for mean values per row in train attributes
sns.distplot(train_df[train_attributes].mean(axis=1),color='blue',kde=True,bins=150,label='train')
#Distribution plot for mean values per row in test attributes
sns.distplot(test_df[test_attributes].mean(axis=1),color='green',kde=True, bins=150, label='test')
plt.title('Distribution of mean values per row in train and test dataset')
plt.legend()
plt.show()


# **Let us look distribution of standard deviation(std) values per rows and columns in train and test dataset**


%%time
#Distribution of std values per column in train and test dataset
plt.figure(figsize=(16,8))
#train attributes
train_attributes=train_df.columns.values[2:202]
#test attributes
test_attributes=test_df.columns.values[1:201]
#Distribution plot for std values per column in train attributes
sns.distplot(train_df[train_attributes].std(axis=0),color='red',kde=True,bins=150,label='train')
#Distribution plot for std values per column in test attributes
sns.distplot(test_df[test_attributes].std(axis=0),color='blue',kde=True,bins=150,label='test')
plt.title('Distribution of std values per column in train and test dataset')
plt.legend()
plt.show()

#Distribution of std values per row in train and test dataset
plt.figure(figsize=(16,8))
#Distribution plot for std values per row in train attributes
sns.distplot(train_df[train_attributes].std(axis=1),color='red',kde=True,bins=150,label='train')
#Distribution plot for std values per row in test attributes
sns.distplot(test_df[test_attributes].std(axis=1),color='blue',kde=True, bins=150, label='test')
plt.title('Distribution of std values per row in train and test dataset')
plt.legend()
plt.show()


# ****


# **Let us look distribution of skewness per rows and columns in train and test dataset**


%%time
#Distribution of skew values per column in train and test dataset
plt.figure(figsize=(16,8))
#train attributes
train_attributes=train_df.columns.values[2:202]
#test attributes
test_attributes=test_df.columns.values[1:201]
#Distribution plot for skew values per column in train attributes
sns.distplot(train_df[train_attributes].skew(axis=0),color='green',kde=True,bins=150,label='train')
#Distribution plot for skew values per column in test attributes
sns.distplot(test_df[test_attributes].skew(axis=0),color='blue',kde=True,bins=150,label='test')
plt.title('Distribution of skewness values per column in train and test dataset')
plt.legend()
plt.show()

#Distribution of skew values per row in train and test dataset
plt.figure(figsize=(16,8))
#Distribution plot for skew values per row in train attributes
sns.distplot(train_df[train_attributes].skew(axis=1),color='green',kde=True,bins=150,label='train')
#Distribution plot for skew values per row in test attributes
sns.distplot(test_df[test_attributes].skew(axis=1),color='blue',kde=True, bins=150, label='test')
plt.title('Distribution of skewness values per row in train and test dataset')
plt.legend()
plt.show()


# **Let us look distribution of kurtosis values per rows and columns in train and test dataset**


%%time
#Distribution of kurtosis values per column in train and test dataset
plt.figure(figsize=(16,8))
#train attributes
train_attributes=train_df.columns.values[2:202]
#test attributes
test_attributes=test_df.columns.values[1:201]
#Distribution plot for kurtosis values per column in train attributes
sns.distplot(train_df[train_attributes].kurtosis(axis=0),color='blue',kde=True,bins=150,label='train')
#Distribution plot for kurtosis values per column in test attributes
sns.distplot(test_df[test_attributes].kurtosis(axis=0),color='red',kde=True,bins=150,label='test')
plt.title('Distribution of kurtosis values per column in train and test dataset')
plt.legend()
plt.show()

#Distribution of kutosis values per row in train and test dataset
plt.figure(figsize=(16,8))
#Distribution plot for kurtosis values per row in train attributes
sns.distplot(train_df[train_attributes].kurtosis(axis=1),color='blue',kde=True,bins=150,label='train')
#Distribution plot for kurtosis values per row in test attributes
sns.distplot(test_df[test_attributes].kurtosis(axis=1),color='red',kde=True, bins=150, label='test')
plt.title('Distribution of kurtosis values per row in train and test dataset')
plt.legend()
plt.show()


# **Missing value analysis**


%%time
#Finding the missing values in train and test data
train_missing=train_df.isnull().sum().sum()
test_missing=test_df.isnull().sum().sum()
print('Missing values in train data :',train_missing)
print('Missing values in test data :',test_missing)


# No missing values are present in both train and test data.


# **Correlation between the attributes**


# We can observed that the correlation between the train attributes is very small.


%%time
#Correlations in train attributes
train_attributes=train_df.columns.values[2:202]
train_correlations=train_df[train_attributes].corr().abs().unstack().sort_values(kind='quicksort').reset_index()
train_correlations=train_correlations[train_correlations['level_0']!=train_correlations['level_1']]
print(train_correlations.head(10))
print(train_correlations.tail(10))


# We can observed that the correlation between the test attributes is very small.


%%time
#Correlations in test attributes
test_attributes=test_df.columns.values[1:201]
test_correlations=test_df[test_attributes].corr().abs().unstack().sort_values(kind='quicksort').reset_index()
test_correlations=test_correlations[test_correlations['level_0']!=test_correlations['level_1']]
print(test_correlations.head(10))
print(test_correlations.tail(10))


# **Correlation plot for train and test data**


# We can observed from correlation distribution plot that the correlation between the train and test attributes is very very small, it means that features are independent each other.


%%time
#Correlations in train data
train_correlations=train_df[train_attributes].corr()
train_correlations=train_correlations.values.flatten()
train_correlations=train_correlations[train_correlations!=1]
test_correlations=test_df[test_attributes].corr()
#Correlations in test data
test_correlations=test_correlations.values.flatten()
test_correlations=test_correlations[test_correlations!=1]

plt.figure(figsize=(20,5))
#Distribution plot for correlations in train data
sns.distplot(train_correlations, color="Red", label="train")
#Distribution plot for correlations in test data
sns.distplot(test_correlations, color="Blue", label="test")
plt.xlabel("Correlation values found in train and test")
plt.ylabel("Density")
plt.title("Correlation distribution plot for train and test attributes")
plt.legend()


# **Feature engineering**


# Let us do some feature engineering by using
# * Permutation importance
# * Partial dependence plots


# **Permutation importance**


# Permutation variable importance measure in a random forest for classification and regression.


#training data
X=train_df.drop(columns=['ID_code','target'],axis=1)
test=test_df.drop(columns=['ID_code'],axis=1)
y=train_df['target']


# Let us build simple model to find features which are more important.


#Split the training data
X_train,X_valid,y_train,y_valid=train_test_split(X,y,random_state=42)

print('Shape of X_train :',X_train.shape)
print('Shape of X_valid :',X_valid.shape)
print('Shape of y_train :',y_train.shape)
print('Shape of y_valid :',y_valid.shape)


# **Random forest classifier**


%%time
#Random forest classifier
rf_model=RandomForestClassifier(n_estimators=10,random_state=42)
#fitting the model
rf_model.fit(X_train,y_train)


# Let us calculate weights and show important features using eli5 library.


%%time
from eli5.sklearn import PermutationImportance
perm_imp=PermutationImportance(rf_model,random_state=42)
#fitting the model
perm_imp.fit(X_valid,y_valid)


# Let us see important features,


%%time
#Important features
eli5.show_weights(perm_imp,feature_names=X_valid.columns.tolist(),top=200)


# Take aways:
# * Importance of the features decreases as we move down the top of the column.
# * As we can see the features shown in green indicate that they have a positive impact on our prediction
# * As we can see the features shown in white indicate that they have no effect on our prediction
# * As we can see the features shown in red indicate that they have a negative impact on our prediction
# * The most important feature is 'Var_81'


# **Partial dependence plots**


# Partial dependence plot gives a graphical depiction of the marginal effect of a variable on the class probability or classification.While feature importance shows what variables most affect predictions, but partial dependence plots show how a feature affects predictions.


# Let us calculate partial dependence plots on random forest


# **Partial dependence plot**


# Let us see impact of the main features which are discovered in the previous section by using the pdpbox.


%%time
#Create the data we will plot 'var_81'
features=[v for v in X_valid.columns if v not in ['ID_code','target']]
pdp_data=pdp.pdp_isolate(rf_model,dataset=X_valid,model_features=features,feature='var_81')
#plot feature "var_81"
pdp.pdp_plot(pdp_data,'var_81')
plt.show()


# **Take aways:**
# * The y_axis is interpreted as change in prediction from what it would be predicted at the baseline.
# * The blue shaded area indicates the level of confidence of 'var_81'.


%%time
#Create the data we will plot 
pdp_data=pdp.pdp_isolate(rf_model,dataset=X_valid,model_features=features,feature='var_109')
#plot feature "var_109"
pdp.pdp_plot(pdp_data,'var_109')
plt.show()


# **Take aways:**
# * The y_axis is interpreted as change in prediction from what it would be predicted at the baseline.
# * The blue shaded area indicates the level of confidence of 'var_109'.


%%time
#Create the data we will plot 
pdp_data=pdp.pdp_isolate(rf_model,dataset=X_valid,model_features=features,feature='var_12')
#plot feature "var_12"
pdp.pdp_plot(pdp_data,'var_12')
plt.show()


# **Take aways:**
# * The y_axis is interpreted as change in prediction from what it would be predicted at the baseline.
# * The blue shaded area indicates the level of confidence of 'var_12'.


# **Handling of imbalanced data**
# 
# Now we are going to explore 6 different approaches for dealing with imbalanced datasets.
# * Change the performance metric
# * Oversample minority class
# * Undersample majority class
# * Synthetic Minority Oversampling Technique(SMOTE)
# * Change the algorithm


# Now let us start with simple Logistic regression model.


# **Split the train data using StratefiedKFold cross validator**


#Training data
X=train_df.drop(['ID_code','target'],axis=1)
Y=train_df['target']
#StratifiedKFold cross validator
cv=StratifiedKFold(n_splits=5,random_state=42,shuffle=True)
for train_index,valid_index in cv.split(X,Y):
    X_train, X_valid=X.iloc[train_index], X.iloc[valid_index]
    y_train, y_valid=Y.iloc[train_index], Y.iloc[valid_index]

print('Shape of X_train :',X_train.shape)
print('Shape of X_valid :',X_valid.shape)
print('Shape of y_train :',y_train.shape)
print('Shape of y_valid :',y_valid.shape)


# **Logistic Regression model**


%%time
#Logistic regression model
lr_model=LogisticRegression(random_state=42)
#fitting the lr model
lr_model.fit(X_train,y_train)


# **Accuracy of model**


#Accuracy of the model
lr_score=lr_model.score(X_train,y_train)
print('Accuracy of the lr_model :',lr_score)


# **Cross validation prediction of lr_model**


%%time
#Cross validation prediction
cv_predict=cross_val_predict(lr_model,X_valid,y_valid,cv=5)
#Cross validation score
cv_score=cross_val_score(lr_model,X_valid,y_valid,cv=5)
print('cross_val_score :',np.average(cv_score))


# Accuracy of the model is not the best metric to use when evaluating the imbalanced datasets as it may be misleading. So, we are going to change the performance metric.


# **Confusion matrix**


#Confusion matrix
cm=confusion_matrix(y_valid,cv_predict)
#Plot the confusion matrix
plot_confusion_matrix(y_valid,cv_predict,normalize=False,figsize=(15,8))


# **Reciever operating characteristics (ROC)-Area under curve(AUC) score and curve**


#ROC_AUC score
roc_score=roc_auc_score(y_valid,cv_predict)
print('ROC score :',roc_score)

#ROC_AUC curve
plt.figure()
false_positive_rate,recall,thresholds=roc_curve(y_valid,cv_predict)
roc_auc=auc(false_positive_rate,recall)
plt.title('Reciver Operating Characteristics(ROC)')
plt.plot(false_positive_rate,recall,'b',label='ROC(area=%0.3f)' %roc_auc)
plt.legend()
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('Recall(True Positive Rate)')
plt.xlabel('False Positive Rate')
plt.show()
print('AUC:',roc_auc)


# When we compare the roc_auc_score and model accuracy , model is not performing well on imbalanced data.


# **Classification report**


#Classification report
scores=classification_report(y_valid,cv_predict)
print(scores)


# We can observed that f1 score is high for number of customers those who will not make a transaction then the who will make a transaction. So, we are going to change the algorithm.


# **Model performance on test data**


%%time
#Predicting the model
X_test=test_df.drop(['ID_code'],axis=1)
lr_pred=lr_model.predict(X_test)
print(lr_pred)


# **Oversample minority class:**
# * It can be defined as adding more copies of minority class.
# * It can be a good choice when we don't have a ton of data to work with.
# * Drawback is that we are adding information.This may leads to overfitting and poor performance on test data.


# **Undersample majority class:**
# * It can be defined as removing some observations of the majority class.
# * It can be a good choice when we have a ton of data -think million of rows.
# * Drawback is that we are removing information that may be valuable.This may leads to underfitting and poor performance on test data.


# Both Oversampling and undersampling techniques have some drawbacks. So, we are not going to use this models for this problem and also we will use other best algorithms.


# **Synthetic Minority Oversampling Technique(SMOTE)**


# SMOTE uses a nearest neighbors algorithm to generate new and synthetic data to used for training the model.


%%time
from imblearn.over_sampling import SMOTE
#Synthetic Minority Oversampling Technique
sm = SMOTE(random_state=42, ratio=1.0)
#Generating synthetic data points
X_smote,y_smote=sm.fit_sample(X_train,y_train)
X_smote_v,y_smote_v=sm.fit_sample(X_valid,y_valid)


# Let us see how baseline logistic regression model performs on synthetic data points.


%%time
#Logistic regression model for SMOTE
smote=LogisticRegression(random_state=42)
#fitting the smote model
smote.fit(X_smote,y_smote)


# **Accuracy of model**


#Accuracy of the model
smote_score=smote.score(X_smote,y_smote)
print('Accuracy of the smote_model :',smote_score)


# Cross validation prediction of smoth_model


%%time
#Cross validation prediction
cv_pred=cross_val_predict(smote,X_smote_v,y_smote_v,cv=5)
#Cross validation score
cv_score=cross_val_score(smote,X_smote_v,y_smote_v,cv=5)
print('cross_val_score :',np.average(cv_score))


# **Confusion matrix**


#Confusion matrix
cm=confusion_matrix(y_smote_v,cv_pred)
#Plot the confusion matrix
plot_confusion_matrix(y_smote_v,cv_pred,normalize=False,figsize=(15,8))


# **Reciever operating characteristics (ROC)-Area under curve(AUC) score and curve**


#ROC_AUC score
roc_score=roc_auc_score(y_smote_v,cv_pred)
print('ROC score :',roc_score)

#ROC_AUC curve
plt.figure()
false_positive_rate,recall,thresholds=roc_curve(y_smote_v,cv_pred)
roc_auc=auc(false_positive_rate,recall)
plt.title('Reciver Operating Characteristics(ROC)')
plt.plot(false_positive_rate,recall,'b',label='ROC(area=%0.3f)' %roc_auc)
plt.legend()
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('Recall(True Positive Rate)')
plt.xlabel('False Positive Rate')
plt.show()
print('AUC:',roc_auc)


# **Classification report**


#Classification report
scores=classification_report(y_smote_v,cv_pred)
print(scores)


# **Model performance on test data**


%%time
#Predicting the model
X_test=test_df.drop(['ID_code'],axis=1)
smote_pred=smote.predict(X_test)
print(smote_pred)


# We can observed that smote model is performing well on imbalance data compare to logistic regression.


# **LightGBM:**
# 
# LightGBM is a gradient boosting framework that uses tree based learning algorithms. We are going to use LightGBM model.


# Let us build LightGBM model


#Training the model
#training data
lgb_train=lgb.Dataset(X_train,label=y_train)
#validation data
lgb_valid=lgb.Dataset(X_valid,label=y_valid)


# **choosing of  hyperparameters**


#Selecting best hyperparameters by tuning of different parameters
params={'boosting_type': 'gbdt', 
          'max_depth' : -1, #no limit for max_depth if <0
          'objective': 'binary',
          'boost_from_average':False, 
          'nthread': 12,
          'metric':'auc',
          'num_leaves': 100,
          'learning_rate': 0.08,
          'max_bin': 950,      #default 255
          'subsample_for_bin': 200,
          'subsample': 1,
          'subsample_freq': 1,
          'colsample_bytree': 0.8,
          'reg_alpha': 1.2, #L1 regularization(>0)
          'reg_lambda': 1.2,#L2 regularization(>0)
          'min_split_gain': 0.5, #>0
          'min_child_weight': 1,
          'min_child_samples': 5,
          'is_unbalance':True,
          'scale_pos_weight': 1,
          }


# **Training the lgbm model**


num_rounds=3000
lgbm= lgb.train(params,lgb_train,num_rounds,valid_sets=[lgb_train,lgb_valid],verbose_eval=100,early_stopping_rounds = 2000)
lgbm


# **lgbm model performance on test data**


X_test=test_df.drop(['ID_code'],axis=1)
#predict the model
#probability predictions
lgbm_predict_prob=lgbm.predict(X_test,random_state=42,num_iteration=lgbm.best_iteration)
#Convert to binary output 1 or 0
lgbm_predict=np.where(lgbm_predict_prob>=0.5,1,0)
print(lgbm_predict_prob)
print(lgbm_predict)


# **Let us plot the important features**


#plot the important features
lgb.plot_importance(lgbm,max_num_features=150,importance_type="split",figsize=(20,50))


# **Conclusion :**
# 
# We tried model with logistic regression,smote and lightgbm. But, both smote and lightgbm model is performing well on imbalanced data compared to other models based on scores of roc_auc_score.


#final submission
sub_df=pd.DataFrame({'ID_code':test_df['ID_code'].values})
sub_df['Target']=lgbm_predict_prob
sub_df.to_csv('submission.csv',index=False)
sub_df.head()

