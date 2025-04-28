# ![](https://storage.googleapis.com/kaggle-media/competitions/santander/atm_image.png)
# 
# ## Introduction
# 
# At Santander their mission is to help people and businesses prosper. Santander is always looking for ways to help customers understand their financial health and identify which products and services might help them achieve their monetary goals.


# ## Objective
# In this challenge, we have to help them identify which customers will make a specific transaction in the future, irrespective of the amount of money transacted. The data provided for the competition has the same structure as the real data Santander have available to solve this problem.


# ## Data
# 
# We are provided with an anonymized dataset containing numeric feature variables, the binary target column, and a string ID_code column.
# 
# The task is to predict the value of target column in the test set.
# 
# **File descriptions**
# ****
# - train.csv - the training set.
# - test.csv - the test set. The test set contains some rows which are not included in scoring.
# - sample_submission.csv - a sample submission file in the correct format.


# ## Approach
# 
# First we will start by acquiring our data. We also need to understand our data better before creating models. We will be using Pandas, Scikit-Learn and other necessary libraries to create a classifier which can classify whether a customer will make a specific transaction in future.


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline


import gc
import os
import warnings
warnings.filterwarnings('ignore')


pd.options.display.precision = 2
pd.options.display.max_columns = 250
pd.set_option('float_format', '{:2f}'.format)
seed = 10


print(os.listdir('../input'))


path = '../input/santander-customer-transaction-prediction'


tr = pd.read_csv(f'{path}/train.csv')
ts = pd.read_csv(f'{path}/test.csv')


# ## Exploratory Data Analysis


# After acquiring we can analyze the data. This is a crucial and the most import part in a Data Science process. The better understanding we have about our data means more meaningful features we can generate and thereby increasing the accuracy of our models. 


tr.head(3)


ts.head(3)


print(f'Train shape: {tr.shape}')
print(f'Test shape: {ts.shape}')


# #### Checking for missing values
# 
# We can see that we have lots of data! It's time to check if there are any missing values somewhere out there.


print(f'Are there any missing values in train? {tr.isnull().sum().any()}')
print(f'Are there any missing values in test? {ts.isnull().sum().any()}')


# There are no null values in our dataset. This is good thing else we need to handle the missing values.


# #### Understanding the target variable
# 
# We know that this is a binary classification problem. But to know how many data samples we have in each of the classes we can use value_counts() function.


sns.countplot(tr['target'])
plt.title(f'Positive class: {round(tr["target"].value_counts()[1]/len(tr) * 100, 2)}%')
plt.show()


# Data is highly imbalanced. We either need to apply under/over sampling or play with class weights.


# #### Checking for correlation
# 
# As you may know, the features are anonymized and because of that we will not be able to know what features add importance to our target variable. We can find the correlation among the features and only select columns that are highly correlated with the target variable.


# Create correlation matrix
corr = tr.corr()['target'][1:].abs()
correlations = pd.DataFrame({'column': corr.index, 'correlation': corr}).sort_values('correlation', ascending=False).reset_index(drop=True)
correlations.head()


plt.figure(figsize=(15, 5))
plt.plot(correlations['column'][:20], correlations['correlation'][:20])
plt.xticks(correlations['column'][:20], correlations['column'][:20], rotation='45')
plt.title('Feature Correlations')
plt.show()


# Let us check the distributions of highly correlated columns.


fig = plt.figure(figsize = (10,10))
ax = fig.gca()
cols = correlations['column'][:10].values
tr[cols].hist(ax = ax)
plt.show()


# ## Base Model
# 
# Using correlation to select features might be tricky for tree models. It is safe to create a base model by passing all the features and considering only the features that are important.


from sklearn.ensemble import RandomForestClassifier


base_model = RandomForestClassifier(random_state=seed, class_weight={0:1, 1:9}, n_estimators=20, verbose=0)
%time base_model.fit(tr.drop(['ID_code', 'target'], 1), tr['target'])


importances = pd.DataFrame({'feature': tr.drop(['ID_code', 'target'], 1).columns, 'importance': base_model.feature_importances_}).sort_values('importance', ascending=False).reset_index(drop=True)
importances[:10]


# We can confirm that the top 9 features overlaps with features we selected using correlation method.


top = 100
selected_features = importances['feature'][:top].values
print(selected_features)


# Now we can proceed with the model creation part.


# ## Modelling


# Before proceeding let us shuffle our data. This is to avoid any bias while splitting the data into train and validation sets. We can easily sample our data through panda's sample() function. The parameter frac denotes the % of data to be selected. Here we need all the data so we choose 1. In other cases we can pass float values between 0 and 1 to select the appropriate % of data.


tr = tr.sample(random_state=seed, frac=1)


features = tr[selected_features]
target = tr['target']

# Payload here represents the actual test data for which we are trying to predict in this challenge
payload = ts[selected_features]


features.head()


features.shape, payload.shape, target.shape


# #### Dimensionality Reduction


# from sklearn.decomposition import PCA
# # from MulticoreTSNE import MulticoreTSNE as TSNE


# decomposed = PCA(n_components=100).fit_transform(features)


from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(features, target, test_size=0.2, random_state=seed, stratify=target)


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


model = CatBoostClassifier(random_state=seed, 
                           scale_pos_weight=10, 
                           silent=True, 
                           max_depth=None, 
                           learning_rate=0.2, 
                           loss_function='Logloss', 
                           n_estimators=2000)
%time model.fit(x_train, y_train, eval_set=(x_val, y_val), early_stopping_rounds=50, plot=True)


y_train_pred = model.predict(x_train)
y_val_pred = model.predict(x_val)


from sklearn.metrics import confusion_matrix, roc_auc_score

pd.DataFrame({'Train Set': roc_auc_score(y_train, y_train_pred)*100, 'Validation Set': roc_auc_score(y_val, y_val_pred)*100}, index=['ROC'])


plt.figure(figsize=(5, 3))
a, b = np.bincount(y_val)
# sns.heatmap(confusion_matrix(y_test, y_test_pred), annot=True, fmt='g')
# plt.show()
sns.heatmap(np.stack([(confusion_matrix(y_val, y_val_pred)[0]/a)*100, (confusion_matrix(y_val, y_val_pred)[1]/b)*100], 0), annot=True, fmt='g')
plt.show()


predictions = model.predict(payload).flatten()

submission = pd.DataFrame({'ID_code': ts['ID_code'], 'target': predictions})

submission.to_csv('submission.csv', index=False)



