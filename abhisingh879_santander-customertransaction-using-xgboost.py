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


# **Importing Some important libraries


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score


# **Let's create a Dataframe for both Train and Test CSV files.


TrainDF = pd.read_csv("../input/train.csv")
TestDF = pd.read_csv("../input/test.csv")
RawData_DF = pd.concat([TrainDF,TestDF])
RawData_DF.describe()


# **We have checked In dataset, null & NaN values are not present !


def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)


display_all(TrainDF)


# **Counts of target variables 0 and 1  in order to see what is the distribution .


((TrainDF['target'].value_counts())/len(TrainDF)).plot.bar()


# As we can data is skewed 90% data having target var as 0 and other 10% have lebel 1.


len(TrainDF[TrainDF['target']==0.0]),len(TrainDF[TrainDF['target']==1.0])


TrainDF['target'].value_counts()


TrainPositive = TrainDF[TrainDF['target'] == 1.0]
TrainPositive.shape


# **Hyperparameter tuning **
# Since data is skewed we are doing sampling kepping 25% data of label 0. In order make robust model.


TrainSF = TrainDF[TrainDF['target'] == 0.0].sample(frac=0.25, random_state=1)
TrainSF.shape


TrainDF = pd.concat([TrainSF,TrainPositive])


TrainDF['target'].value_counts()


# Creating train and validation data. we are making 20% data a validation****


X_train, X_val, y_train, y_val = train_test_split(
    TrainDF.drop(['ID_code','target'], axis=1),
    TrainDF['target'],
    test_size=0.2, random_state=42)


X_train.shape,y_train.shape,X_val.shape,y_val.shape


# XGBoost import****


import xgboost as xgb
import matplotlib.pyplot as pyplot


d_train = xgb.DMatrix(X_train, label=y_train)
d_valid = xgb.DMatrix(X_val, label=y_val)


param = {'max_depth': 4, 'eta': .4, 'silent': 1, 'objective': 'binary:logistic'}
param['nthread'] = 4
param['eval_metric'] = 'auc'
param['min_child_weight'] = 5
param['subsample'] = .5


evallist = [(d_train, 'train'),(d_valid, 'eval')]


# Training XGboost molel below****


num_round = 50
bst = xgb.train(param, d_train, num_round, evallist, 
                early_stopping_rounds=10)


# We can see above that our xgboost model is predicting around 89% ,however on validation data it is 83% accurate.****


 check = bst.predict(xgb.DMatrix(X_val), ntree_limit=bst.best_iteration+1)


# We are plotting below AUC curve for the same****


auc = roc_auc_score(y_val, check)
print('AUC: %.3f' % auc)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_val, check)
# plot no skill
pyplot.title('Receiver Operating Characteristic')
pyplot.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
pyplot.plot(fpr, tpr,marker='.')
pyplot.ylabel('True Positive Rate')
pyplot.xlabel('False Positive Rate')
# show the plot
pyplot.show() 


dictvar = bst.get_score(importance_type='gain')
impft = pd.DataFrame(columns=['feature','importance'])


for column in TrainDF:
    impft = impft.append({'feature' : column , 'importance' : dictvar.get(column)} , ignore_index=True)


impft.sort_values('importance',ascending=False)


# As shown above we can see featurewise importance ****


# We can plot a tree with below code****


xgb.plot_tree(bst, num_trees=1)
fig = pyplot.gcf()
fig.set_size_inches(150, 100)
fig.savefig('tree.png')


fig, ax = pyplot.subplots(figsize=(200, 500))
xgb.plot_importance(bst, ax=ax)


# We are finally predicting on our Test data now****


TestDF['target'] = bst.predict(xgb.DMatrix(TestDF.drop(['ID_code'],axis=1)), ntree_limit=bst.best_iteration+1)


Solution = TestDF[['ID_code','target']]
Solution.to_csv("Santander_CustomerTransaction.csv", index=False)

