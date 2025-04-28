# This is my first attempt to write the notebook from scratch. I have been playing around with exisitng kernels for competition until now. Comments, suggestions, recommendations are all very welcomed. 
# 
# I will be modeling using the following algorithms -
# * Logistic Regression
# * Decision Tree Classifier
# * Random Forest Classifier
# * Light Gradient Boosting Method
# 
# Let's start by importing necessary packages -


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns # Seaborn visualization library
sns.set(style="darkgrid")

import os
print(os.listdir("../input"))

%matplotlib inline

import gc
# Any results you write to the current directory are saved as output.


# ### Import Datasets


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# The dataset consists of an ID_code, 200 input variables (all numeric) and a binary target variable representing the transaction-happened. Since the entire dataset is masked, cannot do much of exploratory data analysis


# Look at first 10 records of the train dataset
train.head(n=10).T


# Check out the shape of the train and test sets
print('Train:', train.shape)
print('Test:', test.shape)


# This is an unbalanced classification problem with only 10% records having target variable = 1. 


# Check the target variable destribution
train['target'].value_counts()


# ## Modeling
# I would be trying a few algorithms, starting from the most simple Logistic Regression, followed by Decision Tree, Random Forest and finally, Light GBM. To build the modeling pipeline, let's import all the necessary packages we would need 


# Imports for Modeling

#from sklearn.preprocessing import Imputer, MinMaxScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, auc, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb


# Let's separate input variables and target variable. Have also created a features list with all input variable names. 


# Target variable from the Training Set
Target = train['target']

# Input dataset for Train and Test 
train_inp = train.drop(columns = ['target', 'ID_code'])
test_inp = test.drop(columns = ['ID_code'])

# List of feature names
features = list(train_inp.columns)


# Split the Train Dataset into training and validation sets for model building. 
# The training set now has 140K records and validation set has 60K records

X_train, X_test, Y_train, Y_test = train_test_split(train_inp, Target, 
                                                    test_size= 0.3, random_state = 2019)


# check the split of train and validation
print('Train:',X_train.shape)
print('Test:',X_test.shape)


# ## Logistic Regression
# We start with most basic algorithm used for classification problems. Initial model with defining only the regularization paramenter (C) yielded 0.6 AUC. Since this is an unbalanced dataset, we need to define another **paramenter 'class_weight = balanced'** which will give equal weights to both the targets irrespective of their reperesentation in the training dataset. We can even define classwise weights using this parameter, if needed 


# Create an object of Logistic Regression with parameters C and class_weight
logist = LogisticRegression(C=0.001, class_weight='balanced')

# Fit the training data on this object
logist.fit(X_train, Y_train)


# Predict the Target for validation dataset 
logist_pred = logist.predict_proba(X_test)[:,1]


logist_pred


# ## Performance Function
# Since we will be building multiple models, it is advisable to create a function that can be called with different outputs of each model. This is a simple function which takes in the Predicted Validation Target and Actual Validation Target. It then gives out classification summary like **confusion matrix and AUC score **


def performance(Y_test, logist_pred):
    logist_pred_var = [0 if i < 0.5 else 1 for i in logist_pred]
    print('Confusion Matrix:')
    print(confusion_matrix(Y_test, logist_pred_var)) 
      
    #print(classification_report(Y_test, logist_pred)) 

    fpr, tpr, thresholds = roc_curve(Y_test, logist_pred, pos_label=1)
    print('AUC:')
    print(auc(fpr, tpr))


# ### Logistic Regresssion Result 
# This model gave out an **AUC of 0.854** on validation set and 0.855 on Public Leaderboard for the test file


performance(Y_test, logist_pred)


# Submission dataframe
logist_pred_test = logist.predict_proba(test_inp)[:,1]

submit = test[['ID_code']]
submit['target'] = logist_pred_test

submit.head()


# Create the Submission File using logistic regression model
submit.to_csv('log_reg_baseline.csv', index = False)


# ## Decision Trees
# Moving on to a slightly advanced algorithm, decision trees. Again, the parameters here are class_weight to deal with unbalanced target variable, random_state for reproducability of same trees. The feature max_features and min_sample_leaf are used to prune the tree and avoid overfitting to the training data. 
# 
# **Max_features** defines what proportion of available input features will be used to create tree. 
# 
# **Min_sample_leaf** restricts the minimum number of samples in a leaf node, making sure none of the leaf nodes has less than 80 samples in it. If leaf nodes have less samples it implies we have grown the tree too much and trying to predict each sample very precisely, thus leading to overfitting.  


# Create Decision Tree Classifier object with few parameters
tree_clf = DecisionTreeClassifier(class_weight='balanced', random_state = 2019, 
                                  max_features = 0.7, min_samples_leaf = 80)

# Fit the object on training data
tree_clf.fit(X_train, Y_train)


# ### Decision Tree Results:
# Basic decision tree is giving us **0.651 AUC score** on the validation set and 0.650 AUC score on the test set submitted on public leaderboard 


# Predict for validation set and check the performance
tree_preds = tree_clf.predict_proba(X_test)[:, 1]
performance(Y_test, tree_preds)


# Submission dataframe
tree_pred_test = tree_clf.predict_proba(test_inp)[:, 1]

submitTree = test[['ID_code']]
submitTree['target'] = tree_pred_test

# Create the Submission File using logistic regression model
submitTree.to_csv('Decision_Tree.csv', index = False)


# Extract feature importances
feature_importance_values = tree_clf.feature_importances_
feature_importances = pd.DataFrame({'feature': features, 'importance': feature_importance_values})
feature_importances.sort_values(by='importance', ascending=False).head(n=10)


# Let's take a look at these features and plot them on a box and whiskrers chart


plt.figure(figsize=(20,8))
sns.boxplot(data=train[['var_81', 'var_139', 'var_12', 'var_26', 'var_146', 'var_110',
                        'var_109', 'var_53', 'var_6', 'var_166']])


# ## Ensemble Learning
# [Ensemble Learning](https://en.wikipedia.org/wiki/Ensemble_learning) refers to the algorithms that created using ensembles of variour learning algorithms. So, to give you an example, random forests are ensembles of many decision tree estimators. 
# 
# There are 2 types of ensemble learning algorithms -
# **1. Bagging Algorithms:** Bagging involves having each model in the ensemble vote with equal weight for the final output. In order to promote model variance, bagging trains each model in the ensemble using a randomly drawn subset of the training set
# **2. Boosting Algorithms:** As Wikipedia defines, boosting involves incrementally building an ensemble by training each new model instance to emphasize the training instances that previous models mis-classified.
# 
# ## Random Forest
# Let's start with building a random forest, with parameters like class_weight, random_state, and hyperparameters like max_features and min_sample_leaf as earlier. We have also defined the n_estimators which is a compulsory parameter. This defines the number of decision trees that will be present in the forest. 


# Create random Forest Object using the mentioned parameters
random_forest = RandomForestClassifier(n_estimators=100, random_state=2019, verbose=1,
                                      class_weight='balanced', max_features = 0.5, 
                                       min_samples_leaf = 100)

# Fit the object on training set 
random_forest.fit(X_train, Y_train)


#  ### Random Forest Results:
#  Basic random forest is giving us **0.787 AUC score** on the validation set and 0.789 AUC score on the test set submitted on public leaderboard


# Predict the validation set target and check the performance
forest_preds = random_forest.predict_proba(X_test)[:, 1]
performance(Y_test, forest_preds)


# Submission dataframe
forest_pred_test = random_forest.predict_proba(test_inp)[:, 1]

submitForest = test[['ID_code']]
submitForest['target'] = forest_pred_test

# Create the Submission File using logistic regression model
submitForest.to_csv('Random_Forest.csv', index = False)


# The feature importance we get from random forest is very similar to the list we got from decision trees 


# Extract feature importances
feature_importance_values = random_forest.feature_importances_
feature_importances = pd.DataFrame({'feature': features, 'importance': feature_importance_values})
feature_importances.sort_values(by='importance', ascending=False).head(n=10)


# ## Light Gradient Boosting Method
# 
# **WHAT IS IT? **
# 
# Light GBM is a gradient boosting framework that uses tree based learning algorithm. It grows tree vertically while other algorithm grows trees horizontally meaning that Light GBM grows tree leaf-wise while other algorithm grows level-wise. Leaf-wise algorithm can reduce more loss than a level-wise algorithm.
# 
# **WHY USE LGB?**
# 
# It is ‘Light’ because of its high speed. It can handle large data, requires low memory to run and focuses on accuracy of results. Also supports GPU learning and thus data scientists/ Kagglers are widely using LGBM for data science application development.
# 
# **TIPS & TRICKS**
# 
# * The algorithm easily overfits and thus, should not be used with small (< 10K rows) datasets.
# * Deal with overfitting using these parameters:
#     1. Small Maximum Depth
#     2. Large Minimum Data in a Leaf
#     3. Small Feature and Bagging Fraction
# * Improve the training speed
#     1. Small Bagging Fraction
#     2. Early Stopping Round 
# * Use small learning_rate with large num_iterations for better accuracy
# * Ideally, the value of num_leaves should be less than or equal to 2^(max_depth). Value more than this will result in overfitting
# * **If you have a big enough dataset, use this algorithm at least once. It’s accuracy has challenged other boosting algorithms**


#custom function to build the LightGBM model.
def run_lgb(X_train, Y_train, X_test, Y_test, test_inp):
    params = {
        "objective" : "binary",
        "metric" : "auc",
        "num_leaves" : 1000,
        "learning_rate" : 0.01,
        "bagging_fraction" : 0.8,
        "feature_fraction" : 0.8,
        "bagging_freq" : 5,
        "reg_alpha" : 1.728910519108444,
        "reg_lambda" : 4.9847051755586085,
        "random_state" : 42,
        "bagging_seed" : 2019,
        "verbosity" : -1,
        "max_depth": 18,
        "min_child_samples":100
       # ,"boosting":"rf"
    }
    
    lgtrain = lgb.Dataset(X_train, label=Y_train)
    lgval = lgb.Dataset(X_test, label=Y_test)
    evals_result = {}
    model = lgb.train(params, lgtrain, 2500, valid_sets=[lgval], 
                      early_stopping_rounds=50, verbose_eval=50, evals_result=evals_result)
    
    pred_test_y = model.predict(test_inp, num_iteration=model.best_iteration)
    return pred_test_y, model, evals_result

# Training the model #
pred_test, model, evals_result = run_lgb(X_train, Y_train, X_test, Y_test, test_inp)


# ### Light GBM Results:
# The AUC Score drastically improves from 0.650 in our Decision Tree model to **an AUC score of 0.89** in our ensemble of trees, Light GBM model. The public leaderboard scores after submitting the test predictions come out to be 0.891
# 
# The feature importance though, it has some variables similar to those we saw in the tree models but majority of them are new in the top 10 most important variable list


# Extract feature importances
feature_importance_values = model.feature_importance()
feature_importances = pd.DataFrame({'feature': features, 'importance': feature_importance_values})
feature_importances.sort_values(by='importance', ascending=False).head(n=10)


# Submission dataframe
pred_test[pred_test>1] = 1
pred_test[pred_test<0] = 0

submitLGB = test[['ID_code']]
submitLGB["target"] = pred_test

# Create the Submission File using Light GBM
submitLGB.to_csv('LightGBM.csv', index = False)

submitLGB.head()


# ## Next Stpes:
# Now that we have a considerably good AUC score to start with, we can improve on it. A very promising approach is to create new features based on the domain knowledge or based on the EDA we usually do as the first step. Tuning the model or creating a more sophisticated stacked architecture helps improve the score too.
# 



