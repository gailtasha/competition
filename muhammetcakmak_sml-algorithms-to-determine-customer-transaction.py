# ## Supervised Machine Learning Classifications
# In this tutorial I will apply supervised machine learning classifications to the canser data sets in order to determine if tested data has heart diseases or not. I will use KNN classification, decision tree classification, random forest classification,Support vector machine, logistig regression and naive bayes algorithms. I will show also how to determine accuracy of the each classificaiton and make evaluation by using confusion matrix.
# 
# 1. [EDA(Exploratory Data Analaysis)](#1)
# 2. [Logistic Regression Classification](#2)
# 4. [Decision Tree Classification](#3)
# 5. [Random Forest Classification](#4)
# 7. [Naive Bayes Classification](#5)
# 6. [Conclusion](#6)


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Read the data and assign it as df_train and df_test 
df_train=pd.read_csv("../input/train.csv")
df_test=pd.read_csv("../input/test.csv")


# <a id="1"></a> <br>
# 1. EDA(Exploratory Data Analaysis)
# 
# EDA is very important to look at what is inside the data. For example, if there is object(string) in the data, we need to change it to integer or float because sci-learn is not handling with object data. There are also missdata in the datasets, we need to handle them.


# Let's have a quick look into data.This code shows first 5 rows and all columns
df_train.head()


# Let's have a quick look into data.This code shows first 5 rows and all columns
df_test.head()


# If there is unknown,missing or unproper data, this codes shows the number of them
# We can also learn about features such as data type of the features
df_train.info()


df_test.info()


# statistical data is important to learn about balance inside or among the features.
df_train.describe()


# Seaborn countplot gives the number of data in the each class
sns.countplot(x="target", data=df_train)


# <a id="2"></a> <br>
# 2.Logistic Regression Classification
# 
# It is very powerfull algorithm to use with binary classification.


# y has target data (clases) such as 1 and 0. 
y_train_data = df_train.target.values.reshape(-1,1)
# This means that take target and ID_code data out from the datasets and assign them to variable
x_train_data = df_train.drop(["target","ID_code"],axis=1)


#Normalization is used to handle with unbalanced features
#This gives the values to the features which range from zero to 1.
x = (x_train_data - np.min(x_train_data))/(np.max(x_train_data)-np.min(x_train_data)).values


# Preperation of testing data
x_test_data = df_test.drop(["ID_code"],axis=1)
#x_test_data.head()


# Build Logistic Logistic Regression Algorithm
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train_data,y_train_data)

y_lr_test_data = lr.predict(x_test_data)


ID_code_data = df_test.ID_code.values
from numpy import array
from numpy import vstack
header=[['ID_code','target']]
lr_array = vstack((ID_code_data, y_lr_test_data)).T
frame_lr = pd.DataFrame(lr_array, columns=header)
print (frame_lr)


# <a id="3"></a> <br>
# 3.Decision Tree Classification
# 
# "Decision tree builds classification or regression models in the form of a tree structure. It breaks down a dataset into smaller and smaller subsets while at the same time an associated decision tree is incrementally developed".
# 
# According to ınformation entropy, we can determine which feature is the most important. And we put the most important one to the top of the related tree.
# 
# Decision tree classification can be used for both binary and multi classes


# Build Decision Tree Classification Model
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state = 42)
dt.fit(x_train_data,y_train_data)
y_dt_test_data = dt.predict(x_test_data)


header=[['ID_code','target']]
dt_array = vstack((ID_code_data, y_dt_test_data)).T
frame_dt = pd.DataFrame(dt_array, columns=header)
print (frame_dt)


# <a id="4"></a> <br>
# 4.Random Forest Classification
# 
# This methods basically use multiple number of decision trees and take the avarage of the results of these decision trees. And we use this avarage to determine the class of the test point.
# 
# This is one of ensamble method which uses multiple classes to predict the target, and very powerfull technique.


# Build Random Forest Classification Model
from sklearn.ensemble import RandomForestClassifier
# n_estimators = 100 means this model will use 100 subsets.
rf = RandomForestClassifier(n_estimators = 100,random_state = 42)
rf.fit(x_train_data,y_train_data)
y_rf_test_data = rf.predict(x_test_data)


header=[['ID_code','target']]
rf_array = vstack((ID_code_data, y_rf_test_data)).T
frame_rf = pd.DataFrame(rf_array, columns=header)
print (frame_rf)


# <a id="5"></a> <br>
# 5.Naive Bayes Classification
# 
# "Naive Bayes classifiers are a family of simple "probabilistic classifiers" based on applying Bayes' theorem with strong (naive) independence assumptions between the features."
# 
# Here we basically determine similarity range and calculate probabilty of the X point in the A feature P(A_feature|x).


# Build Naive Bayes Classification Model
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train_data,y_train_data)
y_nb_test_data = nb.predict(x_test_data)


header=[['ID_code','target']]
nb_array = vstack((ID_code_data, y_nb_test_data)).T
frame_nb = pd.DataFrame(nb_array, columns=header)
print (frame_nb)


# <a id="6"></a> <br>
# 6.Conclusion


# These pandas DataFrames have 2 columns as ID_code and the predicted values of y_test.
Logistic_regression = frame_lr
Decision_tree_classification = frame_dt
Random_forest_classification = frame_rf
Naive_bayes_classification = frame_nb

