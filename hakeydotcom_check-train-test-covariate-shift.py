# ![](https://www.researchgate.net/profile/Patrick_Glauner/publication/322568228/figure/fig1/AS:583987370143745@1516244847412/Example-of-covariate-shift-training-and-test-data-having-different-distributions.png)


# Hi kagglers! Let's check if there are any differences between train and test data. It allow us to determine situation when test data has other distributions, and it is incorrect to build model directly on exisiting data split.
# First load train and test data into dataframes, drop columns that contain labels and not useful data:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

import os
print(os.listdir("../input"))


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


train_df.head()


test_df.head()


train_df.drop(columns=['ID_code', 'target'], inplace=True)
test_df.drop(columns=['ID_code'], inplace=True)


# The main idea of method is to set new label "is_test"=0 for train dataframe and "is_test"=1 for test. Than we should concat this two parts into one huge dataset (train and test with corresponding "is_test" label) and shuffle it randomly. 
# 
# Next step is splitting huge dataset into two parts: **train_2** and **test_2** (for example, in 70/30 ratio). Than we train some classificaton algorithm on dataset **train_2**  using label 'is_test'. Finally predict 'is_test' label on dataset **test_2** and estimate some quality metric. For example, ROC AUC.
# 
# If value or AUC is about 0.5 than there is no sufficient reason to talk about differences in train and test. Otherwise, we need to conduct a deep data analysis.


train_df['is_test'] = 0
test_df['is_test'] = 1
df = pd.concat([train_df, test_df], axis = 0)
X = df.drop(columns=['is_test'])
y = df['is_test']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
print('train2 shape:', X_train.shape, 'test2 shape:', X_test.shape)


#Let's use simple Random Forest as Classifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
y_test_score = rfc.predict_proba(X_test)


# Plot ROC curve and see how well our classifier works:


import scikitplot as skplt
import matplotlib.pyplot as plt

skplt.metrics.plot_roc_curve(y_test, y_test_score)
plt.show()


#AUC score is about 0.5
print('AUC score: ', round(roc_auc_score(y_true=y_test, y_score=y_test_score[:,1]), 4))


# As you can see, classifier can't find any reasonable difference between train and test data. There are no evidence of non-zero covariate shift.
# 
# **Links:**
# 
# [1] https://towardsdatascience.com/how-dis-similar-are-my-train-and-test-data-56af3923de9b
# 
# [2] https://github.com/hakeydotom/Covariate-shift-prediction

