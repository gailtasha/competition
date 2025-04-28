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


# **Import the package that will be needed for the project.**


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline


# **Import the data.**


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# **Summary data analysis including info,description,head name, and column name.**


train.info()


train.describe()


train.head(5)


train.columns


# **As a data quality specialist, always check the data quality issues, e.g. whether there will be missing values, null values, and etc. And if there are a few ways to figure it out, for example, fill in the blank with 0, mean value, or expand the data set to minimize the influence caused by missing values.**


def check_missing_value(df):
    flag=df.isna().sum().any()
    if flag==True:
        total = df.isnull().sum()
        percent = (df.isnull().sum())/(df.isnull().count()*100)
        output = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        data_type = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            data_type.append(dtype)
        output['Types'] = data_type
        return(np.transpose(output))
    else:
        return(False)


# **There are no missing values in the train data set.**


check_missing_value(train)


# **There are no missing values in the test data set either, which makes the data cleansing step easier.**


check_missing_value(test)


# **The visualization part of the data analysis, the only visualization for the train/test data set I can think of is I am trying to find out whether the data is correlated. Below is the finding, unfortunately, they are not correlated AT ALL.**


train_corr = train.drop(["target"], axis=1).corr()
train_corr = train_corr.values.flatten()
train_corr = train_corr[train_corr != 1]

test_corr = test.corr()
test_corr = test_corr.values.flatten()
test_corr = test_corr[test_corr != 1]

plt.figure(figsize=(20,5))
sns.distplot(train_corr, color="Red", label="train")
sns.distplot(test_corr, color="Yellow", label="test")
plt.xlabel("Correlation values found in train")
plt.ylabel("Density")
plt.title("Correlation Relationship"); 
plt.legend();


# **Now is the machine learning part, since the out put is 0/1(yes/no), it will be not a regression model; moreover, when I got some more complicated case, I always try different model to check which one is more accurate; for this case, I choose logistic regression and random forest to predict**


from sklearn.model_selection import train_test_split 


y = train['target']
X = train.iloc[:, 2:]


# **I will separate the train data to train/dev data set to do the machine learning, set the test size, and random state, with a summary of the data set.**


X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.3, random_state=101)
print('X_train shape: ', X_train.shape)
print('y_train shape: ', y_train.shape)
print('X_dev shape: ', X_dev.shape)
print('y_dev shape: ', y_dev.shape)


# **Import the logistic regression model from sklearn**


from sklearn.linear_model import LogisticRegression


# **Fit the model**


lr = LogisticRegression(C = 0.001, class_weight = 'balanced')
lr.fit(X_train, y_train)


# **Make prediction, and see the prediction data set**


prediction_lr = lr.predict_proba(X_dev)[:,1]
prediction_lr


# **See how the model fits the data by classification_report and confusion_matrix**


from sklearn.metrics import classification_report, confusion_matrix
prediction_lr_var = [0 if i < 0.5 else 1 for i in prediction_lr]
print(confusion_matrix(y_dev, prediction_lr_var))
print('\n')
print(classification_report(y_dev, prediction_lr_var))


# **As the model fit the data well, I will use the model to test the test data set, and come up with the prediction**


test_var = test.drop(columns = ['ID_code'])
lr_pred_test = lr.predict_proba(test_var)[:,1]

submit = test[['ID_code']]
submit['target'] = lr_pred_test

submit.head()


# **When I got a project, I always try with mutiple machine learning models, here is the ensembling model which is the random forest part**


from sklearn.ensemble import RandomForestClassifier


rfc = RandomForestClassifier(n_estimators=100, random_state = 101, verbose = 1, class_weight = 'balanced', max_features = 0.5, min_samples_leaf = 100)
rfc.fit(X_train, y_train)


prediction_rfc = rfc.predict(X_dev)


from sklearn.metrics import classification_report, confusion_matrix
prediction_rfc_var = [0 if i < 0.5 else 1 for i in prediction_rfc]
print(confusion_matrix(y_dev, prediction_rfc_var))
print('\n')
print(classification_report(y_dev, prediction_rfc_var))


# **Make prediction by using the random forest model**


rfc_pred_test = rfc.predict_proba(test_var)[:,1]

submit = test[['ID_code']]
submit['target'] = rfc_pred_test

submit.head()


# **Submission**


submit.to_csv('lr_Santnader.csv', index = False)
submit.head()

