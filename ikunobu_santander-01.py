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


train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")


display(train_df.head())
display(train_df.describe())


train_null_s = train_df.isnull().sum()
test_null_s = test_df.isnull().sum()
print(train_null_s[train_null_s>0])
print(test_null_s[test_null_s>0])


sns.countplot(train_df["target"])


train_df.columns.values[2:202]


features = [c for c in train_df.columns if c not in ['ID_code', 'target']]
target = train_df['target']


X_train = train_df.drop(['ID_code', 'target'],axis=1)
y_train = train_df["target"]


y_pred = rfc.predict(test_df[])


from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(random_state=0)
rfc.fit(X_train,y_train)


X_test = test_df.drop('ID_code',axis=1)
y_test_pred = rfc.predict(X_test)


print(y_test_pred)


df_sub = pd.read_csv("./submission.csv")
sns.countplot(df_sub["target"])


my_submission = pd.DataFrame({'ID_code': test_df["ID_code"].values, 'target': y_test_pred})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)



