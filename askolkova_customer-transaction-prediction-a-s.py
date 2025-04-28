# ### Preliminary setup


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

import os
print(os.listdir("../input"))


# ## 1. Reading data


data_path = '../input/train.csv'
df = pd.read_csv(data_path)


# ### 1.1 Brief look at the data


df.head()


df.describe()


df.info()


# So, we don't understand the meaning of the features at all. Ok.


y = df.target
features = [col for col in df.columns if 'var' in col]
X = df[features]

train_X, val_X, train_y, val_y = train_test_split(X, y, test_size = 0.3, random_state=7)


# ### 1.2 Pairwise correlations for X


#fig,ax = plt.subplots(figsize=(200, 200))
#sb.heatmap(df.corr(), ax=ax, annot=True, linewidths=0.05, fmt= '.2f',cmap="magma")
#plt.show()
# -> bad idea, it's too big

#fig,ax = plt.subplots(figsize=(50, 50))
#sb.heatmap(df[0:50].corr(), ax=ax, annot=True, linewidths=0.05, fmt= '.2f',cmap="magma")
#plt.show()
# -> also too big




# ## 2. Basic Linear Regression


lr = LinearRegression()
lr.fit(train_X, train_y)
lr_predicts = lr.predict(val_X)
lr_mse = mse(lr_predicts, val_y)
print(lr_mse)


# Linear regression: MSE = 0.0742733 (score = 0.86096)


# submitting the very first predictions
test_data_path = '../input/test.csv'
test_data = pd.read_csv(test_data_path)
test_X = test_data[features]
#test_preds = lr.predict(test_X)

#output = pd.DataFrame({'ID_code': test_data.ID_code,
#                       'target': test_preds})
#output.to_csv('submission.csv', index=False)


# ## 3. Decision Trees


#dt = DecisionTreeRegressor(max_leaf_nodes=100, random_state=0)
#dt.fit(train_X, train_y)
#dt_predicts = dt.predict(val_X)
#mse_dt = mse(dt_predicts, val_y)
#print(mse_dt)


# Worse than linear regression.
# ## 4. Ridge Regression with 10fold CV


from sklearn.linear_model import RidgeCV
rCV = RidgeCV(normalize=True, cv=10)
rCV.fit(train_X, train_y)
rCV_predicts = rCV.predict(val_X)
mse_rCV=mse(rCV_predicts, val_y)
print(mse_rCV)


print(mse_rCV)


# Ridge with CV: MSE = 0.07439 (almost no improvement over the simple linear regression)


test_data_path = '../input/test.csv'
test_data = pd.read_csv(test_data_path)
test_X = test_data[features]
test_preds = rCV.predict(test_X)

output = pd.DataFrame({'ID_code': test_data.ID_code,
                       'target': test_preds})
output.to_csv('submission.csv', index=False)


# Private ranking is slightly better, BUT
# obviously, it's better to account for the fact that the target variable is binary.
# 
# ## 5. LogisticRegression


from sklearn.linear_model import LogisticRegression
logr = LogisticRegression()
logr.fit(train_X, train_y)
logr_predicts = logr.predict(val_X)

from sklearn.metrics import accuracy_score
acc_logr = accuracy_score(logr_predicts, val_y)
print(acc_logr)




test_data_path = '../input/test.csv'
test_data = pd.read_csv(test_data_path)
test_X = test_data[features]
test_preds = logr.predict(test_X)

output = pd.DataFrame({'ID_code': test_data.ID_code,
                       'target': test_preds})
output.to_csv('submission.csv', index=False)

