# <img src="https://cached.imagescaler.hbpl.co.uk/resize/scaleWidth/743/cached.offlinehbpl.hbpl.co.uk/news/OMC/35588D26-DA0A-7761-CEE8D2C3A142E700.JPG" width="400px"/>


# ## Business Goal:
# ### Identify which customers will make a specific transaction in the future, irrespective of the amount of money transacted. 


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import gc
import os
import logging
import datetime
import warnings

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.offline as py 
from plotly.offline import init_notebook_mode, iplot
py.init_notebook_mode(connected=True) # this code, allow us to work with offline plotly version
import plotly.graph_objs as go # it's like "plt" of matplot

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


train_df = pd.read_csv('../input/train.csv')
train_df.head()


test_df = pd.read_csv('../input/test.csv')
test_df.head()


train_df['target'].value_counts()


sns.countplot(train_df['target'])


# There is major imbalance in target values. 


# ### Data Preprocessing


train_df.shape


train_df.describe()


# #### Any missing values?


train_df.isnull().any().any()


# ### variable distribution.


train_df.hist(figsize = (20,20), bins = 20)
plt.subplots_adjust(bottom=1.5, right=1.5, top=3)
plt.show()


# ## Feature Enginnering and Prediction
# ### Random forest classifier
# A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.


label = train_df.target
features = [c for c in train_df.columns if c not in ['ID_code','target']]

X_train, X_test, y_train, y_test = train_test_split(train_df[features], label, test_size = 0.02, random_state = 7)
X_train1, y_train1 = X_train, y_train
X_test1, y_test1 = X_test, y_test

model1 = RandomForestClassifier(n_estimators = 50, random_state = 0).fit(X_train1, y_train1)
y_pred = model1.predict(X_test1)


# ### checking /model performance or accuracy score.


from sklearn.metrics import accuracy_score,roc_curve, auc
accuracy_score(y_test1, y_pred)


# Accuracy score is nearly 90%, which is good. for further investigation we can check relative feature importance


# ### Feature Importance


feature_importances = pd.DataFrame(model1.feature_importances_, index = X_train.columns, columns = ['importance'])
feature_importances = feature_importances.sort_values('importance' , ascending = False)
#feature_importances.head()

colors = ['grey'] * 47 + ['green'] * 50
trace1 = go.Bar(x = feature_importances.importance[:97][::-1],
               y = [x.title()+"  " for x in feature_importances.index[:97][::-1]],
               name = 'feature importnace (relative)',
               marker = dict(color = colors, opacity=0.4), orientation = 'h')

data = [trace1]

layout = go.Layout(
    margin=dict(l=400), width = 1000, height = 1000,
    xaxis=dict(range=(0.0,0.015)),
    title='Feature Importance (Which Features are important to make predictions ?)',
    barmode='group',
    bargap=0.25
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# We can further analys more important features


submission = pd.read_csv('../input/sample_submission.csv')
#submission['target'] = model1.predict_proba(X_test1)[:,1]
#submission.to_csv('submission_gnb.csv', index=False)


# Feature enginnering plot reference kernel:  [Shivam Bansal](https://www.kaggle.com/shivamb/an-insightful-story-of-crowdfunding-projects) 



