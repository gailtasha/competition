# # Learning Data Science Part 2.0
# 
# This kernel I create to train my data science skills. I want to train my step course to apply them on this data set. I never mind about the accuracy of my model, but, as long as my progress on this kernel, I try to improve that. So, task for this competition is to predict probability customer of santander who will conducted transaction on future and their nominal.


import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from IPython.core.interactiveshell import InteractiveShell
import warnings

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from keras.layers import Dense
from keras.models import Sequential
import gc
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold


pd.options.display.max_columns = None
warnings.filterwarnings('ignore')


traindf = pd.read_csv('../input/santander-customer-transaction-prediction/train.csv')
testdf = pd.read_csv('../input/santander-customer-transaction-prediction/test.csv')


traindf['target'].value_counts().plot(kind='bar', title='Unbalance target variable')


fig = plt.figure(figsize=(20,10))

for i in range(0,50):
    fig.add_subplot(5,10,i+1)
    plt.title('Distribution on var_'+str(50+i))
    sns.distplot(traindf['var_'+str(50+i)], color="m")

fig.tight_layout(pad=0.1)


# Based on graph above, I think variable have normal distribution. But, they have different range. So, in the next step, i will apply standard scaler to those variable.


# ## Build Model


cols=[c for c in traindf.columns if c not in ['ID_code', 'target']]
y = traindf["target"]
x = traindf[cols]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)


rb = RobustScaler(with_centering=True,copy=False)
x_train = rb.fit_transform(x_train)
x_test = rb.fit_transform(x_test)
transform_2 = rb.fit_transform(testdf[cols])
testdf[cols] = transform_2


# ## Light Gredient Boost Decision Tree


params = {
    'subsample': 0.95,
    'subsample_freq': 100,
    'num_iterations': 25000,
    'learning_rate': 0.01,
    'early_stopping_rounds':2500,
    'max_bin':20,
    'min_data_in_leaf':80,
    'objective': 'binary',
    'metric': 'auc',
    'boosting' : 'gbdt',
    'is_unbalance': True,
    'num_threads': 8,
    'verbosity': 1,
    'num_leaves': 16,
    'min_hessian': 80,
    'tree_learner': 'serial',
    'max_depth': 4,
    'feature_fraction': 0.95,
}


train_data = lgb.Dataset(x_train, label=y_train)
valid_data = lgb.Dataset(x_test, label=y_test, reference=train_data)

lgbmodel = lgb.train(params, train_data,                     
                 valid_sets=[valid_data],
                 valid_names=['valid'],
                 verbose_eval=1000)
score = lgbmodel.best_score['valid']['auc']
y_pred = lgbmodel.predict(x_test)
print("Accuracy on test data:",metrics.accuracy_score(y_test, y_pred.round(0).astype(int)))
print("Model ROC_AUC on test data: {:.2f}%".format(roc_auc_score(y_test, y_pred.round(0).astype(int))*100))
print('Best AUC score {}'.format(score*100))


# ## Neural Network


model = Sequential()
model.add(Dense(36, input_dim=200,activation='relu',kernel_initializer='glorot_normal',bias_initializer='random_normal'))
model.add(Dense(20, activation='relu',kernel_initializer='glorot_uniform',bias_initializer='random_normal'))
model.add(Dense(16, activation='relu',kernel_initializer='glorot_uniform',bias_initializer='random_normal'))
model.add(Dense(8, activation='relu',kernel_initializer='glorot_uniform',bias_initializer='random_normal'))
model.add(Dense(1, activation='sigmoid',kernel_initializer='glorot_uniform',bias_initializer='random_normal'))


model.compile(loss = 'binary_crossentropy', optimizer = 'Adam', metrics = ['accuracy'])


model.fit(x_train, y_train, epochs=20, batch_size=1000)
_, accuracy = model.evaluate(x_test, y_test)
print('Accuracy: %.2f' % (accuracy*100))


y_pred = model.predict_classes(x_test)
print("Accuracy on test data:",metrics.accuracy_score(y_test, y_pred))
print("Model ROC_AUC on test data: {:.2f}%".format(roc_auc_score(y_test, y_pred)))


# ## Logistic Regression


logreg = LogisticRegression(max_iter=10000, C=50)
logreg.fit(x_train, y_train.ravel())

y_pred = logreg.predict(x_test)
print("Accuracy on test data:",metrics.accuracy_score(y_test.ravel(), y_pred))
print("Model ROC_AUC on test data: {:.2f}%".format(roc_auc_score(y_test, y_pred)))


# ## For Submission


y_pred_1 = logreg.predict(testdf[cols].to_numpy())
y_pred_2 = model.predict_classes(testdf[cols].to_numpy())
y_pred_3 = lgbmodel.predict(testdf[cols].to_numpy()).round(0).astype(int)

testdf['target'] = y_pred_1
testdf[['ID_code','target']].to_csv('SantanderSubmission1.csv', index=False)

testdf['target'] = y_pred_2
testdf[['ID_code','target']].to_csv('SantanderSubmission2.csv', index=False)

testdf['target'] = y_pred_3
testdf[['ID_code','target']].to_csv('SantanderSubmission3.csv', index=False)

