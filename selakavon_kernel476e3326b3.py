import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score


df = pd.read_csv('../input/train.csv')
x_data = df.drop(['ID_code','target'],axis=1).values
y_data = df['target'].values
priors = np.array([0.01,0.99])
var_smoothing=1e-12
model = GaussianNB(priors=priors, var_smoothing=var_smoothing)
model.fit(x_data, y_data)


df = pd.read_csv('../input/test.csv')
x_test = df.drop('ID_code',axis=1).values
y_test = model.predict_proba(x_test)
subm = pd.DataFrame({'ID_code':df['ID_code'], 'target':pd.Series(y_test[:,1])})


subm.to_csv('submission_gnb.csv',index=False)

