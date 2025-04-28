import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression,Lasso,LassoCV,Ridge,RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


data=pd.read_csv('../input/train.csv')


(data['target'].value_counts()/data['target'].count()*100).plot(kind='hist',color='r')


data['target'].value_counts()/data['target'].count()


data.corr()


data.describe()


from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


X = add_constant(data)
X.shape[1]


for i in range(1,data.shape[1]):
    print(data.iloc[:,i].plot(kind='hist'))
    plt.xlabel(data.columns[i])
    plt.show()


X=data.drop(columns=['target','ID_code'])


X.head()


Y=data.iloc[:,1]


from sklearn.model_selection import train_test_split


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3)


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()


lr.fit(x_train,y_train)


pred=lr.predict(x_test)


from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score


accuracy_score(y_test,pred)


recall_score(y_test,pred)


precision_score(y_test,pred)


f1_score(y_test,pred)


!pip install imblearn


from imblearn.over_sampling import SMOTE
sm=SMOTE()


x_train1,ytrain1=sm.fit_sample(x_train,y_train)


lr.fit(x_train1,ytrain1)


pred1=lr.predict(x_test)




accuracy_score(y_test,pred1),recall_score(y_test,pred1),precision_score(y_test,pred1),f1_score(y_test,pred1)


 from sklearn.metrics import precision_recall_curve,roc_auc_score


roc_auc_score(y_test,pred1),roc_auc_score(y_test,pred)









