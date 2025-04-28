# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
data=pd.read_csv('../input/train.csv')
print(data.head())

# Any results you write to the current directory are saved as output.


X = data.drop(["target","ID_code"], axis=1)
Y = data["target"]


import seaborn as sns
corr = data.corr()
sns.heatmap(corr)


import statsmodels.api as sm
logit_model=sm.Logit(Y,X)
result=logit_model.fit()
print(result.summary2())


X = data.drop(["target","ID_code", "var_7", "var_10","var_14", "var_17","var_27", "var_30", "var_38","var_39","var_41",
               "var_96","var_98","var_100","var_103","var_117","var_124","var_126","var_136","var_153","var_158","var_160", "var_183","var_185", ], axis=1)


# As you can see p-values for most of the variables are smaller than 0.05, except var_7, var_10, var_17, var_27, var_30, var_38, var_39, var_41, var_96, var_98, var_100, var_103, var_117, var_126, var_136, var_153, var_158, var_160, var_183, var_185, 
# We will remove these



from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
X_resampled, Y_resampled = ros.fit_resample(X, Y)
from collections import Counter
print(sorted(Counter(Y_resampled).items()))


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, Y_resampled, test_size=0.15, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)


y_pred_lr = logreg.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_lr))
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred_lr)
print(confusion_matrix)


from sklearn import svm

model = svm.LinearSVC(C=2)
# there is various option associated with it, like changing kernel, gamma and C value. Will discuss more # about it in next section.Train the model using the training sets and check score

model.fit(X_train, y_train)
model.score(X_train, y_train)
#Predict Output
y_pred_svm= model.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_svm))
print('Accuracy of SVM classifier on test set: {:.2f}'.format(model.score(X_test, y_test)))


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred_svm)
print(confusion_matrix)


from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)


# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors=5)  
classifier.fit(X_train, y_train)  

y_pred = classifier.predict(X_test)  

from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 


import pickle
with open ("model_pickle", "wb") as f:
    pickle.dump(clf,f)
with open ()

