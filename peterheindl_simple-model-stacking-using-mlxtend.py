# **Simple model stacking using mlxtend**
# 
# Ensemble learning can be highly useful to improve the accuracy of classification. The basic idea is that more than one model work together to predict some outcome. Actually, boosting as used in xgboost or lightgbm was originally proposed as an ensemble learning technique. A good introduction to ensemble learning can be found in Hastie et al. (2017), “Elements of Statistical Learning” (Chapter 16 http://web.stanford.edu/~hastie/ElemStatLearn/). Averaging and Stacking are described in Section 8.8 of the book. 
# 
# A nice “hands on” description of Stacking can be found at the Kaggle Blog, posted by Ben Gorman http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/. 
# 
# A good overview on ensemble methods is provided here: https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-for-ensemble-models/.
# 
# Writing a stacking code on your own can be fun. However, there are many little helpers out there. I use *mlxtend* in this post: http://rasbt.github.io/mlxtend/user_guide/classifier/StackingClassifier/.
# 
# Let’s load and prepare some data first. Please note that I only use 5000 obs. here for demonstration:


import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB 
from sklearn import model_selection
import numpy as np
from mlxtend.classifier import StackingCVClassifier

# Load data for training 
mydata = pd.read_csv('../input/train.csv', sep=',')
# Select only 5000 obs. to let the kernel run here
mydata = mydata.head(5000)
mydata = mydata.drop('ID_code', 1)

# Load prediction data
preddata = pd.read_csv('../input/test.csv', sep=',')
predids = preddata[['ID_code']] 
iddf = preddata[['ID_code']] 
preddata = preddata.drop('ID_code', 1)

# Format train data
y_train = mydata['target']
x_train = mydata.drop('target', 1)

# Scale data
scaler = preprocessing.StandardScaler()
scaled_df = scaler.fit_transform(x_train)
x_train = pd.DataFrame(scaled_df)
scaled_df = scaler.fit_transform(preddata)
preddata = pd.DataFrame(scaled_df)

# x,y to np (needed for scipy CV)
x_train = x_train.values
y_train = y_train.values


# The next step is to train and stack some models. Here I use KNN, RF, and NB. The tree models will be stacked using Logit. In the code below, the models and the stacking classifier are defined first. Then each model is trained using CV.
# 
# Finally, the stacking classifier is fitted and predictions are obtained. All that we need to do now is to get the predictions into the right shape to make a submission.
# 
# As you can see, the AUC is rather low here and not at all competitive compared to the LB. However, its just a minimal example.
# 
# Let me know if you have any comments: happy coding!


# Set up models
clf1 = KNeighborsClassifier(n_neighbors=600)
clf2 = RandomForestClassifier(random_state=1, n_estimators=300)
clf3 = GaussianNB()
# Logit will be used for stacking
lr = LogisticRegression(solver='lbfgs')
sclf = StackingCVClassifier(classifiers=[clf1, clf2, clf3], meta_classifier=lr, use_probas=True, cv=3)

# Do CV
for clf, label in zip([clf1, clf2, clf3, sclf], 
                      ['KNN', 
                       'Random Forest', 
                       'Naive Bayes',
                       'StackingClassifier']):

    scores = model_selection.cross_val_score(clf, x_train, y_train, cv=3, scoring='roc_auc')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

# Fit on train data / predict on test data
sclf_fit = sclf.fit(x_train, y_train)
mypreds = sclf_fit.predict_proba(preddata)
# "predict" delivers classes, "predict_proba" delivers probabilities

# Probabilities for classes (1,0)
zeros = [i[0] for i in mypreds]
ones  = [i[1] for i in mypreds]

# Get IDs and predictions
y_id = predids.values.tolist()
preddf = pd.DataFrame({'ID_code': y_id,'target': ones})
preddf['ID_code'] = preddf['ID_code'].map(lambda x: str(x)[:-2])
preddf['ID_code'] = preddf['ID_code'].map(lambda x: str(x)[2:])

# Look at predictions
print(preddf.head())

# Save DF
preddf.to_csv('submission.csv', index=False)

