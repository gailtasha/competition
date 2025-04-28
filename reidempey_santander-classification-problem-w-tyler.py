# # Walkthrough of Tyler's Example Code
# 
# This is going to be a version of what Tyler did in class with comments explaining each step. It is slightly different from Tyler's exact code becasue I was not able to save the notebook he worked on. However, even though some of the code is written a little differently, all of the main ideas and important steps he took are here and have been explained.



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))


# First, we are going to read in the data and take a sample of our data so that we can run our models faster. Before doing this, we ran through this process once and got validation scores with a model. Afterwards, we come back to this point and take a smaller sample of our data. From here, we can run through the process again. If our model performs somewhat similarly to how it did with all of the data, we can use a smaller sample to get through iterations of our model quicker, which is what we will do here. 


df_train = pd.read_csv('../input/train.csv')
df_sample = df_train.sample(frac=.3)
y = df_sample['target']


# When doing classification problems it is always important to know the distribution of your classes. Here we can see we have a lot more observations classified as zeros than we have classified as ones. We will take this into account when building a model.


y.value_counts(normalize=True)


# A good first step when exploring your data is to look at what variables matter most when predicting the target variable. It can give good intuition into how your model is going to act and helps understand what variables may not be needed when modeling if we need to drop features. Here we will look at the 10 most correlated variables. In this example it is hard to get much information because the variables have been made anonymous, but we can see that there are no variables that are extremely correlated with the target variable.


df_train.corr()['target'].sort_values(ascending=False)[0:10]


# We never want the target variable or ID variables in our training data. Leaving your target variable in will cause your model to perform perfectly on test data, which is bad. The ID will never give your model real information and may throw off predictions if there is random correlation between it and the target variable.


X = df_sample.drop(['target', 'ID_code'], axis=1).copy()


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, roc_auc_score


# Now it is time to start modeling. We are going to split our data up twice. First, we need a training set and a testing set. We are going to put the testing set off to the side and not touch it until we feel we have a model that is ready for production. Then, we are going to take the training data and split it up into a new training and validation set to help us tune our model.


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
x_tr, x_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=.2, random_state=42)


y_train.value_counts()


# We are going to use a Random Forest Classifier from sklearn to do our modeling. At this point, we are going to start iterating through a lot of versions of models to try to best predict our data. We started with a basic random forest that was overfitting by a lot. We change the class weights (because of the imbalanced data), number of estimators, and max depth of the trees to create a better model. 


param_dictionary = {"n_estimators": [1000]}
clf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=3)
# Press Shift-Tab to look at what the arguments are for a function, as well as the defaults for each argument
gs = GridSearchCV(clf, param_dictionary, scoring='roc_auc', n_jobs=1, verbose=2, cv=2)
gs.fit(x_tr, y_tr)
# max depth 5, n estimators 500


#gs.best_index_


#gs.cv_results_


# First we are going to look at the score we get when predicting on our data that we trained on.


train_predictions = gs.predict(x_tr)
cr = classification_report(y_tr, train_predictions)
roc_auc = roc_auc_score(y_tr, train_predictions)
print("Training Scores:")
print(cr)
print("-"*50)
print("ROC AUC Score: {}".format(roc_auc))


# Now we are going to look at the validation scores and compare them to our training scores. We can see that the training score is a little bit higher, indicating some overfitting. There is not that big of a difference though and it is the best model we have built so far. 


val_predictions = gs.predict(x_val)
cr = classification_report(y_val, val_predictions)
roc_auc = roc_auc_score(y_val, val_predictions)
print('Validation Scores:')
print(cr)
print('-'*50)
print("ROC AUC Score: {}".format(roc_auc))


# We can also look at feature importances for our model after training it. The feature importances can be useful if we need to drop variables for some reason. We can use it to select the best variables that we want to keep in our model. 


feat_imports = sorted(list(zip(X_train.columns, gs.best_estimator_.feature_importances_)), key=lambda x:x[1], reverse=True)
feat_imports[0:10]


clf = RandomForestClassifier(n_jobs=-1, max_depth=5, n_estimators=1000, class_weight='balanced', verbose=1)
clf.fit(X_train, y_train)


# Now that we have selected a model, we are going to train it on all of our training data and predict on our holdout set of test data. The hope is that this score is similar to our validation score above and will also be a good representation of what our model would do in the real world, or in this case our Kaggle leaderboard. We only use this data when we have built a good model. We want to touch it as little as possible to avoid overfitting to it. We can see it is slightly lower than our validation score above which is to be expected since we used our validation score to tune our model.


test_predictions = clf.predict(X_test)
roc_auc = roc_auc_score(y_test, test_predictions)
print('ROC AUC Test Score: {}'.format(roc_auc))


# Next steps for building a better model would include some error analysis. We can predict our probabilities for each test observation instead of classes and use them to see which observations we predicted the most wrong. We would go through the process of trying to find patterns and understanding why we got some of these observations wrong and explaining them. 


#Error Analysis
probabilities = clf.predict_proba(X_test)


probabilities = probabilities[:,1]
errors = pd.DataFrame()
errors['probs']=probabilities
errors['truth']=y_test.values
errors.head()


errors[errors.truth==1].sort_values(by='probs')[0:10]


X_test.iloc[3995]['var_81']

