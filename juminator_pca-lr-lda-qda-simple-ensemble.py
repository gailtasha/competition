import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
%matplotlib inline


# Read in Train Data
train = pd.read_csv("../input/train.csv")


# Read in Test Data
test = pd.read_csv("../input/test.csv")


# Number of rows and columns of training and test data
train.shape, test.shape


# # **Exploratory Data Analysis (EDA)**


train.head()


train.info()


train.select_dtypes(include="object").columns


# The only categorical feature is the ID_code variable and all other variables are numerical!


# Checking if ID_code is unique
train.ID_code.nunique() == train.shape[0]


# So every observation is an unique customer record!


# > ### **Target Variable** (What we want to predict)


sns.countplot(train.target)


train.target.value_counts() *100 / train.target.count()


# 89% for target equal to 0 and 10% for target equal to 1. Pretty unbalanced! We might want to take this unbalance into consideration! (Some algorithms don't perform well with class unbalance). Algorithms like KNN, Boosting, Random forest might work better than others. But the model evaluation metric here is "AUC" which is less sensitive to class imbalance (other recommended metrics for unbalanced classes are f1-score and logloss)


train.groupby("target").mean()


train.groupby("target").median()


# Just by first glance, observations with target==1 seem to have higher mean & median  values for each variable in general than those with target==0. Let's see if that is the case.


np.mean(train.groupby("target").mean().iloc[1] >= train.groupby("target").mean().iloc[0])


np.mean(train.groupby("target").median().iloc[1] >= train.groupby("target").mean().iloc[0])


# 52% and 51% of the variables have higher mean and median values respectively for observations with target==1.


# ### **Distributions of variables**


# I am interested in which variables are not linear not likely from a Gaussian Distribution because some ML algorithms work better if each feature is normally distributed (and if not, we might want to log transform it!) There are multiple normality tests but Shapiro test is appropriate for small dataset(N<5000), so I will use the D’Agostino’s K^2 Test!


features = train.columns.values[2:203]


from scipy.stats import normaltest


# # D’Agostino’s K^2 Test on TRAIN DATA
# non_normal_features = []
# for feature in features:
#     stat, p = normaltest(train[feature])
#     if p <= 0.01:
#         print(feature,"not normal")
#         non_normal_features.append(feature)


# # D’Agostino’s K^2 Test on TEST DATA
# non_normal_features_test_data = []
# for feature in test.columns.values[1:202]:
#     stat, p = normaltest(test[feature])
#     if p <= 0.05:
#         print(feature,"not normal")
#         non_normal_features_test_data.append(feature)


# You may want to log or squared transform these non-normal features!


# > ### **Missing Data** 


train.isnull().sum().sum()


# There is no missing data!


# ### **Correlations amongst Variables** (Credits to Gebriel Preda's Kernel "Santander EDA and Prediction")


correlations = train[features].corr().abs().unstack().sort_values(kind="quicksort").reset_index()
correlations = correlations[correlations['level_0'] != correlations['level_1']]
correlations.tail(10)


# Even the top 10 pairs with highest correlation have absolute values of 0.008. This is a very weak correlation. Multicollinearity issue doesn't seem to be a problem here!


# ### **PCA**


# PCA is a dimensionality reduction technique that reduces noise and extracts features that are independent(orthogonal). But PCA is sensitive to variance and different scales, so standardizing will help PCA perform better! HOWEVER, we found that the correlation between different features in the training dataset is not that significant, so using PCA might not be meaningful (because PCA is best when the dimension p is very large and a lot of features are correlated to one another a lot)


from sklearn.preprocessing import StandardScaler
standardized_train = StandardScaler().fit_transform(train.set_index(['ID_code','target']))


standardized_train = pd.DataFrame(standardized_train, columns=train.set_index(['ID_code','target']).columns)
standardized_train = standardized_train.join(train[['ID_code','target']])


# We have to determine the number of features we are going to extract with PCA! We use the cumulative variance explained and find the number of features where the variance doesn't increase as much.


from sklearn.decomposition import PCA
k=80
pca = PCA(n_components=k, random_state=42, whiten=True)
pca.fit(standardized_train.set_index(['ID_code','target']))


plt.figure(figsize=(25,5))
plt.plot(pca.explained_variance_ratio_)
plt.xticks(range(k))
plt.xlabel("Number of Features")
plt.ylabel("Proportion of variance explained by additional feature")


sum(pca.explained_variance_ratio_)


# Normally, if there is a elbow looking point in the graph above, the x value(number of features) of that point is usually the ideal number of components for PCA. However in this case, each principal component explains very little of the total variance (e.g. first principal component only explains abou 0.6% of the total variance). Even when we sum up all the variance explained by the 80 principal components, it only amounts to 40%. Let's increase the k and see what happens.


sum(PCA(n_components=120, random_state=42, whiten=True).fit(standardized_train.set_index(['ID_code','target'])).\
explained_variance_ratio_)


sum(PCA(n_components=160, random_state=42, whiten=True).fit(standardized_train.set_index(['ID_code','target'])).\
explained_variance_ratio_)


# 80% of the total variance is explained if we use 160 principal components. 80% is not bad! Let's reduce 200 features to 160 by setting k=160 for PCA.


pca = PCA(n_components=160).fit_transform(standardized_train.set_index(['ID_code','target']))


pca_col_names = []
for i in range(160):
    pca_col_names.append("pca_var_" + str(i))
pca_col_names


# Save PCA transformed train dataset just in case
pca_train = pd.DataFrame(pca, columns=pca_col_names).join(train[['ID_code','target']])
pca_train.to_csv("pca_train.csv")


# Standardize the test data as well
standardized_test = StandardScaler().fit_transform(test.set_index(['ID_code']))
standardized_test = pd.DataFrame(standardized_test, columns=test.set_index(['ID_code']).columns)
standardized_test = standardized_test.join(test[['ID_code']])


pca = PCA(n_components=160).fit_transform(standardized_test.set_index(['ID_code']))


pca_col_name_for_test = []
for i in range(160):
    pca_col_name_for_test.append("pca_var_" + str(i))


# Save PCA transformed test dataset just in case
pca_test = pd.DataFrame(pca, columns=pca_col_name_for_test).join(train[['ID_code']])
pca_test.to_csv("pca_test.csv")


# # **Modelling**


# from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_score


# X = standardized_train.drop('target',axis=1).set_index('ID_code')
# y = standardized_train[['target']]


# # Split training dataset to train and validation set
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Split Train Dataset into Predictor variables Matrix and Target variable Matrix
X_train = standardized_train.set_index(['ID_code','target']).values.astype('float64')
y_train = standardized_train['target'].values


# #### Logistic Regression


# Logistic Regression
from sklearn.linear_model import LogisticRegression
logit_clf = LogisticRegression(random_state=42).fit(X_train,y_train)


plt.figure(figsize=(10, 10))
fpr, tpr, thr = roc_curve(y_train, logit_clf.predict_proba(X_train)[:,1])
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operator Characteristic Plot', fontsize=20, y=1.05)
auc(fpr, tpr)


cross_val_score(logit_clf, X_train, y_train, scoring='roc_auc', cv=10).mean()


# #### Linear Discriminant Analysis (LDA)
# - LDA aims to find the directions that maximize the separation (or discrimination) between different classes
# - LDA tries to determine a suitable feature (sub)space in order to distinguish between patterns that belong to different classes
# - Estimate parameters with maximum likelihood (those parameters minimize Squared Mahalanobis Distance)
# - Models the distribution of predictors separately in each of the response classes, and then it uses Bayes’ theorem to estimate the probability
# - Both LDA and QDA assume the the predictor variables X are drawn from a multivariate Gaussian (aka normal) distribution.
# - (Compared to QDA) LDA is more suitable for smaller data sets, and it has a higher bias, and a lower variance.
# - If n is small and the distribution of the predictors X is approximately normal in each of the classes, the LDA model is more stable than logistic.
# - When the classes are well-separated, the parameter estimates for the logistic model are surprisingly unstable. LDA does not suffer from this.
# - LDA and QDA are attractive because they have closed-form solutions that can be easily computed, are inherently multiclass, have proven to work well in practice, and have no hyper-parameters to tune.


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

lda_clf = LinearDiscriminantAnalysis()
lda_clf.fit(X_train, y_train)


plt.figure(figsize=(6, 6))
fpr, tpr, thr = roc_curve(y_train, lda_clf.predict_proba(X_train)[:,1])
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operator Characteristic Plot', fontsize=20, y=1.05)
auc(fpr, tpr)


cross_val_score(lda_clf, X_train, y_train, scoring='roc_auc', cv=10).mean()


# #### Quadratic Discriminant Analysis
# - A classifier with a quadratic decision boundary, generated by fitting class conditional densities to the data and using Bayes’ rule.
# - The model fits a Gaussian density to each class.
# - QDA is a better option for large data sets (compared to LDA), as it tends to have a lower bias and a higher variance.


qda_clf = QuadraticDiscriminantAnalysis()
qda_clf.fit(X_train, y_train)


plt.figure(figsize=(6, 6))
fpr, tpr, thr = roc_curve(y_train, qda_clf.predict_proba(X_train)[:,1])
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operator Characteristic Plot', fontsize=20, y=1.05)
auc(fpr, tpr)


cross_val_score(qda_clf, X_train, y_train, scoring='roc_auc', cv=10).mean()


# LDA has the highest AUC for cross validation among the three ML algorithms (Logis****tic regression, LDA, QDA) I tried so far! 


X_test = standardized_test.set_index('ID_code').values.astype('float64')
submission = pd.read_csv('../input/sample_submission.csv')
submission['target'] = logit_clf.predict_proba(X_test)[:,1]
submission.to_csv('LR.csv', index=False)


X_test = standardized_test.set_index('ID_code').values.astype('float64')
submission = pd.read_csv('../input/sample_submission.csv')
submission['target'] = lda_clf.predict_proba(X_test)[:,1]
submission.to_csv('lda.csv', index=False)


X_test = standardized_test.set_index('ID_code').values.astype('float64')
submission = pd.read_csv('../input/sample_submission.csv')
submission['target'] = qda_clf.predict_proba(X_test)[:,1]
submission.to_csv('lda.csv', index=False)


# # **Simple Ensemble Method**


# There are various Ensemble methods and one way is to use the mean probability of all the models. That is, for each observation, different ML algorithms predict the probability of that observation being part of class 1 and we calculate the mean of all those probabilities.


X_test = standardized_test.set_index('ID_code').values.astype('float64')
submission = pd.read_csv('../input/sample_submission.csv')

logit_pred = logit_clf.predict_proba(X_test)[:,1]
lda_pred = lda_clf.predict_proba(X_test)[:,1]
qda_pred = qda_clf.predict_proba(X_test)[:,1]


submission = \
submission.join(pd.DataFrame(qda_pred, columns=['target1'])).join(pd.DataFrame(logit_pred, columns=['target2'])).\
join(pd.DataFrame(lda_pred, columns=['target3']))


submission['target'] = (submission.target1 + submission.target2 + submission.target3) / 3


submission.head()


del submission['target1']
del submission['target2']
del submission['target3']


submission.head()


submission.to_csv('logit_lda_qda_mean_ensemble.csv', index=False)

