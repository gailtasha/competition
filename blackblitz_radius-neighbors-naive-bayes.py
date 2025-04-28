# # Introduction
# 
# In this kernel, we implement the radius neighbors naive Bayes, which uses a trick from [cdeotte's kernel](https://www.kaggle.com/cdeotte/modified-naive-bayes-santander-0-899) that bypasses likelihood calculation and directly approximates the posterior. This is an attempt to improve over the [Gaussian naive Bayes classifier](https://www.kaggle.com/blackblitz/gaussian-naive-bayes).
# 
# We implement the radius neighbors naive Bayes model to predict Santander Customer Transaction Prediction data. The problem has a binary target and 200 continuous features, and we assume that these features are independent. We model the target $Y$ as Bernoulli, taking values $0$ (negative) and $1$ (positive). The features $X_0,X_1,\ldots,X_{199}$ are modelled as continuous random variables. Recall the Bayes rule:
# 
# $$p_{Y|X_0,X_1,\ldots,X_{199}}(y|x_0,x_1,\ldots,x_{199})=\frac{p_Y(y)\prod_{i=0}^{199}f_{X_i|Y}(x_i|y)}{\sum_{y'=0}^1p_Y(y')\prod_{i=0}^{199}f_{X_i|Y}(x_i|y')}$$
# 
# The trick is to directly approximate the posterior given each feature $p_{Y|X_i}(y|x_i)$ and rewrite the above rule in the following form:
# 
# $$p_{Y|X_0,X_1,\ldots,X_{199}}(y|x_0,x_1,\ldots,x_{199})=\frac{p_Y(y)\prod_{i=0}^{199}f_{X_i|Y}(x_i|y)}{\prod_{i=0}^{199}f_{X_i}(x_i)}=\frac{\prod_{i=0}^{199}p_{Y|X_i}(y|x_i)}{(p_Y(y))^{199}}$$
# 
# The first equality is becuase of the assumption that the features are (unconditionally) independent and the second equality is obtained by applying Bayes rule to each feature and substituting $\frac{f_{X_i|Y}(x_i|y)}{f_{X_i}(x_i)}=\frac{p_{Y|X_i}(y|x_i)}{p_Y(y)}$.
# 
# We can use a similarity-based method to estimate the posterior probabilities given each feature $p_{Y|X_i}(y|x_i)$ and plug into the above formula to get the posterior probabilities given all the features $p_{Y|X_0,X_1,\ldots,X_{199}}(y|x_0,x_1,\ldots,x_{199})$. However, this step is not legal since we are using approximate values in the Bayes rule instead of computing from a fully specified model. The output is not guaranteed to satisfy the axioms of probability so we need to normalize in the end. This hack enables us to avoid the curse of dimensionality at the expense of the independence assumption.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline


plt.style.use('bmh')
plt.rcParams['figure.figsize'] = (10, 10)
title_config = {'fontsize': 20, 'y': 1.05}


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


X_train = train.iloc[:, 2:].values.astype('float64')
y_train = train['target'].values
X_test = test.iloc[:, 1:].values.astype('float64')


# # Implementing the Model
# 
# We will use the proportion of the classes as the prior $p_Y(y)$ and the proportion of radius neighbors as the approximate posterior given each feature $p_{Y|X_i}(y|x_i)$. Then we use the following update rule to predict the probability of each class.
# 
# $$p_{Y|X_0,X_1,\ldots,X_{199}}(y|x_0,x_1,\ldots,x_{199})=\frac{\prod_{i=0}^{199}p_{Y|X_i}(y|x_i)}{(p_Y(y))^{199}}$$
# 
# We will work on the log scale for stability. The log posterior is calculated as follows:
# 
# $$\ln p_{Y|X_0,X_1,\ldots,X_{199}}(y|x_0,x_1,\ldots,x_{199})=\sum_{i=0}^{199}\ln p_{Y|X_i}(y|x_i)-199\ln p_Y(y)$$
# 
# Key points in the implementation are:
# * The log prior is set as the logarithm of the proportion of different classes.
# * The features are standardized and a grid from -5 to 5 with size `steps` is constructed.
# * For each point in the grid, the posterior probability given each feature is approximated by using the proportion of different classes of points falling in `radius` if there are at least `threshold` points. Otherwise, the posterior probability is set as the prior probability.
# * [numpy.searchsorted](https://docs.scipy.org/doc/numpy/reference/generated/numpy.searchsorted.html) (which uses binary search) is used to look up the probability and the update rule is applied to obtain the posterior probabilities given all the features.
# * Since the radius neighbors is just an approximation, the output is not an true probability - it can violate the axioms of probability (exceeding one and not summing to one). So we need to renormalize the log posterior by subtracting its logsumexp before exponentiation.


from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.special import logsumexp

class RadiusNeighborsNB(BaseEstimator, ClassifierMixin):
    def __init__(self, radius=1.0, steps=100, threshold=500):
        self.radius = radius
        self.steps = steps
        self.threshold = threshold
    def fit(self, X, y):
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        self.log_prior_ = np.log(np.bincount(y)) - np.log(len(y))
        self.grid_ = np.linspace(-5, 5, self.steps)
        # shape of self.log_prob_grid_
        shape = (self.steps, X.shape[1], len(self.log_prior_))
        self.log_prob_grid_ = np.full(shape, self.log_prior_)
        for i in range(shape[0]):
            for j in range(shape[1]):
                mask = np.abs(X[:, j] - self.grid_[i]) < self.radius
                total = mask.sum()
                if total >= self.threshold:
                    self.log_prob_grid_[i, j] = (np.log(np.bincount(y[mask]))
                                                 - np.log(total))
        return self
    def predict_proba(self, X):
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        # shape of log_prob
        shape = (*X.shape, len(self.log_prior_))
        log_prob = np.empty(shape)
        for j in range(shape[1]):
            lookup = np.searchsorted(self.grid_, X[:, j])
            lookup[lookup == len(self.grid_)] -= 1
            log_prob[:, j] = self.log_prob_grid_[lookup, j]
        log_posterior = log_prob.sum(axis=1) - (X.shape[1] - 1) * self.log_prior_
        return np.exp(log_posterior - logsumexp(log_posterior))
    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


# # Training and Evaluating the Model
# 
# We train and evaluate the model by using the training AUC and validation AUC. The time taken to train and predict depends on the hyperparameters (especially `steps`) as well as the size of the data. In order to speed up the hyperparameter search, we will use validation, which is k times faster than k-fold cross-validation. Feel free to use cross-validation if you have the time (and tell me if you find better hyperparameters).


from sklearn.model_selection import StratifiedShuffleSplit

i_train, i_valid = next(StratifiedShuffleSplit(n_splits=1).split(X_train, y_train))


from sklearn.metrics import roc_auc_score

model = RadiusNeighborsNB(radius=0.35, threshold=30)
model.fit(X_train[i_train], y_train[i_train])
print(f'Training AUC is {roc_auc_score(y_train[i_train], model.predict_proba(X_train[i_train])[:, 1])}.')
print(f'Validation AUC is {roc_auc_score(y_train[i_valid], model.predict_proba(X_train[i_valid])[:, 1])}.')


# # Submitting the Test Predictions
# 
# We retrain using all the data and submit the test predictions for the competition.


model.fit(X_train, y_train)


submission = pd.read_csv('../input/sample_submission.csv')
submission['target'] = model.predict_proba(X_test)[:, 1]
submission.to_csv('submission.csv', index=False)


# # Conclusion
# 
# The radius neighbors naive Bayes performs very well and is an improvement over the Gaussian naive Bayes, although it takes a little more time to train. It has the advantage that it is more flexible and does not require that the data come from a normal distribution. The only assumption is that the features are independent. This is an unusual Bayesian method since we do not specify the likelihood model. The calculations are approximate so the outputs need to be renormalized. If we want to specify a likelihood model based on the data, we can use the Gaussian mixture model (see [here](https://www.kaggle.com/blackblitz/gaussian-mixture-naive-bayes) for implementation) or kernel density estimation to model the likelihood distribution. Whichever method we use, the goal is to have a model that is simple (easily understood), tractable (easily computed) and accurate (represents reality very well).

