# Hi everyone! 
# 
# Here is another take on trying to get the most out of the data when the features are assumed to be independent. The idea is to fit 200 classifiers, each using only one of the features, and then combining the predictions using Bayes' rule as in Naive Bayes methods. I've previously experimented by using kernel density estimates to obtain the individual classifiers as in Chris Deotte's kernel [here](http://www.kaggle.com/cdeotte/modified-naive-bayes-santander-0-899), but achieved better results when the individual classifiers where gradient boosted trees. Enjoy!


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb

import os
print(os.listdir("../input"))


train_df = pd.read_csv("../input/train.csv")


# LightGBM parameters found manually. 
# I picked one feature (var_12) and tuned the parameters only on that feature. 
# I tried to keep n_iterations fairly low (~600) while still achieving good results.
params = {
    'boosting':'gbdt', 
    'bagging_freq':5,
    'bagging_fraction':0.5, # important for creating fairly smooth prediction functions.
                # without bagging, tree splits always happens at the same 1023 places and
                # the result is a very "rugged" function
    'num_leaves':2,
    'reg_lambda':100.0,
    'learning_rate':0.01, 
    'max_bin':1023, # seems to allow fitting the tails better when using more bins
    'seed':3366
}


# One could of course try to tune the LGBM parameters for each feature individually, but that's just too tedious. So I prayed that the same parameters work for every feature as long as I allow the number of number of trees to vary for each feature. The following function estimates the optimal number of rounds for each feature.


def optimal_rounds(X, verbose=False):
    """
    Runs LGBM with 3-fold CV on each feature separately with early stopping and
    determines the optimal number of trees for each feature.
    """
    rounds = []
    for i in range(200):
        if verbose:
            print("Feature ", i)
        cv_res = lgb.cv(params, 
           lgb.Dataset(X[['var_'+str(i)]], X['target']),
           nfold=3, # increasing this doesn't seem to matter much, so keep it low for speed
           num_boost_round=100000,
           metrics='binary_logloss',
           verbose_eval=100 if verbose else None,
           early_stopping_rounds=100
          )
        rounds.append(len(cv_res['binary_logloss-mean']))
    return rounds


# It is important that the metric used for the early stopping is the binary logloss (or something similar) and not the AUC score! For many features, the probability function to predict seem to be monotonic. That is, the larger the value of that feature the more likely that the class value is 1 (or 0). In these cases, **any** monotonic predictor will have the same (and best possible) AUC score, since all induce the same ranking of the observations. But a random monotonic function is not good for anything, since it is crucial for Naive Bayes is to estimate the individual conditional probabilities as accurately as possible.


opt_rounds = optimal_rounds(train_df, verbose=True)


print("Optimal number of rounds: ", opt_rounds)
print("Optimal number of rounds for var_108: ", opt_rounds[108])
print("Optimal number of rounds for var_30: ", opt_rounds[30])


# As we can see, the optimal number of rounds varies greatly. 
# 
# There are features that require quite a long training. For example, it seems that the distibution of var_108 has a sharp spike somewhere at the middle and that takes a long time to take into account. (See the end of this notebook for a figure.) The kernel density estimate in Chris's [kernel](http://www.kaggle.com/cdeotte/modified-naive-bayes-santander-0-899) does seem to see this spike very well, so this might illustrate why this version of Naive Bayes performs slightly better.
# 
# Interestingly, there are also features where the optimal number of trees is less then 10, indicating that those features might be very weak predictors (or perhaps that the chosen LGBM parameters just don't work well for those features).


# Next, we implement our LGB Naive Bayes classifier.


from sklearn.utils.fixes import logsumexp

num_ones = np.sum(train_df['target'] == 1)
num_zeros = np.sum(train_df['target'] == 0)

class LGBNaiveBayes:
    def fit(self,X_train, y_train, opt_rounds):
        self.clfs = []
        for i in range(200):
            if i%20 == 0:
                print("Fitting var_"+ str(i)+"...")
            params['n_estimators'] = opt_rounds[i]

            lgb_clf = lgb.LGBMClassifier(**params)
            lgb_clf.fit(X_train[['var_'+str(i)]], y_train)

            self.clfs.append(lgb_clf)
            
    def predict_proba(self,X):
        log_sum = np.zeros((X.shape[0],2))
        for i in range(200):
            # Adding up the log-probabilities to compute the joint probabilities. 
            # This assumes independence of the features just like other Naive Bayes methods.
            log_sum += np.log(self.clfs[i].predict_proba(X[['var_'+str(i)]]))
            
            # Correcting with the apriori log-probabilities of the two classes. This does not
            # have an effect on the AUC score since it shifts all log-probabilities with the 
            # same amount and hence does not change the ranking of the observations.
            # Without these correction though, the predicted probabilities of class 1 would be
            # extremely small though which is both incorrect and inconvenient when comparing to
            # predictions obtained from other classifiers.
            log_sum += np.array([np.log(num_ones) - np.log(200000), np.log(num_zeros) - np.log(200000)])
        # One last correction term for the final log-probabilities. Again, it is not important for
        # the AUC score.
        log_sum -= np.array([np.log(num_ones) - np.log(200000), np.log(num_zeros) - np.log(200000)])
        
        # Applying numerically stable softmax using logs, inspired by the scikit-learn
        # implementation of Naive Bayes.
        log_prob_x = logsumexp(log_sum, axis=1)
        return np.exp(log_sum - np.atleast_2d(log_prob_x).T)



clf = LGBNaiveBayes()


# Performing 5-fold CV.


from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score

features = train_df.columns[2:].values

def cross_validate(nfolds):
    sss = StratifiedShuffleSplit(nfolds)
    aucs = []
    for train, test in sss.split(train_df[features], train_df['target']):
        clf.fit(train_df.loc[train][features], train_df.loc[train]['target'], opt_rounds)
        y_true = train_df.loc[test]['target']
        y_pred = clf.predict_proba(train_df.loc[test][features])[:,1]
        test_auc = roc_auc_score(y_true, y_pred)
        aucs.append(test_auc)
        print("Test AUC:", test_auc)
    print("Mean test AUC: ", np.mean(aucs))
    
cross_validate(5)


# Creating submission:


test_df = pd.read_csv('../input/test.csv')
clf.fit(train_df[features], train_df['target'], opt_rounds)
pred = clf.predict_proba(test_df.iloc[:][features])
sub_df = pd.DataFrame({"ID_code":test_df["ID_code"].values})
sub_df["target"] = pred[:,1]
sub_df.to_csv("submission.csv", index=False)


# Finally, here is the prediction plot for var_108. The spike in the middle is nicely recognized.


import matplotlib.pyplot as plt
x1 = min(train_df['var_108'])
x2 = max(train_df['var_108'])
data = np.arange(x1, x2, 0.001)
fig = plt.figure(figsize=(18,12))
plt.plot(data, clf.clfs[108].predict_proba(data.reshape(-1,1))[:,1])


import seaborn as sns
plt.figure(figsize=(18,12))
sns.distplot(train_df.loc[train_df['target']==0]['var_108'], bins=300)
sns.distplot(train_df.loc[train_df['target']==1]['var_108'], bins=300)



