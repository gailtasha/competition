# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


train = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')
test = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')


train.info()


test.info()


train_32 = train.drop(['ID_code', 'target'], axis = 1).astype('float32')


train_32.info()


test_32 = train.drop(['ID_code'], axis = 1).astype('float32')


test_32.info()


train.head(2)


train_32.head(2)


test.head(2)


test_32.head(2)


train_32.shape, test_32.shape


train_32.describe()


test_32.describe()


# **Checking target**


import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(x='target',data=train, palette='hls')
plt.show()


# **Logistic Regression**


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


X = train.iloc[:, 2:].values
y = train.target.values
X_test = test.iloc[:, 1:].values


X_train = X
y_train = y


#create an instance and fit the model 
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)


predictions = logmodel.predict(X_test)


from sklearn.metrics import classification_report
print(classification_report(y_train,predictions))


sub_df = pd.DataFrame({'ID_code':test.ID_code.values})
sub_df['target'] = predictions
sub_df.to_csv('submission.csv', index=False)


# **Gaussian Naive Bayes
# **


# When dealing with continuous variable, how do you calculate the probabilty for a given value(e.g. x=1)? The probability should be zero. So we'd better calculate the probability of a interval(e.g.  1−Δ<x<1+Δ ).


%matplotlib inline
from scipy.stats import norm


x = np.linspace(-5, 5)
y = norm.pdf(x)
plt.plot(x, y)
plt.vlines(ymin=0, ymax=0.4, x=1, colors=['red'])


# If the interval is small enough(i.e.  Δ→0 ), the probability of a given value(e.g. x=1) can be represented by probability density(pdf) value. How can we know the probability function of a variable? The convenient way is to estimate using normal distribution. This is the Gaussian Naive Bayes.


# Let's apply Gaussian Naive Bayes to our Santander data.
# 


target = train.target.values
train.drop('target', axis=1, inplace=True)
train.shape, target.shape, test.shape


# Calculate mean/sd of train data for each each feature.
# 


pos_idx = (target == 1)
neg_idx = (target == 0)
stats = []
for col in train_32.columns:
    stats.append([
        train_32.loc[pos_idx, col].mean(),
        train_32.loc[pos_idx, col].std(),
        train_32.loc[neg_idx, col].mean(),
        train_32.loc[neg_idx, col].std()
    ])
    
stats_df = pd.DataFrame(stats, columns=['pos_mean', 'pos_sd', 'neg_mean', 'neg_sd'])
stats_df.head()


# Using normal distribution to estimate each feature
# 


# priori probability
ppos = pos_idx.sum() / len(pos_idx)
pneg = neg_idx.sum() / len(neg_idx)

def get_proba(x):
    # we use odds P(target=1|X=x)/P(target=0|X=x)
    return (ppos * norm.pdf(x, loc=stats_df.pos_mean, scale=stats_df.pos_sd).prod()) /\
           (pneg * norm.pdf(x, loc=stats_df.neg_mean, scale=stats_df.neg_sd).prod())


tr_pred = train_32.apply(get_proba, axis=1)


from sklearn.metrics import roc_auc_score
roc_auc_score(target, tr_pred)


# Gaussian Naive Bayes can give us 0.876 AUC, which is quite good!
# 


# Remove the Gaussian constrain


# Infact our data is not normal distributed, we can achive better score with Gaussian constran removed. let's take var_0 as an example.


plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
sns.distplot(train.loc[pos_idx, 'var_0'])
plt.plot(np.linspace(0, 20), norm.pdf(np.linspace(0, 20), loc=stats_df.loc[0, 'pos_mean'], scale=stats_df.loc[0, 'pos_sd']))
plt.title('target==1')
plt.subplot(1, 2, 2)
sns.distplot(train.loc[neg_idx, 'var_0'])
plt.plot(np.linspace(0, 20), norm.pdf(np.linspace(0, 20), loc=stats_df.loc[0, 'neg_mean'], scale=stats_df.loc[0, 'neg_sd']))
plt.title('target==0')


# We can see that the data is very different from normal distribution, we need use more accurate probability density function to estimate, this can be done by kernel function estimation. Let's use scipy.stats.kde.gaussian_kde


from scipy.stats.kde import gaussian_kde


plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
sns.distplot(train.loc[pos_idx, 'var_0'])
kde = gaussian_kde(train.loc[pos_idx, 'var_0'].values)
plt.plot(np.linspace(0, 20), kde(np.linspace(0, 20)))
plt.title('target==1')
plt.subplot(1, 2, 2)
sns.distplot(train.loc[neg_idx, 'var_0'])
kde = gaussian_kde(train.loc[neg_idx, 'var_0'].values)
plt.plot(np.linspace(0, 20), kde(np.linspace(0, 20)))
plt.title('target==0')


# Kernel funtion can fit the data better, which will give us better accuracy.
# 


stats_df['pos_kde'] = None
stats_df['neg_kde'] = None
for i, col in enumerate(train_32.columns):
    stats_df.loc[i, 'pos_kde'] = gaussian_kde(train_32.loc[pos_idx, col].values)
    stats_df.loc[i, 'neg_kde'] = gaussian_kde(train_32.loc[neg_idx, col].values)


def get_col_prob(df, coli, bin_num=100):
    bins = pd.cut(df.iloc[:, coli].values, bins=bin_num)
    uniq = bins.unique()
    uniq_mid = uniq.map(lambda x: (x.left + x.right) / 2)
    dense = pd.DataFrame({
        'pos': stats_df.loc[coli, 'pos_kde'](uniq_mid),
        'neg': stats_df.loc[coli, 'neg_kde'](uniq_mid)
    }, index=uniq)
    return bins.map(dense.pos).astype(float) / bins.map(dense.neg).astype(float)


tr_pred = ppos / pneg
for i in range(200):
    tr_pred *= get_col_prob(train_32, i)


roc_auc_score(target, tr_pred)


# Using more accurate kernel function, we can achieve 0.909 AUC(maybe a little overfit, since we fit train's data, but it's not too much). Let's use this model to predict the test data.


te_pred = ppos / pneg
for i in range(200):
    te_pred *= get_col_prob(test_32, i)


pd.DataFrame({
    'ID_code': test.index,
    'target': te_pred
}).to_csv('sub.csv', index=False)


# **Conclusion.**


# 
# In this kernel we demonstrate how Naive bayes works, we build Gaussian Naive Bayes, which gives us 0.890 AUC. By remove Gaussian constrain and choosing more accurate kernel function, we can get better performance



