# In the kernel [Modified Naive Bayes scores 0.899 LB - Santander](https://www.kaggle.com/cdeotte/modified-naive-bayes-santander-0-899), [Chris Deotte](https://www.kaggle.com/cdeotte) has demonstrated that Naive Bayes can be a simple but powful method when there is little or no interaction between the features. I think it's a good time to study Naive Bayes Method.
# 
# In this kernel, I'll try to introduce Naive Bayes method step by step.


# ### Bayes' theorem
# 
# Bayes' theorem can be deduced by conditional probability:
# 
# $$
# P(A|B) = \frac {P(A \bigcap B)}{P(B)}
# $$
# 
# $$
# P(B|A) = \frac {P(A \bigcap B)}{P(A)}
# $$
# 
# $$
# \Rightarrow P(A|B) = \frac {P(B|A) \cdot P(A)} {P(B)}
# $$


# Suppose we have features $X \in R^n$, target $y \in \{+1, -1\}$, for a given $x_0$ our goal is to predict $y=+1$ or $-1$.
# 
# we can achive this by calculate
# 
# $$P(y=+1 | X=x_0)$$ 
# and
# $$P(y=-1 |  X=x_0)$$
# 
# then choose the one with bigger probability.


# How can we calculate $P(y=+1|X=x_0)$ ? We can use Bayes's theorem:
# $$
# P(y=+1|X=x_0) = \frac {P(X=x_0 | y=+1) \cdot P(y=+1)} {P(X=x_0)}
# $$
# 
# Because $P(X=X_0)$ is the probability(or frequency) of a sample in test set, it's the same for every sample, so we can simply write the formular as:
# 
# $$
# P(y=+1|X=x_0) = P(X=x_0 | y=+1) \cdot P(y=+1)
# $$
# 
# $P(y=+1/-1)$ is called priori probability and $P(X=x_0 | y=+1)$ is called conditional probability. Training process is to estimate this two kind of probability.


# ### Naive Bayes
# 
# If we have many features, then:
# $$
# P(X=x_0 | y=+1) = P(X^{(1)}=x_0^{(1)}, X^{(2)}=x_0^{(2)}, ..., X^{(n)}=x_0^{(n)} | y=+1)
# $$
# 
# To simplify this problem, we assume the features as independent(this is the Naive means):
# $$
# P(X=x_0 | y=+1) = P(X^{(1)}=x_0^{(1)} | y=+1) \cdot P(X^{(2)}=x_0^{(2)} | y=+1) \cdot, ..., \cdot P(X^{(n)}=x_0^{(n)} | y=+1)
# $$


# ### Discrete variable
# 
# We will use a sample example to demostrate the training and predicting process of Naive Bayes for discrete variable.


import pandas as pd
import numpy as np


data = pd.DataFrame({
    'X1': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
    'X2': ['S', 'M', 'M', 'S', 'S', 'S', 'M', 'M', 'L', 'L', 'L', 'M', 'M', 'L', 'L'],
    'y': [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]
})


# Try to use data to training a Naive Bayes classifier, and classify the sample $X_0=(2, S)$.


# Calculate priori probability $P(y=1)$ and $P(y=-1)$


data.y.value_counts()


# $$
# P(y=1) = \frac {9}{15}, P(y=-1) = \frac {6}{15}
# $$


# Calculate conditional probability


data[data.y == 1].X1.value_counts()


# $$
# P(X1=1 | y=1) = \frac{2}{9}, P(X1=2 | y=1) = \frac{3}{9}, P(X1=3 | y=1) = \frac{4}{9}
# $$


data[data.y == 1].X2.value_counts()


# $$
# P(X2=S | y=1) = \frac{1}{9}, P(X2=M | y=1) = \frac{4}{9}, P(X2=L | y=1) = \frac{4}{9}
# $$


data[data.y == -1].X1.value_counts()


# $$
# P(X1=1 | y=-1) = \frac{3}{6}, P(X1=2 | y=-1) = \frac{2}{6}, P(X1=3 | y=-1) = \frac{1}{6}
# $$


data[data.y == -1].X2.value_counts()


# $$
# P(X2=S | y=1) = \frac{3}{6}, P(X2=M | y=1) = \frac{2}{6}, P(X2=L | y=1) = \frac{1}{6}
# $$


# For the given sample $X_0=(2, S)$:
# $$
# P(y=1 | X=X_0) = P(y=1) \cdot P(X1=2|y=1) \cdot P(X2=S|y=1) = \frac{9}{15} \cdot \frac{3}{9} \cdot \frac{1}{9} = \frac{1}{45}
# $$
# 
# $$
# P(y=-1 | X=X_0) = P(y=-1) \cdot P(X1=2|y=-1) \cdot P(X2=S|y=-1) = \frac{6}{15} \cdot \frac{2}{6} \cdot \frac{3}{6} = \frac{1}{15}
# $$
# 
# Because $P(y=-1|X=X_0) > P(y=1|X=X_0)$, so $y=-1$
# 
# This is Naive Bayes for discrete variable, pretty simple.


# ### Gaussian Naive Bayes


# When dealing with continuous variable, how do you calculate the probabilty for a given value(e.g. x=1)? The probability should be zero. So we'd better calculate the probability of a interval(e.g. $1-\Delta < x < 1+\Delta$).


import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from scipy.stats import norm


x = np.linspace(-5, 5)
y = norm.pdf(x)
plt.plot(x, y)
plt.vlines(ymin=0, ymax=0.4, x=1, colors=['red'])


# If the interval is small enough(i.e. $\Delta \rightarrow 0$), the probability of a given value(e.g. x=1) can be represented by probability density(pdf) value. How can we know the probability function of a variable? The convenient way is to estimate using normal distribution. This is the **Gaussian Naive Bayes**.  


# Let's apply Gaussian Naive Bayes to our Santander data.


train = pd.read_csv('../input/train.csv', index_col=0)
test = pd.read_csv('../input/test.csv', index_col=0)
target = train.target.values
train.drop('target', axis=1, inplace=True)
train.shape, target.shape, test.shape


# Calculate mean/sd of train data for each each feature.


pos_idx = (target == 1)
neg_idx = (target == 0)
stats = []
for col in train.columns:
    stats.append([
        train.loc[pos_idx, col].mean(),
        train.loc[pos_idx, col].std(),
        train.loc[neg_idx, col].mean(),
        train.loc[neg_idx, col].std()
    ])
    
stats_df = pd.DataFrame(stats, columns=['pos_mean', 'pos_sd', 'neg_mean', 'neg_sd'])
stats_df.head()


# Using normal distribution to estimate each feature


# priori probability
ppos = pos_idx.sum() / len(pos_idx)
pneg = neg_idx.sum() / len(neg_idx)

def get_proba(x):
    # we use odds P(target=1|X=x)/P(target=0|X=x)
    return (ppos * norm.pdf(x, loc=stats_df.pos_mean, scale=stats_df.pos_sd).prod()) /\
           (pneg * norm.pdf(x, loc=stats_df.neg_mean, scale=stats_df.neg_sd).prod())


tr_pred = train.apply(get_proba, axis=1)


from sklearn.metrics import roc_auc_score
roc_auc_score(target, tr_pred)


# Gaussian Naive Bayes can give us 0.890 AUC, which is quite good!


# ### Remove the Gaussian constrain


# Infact our data is not normal distributed, we can achive better score with Gaussian constran removed. let's take `var_0` as an example.


plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
sns.distplot(train.loc[pos_idx, 'var_0'])
plt.plot(np.linspace(0, 20), norm.pdf(np.linspace(0, 20), loc=stats_df.loc[0, 'pos_mean'], scale=stats_df.loc[0, 'pos_sd']))
plt.title('target==1')
plt.subplot(1, 2, 2)
sns.distplot(train.loc[neg_idx, 'var_0'])
plt.plot(np.linspace(0, 20), norm.pdf(np.linspace(0, 20), loc=stats_df.loc[0, 'neg_mean'], scale=stats_df.loc[0, 'neg_sd']))
plt.title('target==0')


# We can see that the data is very different from normal distribution, we need use more accurate probability density function to estimate, this can be done by kernel function estimation. Let's use `scipy.stats.kde.gaussian_kde` 


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


stats_df['pos_kde'] = None
stats_df['neg_kde'] = None
for i, col in enumerate(train.columns):
    stats_df.loc[i, 'pos_kde'] = gaussian_kde(train.loc[pos_idx, col].values)
    stats_df.loc[i, 'neg_kde'] = gaussian_kde(train.loc[neg_idx, col].values)


def get_proba2(x):
    proba = ppos / pneg
    for i in range(200):
        proba *= stats_df.loc[i, 'pos_kde'](x[i]) / stats_df.loc[i, 'neg_kde'](x[i])
    return proba


%%time
get_proba2(train.iloc[0].values)


# It's too slow, we can speed up by binize the variable values.


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
    tr_pred *= get_col_prob(train, i)


roc_auc_score(target, tr_pred)


# Using more accurate kernel function, we can achieve 0.909 AUC(maybe a little overfit, since we fit train's data, but it's not too much). Let's use this model to predict the test data.


te_pred = ppos / pneg
for i in range(200):
    te_pred *= get_col_prob(test, i)


pd.DataFrame({
    'ID_code': test.index,
    'target': te_pred
}).to_csv('sub.csv', index=False)


# ### Conclusion
# 
# In this kernel we demonstrate how Naive bayes works, we build Gaussian Naive Bayes, which gives us 0.890 AUC. By remove Gaussian constrain and choosing more accurate kernel function, we can get better performance.
# 
# Holp this can help, thanks!

