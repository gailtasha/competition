# ## Abstract
# The aim of this notebook is to look at target distribution versus all variables. Many people have already done similar job, so, first of all, I kindly invite you to check their efforts:
# * [Modified Naive Bayes - Santander - [0.899]](https://www.kaggle.com/cdeotte/modified-naive-bayes-santander-0-899)
# * [Gaussian Naive Bayes](https://www.kaggle.com/blackblitz/gaussian-naive-bayes)
# * [Fast PDF calculation with correlation matrix](https://www.kaggle.com/jiweiliu/fast-pdf-calculation-with-correlation-matrix)
# * [Are vars mixed up time intervals?](https://www.kaggle.com/sibmike/are-vars-mixed-up-time-intervals)
# * [Boosting creativity towards feature engineering](https://www.kaggle.com/felipemello/boosting-creativity-towards-feature-engineering)
# 
# Yet many results have already been shown, there is always place for exploration. In this kernel I will try to look at variables from machines perspective.


# ## Charts and cool stuff
# It is not surprise that current **.900** score can be achieved by using LGBM with shallow (only 2 or 3 leaves) trees. More investigation into this area shows that standard **max_bins=255** parameter can be reduced down to **max_bins=25** without any loss in quality. It means that 25 possible points for split is more than enough for this data. (i.e. algorithm sees no point in complicated splits like top 0.5% values vs 99.5% rest). However, the tails of the variables is the most interesting part in this task. Usually they have increased number of occurrences of the positive class. 
# 
# Let's look how positive class is distributed across variables.


# Load packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
sns.set_style('whitegrid')
pd.set_option('display.max_columns', 500)


# Read data. Bin variable values into 50 and 100 bins.
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv') 

coltouse = [col for col in train.columns.tolist() if col not in ['ID_code', 'target']]

for col in tqdm(coltouse):
    out = pd.qcut(train[col], q=50, labels=False, precision=5)
    train[f'{col}_bin50'] = out.values
    
for col in tqdm(coltouse):
    out = pd.qcut(train[col], q=100, labels=False, precision=5)
    train[f'{col}_bin100'] = out.values    


# Draw charts. Each chart represents dinamics of average target per bin.
# Each bin consists of almost equal number of observations.
# Green dots - bin values (i.e. average target for that bin)
# Red line - average target across all dataset.
# Purple lines - conditional borders equals to average target +/- 0.01.
# Each variable is plotted on two scales - 50 and 100 bins. 
# Plots are hidden. If you'd like to look at them - press "Output" button.
for col in coltouse:
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 6))
    plt.tight_layout()
    
    ax1.set_title(f"Distribution of {col}_bin50. Nunique is {train[col].nunique()}")
    ax1.plot(train.groupby([f'{col}_bin50'])['target'].agg('mean'), 'b:')
    ax1.plot(train.groupby([f'{col}_bin50'])['target'].agg('mean'), 'go')
    ax1.axhline(y=train['target'].mean(), color='r', linestyle='-')
    ax1.axhline(y=train['target'].mean()+0.01, color='purple', linestyle='--')
    ax1.axhline(y=train['target'].mean()-0.01, color='purple', linestyle='--')
    
    ax2.set_title(f"Distribution of {col}_bin100. Nunique is {train[col].nunique()}")
    ax2.plot(train.groupby([f'{col}_bin100'])['target'].agg('mean'), 'b:')
    ax2.plot(train.groupby([f'{col}_bin100'])['target'].agg('mean'), 'go')
    ax2.axhline(y=train['target'].mean(), color='r', linestyle='-')
    ax2.axhline(y=train['target'].mean()+0.01, color='purple', linestyle='--')
    ax2.axhline(y=train['target'].mean()-0.01, color='purple', linestyle='--')
    
    plt.show()


# ### Three types of distribution
# As we can see target seems to have 3 types of distributions across variables.
# 
# ### 1st type. Nice and cosy exponential-like monotonous-like distribution.
# These type of distribution seems to growth (or decrease) exponentially as variable increases in value. Probably, that's where models took most information from. It is easy to build good separation rules with these varaibles. Good examples are - **var_34** and **var_40**.


f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 6))
plt.tight_layout()

ax1.set_title(f"Distribution of var_34_bin50. Nunique is {train[col].nunique()}")
ax1.plot(train.groupby([f'var_34_bin50'])['target'].agg('mean'), 'b:')
ax1.plot(train.groupby([f'var_34_bin50'])['target'].agg('mean'), 'go')
ax1.axhline(y=train['target'].mean(), color='r', linestyle='-')
ax1.axhline(y=train['target'].mean()+0.01, color='purple', linestyle='--')
ax1.axhline(y=train['target'].mean()-0.01, color='purple', linestyle='--')

ax2.set_title(f"Distribution of var_40_bin100. Nunique is {train[col].nunique()}")
ax2.plot(train.groupby([f'var_40_bin100'])['target'].agg('mean'), 'b:')
ax2.plot(train.groupby([f'var_40_bin100'])['target'].agg('mean'), 'go')
ax2.axhline(y=train['target'].mean(), color='r', linestyle='-')
ax2.axhline(y=train['target'].mean()+0.01, color='purple', linestyle='--')
ax2.axhline(y=train['target'].mean()-0.01, color='purple', linestyle='--')

plt.show()


# ### 2nd type. Saw-like distribution.
# This type of distribution represents contradictory information. It is concentrated around average target value. Consecutive bin values tend to vary a lot. Unlikely this type of variables presents any usefull information to the model. Moreover, I did local experiment where I dropped 30 saw-like variables and still got **.900** score on 5-folds CV. Good examples are - **var_29** and **var_38**.


f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 6))
plt.tight_layout()

ax1.set_title(f"Distribution of var_29_bin50. Nunique is {train[col].nunique()}")
ax1.plot(train.groupby([f'var_29_bin50'])['target'].agg('mean'), 'b:')
ax1.plot(train.groupby([f'var_29_bin50'])['target'].agg('mean'), 'go')
ax1.axhline(y=train['target'].mean(), color='r', linestyle='-')
ax1.axhline(y=train['target'].mean()+0.01, color='purple', linestyle='--')
ax1.axhline(y=train['target'].mean()-0.01, color='purple', linestyle='--')

ax2.set_title(f"Distribution of var_38_bin100. Nunique is {train[col].nunique()}")
ax2.plot(train.groupby([f'var_38_bin100'])['target'].agg('mean'), 'b:')
ax2.plot(train.groupby([f'var_38_bin100'])['target'].agg('mean'), 'go')
ax2.axhline(y=train['target'].mean(), color='r', linestyle='-')
ax2.axhline(y=train['target'].mean()+0.01, color='purple', linestyle='--')
ax2.axhline(y=train['target'].mean()-0.01, color='purple', linestyle='--')

plt.show()


# ### 3rd type. Nice and cosy with an outlier.
# Probably this is the most interesting type of distribution. It behaves like 1st type, but has a spike somewhere near median. This can heavily affect LGBM ability to identify optimal split. I don't know why this pattern cannot be seen in Chris's kernel. Some variables has enourmous outliers in the center. Good examples are - **var_80** and **var_108**.


f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 6))
plt.tight_layout()

ax1.set_title(f"Distribution of var_80_bin50. Nunique is {train[col].nunique()}")
ax1.plot(train.groupby([f'var_80_bin50'])['target'].agg('mean'), 'b:')
ax1.plot(train.groupby([f'var_80_bin50'])['target'].agg('mean'), 'go')
ax1.axhline(y=train['target'].mean(), color='r', linestyle='-')
ax1.axhline(y=train['target'].mean()+0.01, color='purple', linestyle='--')
ax1.axhline(y=train['target'].mean()-0.01, color='purple', linestyle='--')

ax2.set_title(f"Distribution of var_108_bin100. Nunique is {train[col].nunique()}")
ax2.plot(train.groupby([f'var_108_bin100'])['target'].agg('mean'), 'b:')
ax2.plot(train.groupby([f'var_108_bin100'])['target'].agg('mean'), 'go')
ax2.axhline(y=train['target'].mean(), color='r', linestyle='-')
ax2.axhline(y=train['target'].mean()+0.01, color='purple', linestyle='--')
ax2.axhline(y=train['target'].mean()-0.01, color='purple', linestyle='--')

plt.show()


# ### 4th type. (Arbitrary)
# Whereas there are three clear patterns, someone might observe other patterns. For example, some features are stable for the first 50% of values and then start behave like 1st type. (or it might be binning effect).


# ## Closing points
# * Score is already high. Decision trees are capable of capturing higher target distribution in tails on their own. No point in trying to generate any features like quantile, quintile, rounding etc.
# * Despite the fact that dropping saw-like features didn't affect the score, it is still question whether these features might be usefull for feature engineering.
# * Discussions [here](https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/80996) and [here](https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/86736) indicates that one way to increase score (at least up to **.910** level) is to create additional 200 features. (i.e. apply some kind of transformation to original 200 features).
# * People who broke out of **.901** club also state that they observed weird patterns of [local CV increase whereas public LB stayed on the same level](https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/80996#502449). What kind of transformation can lead to such results?
# * No target encoding reportedly been used. 
# * What if vars are time series and FE might be related to order and aggregate features based on their target distribution plot?

