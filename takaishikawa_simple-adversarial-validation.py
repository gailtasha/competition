# ***[Please upvote this kernel if you like it!]***
# 
# ## **[Adversarial validation](http://fastml.com/adversarial-validation-part-one/)**
# 
# Here, we will cofirm how different training and test datasets are by the adversarial validation.  
# 
# ***If we attempted to train a classifier to distinguish training datasets from test datasets, it would perform no better than random. This would correspond to ROC AUC of 0.5.***
# 
# ( I may be not correct, so I'll welcome any comments. )


import numpy as np; np.random.random(42)
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import warnings; warnings.filterwarnings("ignore")

%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


plt.rcParams['figure.figsize'] = [10, 5]
plt.rcParams['font.size'] = 12


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
print(train.shape, test.shape)


train.head(3)


print( train.target.value_counts() / train.shape[0] * 100 )


feature_names = train.columns[2:]


# **We start by setting the labels according to the task. It’s as easy as:**


train["train_test"] = 1
test["train_test"] = 0


# **Then we concatenate both frames and shuffle the examples:**


data = pd.concat(( train, test ))

np.random.seed(42)
data = data.iloc[ np.random.permutation(len( data )) ]
data.reset_index( drop = True, inplace = True )

x = data.drop( [ 'target', 'ID_code','train_test' ], axis = 1 )
y = data.train_test


# **Finally we create a new train/test split:**


train_examples = len(train)

x_train = x[:train_examples]
x_test = x[train_examples:]
y_train = y[:train_examples]
y_test = y[train_examples:]


# **Come to think of it, there’s a shorter way (no need to shuffle examples beforehand, too):**


x_train, x_test, y_train, y_test = train_test_split( x, y, train_size = train_examples, random_state=42 )


# **Now we’re ready to train and evaluate. Here are the scores:**


clf = LogisticRegression(penalty="l1", C=0.1, solver="liblinear", random_state=42)
clf.fit(x_train, y_train)
y_pred = clf.predict_proba(x_test)[:, 1]
roc_auc_score(y_test, y_pred)


clf = RandomForestClassifier(n_estimators=10, random_state=42)
clf.fit(x_train, y_train)
y_pred = clf.predict_proba(x_test)[:, 1]
print("AUC:",round(roc_auc_score(y_test, y_pred)*100,2),"%")


clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(x_train, y_train)
y_pred = clf.predict_proba(x_test)[:, 1]
print("AUC:",round(roc_auc_score(y_test, y_pred)*100,2),"%")


# ***Although There's room for improvement in these models, but, for now, we can't distinguish train and test datasets.***
# 
# ***So, "Trust CV" may be also very true to this competiton!***



