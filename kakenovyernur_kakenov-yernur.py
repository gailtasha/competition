# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, roc_curve
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.



from shutil import copyfile

copyfile(src = "../input/thinkstats/thinkstats2.py", dst = "../working/thinkstats2.py")
copyfile(src = "../input/thinkstats/thinkplot.py", dst = "../working/thinkplot.py")

from thinkstats2 import *
from thinkplot import *


import thinkplot
import thinkstats2


train = pd.read_csv("/kaggle/input/santander-customer-transaction-prediction/train.csv")
test = pd.read_csv("/kaggle/input/santander-customer-transaction-prediction/test.csv")


train.head()


test.head()


train.shape, test.shape


train.isnull().sum()


test.isnull().sum()


X_train = train.iloc[:, 2:].values
y_train = train.target.values


X_train


y_train


X_test = test.iloc[:, 1:].values


X, y = train.iloc[:,2:], train.iloc[:,1]


X_test


train.info()


train.count().tail()


train.describe()


test.describe()


train.nunique()


test.nunique()


def isnull_data(dataset):
    nulls = dataset.isnull().sum() 
    tot = pd.concat([nulls], axis=1, keys=['Nulls']) 
    types = []
    for columns in dataset.columns:
        dtype = str(dataset[columns].dtype)
        types.append(dtype)
    tot['Types'] = types
    return(np.transpose(tot))


isnull_data(train)


isnull_data(test)


sns.countplot(train.target)


col = train.columns.values[2:202]





thinkplot.Scatter(train[col[0]],test[col[0]], alpha=1)
thinkplot.Config(legend=False)


thinkplot.Scatter(train[col[1]],test[col[1]], alpha=1)
thinkplot.Config(legend=False)


thinkplot.Scatter(train[col[2]],test[col[2]], alpha=1)
thinkplot.Config(legend=False)


thinkplot.Scatter(train[col[3]],test[col[3]], alpha=1)
thinkplot.Config(legend=False)


plt.figure(figsize=(16,6))
plt.title("Distribution of mean values per row in the train and test set")
sns.distplot(train[col].mean(axis=0),color="green", kde=True,bins=120, label='train')
sns.distplot(test[col].mean(axis=0),color="blue", kde=True,bins=120, label='test')
plt.legend()
plt.show()


plt.figure(figsize=(16,6))
plt.title("Distribution of min values per column in the train and test set")
sns.distplot(train[col].min(axis=0),color="blue", kde=True,bins=120, label='train')
sns.distplot(test[col].min(axis=0),color="red", kde=True,bins=120, label='test')
plt.legend()
plt.show()


targetEqZero = train.loc[train['target'] == 0][col]
targetEqOne = train.loc[train['target'] == 1][col]


plt.figure(figsize=(16,6))
plt.title("Distribution of max values per column in the train and test set")
sns.distplot(train[col].max(axis=0),color="blue", kde=True,bins=120, label='train')
sns.distplot(test[col].max(axis=0),color="red", kde=True,bins=120, label='test')
plt.legend()
plt.show()


plt.figure(figsize=(16,6))
plt.title("Distribution of min values per row in the train set")
sns.distplot(targetEqZero.min(axis=1),color="orange", kde=True,bins=120, label='target = 0')
sns.distplot(targetEqOne.min(axis=1),color="darkblue", kde=True,bins=120, label='target = 1')
plt.legend(); plt.show()


plt.figure(figsize=(16,6))
plt.title("Distribution of max values per row in the train set")
sns.distplot(targetEqZero.max(axis=1),color="orange", kde=True,bins=120, label='target = 0')
sns.distplot(targetEqOne.max(axis=1),color="darkblue", kde=True,bins=120, label='target = 1')
plt.legend(); plt.show()


unique_max_train = []
unique_max_test = []
for col in col:
    values = train[col].value_counts()
    unique_max_train.append([col, values.max(), values.idxmax()])
    values = test[col].value_counts()
    unique_max_test.append([col, values.max(), values.idxmax()])


np.transpose((pd.DataFrame(unique_max_train, columns=['Column', 'Max duplicates', 'Value'])).sort_values(by = 'Max duplicates', ascending=False).head(10))


features = train.columns.values[2:202]


train_standart = pd.DataFrame()
test_standart = pd.DataFrame()


idx = train.columns.values[2:202]

for df in [train]:
    train_standart['target'] = df.target
    train_standart['sum'] = df[idx].sum(axis=1)  
    train_standart['min'] = df[idx].min(axis=1)
    train_standart['max'] = df[idx].max(axis=1)
    train_standart['mean'] = df[idx].mean(axis=1)
    train_standart['std'] = df[idx].std(axis=1)
 


train_standart.head


idx = train.columns.values[2:202]
for df in [test]:
    test_standart['sum'] = df[idx].sum(axis=1)  
    test_standart['min'] = df[idx].min(axis=1)
    test_standart['max'] = df[idx].max(axis=1)
    test_standart['mean'] = df[idx].mean(axis=1)
    test_standart['std'] = df[idx].std(axis=1)
 


test_standart.head


t0 = train_standart.loc[train_standart['target'] == 0]
t1 = train_standart.loc[train_standart['target'] == 1]


pdf_min_t0 = thinkstats2.EstimatedPdf(t0['min'])
pdf_min_t1 = thinkstats2.EstimatedPdf(t1['min'])
thinkplot.Pdf(pdf_min_t0, label='target 0')
thinkplot.Pdf(pdf_min_t1, label='target 1')
thinkplot.Config(xlabel='Min', ylabel='PDF')


pdf_sum_t0 = thinkstats2.EstimatedPdf(t0['sum'])
pdf_sum_t1 = thinkstats2.EstimatedPdf(t1['sum'])
thinkplot.Pdf(pdf_sum_t0, label='target 0')
thinkplot.Pdf(pdf_sum_t1, label='target 1')
thinkplot.Config(xlabel='Sum', ylabel='PDF')


pdf_max_t0 = thinkstats2.EstimatedPdf(t0['max'])
pdf_max_t1 = thinkstats2.EstimatedPdf(t1['max'])
thinkplot.Pdf(pdf_max_t0, label='target 0')
thinkplot.Pdf(pdf_max_t1, label='target 1')
thinkplot.Config(xlabel='Max', ylabel='PDF')


pdf_mean_t0 = thinkstats2.EstimatedPdf(t0['mean'])
pdf_mean_t1 = thinkstats2.EstimatedPdf(t1['mean'])
thinkplot.Pdf(pdf_mean_t0, label='target 0')
thinkplot.Pdf(pdf_mean_t1, label='target 1')
thinkplot.Config(xlabel='Mean', ylabel='PDF')


pdf_std_t0 = thinkstats2.EstimatedPdf(t0['std'])
pdf_std_t1 = thinkstats2.EstimatedPdf(t1['std'])
thinkplot.Pdf(pdf_std_t0, label='target 0')
thinkplot.Pdf(pdf_std_t1, label='target 1')
thinkplot.Config(xlabel='Std', ylabel='PDF')


correlation = train.corr(method='pearson', min_periods=1).abs().unstack().sort_values(kind="quicksort").reset_index()
correlation.head(10)


sns.FacetGrid(train, hue="target", size=5) \
   .map(plt.scatter, "var_1", "var_2") \
   .add_legend()


def Cov(xs, ys, meanx=None, meany=None):
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    if meanx is None:
        meanx = np.mean(xs)
    if meany is None:
        meany = np.mean(ys)
        
    cov = np.dot(xs-meanx, ys-meany) / len(xs)
    return cov


def Cov(xs, ys, meanx=None, meany=None):
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    if meanx is None:
        meanx = np.mean(xs)
    if meany is None:
        meany = np.mean(ys)
        
    cov = np.dot(xs-meanx, ys-meany) / len(xs)
    return cov


def Corr(xs, ys):
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    meanx, varx = thinkstats2.MeanVar(xs)
    meany, vary = thinkstats2.MeanVar(ys)

    corr = Cov(xs, ys, meanx, meany) / np.sqrt(varx * vary)
    return corr


Corr(train['var_0'], train['var_1'])


np.corrcoef(train['var_0'], train['var_1'])


def plot_feature_importances(model, columns):
    nr_f = columns.shape[0]
    imp = pd.Series(data = model.feature_importances_, 
                    index=columns).sort_values(ascending=False)
    plt.figure(figsize=(7,5))
    plt.title("Feature importance")
    ax = sns.barplot(y=imp.index[:nr_f], x=imp.values[:nr_f], orient='h')


from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier



import pickle


xgb_s = XGBClassifier()


%%time
xgb_s.fit(X_train,y_train)


preds_xgb = xgb.predict(X_test)


xgb_s_y_preds = xgb_s.predict(X_test)


pickle.dump(xgb_s, open("pima.xgb_s.dat", "wb"))


loaded_model = pickle.load(open("pima.xgb_s.dat", "rb"))


plot_feature_importances(loaded_model, train.drop('Target', axis=1).columns)


pickle.dump(xgb, open("pima.pickle.dat", "wb"))


sub_df = pd.DataFrame({"ID_code": test["ID_code"],
                       "target": xgb_s_y_preds
                      })
sub_df.to_csv("submission_xgb.csv", index=False)


sub_xbs = pd.read_csv("submission_xgb.csv")


sns.countplot(sub_xbs.target)


from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


%%time
clf = tree.DecisionTreeClassifier(max_depth=10)
clf = clf.fit(X_train, y_train)


clf_pred = clf.predict(X_test)


dtc_df = pd.DataFrame({"ID_code": test["ID_code"],
                       "target": clf_pred
                      })
dtc_df.to_csv("submission_dtc.csv", index=False)


dtc_data = pd.read_csv("submission_dtc.csv")


sns.countplot(dtc_data.target)
    


from sklearn.neighbors import KNeighborsClassifier


knn = KNeighborsClassifier()


%%time
knn.fit(X_train,y_train)


knn_preds = knn.predict(X_test)


knn_df = pd.DataFrame({"ID_code": test["ID_code"],
                       "target": knn_preds
                      })
knn_df.to_csv("submission_knn.csv", index=False)


knn_data = pd.read_csv("submission_knn.csv")


sns.countplot(knn_data.target)


from sklearn.linear_model import LogisticRegression


log_reg_cls = LogisticRegression()


%%time
log_reg_cls.fit(X_train, y_train)


y_preds_log_reg = log_reg_cls.predict(X_test)


log_df = pd.DataFrame({"ID_code": test["ID_code"],
                       "target": y_preds_log_reg
                      })
log_df.to_csv("submission_log.csv", index=False)


log_data = pd.read_csv("submission_log.csv")


sns.countplot(log_data.target)

