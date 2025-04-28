# Basic packages
import pandas as pd
import numpy as np
import warnings
import time
import glob
import sys
import os
import gc

# ML packages
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from IPython.display import display
import matplotlib.patches as patch
import matplotlib.pyplot as plt
from sklearn.svm import NuSVR
from scipy.stats import norm
import lightgbm as lgb
import eli5
from eli5.sklearn import PermutationImportance
from scipy.stats import kurtosis, skew

from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

# visualization packages
import seaborn as sns
import matplotlib.pyplot as plt

# execution progress bar
from tqdm import tqdm_notebook, tnrange
from tqdm.auto import tqdm
tqdm.pandas()


# System Setup
%matplotlib inline
%precision 4
warnings.filterwarnings('ignore')
plt.style.use('ggplot')
np.set_printoptions(suppress=True)
pd.set_option("display.precision", 15)


# ## Load Data


print(os.listdir("../input/"))


# import Dataset to play with it
train= pd.read_csv("../input/train.csv")
test = pd.read_csv('../input/test.csv')
sample_submission = pd.read_csv('../input/sample_submission.csv')


train.shape, test.shape, sample_submission.shape


train.head(5)


# ##   Data Exploration


train.columns


print(len(train.columns))


print(train.info())


train.describe()


# distribution of targets
colors = ['darkseagreen','lightcoral']
plt.figure(figsize=(6,6))
plt.pie(train["target"].value_counts(), explode=(0, 0.25), labels= ["0", "1"], startangle=45, autopct='%1.1f%%', colors=colors)
plt.axis('equal')
plt.show()


# correlation with target
labels = []
values = []

for col in train.columns:
    if col not in ['ID_code', 'target']:
        labels.append(col)
        values.append(spearmanr(train[col].values, train['target'].values)[0])

corr_df = pd.DataFrame({'col_labels': labels, 'corr_values' : values})
corr_df = corr_df.sort_values(by='corr_values')

corr_df = corr_df[(corr_df['corr_values']>0.03) | (corr_df['corr_values']<-0.03)]

ind = np.arange(corr_df.shape[0])
width = 0.9
fig, ax = plt.subplots(figsize=(12,12))
rects = ax.barh(ind, np.array(corr_df.corr_values.values), color='darkseagreen')
ax.set_yticks(ind)
ax.set_yticklabels(corr_df.col_labels.values, rotation='horizontal')
ax.set_xlabel("Correlation coefficient")
ax.set_title("Variable correlation to Target")
plt.show()


# check covariance among importance variables
cols_to_use = corr_df[(corr_df['corr_values']>0.05) | (corr_df['corr_values']<-0.05)].col_labels.tolist()

temp_df = train[cols_to_use]
corrmat = temp_df.corr(method='spearman')
f, ax = plt.subplots(figsize=(18, 18))

#Draw the heatmap using seaborn
sns.heatmap(corrmat, vmax=1., square=True, cmap="Blues", annot=True)
plt.title("Important variables correlation map", fontsize=15)
plt.show()


# ## Data Preprocessing


# Check missing data for test & train
def check_missing_data(df):
    flag=df.isna().sum().any()
    if flag==True:
        total = df.isnull().sum()
        percent = (df.isnull().sum())/(df.isnull().count()*100)
        output = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        data_type = []
        # written by MJ Bahmani
        for col in df.columns:
            dtype = str(df[col].dtype)
            data_type.append(dtype)
        output['Types'] = data_type
        return(np.transpose(output))
    else:
        return(False)


print('missing in train: ',check_missing_data(train))
print('missing in test: ',check_missing_data(test))


train.head()


# ## Variable Engineering


# #### PCA


'''pca_df = preprocessing.normalize(train.drop(['ID_code','target'],axis=1))
pca_test_df = preprocessing.normalize(train.drop(['ID_code'],axis=1))

def _get_number_components(model, threshold):
    component_variance = model.explained_variance_ratio_
    explained_variance = 0.0
    components = 0

    for var in component_variance:
        explained_variance += var
        components += 1
        if(explained_variance >= threshold):
            break
    return components

### Get the optimal number of components
pca = PCA()
train_pca = pca.fit_transform(pca_df)
test_pca = pca.fit_transform(pca_test_df)
components = _get_number_components(pca, threshold=0.9)
components'''


# Implement PCA 
#obj_pca = model = PCA(n_components = components)
#X_pca = obj_pca.fit_transform(pca_df)
#X_t_pca = obj_pca.fit_transform(pca_test_df)


'''# add the decomposed features in the train dataset
def _add_decomposition(df, decomp, ncomp, flag):
    for i in range(1, ncomp+1):
        df[flag+"_"+str(i)] = decomp[:, i - 1]'''


'''pca_train = train[['ID_code','target']]
pca_test = test[['ID_code']]

_add_decomposition(pca_train, X_pca, 90, 'pca')
_add_decomposition(pca_test, X_t_pca, 90, 'pca')'''


#pca_train.head()


#pca_test.head()


# #### Summary Stats


'''#train_df.reset_index(drop=True, inplace=True)
train_stats_var = pd.concat([train[['ID_code', 'target']],
                             train.sum(axis=1),
                             train.mean(axis=1),
                             train.min(axis=1),
                             train.max(axis=1),
                             train.median(axis=1),
                             train.var(axis=1),
                             train.skew(axis=1),
                             train.apply(kurtosis, axis=1)
                            ], axis=1)

train_stats_var.columns = ['ID_code','target', 'txn_ttl','avg_txn', 'min_txn', 'max_txn', 'med_txn','var_txn','skew', 'kurt']

train_stats_var.loc[train_stats_var['var_txn'].isnull(), 'var_txn'] = 0
train_stats_var.loc[train_stats_var['skew'].isnull(), 'skew'] = 0
train_stats_var.loc[train_stats_var['kurt'].isnull(), 'kurt'] = 0

train_stats_var['std_txn'] = train_stats_var['var_txn']**(.5)
train_stats_var.loc[train_stats_var['std_txn'].isnull(), 'std_txn'] = 0

train_stats_var.head()'''


# ## Feature importance


cols=["target","ID_code"]
X = train.drop(cols,axis=1)
y = train["target"]

#cols=["target","ID_code"]
#X = pca_train.drop(cols,axis=1)
#y = pca_train["target"]


X_test  = test.drop("ID_code",axis=1)
#X_test  = pca_test.drop("ID_code",axis=1)


train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)


# rfc_model = RandomForestClassifier(random_state=0).fit(train_X, train_y)


# ### Permutation Importance


'''perm = PermutationImportance(rfc_model, random_state=1).fit(val_X, val_y)
eli5.show_weights(perm, feature_names = val_X.columns.tolist())'''


# features = [c for c in train.columns if c not in ['ID_code', 'target']]


#  ## Model Development


# for get better result chage fold_n to 5
fold_n=5
folds = StratifiedKFold(n_splits=fold_n, shuffle=True, random_state=10)


# ### lightgbm


# https://www.kaggle.com/dromosys/sctp-working-lgb
params = {'num_leaves': 9,
         'min_data_in_leaf': 42,
         'objective': 'binary',
         'max_depth': 12,
         'learning_rate': 0.05,
         'boosting': 'gbdt',
         'bagging_freq': 5,
         'bagging_fraction': 0.8,
         'feature_fraction': 0.8201,
         'bagging_seed': 11,
         'reg_alpha': 5,
         'reg_lambda': 5,
         'random_state': 42,
         'metric': 'auc',
         'verbosity': -1,
         'subsample': 0.81,
         'min_gain_to_split': 0.01077313523861969,
         'min_child_weight': 19.428902804238373,
         'num_threads': 4}


%%time
y_pred_lgb = np.zeros(len(X_test))
for fold_n, (train_index, valid_index) in enumerate(folds.split(X,y)):
    print('Fold', fold_n, 'started at', time.ctime())
    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
    
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid)
        
    lgb_model = lgb.train(params,train_data,num_boost_round=2000,
                    valid_sets = [train_data, valid_data],verbose_eval=300,early_stopping_rounds = 200)
            
    y_pred_lgb += lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)/5


# ### Neural Net


train_features = train.drop(['target','ID_code'], axis = 1)
test_features = test.drop(['ID_code'],axis = 1)
train_target = train['target']

sc = StandardScaler()
train_features = sc.fit_transform(train_features)
test_features = sc.transform(test_features)

n_splits = 5 # Number of K-fold Splits
splits = list(StratifiedKFold(n_splits=n_splits, shuffle=True).split(train_features, train_target))


class CyclicLR(object):
    def __init__(self, optimizer, base_lr=1e-3, max_lr=6e-3,
                 step_size=2000, mode='triangular', gamma=1.,
                 scale_fn=None, scale_mode='cycle', last_batch_iteration=-1):

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(base_lr, list) or isinstance(base_lr, tuple):
            if len(base_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} base_lr, got {}".format(
                    len(optimizer.param_groups), len(base_lr)))
            self.base_lrs = list(base_lr)
        else:
            self.base_lrs = [base_lr] * len(optimizer.param_groups)

        if isinstance(max_lr, list) or isinstance(max_lr, tuple):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} max_lr, got {}".format(
                    len(optimizer.param_groups), len(max_lr)))
            self.max_lrs = list(max_lr)
        else:
            self.max_lrs = [max_lr] * len(optimizer.param_groups)

        self.step_size = step_size

        if mode not in ['triangular', 'triangular2', 'exp_range'] \
                and scale_fn is None:
            raise ValueError('mode is invalid and scale_fn is None')

        self.mode = mode
        self.gamma = gamma

        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = self._triangular_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = self._triangular2_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = self._exp_range_scale_fn
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self.batch_step(last_batch_iteration + 1)
        self.last_batch_iteration = last_batch_iteration

    def batch_step(self, batch_iteration=None):
        if batch_iteration is None:
            batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = batch_iteration
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def _triangular_scale_fn(self, x):
        return 1.

    def _triangular2_scale_fn(self, x):
        return 1 / (2. ** (x - 1))

    def _exp_range_scale_fn(self, x):
        return self.gamma**(x)

    def get_lr(self):
        step_size = float(self.step_size)
        cycle = np.floor(1 + self.last_batch_iteration / (2 * step_size))
        x = np.abs(self.last_batch_iteration / step_size - 2 * cycle + 1)

        lrs = []
        param_lrs = zip(self.optimizer.param_groups, self.base_lrs, self.max_lrs)
        for param_group, base_lr, max_lr in param_lrs:
            base_height = (max_lr - base_lr) * np.maximum(0, (1 - x))
            if self.scale_mode == 'cycle':
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(self.last_batch_iteration)
            lrs.append(lr)
        return lrs


class Simple_NN(nn.Module):
    def __init__(self ,input_dim ,hidden_dim, dropout = 0.2):
        super(Simple_NN, self).__init__()
        
        self.inpt_dim = input_dim
        self.hidden_dim = hidden_dim
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, int(hidden_dim/2))
        self.fc3 = nn.Linear(int(hidden_dim/2), int(hidden_dim/4))
        self.fc4 = nn.Linear(int(hidden_dim/4), int(hidden_dim/8))
        self.fc5 = nn.Linear(int(hidden_dim/8), 1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(int(hidden_dim/2))
        self.bn3 = nn.BatchNorm1d(int(hidden_dim/4))
        self.bn4 = nn.BatchNorm1d(int(hidden_dim/8))
    
    def forward(self, x):
        y = self.fc1(x)
        y = self.relu(y)
        #y = self.bn1(y)
        y = self.dropout(y)
        
        y = self.fc2(y)
        y = self.relu(y)
        #y = self.bn2(y)
        y = self.dropout(y)
        
        y = self.fc3(y)
        y = self.relu(y)
        #y = self.bn3(y)
        y = self.dropout(y)
        
        y = self.fc4(y)
        y = self.relu(y)
        #y = self.bn4(y)
        y = self.dropout(y)
        
        out= self.fc5(y)
        
        return out


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


model = Simple_NN(200,512)
model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0002) # Using Adam optimizer


from torch.optim.optimizer import Optimizer
n_epochs = 40
batch_size = 25000

train_preds = np.zeros((len(train_features)))
test_preds = np.zeros((len(test_features)))

x_test = np.array(test_features)
x_test_cuda = torch.tensor(x_test, dtype=torch.float).cuda()
test = torch.utils.data.TensorDataset(x_test_cuda)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

avg_losses_f = []
avg_val_losses_f = []

for i, (train_idx, valid_idx) in enumerate(splits):  
    x_train = np.array(train_features)
    y_train = np.array(train_target)
    
    x_train_fold = torch.tensor(x_train[train_idx.astype(int)], dtype=torch.float).cuda()
    y_train_fold = torch.tensor(y_train[train_idx.astype(int), np.newaxis], dtype=torch.float32).cuda()
    
    x_val_fold = torch.tensor(x_train[valid_idx.astype(int)], dtype=torch.float).cuda()
    y_val_fold = torch.tensor(y_train[valid_idx.astype(int), np.newaxis], dtype=torch.float32).cuda()
    
    loss_fn = torch.nn.BCEWithLogitsLoss()
    
    step_size = 300
    base_lr, max_lr = 0.0001, 0.001  
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                             lr=max_lr)
    
    ################################################################################################
    scheduler = CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr,
               step_size=step_size, mode='exp_range',
               gamma=0.99994)
    ###############################################################################################

    train = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)
    valid = torch.utils.data.TensorDataset(x_val_fold, y_val_fold)
    
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)
    
    print(f'Fold {i + 1}')
    for epoch in range(n_epochs):
        start_time = time.time()
        model.train()
        avg_loss = 0.
        #avg_auc = 0.
        for i, (x_batch, y_batch) in enumerate(train_loader):
            y_pred = model(x_batch)
            #########################
            if scheduler:
                #print('cycle_LR')
                scheduler.batch_step()
            ########################
            loss = loss_fn(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            avg_loss += loss.item()/len(train_loader)
            #avg_auc += round(roc_auc_score(y_batch.cpu(),y_pred.detach().cpu()),4) / len(train_loader)
        model.eval()
        
        valid_preds_fold = np.zeros((x_val_fold.size(0)))
        test_preds_fold = np.zeros((len(test_features)))
        
        avg_val_loss = 0.
        #avg_val_auc = 0.
        for i, (x_batch, y_batch) in enumerate(valid_loader):
            y_pred = model(x_batch).detach()
            
            #avg_val_auc += round(roc_auc_score(y_batch.cpu(),sigmoid(y_pred.cpu().numpy())[:, 0]),4) / len(valid_loader)
            avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
            valid_preds_fold[i * batch_size:(i+1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]
            
        elapsed_time = time.time() - start_time 
        print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(
            epoch + 1, n_epochs, avg_loss, avg_val_loss, elapsed_time))
        
    avg_losses_f.append(avg_loss)
    avg_val_losses_f.append(avg_val_loss) 
    
    for i, (x_batch,) in enumerate(test_loader):
        y_pred = model(x_batch).detach()

        test_preds_fold[i * batch_size:(i+1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]
        
    train_preds[valid_idx] = valid_preds_fold
    test_preds += test_preds_fold / len(splits)

auc  =  round(roc_auc_score(train_target,train_preds),4)      
print('All \t loss={:.4f} \t val_loss={:.4f} \t auc={:.4f}'.format(np.average(avg_losses_f),np.average(avg_val_losses_f),auc))


esemble = 0.6*y_pred_lgb + 0.4* train_preds


print('NN auc = {:<8.5f}'.format(auc))
print('LightBGM auc = {:<8.5f}'.format(roc_auc_score(train_target, y_pred_lgb)))
print('NN+LightBGM auc = {:<8.5f}'.format(roc_auc_score(train_target, esemble)))


# ## Submission Files


submission_lgb = pd.DataFrame({
        "ID_code": test["ID_code"],
        "target": y_pred_lgb
    })
submission_lgb.to_csv('submission_lgb.csv', index=False)


submission_nn = pd.DataFrame({
        "ID_code": test["ID_code"],
        "target": test_preds
    })
submission_nn.to_csv('submission_nn.csv', index=False)


submission_ens = pd.DataFrame({
        "ID_code": test["ID_code"],
        "target": esemble
    })
submission_ens.to_csv('submission_ens.csv', index=False)


#  <a id="55"></a> <br>
# ## Stacking


'''submission_rfc_cat = pd.DataFrame({
        "ID_code": test["ID_code"],
        "target": (y_pred_rfc +y_pred_cat)/2
    })
submission_rfc_cat.to_csv('submission_rfc_cat.csv', index=False)'''


# # References & credits
# Thanks fo following kernels that help me to create this kernel.


# 1. [https://www.kaggle.com/dansbecker/permutation-importance](https://www.kaggle.com/dansbecker/permutation-importance)
# 1. [https://www.kaggle.com/dansbecker/partial-plots](https://www.kaggle.com/dansbecker/partial-plots)
# 1. [https://www.kaggle.com/miklgr500/catboost-with-gridsearch-cv](https://www.kaggle.com/miklgr500/catboost-with-gridsearch-cv)
# 1. [https://www.kaggle.com/dromosys/sctp-working-lgb](https://www.kaggle.com/dromosys/sctp-working-lgb)
# 1. [https://www.kaggle.com/gpreda/santander-eda-and-prediction](https://www.kaggle.com/gpreda/santander-eda-and-prediction)
# 1. [permutation-importance](https://www.kaggle.com/dansbecker/permutation-importance)
# 1. [partial-plots](https://www.kaggle.com/dansbecker/partial-plots)
# 1. [https://www.kaggle.com/dansbecker/shap-values](https://www.kaggle.com/dansbecker/shap-values)
# 1. [algorithm-choice](https://docs.microsoft.com/en-us/azure/machine-learning/studio/algorithm-choice)


# Go to first step: [Course Home Page](https://www.kaggle.com/mjbahmani/10-steps-to-become-a-data-scientist)
# 
# Go to next step : [Titanic](https://www.kaggle.com/mjbahmani/a-comprehensive-ml-workflow-with-python)


# # Not Completed yet!!!

