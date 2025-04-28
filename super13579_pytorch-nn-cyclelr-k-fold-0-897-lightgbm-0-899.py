# ## Pytorch to implement simple feed-forward NN model (0.89+)
# 
# * As below discussion, NN model can get lB 0.89+
# * https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/82499#latest-483679
# * Add Cycling learning rate , K-fold cross validation (0.85 to 0.86)
# * Add flatten layer as below discussion (0.86 to 0.897)
# * https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/82863
# 
# ## LightGBM (LB 0.899)
# 
# * Fine tune parameters (0.898 to 0.899)
# * Reference this kernel : https://www.kaggle.com/chocozzz/santander-lightgbm-baseline-lb-0-899
# 
# 
# ## Plan to do
# * Modify model structure on NN model
# * Focal loss
# * Feature engineering
# * Tune parameters oof LightGBM


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import lightgbm as lgb
from sklearn.metrics import mean_squared_error

import os
print(os.listdir("../input"))


# ## Load Data


#Load data
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


train_df.shape, test_df.shape


train_df.head()


train_features = train_df.drop(['target','ID_code'], axis = 1)
test_features = test_df.drop(['ID_code'],axis = 1)
train_target = train_df['target']


train_features.shape,test_features.shape,train_target.shape


#### Scaling feature #####
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train_features = sc.fit_transform(train_features)
test_features = sc.transform(test_features)


# ## Split K- fold validation


# Implement K-fold validation to improve results
n_splits = 5 # Number of K-fold Splits

splits = list(StratifiedKFold(n_splits=n_splits, shuffle=True).split(train_features, train_target))
splits[:3]


# ## Cycling learning rate
# 
# *copy from ==> https://github.com/anandsaha/pytorch.cyclic.learning.rate/blob/master/cls.py


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


# ## Build Simple NN model (Pytorch)
# 
# * add flatten layer before fc layer (improve to 0.89+)
# * https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/82863
# 
# * Model structure
# * (batch_size, 200) ==> Flatten ==> (batch_size* 200,1) ==> fc1 ==> (batch_size* 200, hidden_layer) ==>Reshape ==>(batch_size, hidden_layer * 200) ==> fc2 ==> (batch_size, 1)


class Simple_NN(nn.Module):
    def __init__(self ,input_dim ,hidden_dim, dropout = 0.75):
        super(Simple_NN, self).__init__()
        
        self.inpt_dim = input_dim
        self.hidden_dim = hidden_dim
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(int(hidden_dim*input_dim), 1)
        #self.fc3 = nn.Linear(int(hidden_dim/2*input_dim), int(hidden_dim/4))
        #self.fc4 = nn.Linear(int(hidden_dim/4*input_dim), int(hidden_dim/8))
        #self.fc5 = nn.Linear(int(hidden_dim/8*input_dim), 1)
        #self.bn1 = nn.BatchNorm1d(hidden_dim)
        #self.bn2 = nn.BatchNorm1d(int(hidden_dim/2))
        #self.bn3 = nn.BatchNorm1d(int(hidden_dim/4))
        #self.bn4 = nn.BatchNorm1d(int(hidden_dim/8))
    
    def forward(self, x):
        b_size = x.size(0)
        x = x.view(-1, 1)
        y = self.fc1(x)
        y = self.relu(y)
        y = y.view(b_size, -1)
        
        out= self.fc2(y)
        
        return out


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ## Start training
# * Epoch = 40
# * Batch size = 256
# * Cycling step = 150


from torch.optim.optimizer import Optimizer
## Hyperparameter
n_epochs = 40
batch_size = 256

## Build tensor data for torch
train_preds = np.zeros((len(train_features)))
test_preds = np.zeros((len(test_features)))

x_test = np.array(test_features)
x_test_cuda = torch.tensor(x_test, dtype=torch.float).cuda()
test = torch.utils.data.TensorDataset(x_test_cuda)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

avg_losses_f = []
avg_val_losses_f = []

## Start K-fold validation
for i, (train_idx, valid_idx) in enumerate(splits):  
    x_train = np.array(train_features)
    y_train = np.array(train_target)
    
    x_train_fold = torch.tensor(x_train[train_idx.astype(int)], dtype=torch.float).cuda()
    y_train_fold = torch.tensor(y_train[train_idx.astype(int), np.newaxis], dtype=torch.float32).cuda()
    
    x_val_fold = torch.tensor(x_train[valid_idx.astype(int)], dtype=torch.float).cuda()
    y_val_fold = torch.tensor(y_train[valid_idx.astype(int), np.newaxis], dtype=torch.float32).cuda()
    
    ##Loss function
    #loss_fn = FocalLoss(2)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    
    #Build model, initial weight and optimizer
    model = Simple_NN(200,16)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001,weight_decay=1e-5) # Using Adam optimizer
    
    
    ######################Cycling learning rate########################

    step_size = 2000
    base_lr, max_lr = 0.001, 0.005  
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                             lr=max_lr)
    
    scheduler = CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr,
               step_size=step_size, mode='exp_range',
               gamma=0.99994)

    ###################################################################

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
            ###################tuning learning rate###############
            if scheduler:
                #print('cycle_LR')
                scheduler.batch_step()

            ######################################################
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


# ## LightGBM Model
# * reference this kernel : https://www.kaggle.com/chocozzz/santander-lightgbm-baseline-lb-0-899 


## Use no scaling data to train LGBM
train_features = train_df.drop(['target','ID_code'], axis = 1)
test_features = test_df.drop(['ID_code'],axis = 1)
train_target = train_df['target']


#LGBM Paramater tuning
param = {
        'num_leaves': 7,
        'learning_rate': 0.01,
        'feature_fraction': 0.04,
        'max_depth': 17,
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'metric': 'auc',
    }


# ## LGBM training


oof = np.zeros(len(train_df))
predictions = np.zeros(len(test_df))
feature_importance_df = pd.DataFrame()
features = [c for c in train_df.columns if c not in ['ID_code', 'target']]

for i, (train_idx, valid_idx) in enumerate(splits):  
    print(f'Fold {i + 1}')
    x_train = np.array(train_features)
    y_train = np.array(train_target)
    trn_data = lgb.Dataset(x_train[train_idx.astype(int)], label=y_train[train_idx.astype(int)])
    val_data = lgb.Dataset(x_train[valid_idx.astype(int)], label=y_train[valid_idx.astype(int)])
    
    num_round = 15000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 100)
    oof[valid_idx] = clf.predict(x_train[valid_idx], num_iteration=clf.best_iteration)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = i + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    predictions += clf.predict(test_features, num_iteration=clf.best_iteration) / 5

print("CV score: {:<8.5f}".format(roc_auc_score(train_target, oof)))




# ## Ensemble two model (NN+ LGBM)
# * NN model accuracy is too low, ensemble looks don't work.


esemble = 0.6*oof + 0.4* train_preds
print('NN auc = {:<8.5f}'.format(auc))
print('LightBGM auc = {:<8.5f}'.format(roc_auc_score(train_target, oof)))
print('NN+LightBGM auc = {:<8.5f}'.format(roc_auc_score(train_target, esemble)))


test_preds.shape,predictions.shape


esemble_pred = 0.4* test_preds+ 0.6 *predictions


id_code_test = test_df['ID_code']


# ## Create submit file


my_submission_nn = pd.DataFrame({"ID_code" : id_code_test, "target" : test_preds})
my_submission_lbgm = pd.DataFrame({"ID_code" : id_code_test, "target" : predictions})
my_submission_esemble = pd.DataFrame({"ID_code" : id_code_test, "target" : esemble_pred})


my_submission_nn.to_csv('submission_nn.csv', index = False, header = True)
my_submission_lbgm.to_csv('submission_lbgm.csv', index = False, header = True)
my_submission_esemble.to_csv('submission_esemble.csv', index = False, header = True)



