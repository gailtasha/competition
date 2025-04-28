import os
print(os.listdir("../input"))
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')


features = [c for c in train.columns if c not in ['ID_code', 'target']]


y=train['target']
X = train.drop(['target', 'ID_code'], axis=1)


ID_code=test['ID_code']
X_test = test.drop(['ID_code'],axis = 1)


n_splits = 5 # Number of K-fold Splits
splits = list(StratifiedKFold(n_splits=n_splits, shuffle=False).split(X, y))


import xgboost as xgb


params ={
    booster = "gbtree",
    objective = "binary:logistic",
    eta=0.02,
               #gamma=80,
               max_depth=2,
               min_child_weight=1, 
               subsample=0.5,
               colsample_bytree=0.1,
               scale_pos_weight = round(sum(!y) / sum(y), 2))


y.value_counts()


xgb_param = {
    'booster':"gbtree",
    'min_child_weight': 1,
    'objective': 'binary:logistic',
    'max_depth': 2,
    'subsample':0.5,
    #'max_delta_step': 1.8,
    'colsample_bytree': 0.1,
    'eta': 0.02,
    #'gamma': 0.65,
    'eval_metric':'auc',
    'scale_pos_weight': 0.1004
        }


oof = np.zeros(len(X))
predictions = np.zeros(len(X_test))
feature_importance_df = pd.DataFrame()

for i, (train_idx, valid_idx) in enumerate(splits):  
    print(f'Fold {i + 1}')
    x_train = np.array(X)
    y_train = np.array(y)
    trn_data = xgb.DMatrix(x_train[train_idx.astype(int)], label=y_train[train_idx.astype(int)])
    val_data = xgb.DMatrix(x_train[valid_idx.astype(int)], label=y_train[valid_idx.astype(int)])
    watchlist = [(trn_data, 'train'), (val_data, 'valid')]
                 
    #num_round = 15000
    clf=xgb.train(xgb_param, trn_data, 30000, evals=watchlist, early_stopping_rounds=500,verbose_eval=1000)
    
    xgb_valid=xgb.DMatrix(x_train[valid_idx])
    oof[valid_idx] = clf.predict(xgb_valid, ntree_limit=clf.best_ntree_limit)
    
   
    xgb_test=xgb.DMatrix(X_test)
    predictions+=clf.predict(xgb_test,ntree_limit=clf.best_ntree_limit)/5
    #predictions += clf.predict(X_test, num_iteration=clf.best_iteration) / 5

print("CV score: {:<8.5f}".format(roc_auc_score(y, oof)))


xgb_test=xgb.DMatrix(X_test.values)
predictions=clf.predict(xgb_test,ntree_limit=clf.best_ntree_limit)


submission=pd.DataFrame()
submission['ID_code']=ID_code
submission['target']=predictions
submission.to_csv('sub_xgb_v1.csv', index=False)


for i, (train_idx, valid_idx) in enumerate(splits): 
    xgb_valid=xgb.DMatrix(x_train[valid_idx])
    oof[valid_idx] = clf.predict(xgb_valid, ntree_limit=clf.best_ntree_limit)


print("CV score: {:<8.5f}".format(roc_auc_score(y, oof)))


params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        'learning_rate':[0.01,0.02,0.03]
        }


from sklearn.model_selection import RandomizedSearchCV
xgb_clf = XGBClassifier(n_estimators=600, objective='binary:logistic',
                    silent=True, nthread=1)


splits = StratifiedKFold(n_splits=3, shuffle = False)

random_search = RandomizedSearchCV(xgb_clf, param_distributions=params, n_iter=100, scoring='roc_auc', 
                                   n_jobs=-1, cv=splits.split(X,Y), verbose=3)

random_search.fit(X, Y)

