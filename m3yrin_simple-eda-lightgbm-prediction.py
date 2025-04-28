# # Simple EDA + lightGBM Prediction for Santander Customer Transaction Prediction
# Auther : @m3yrin  
# Comments are mainly written in Japanese, Sorry.


# ## 0. コンペティション概要
# Santander Customer Transaction Prediction  
# https://www.kaggle.com/c/santander-customer-transaction-prediction  
# 
# **タスク**    
# サンタンデールという米国の銀行が開催するコンペ。  
# 与えられたデータをもとに、そのユーザが将来に特定の取引を行うかを予測する。
# 
# **ルール**  
# 評価方法はAUC(ROCカーブの面積), サブミットは一日に3回まで。


# ## 1. ライブラリのインポート


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings('ignore')

import os
print(os.listdir("../input"))


# データは 'test.csv', 'train.csv', 'sample_submission.csv' の3種類。


# ## 2. データをPandasで取得


train = pd.read_csv("../input/train.csv")
test = pd.read_csv('../input/test.csv')
sample_submission = pd.read_csv('../input/sample_submission.csv')


train.head(10)


# ## 3. データの外観を確認(EDA)


#すべて分析をかけると重いのでデータ数を絞る
max_row = 50000

# 0 ~ 50000行までのデータをtrain_pdpに詰める
train_pdp = train.iloc[:max_row, :]


import pandas_profiling as pdp
pdp.ProfileReport(train_pdp)


# **雑な外観**
# * 欠損値はなく補完は不要
# * target == 0 と target == 1 のデータ数がざっくり10倍違うので、不均衡なデータ。
# * パラメータの名前と値は元情報がわからないようになっているので、ドメインの知識を使った分析は難しい


# ## 4. lightGBMによる学習と予測


from sklearn.model_selection import train_test_split
from sklearn import metrics
import lightgbm as lgb


# 学習データと正解データを分ける


train_y = train['target']
train_x = train.drop(['ID_code','target'], axis=1)
test_x = test.drop('ID_code', axis=1)


# train_test_splitで評価データを分ける


train_X, eval_X, train_Y, eval_Y = train_test_split(train_x, train_y, test_size=0.1, random_state=42)


# lgb.Datasetでlightgbmで使いやすいデータ形式にパース


lgb_train = lgb.Dataset(train_X, train_Y)
lgb_eval = lgb.Dataset(eval_X, eval_Y, reference=lgb_train)


# 学習パラメータを定義  
# * objective : 今回は二値分類
# * is_unbalance : 不均衡データなので指定
# * metric : 評価はAUCなのでaucを指定  
# 
# 最適化してないのでパラメータは探索する必要あり。
# 少なくとも学習率はもっと小さくていい。0.01くらい。


params = {
    'objective': 'binary',
    'max_depth': 16,
    'learning_rate': 0.1,
    'is_unbalance': True,
    'random_state': 42,
    'metric': 'auc',
    'num_threads': 4}


# 学習


gbm = lgb.train(params, 
                lgb_train,
                valid_sets=lgb_eval,
                num_boost_round=2000,
                verbose_eval=100,
                early_stopping_rounds = 100
               )


# 学習済みのモデルを使ってtest.csvのデータを予測


y_pred = gbm.predict(test_x, num_iteration=gbm.best_iteration)   


# 予測結果をsample_submission.csvと同じ形式に整形


submission_lgbm = pd.DataFrame({
        "ID_code": test["ID_code"],
        "target": y_pred
    })
submission_lgbm.to_csv('submission_lgbm.csv', index=False)


print("#All  : " + str(len(submission_lgbm)))
print("#True : " + str(len(submission_lgbm[submission_lgbm['target'] > 0.5])))

