# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # 数据概览


import pandas as pd

data = pd.read_csv('../input/train.csv',index_col=False)
data.head(5)


# ## 数据总量和特征数量


print('共有数据: %d 条'%len(data.index))
print('特征总数为: %d 个'%(len(data.columns)-2))


# ## 正负类别比例


from collections import Counter

Counter(data['target'])


print('负类和正类的比例: {}'.format(float(179902)/20098))


# 负类的数量是正类的9倍，也就是正类占总量的只有10%左右


# ## 查看是否有缺失值


isnull = data.isnull().any()
isnull.values


# ## 将类别和特征分离


labels = data['target'].values
data = data.drop(['target','ID_code'],axis=1)
data.loc[:10]
len(labels),len(data)


# # 标准化: 对特征进行标准化


from sklearn import preprocessing

scaler = preprocessing.StandardScaler().fit(data)
scaler # 缩放器，便于对test.csv中的数据做同样的标准化变换


data = scaler.transform(data)
data[:10]


# # 不平衡数据处理: 对正类过采样


from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from collections import Counter

#sampler = RandomOverSampler(random_state=0) # 默认重复采样
sampler = SMOTE()
X_resampled,Y_resampled = sampler.fit_sample(data,labels)
print(X_resampled.shape,Y_resampled.shape)
Counter(Y_resampled)


# # 分割数据集:训练集和测试集


from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X_resampled,Y_resampled,test_size=0.2)
len(Y_train),len(Y_test)


# # 选择算法:BayesianRidge


from sklearn.linear_model import BayesianRidge


# ## 训练分类器


model = BayesianRidge()
model.fit(X_train,Y_train)


# ## 在测试集上的结果


from sklearn.metrics import roc_auc_score

result = model.predict(X_test,return_std=False)
roc_score = roc_auc_score(Y_test,result)
print('在测试集上的ROC: %.3f'%roc_score)


# # 预测


import pandas as pd

test_data = pd.read_csv('../input/test.csv',index_col=0)
test_data[:10]


# ## 保存客户编码索引


test_code = test_data.index
test_code


# ## 标准化:缩放比例要与训练数据集相同


test_data = scaler.transform(test_data)
test_data[:10]


# ## 预测结果


result = model.predict(test_data,return_std=False)


# ## 提交结果


sub = pd.DataFrame(data={'ID_code':test_code,'target':result})
sub.to_csv('sample_submission.csv',index=False)



