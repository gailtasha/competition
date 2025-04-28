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


# # 数据探索


# ## 数据概览


import pandas as pd

data = pd.read_csv('../input/train.csv',index_col=False)
data.loc[:10]


# ## 数据总量和特征数量


print('共有数据: %d 条'%len(data.index))
print('特征总数为: %d 个'%(len(data.columns)-2))


# ## 正负类别比例


from collections import Counter

Counter(data['target'])


print('负类和正类的比例: {}'.format(float(179902)/20098))


# 负类的数量是正类的9倍，也就是正类占总量的只有10%左右


# ## 查看是否有缺失值:类别列和特征列


isnull = data.isnull().any()
isnull.values


# 显然不存在缺失值


# ## 将特征和类别分离


labels = data['target'].values
data = data.drop(['target','ID_code'],axis=1)
data.loc[:10]
len(labels),len(data)


# ## 检测特征列是否存在异常值


# ### 查看每个特征的统计信息


data.describe()


# 观察每个特征列(除去target)的统计信息，可以发现这些特征的均值和标准差彼此间相差很大，这提醒我们可以对每个特征列做标准化，以减弱绝对值和浮动大的特征对预测结果的扰动。


# ### 标准差最大的前三个特征


stds = data.describe().loc['std'] # 存储所有特征的标准差
result = [] # 存储标准差最大前三个特征
for i in range(3):
    index = stds.argmax()
    result.append(index)
    stds = stds.drop([index]) # 删除该最大值
print('标准差最大的前三个特征是: ',result)


# ### 用箱型图查看每个特征的异常值分布情况：选取方差最大的前三个特征查看


import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

p = sns.boxplot(data=data[result]) 


# 从上图可以看出，这三个特征都存在着不少异常值


# ### 统计各个特征的异常值个数


# 箱型图识别异常值的标准：
# 
# 异常值通常被定义为小于$Q_{L}-1.5IQR$或大于$Q_{U}+1.5IQR$的值。
# 
# $Q_{L}$称为下四分位数，表示全部观察值中有四分之一的数据取值比它小；
# 
# $Q_{U}$称为上四分位数，表示全部观察值中有四分之一的数据取值比它大；
# 
# $IQR$称为四分位数间距，是上四分位数$Q_{U}$与下四分位数$Q_{L}$之差，其间包含了全部观察值的一半。


def normalRange(q_l,q_u):
    '正常值的范围'
    iqr = q_u - q_l
    return q_l - 1.5*iqr,q_u + 1.5*iqr # 正常值的下限，正常值的上限


statis = data.describe()
outlier_values = [] # 存储每个特征的异常值
for column in data.columns:
    num_lower,num_upper = normalRange(statis.loc['25%'][column],statis.loc['75%'][column])
    temp = data[(data[column]<num_lower) | (data[column] > num_upper)] # 只保留异常值的行
    outlier_values.append(len(temp))
outlier = pd.DataFrame(index=['outlier'],columns=data.columns)
outlier.loc['outlier'] = outlier_values
print('每个特征的异常值数量如下:')
outlier


# ### 异常值数量超过100的特征


count = 0
for column in outlier.columns:
    if outlier.loc['outlier'][column] >= 100:
        print(column,end=',')
        count += 1
print()
print()
print('异常值数量超过100的特征有: %d 个'%count)


# 异常值数量超过100的特征有77个，占总特征数量的比例38.5%，比例非常高，但考虑到总数据量有20万条，每个异常值占比并不高。


# ## 相关度分析


# ### 特征之间的相关系数矩阵


result = data.corr(method='pearson')
result


# ### 查找那些相关系数比较高的特征


# 由于特征太多，不可能完全展示所有特征之间的关系，我们寻找那些相关系数比较高的变量，查看他们之间的散点图矩阵和热图


var_names = [] # 存储特征序列，只要两个特征彼此间的相关系数在某个特征里最大，便将两者加入进来，用于后期统计出现次数最多的特征
# 寻找每一行相关系数最大的值对应的列名，必须排除相关系数为1的列名
for index in result.index:
    var_name = result.loc[index].argmax()
    temp = result.drop([var_name],axis=1)
    var_name = temp.loc[index].argmax() # 与当前特征相关系数最大的特征
    print(index,' and ',var_name,' : ',temp.loc[index][var_name])
    var_names.extend([index,var_name])


# 与其他特征相关性最高的前20个特征:具体是指，这些特征与许多其他特征的相关系数在对应特征的相关系数中都是最大


Counter(var_names).most_common(20)


show_vars = [] # 用来展示的特征
for name,count in Counter(var_names).most_common(5):
    show_vars.append(name)
show_vars


# ### 从正类和负类中，分别随机抽取100条数据用来展示


import numpy as np
np.random.seed(2)

neg_labels = [] # 负类的索引
pos_labels = [] # 正类的索引
count = 0
for label in labels.tolist():
    if label == 0:
        neg_labels.append(count)
    elif label == 1:
        pos_labels.append(count)
    count += 1

neg_index = np.random.randint(len(neg_labels),size=100)
pos_index = np.random.randint(len(pos_labels),size=100)
neg_index = np.array(neg_labels)[neg_index].tolist()
pos_index = np.array(pos_labels)[pos_index].tolist()
index = neg_index + pos_index
temp_data = data.loc[index]
temp_data # 用于展示特征之间的关系的数据，其中正负类各500条


# ### 绘制散点图矩阵


# 可以利用**散点图矩阵**同时绘制各变量间的散点图，从而快速发现**多个变量间的主要相关性**，这在进行**多元线性回归**时显得尤为重要


pd.plotting.scatter_matrix(temp_data[show_vars],diagonal='kde',c='r') # diagonal='kde' 对角线上是 核密度估计
plt.show()


# # 异常值处理


# 可用均值，中位数，众数，拉格朗日插值法来处理异常值


from scipy.interpolate import lagrange

# 拉格朗日插值法
def ploy(data,n,k=6): # data是待插值的序列值,n是data中待插值的位置,k是取待插值前后k个数作为插值的候选项
    y = data[list(range(n-k,n))+list(range(n+1,n+1+k))] #取当前异常值前 k个数，和其后 k个数
    y = y[y.notnull()] 
    return lagrange(y.index,list(y))(n)


statis = data.describe()
for column in data.columns:
    num_lower,num_upper = normalRange(statis.loc['25%'][column],statis.loc['75%'][column])
    indexs = data[(data[column]<num_lower) | (data[column] > num_upper)].index # 获取异常值的行索引
    for index in indexs:
        data.loc[index][column] = statis.loc['mean'][column] # 用均值来填充
        #data.loc[index][column] = statis.loc['50%'][column] # 用中位数来填充
        #data.loc[index][column] = ploy(data[column],index) # 拉格朗日插值法填充
data[:10]


# # 标准化:对特征进行标准化


'''
statis = data.describe()
for column in data.columns:
    data[column] = (data[column] - statis.loc['mean'][column]) / statis.loc['std'][column] # 
data[:10]
'''

from sklearn import preprocessing

scaler = preprocessing.StandardScaler().fit(data)
scaler # 缩放器，便于对test.csv中的数据做同样的标准化变换


data = scaler.transform(data)
data[:10]


# # 不平衡数据的处理:对负类欠采样


from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

sampler = RandomUnderSampler(random_state=0) # 默认重复采样
X_resampled,Y_resampled = sampler.fit_sample(data,labels)
print(X_resampled.shape,Y_resampled.shape)
Counter(Y_resampled)


# # 分割数据集:训练集和测试集


from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X_resampled,Y_resampled,test_size=0.3)
len(Y_train),len(Y_test)


# # 选择算法:随机森林


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier


# # 调参: 交叉验证 + 贝叶斯优化


def target(n_estimators,min_samples_split,max_features,max_depth):
    target_value = cross_val_score(
      RandomForestClassifier(n_estimators=int(n_estimators),
                             min_samples_split=int(min_samples_split),
                             max_features=min(max_features,0.999),# float
                             max_depth=int(max_depth),
                             random_state=2
                            ),
      X_train, # 训练数据
      Y_train, # 类别
      scoring='roc_auc', # 以AUC来评估每轮交叉验证的结果
      cv=10 # 10折交叉验证
    ).mean() # 将 10折交叉验证每轮AUC的均值 作为待优化的值
    return target_value


# ## 待调参数的范围


params = {'n_estimators': (10, 50),
          'min_samples_split': (2, 25),
          'max_features': (0.1, 0.999),
          'max_depth': (5, 10)
         }


# ## 实例化贝叶斯调参对象


from bayes_opt import BayesianOptimization

model_bo = BayesianOptimization(
  f=target,
  pbounds=params
)


# ## 输出贝叶斯优化过程


model_bo.maximize(n_iter=5) # n_iter指代贝叶斯优化次数，次数越多越容易找到最优参数


# ## 获取最优参数


params = model_bo.max['params']
params


# ## 用最优参数重新训练分类器


model = RandomForestClassifier(n_estimators=int(params['n_estimators']),
                             min_samples_split=int(params['min_samples_split']),
                             max_features=min(params['max_features'],0.999),# float
                             max_depth=int(params['max_depth']),
                             random_state=2) # 随机森林分类器
model.fit(X_train,Y_train)


# ## 在测试集上的结果


from sklearn.metrics import roc_auc_score

result = model.predict_proba(X_test)
result = [max(pair) for pair in result]
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


result = model.predict_proba(test_data)
result = [max(pair) for pair in result]


# ## 提交结果


sub = pd.DataFrame(data={'ID_code':test_code,'target':result})
sub.to_csv('sample_submission.csv',index=False)

