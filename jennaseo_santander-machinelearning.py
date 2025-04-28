# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # 1. 머리말 Introduction
# 
#  Santander의 사명은 사람과 사업이 번영하도록 돕는 것입니다. 그들은 항상 고객이 재무 건전성을 이해하고 금전적 목표를 달성하는 데 어떠한 제품 및 서비스가 도움이 되는가를 식별 할 방법을 찾고 있습니다.
#     


# ![](https://www.santander.co.uk/assets/s3fs-public/images/all_together_now_hero_banner.jpg)


# Santander의 데이터 과학 팀은 머신러닝 알고리즘에 지속적으로 도전하고 있으며, 글로벌 데이터 과학 커뮤니티와 협력하여 가장 일반적인 과제를 해결하기 위한 새로운 방법을 보다 정확하게 식별할 수 있도록 지원하고 있습니다. 


# # 2. 패키지 불러오기


# ## 2-1.Import


from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier,Pool
from sklearn.metrics import roc_curve, auc
from IPython.display import display
import matplotlib.patches as patch
import matplotlib.pyplot as plt
from sklearn.svm import NuSVR
from scipy.stats import norm
from sklearn import svm
import lightgbm as lgb
import xgboost as xgb
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import time
import glob
import sys
import os
import gc


# ## 2-2.Version


print('pandas: {}'.format(pd.__version__))
print('numpy: {}'.format(np.__version__))
print('Python: {}'.format(sys.version))


# # 3. 목표설정
# 
# 해당 Kernel은, 거래 금액과 관계없이 향후 어떤 고객이 특정 거래를 할 것인지 머신러닝을 이용하여 분석합니다.
# 
# -> [test dataset의 target 컬럼 값을 예측하기]


# # 4. 탐색적 데이터 분석(EDA)


# ## 4-1. 데이터 불러오기


print(os.listdir("../input/"))


print(os.listdir("../input/santander-customer-transaction-prediction"))


# 데이터 불러오기 Pandas
train= pd.read_csv('../input/santander-customer-transaction-prediction/train.csv')
test = pd.read_csv('../input/santander-customer-transaction-prediction/test.csv')


#데이터 확인
train.shape, test.shape


# ## 4-2 데이터 분석
# 


# train 데이터 (행:20만개 열: 202개)
# * ID_code(스트링);
# * 타겟;
# * 200개의 변수(수치):var_0 ~ var_199


print(train.info())


# 
# test 데이터 (행:20만개 열: 201개)
# * ID_code(스트링);
# * 200개의 변수(수치):var_0 ~ var_199


print(test.info())


# ### 4-2-1. Descibe()
# 
# 
# * 표준 편차는 열차 및 시험 변수 데이터 모두에서 상대적으로 크다.
# * 열차 및 시험 데이터의 최소값, 최대값, 평균값, 표준값이 상당히 가깝게 보인다.
# * 평균 값은 큰 범위로 분포한다.


%%time
test.describe()


# ### 4-2-2 결측값 확인
# 
# -> 누락데이터 없음


def missing_check(data):
    tf=data.isna().sum().any()
    if tf==True:
        total = data.isnull().sum()
        percent = (data.isnull().sum()/data.isnull().count()*100)
        output = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        data_type = []
        for col in data.columns:
            dtype = str(data[col].dtype)
            data_type.append(dtype)
        output['Types'] = data_type
        return(np.transpose(output))
    else:
        return(False)


missing_check(train)


missing_check(test)


# ### 4-2-3.종속변수 Target Variable 


sns.countplot(train.target)
plt.title("")


#수치
train['target'].value_counts()


#비율
train.target.value_counts() *100/ train.target.count()


# **=>비대칭 데이터(Imbalanced Data)**
# 
# *거래하지 않을 고객 수(0)가 거래할 것으로 예상되는 고객(1)보다 훨씬 많다.*


# ## 4-3.데이터 다운사이징
# 변환 전 약 300MB -> 변환 후 약 150MB
# 
# 데이터 다운사이징 : 참고// https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65


#Based on this great kernel https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65
def reduce_mem_usage(df):
    start_mem_usg = df.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in df.columns:
        if df[col].dtype != object:  # Exclude strings
            
            # Print current column type
            print("******************************")
            print("Column: ",col)
            print("dtype before: ",df[col].dtype)
            
            # make variables for Int, max and min
            IsInt = False
            mx = df[col].max()
            mn = df[col].min()
            
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(df[col]).all(): 
                NAlist.append(col)
                df[col].fillna(mn-1,inplace=True)  
                   
            # test if column can be converted to an integer
            asint = df[col].fillna(0).astype(np.int64)
            result = (df[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        df[col] = df[col].astype(np.uint8)
                    elif mx < 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif mx < 4294967295:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)    
            
            # Make float datatypes 32 bit
            else:
                df[col] = df[col].astype(np.float32)
            
            # Print new column type
            print("dtype after: ",df[col].dtype)
            print("******************************")
    
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = df.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return df, NAlist


test, NAlist = reduce_mem_usage(test)
print("_________________")
print("")
print("Warning: the following columns have missing values filled with 'df['column_name'].min() -1': ")
print("_________________")
print("")
print(NAlist)


train, NAlist = reduce_mem_usage(train)
print("_________________")
print("")
print("Warning: the following columns have missing values filled with 'df['column_name'].min() -1': ")
print("_________________")
print("")
print(NAlist)


train.info()


test.info()


# # 5. Modelling
# 
# 예측에 가장 큰 영향을 미치는 특징은?
# 
# 모델에서 통찰력을 추출하는 방법?
# 
# * RandomForestClassifier
# * DecisionTreeClassifier
# * +Eli5


columns=["target","ID_code"]
X = train.drop(columns,axis=1)
y = train["target"]


X_test  = test.drop("ID_code",axis=1)


# ### 5-1. RandomForestClassifier


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.5, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train,test_size=0.5, random_state=1)
rfc = RandomForestClassifier(random_state=0).fit(X_train, y_train)


rfc


# ### 5-2. Eli5
# 
# ELI5는 Python 라이브러리로서 통합 API를 이용하여 다양한 머신러닝 모델을 시각화하고 디버그할 수 있다. 
# 
# * 어떤 변수가 가장 큰 영향을 미치는가를 계산하는 방법 중 하나
# 
# 분류 모형 또는 회귀 모형을 보는 두 가지 주요 방법이 있다.
# 
# 1. 모델 매개변수를 검사하고 모델이 전세계적으로 어떻게 작동하는지 알아내려고 노력한다.
# 2. 모델의 개별적인 예측을 검사하고, 모델이 결정을 내리는 이유를 알아내려고 노력한다.


import eli5
from eli5.sklearn import PermutationImportance

perm_imp = PermutationImportance(rfc, random_state=1).fit(X_test, y_test)


eli5.show_weights(perm_imp, feature_names = X_test.columns.tolist(), top=200)


# ### 5-3. DecisionTreeClassifier


Dec_tree = DecisionTreeClassifier(random_state=0, max_depth=5, min_samples_split=5).fit(X_train, y_train)


features = [c for c in train.columns if c not in ['ID_code', 'target']]


from sklearn import tree
import graphviz
tree_graph = tree.export_graphviz(Dec_tree, out_file=None, feature_names=features)
graphviz.Source(tree_graph)


# *pdp박스를 사용하여 이전 절에서 발견된 주요 변수의 영향 확인*
# 
# -> 트리모델분석에 따르면, Var_81이 모델에 더 유효함을 알 수 있다.


from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots

# Create the data that we will plot
pdp_goals = pdp.pdp_isolate(model=Dec_tree, dataset= X_train, model_features=features, feature='var_110')

# plot it
pdp.pdp_plot(pdp_goals, 'var_110')
plt.show()


from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots

# Create the data that we will plot
pdp_goals = pdp.pdp_isolate(model=Dec_tree, dataset= X_train, model_features=features, feature='var_81')

# plot it
pdp.pdp_plot(pdp_goals, 'var_81')
plt.show()


# *While <feature importance> shows which <variables> most affect predictions, /partial dependence/plots show how a feature affects predictions.*


# # 6. Logistic Regression


logit_clf = LogisticRegression(random_state=42).fit(X_train,y_train)
logit_clf


# #### predict_proba


plt.figure(figsize=(10, 10))
fpr, tpr, thr = roc_curve(y_train, logit_clf.predict_proba(X_train)[:,1])
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operator Characteristic Plot', fontsize=20, y=1.05)
auc(fpr, tpr)


# #### cross_validate


cross_val_score(logit_clf, X_train, y_train, scoring='roc_auc', cv=10).mean()


# #### Linear Discriminant Analysis (LDA)


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

lda_clf = LinearDiscriminantAnalysis()
lda_clf.fit(X_train, y_train)


plt.figure(figsize=(6, 6))
fpr, tpr, thr = roc_curve(y_train, lda_clf.predict_proba(X_train)[:,1])
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operator Characteristic Plot', fontsize=20, y=1.05)
auc(fpr, tpr)


cross_val_score(lda_clf, X_train, y_train, scoring='roc_auc', cv=10).mean()


# Quadratic Discriminant Analysis


qda_clf = QuadraticDiscriminantAnalysis()
qda_clf.fit(X_train, y_train)


plt.figure(figsize=(6, 6))
fpr, tpr, thr = roc_curve(y_train, qda_clf.predict_proba(X_train)[:,1])
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operator Characteristic Plot', fontsize=20, y=1.05)
auc(fpr, tpr)


cross_val_score(qda_clf, X_train, y_train, scoring='roc_auc', cv=10).mean()


# ### Model Combining(모형 결합)


from sklearn.preprocessing import StandardScaler
standardized_train = StandardScaler().fit_transform(train.set_index(['ID_code','target']))
standardized_test = StandardScaler().fit_transform(test.set_index(['ID_code']))
standardized_test = pd.DataFrame(standardized_test, columns=test.set_index(['ID_code']).columns)
standardized_test = standardized_test.join(test[['ID_code']])


X_test = standardized_test.set_index('ID_code').values.astype('float64')
submission = pd.read_csv('../input/santander-customer-transaction-prediction/sample_submission.csv')

logit_pred = logit_clf.predict_proba(X_test)[:,1]
lda_pred = lda_clf.predict_proba(X_test)[:,1]
qda_pred = qda_clf.predict_proba(X_test)[:,1]


submission = \
submission.join(pd.DataFrame(qda_pred, columns=['target1'])).join(pd.DataFrame(logit_pred, columns=['target2'])).\
join(pd.DataFrame(lda_pred, columns=['target3']))


submission['target'] = (submission.target1 + submission.target2 + submission.target3) / 3


submission.head()


del submission['target1']
del submission['target2']
del submission['target3']


submission.head()


submission.to_csv('logit_lda_qda_mean_ensemble.csv', index=False)

