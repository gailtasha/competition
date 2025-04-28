# *Motivation*:
# 
# I would like to find something with special relationships from the Feature.


import numpy as np
import pandas as pd

print('read data')
df = pd.read_csv('../input/train.csv')


# There seems to be some relationship between these features...


engineering_feats = {
'var_0': ['var_2', 'var_198', 'var_179', 'var_191', 'var_22'],
'var_2': ['var_0', 'var_179', 'var_198', 'var_146', 'var_22', 'var_115'],
'var_26': ['var_44', 'var_155', 'var_157', 'var_163', 'var_180', 'var_123', 'var_87'],
'var_44': ['var_26', 'var_123', 'var_173', 'var_180', 'var_87', 'var_155', 'var_157', 'var_163', 'var_35', 'var_196', 'var_75'],
'var_86': ['var_21', 'var_51', 'var_40', 'var_135', 'var_139', 'var_67', 'var_167', 'var_76'],
'var_139': ['var_21', 'var_80', 'var_86', 'var_174', 'var_40', 'var_76'],
'var_172': ['var_83', 'var_167', 'var_19', 'var_67', 'var_118'],
}


# fmm...


k = 'var_44'
v = engineering_feats[k]
T = df[v].copy()
for i, fe in enumerate(v):
    T['%s_%s'%(k.split('_')[1],fe.split('_')[1])] = df[k]+df[fe]
T = T.drop(v,axis=1)
T.corr()


S = T.corr()
S[S!=1.0].mean()


# It's Dummy.


k = 'var_44'
v = df.columns[2:14]
T = df[v].copy()
for i, fe in enumerate(v):
    T['%s_%s'%(k.split('_')[1],fe.split('_')[1])] = df[k]+df[fe]
T = T.drop(v,axis=1)
T.corr()


S = T.corr()
S[S!=1.0].mean()


# Correlation is large because it becomes similar value when adding together. So it's less meaningful to include than the top set...?


# **Expected value of GaussianNB by cross validation**
# 
# without Feature Engineering, It's 0.8893809240152594


from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import QuantileTransformer
X = df.values[:,2:].astype(np.float32)
Y = df.values[:,1].astype(np.float32)
all_roc = 0
all_cnt = 0
for fold_id, (IDX_train, IDX_test) in enumerate(KFold(n_splits=5, random_state=12, shuffle=True).split(Y)):
    X_train = X[IDX_train]
    X_test = X[IDX_test]
    Y_train = Y[IDX_train]
    Y_test = Y[IDX_test]
    clf = make_pipeline(QuantileTransformer(output_distribution='normal',random_state=12), GaussianNB())
    clf.fit(X_train,Y_train)
    Z = clf.predict_proba(X_test)[:,1]
    all_roc += roc_auc_score(Y_test,Z)
    all_cnt += 1
print('CV score:',all_roc/all_cnt)


# Replace the features to theres sum and difference


combination_feats = [('var_26','var_44')]
print(combination_feats)


df_e = df.copy()
for fe in combination_feats:
    df_e['%s_plus_%s'%fe] = df_e[fe[0]]+df_e[fe[1]]
    df_e['%s_minus_%s'%fe] = df_e[fe[1]]-df_e[fe[0]]
df_e = df_e.drop(list(set([i for s in combination_feats for i in s])),axis=1)
df_e.head()


# The number of Feature has not changed


X = df_e.values[:,2:].astype(np.float32)
Y = df_e.values[:,1].astype(np.float32)
print(X.shape)


# QuantileTransfExpected value of GaussianNB by cross validationrmer


all_roc = 0
all_cnt = 0
for fold_id, (IDX_train, IDX_test) in enumerate(KFold(n_splits=5, random_state=12, shuffle=True).split(Y)):
    X_train = X[IDX_train]
    X_test = X[IDX_test]
    Y_train = Y[IDX_train]
    Y_test = Y[IDX_test]
    clf = make_pipeline(QuantileTransformer(output_distribution='normal',random_state=12), GaussianNB())
    clf.fit(X_train,Y_train)
    Z = clf.predict_proba(X_test)[:,1]
    all_roc += roc_auc_score(Y_test,Z)
    all_cnt += 1
print('CV score:',all_roc/all_cnt)


# CV Score gains.*(@_@)* 


# **It has not been completed yet.
# I am halfway!**





