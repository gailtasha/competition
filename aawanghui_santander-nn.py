# Basic packages
import pandas as pd
import numpy as np
import warnings
import time
import random
import glob
import sys
import os
import gc

import pandas as pd
import numpy as np
import scipy
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Flatten, Dropout,Conv1D, BatchNormalization ,Activation, Add,Reshape, Average,Lambda, concatenate
from keras import callbacks
from keras import optimizers
import keras.backend as K

# visualization packages
import seaborn as sns
import matplotlib.pyplot as plt

# execution progress bar
from tqdm import tqdm_notebook, tnrange
from tqdm.auto import tqdm
tqdm.pandas()
import tensorflow as tf


# System Setup
%matplotlib inline
%precision 4
warnings.filterwarnings('ignore')
plt.style.use('ggplot')
np.set_printoptions(suppress=True)
pd.set_option("display.precision", 15)


# ## Load Data


# ### Neural Net


#LOAD DATA
df_train = pd.read_csv('../input/santander-customer-transaction-prediction/train.csv', index_col=0)
y_train = df_train.pop('target')


len_train = len(df_train)
df_test = pd.read_csv('./../input/santander-customer-transaction-prediction/test.csv', index_col=0)
df_all = pd.concat((df_train, df_test), sort=False)
prev_cols = df_all.columns

# PREPROCESS
scaler = StandardScaler()
df_all[prev_cols] = scaler.fit_transform(df_all[prev_cols])
df_train = df_all[0:len_train]
df_test = df_all[len_train:]



 def augment_train(df_train, y_train):   
    t0 = df_train[y_train == 0].copy()
    t1 = df_train[y_train == 1].copy()
    i = 0
    N = 3
    for I in range(0):  # augment data into 2x
        for col in df_train.columns:
            i = i + 1000
            np.random.seed(i)
            np.random.shuffle(t0[col].values)
            np.random.shuffle(t1[col].values)
        df_train = pd.concat([df_train, t0.copy()])
        df_train = pd.concat([df_train, t1.copy()])
        y_train = pd.concat([y_train, pd.Series([0] * t0.shape[0]), pd.Series([1] * t1.shape[0])])
    return df_train, y_train


features = [c for c in df_train.columns if c not in ["ID_code","target"]]
def detect_test(test_df):
    df_test=test_df.values
    unique_count = np.zeros_like(df_test)
    for feature in tqdm(range(df_test.shape[1])):
        _, index_, count_ = np.unique(df_test[:, feature], return_counts=True, return_index=True)
        unique_count[index_[count_ == 1], feature] += 1

    # Samples which have unique values are real the others are fake
    real_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) > 0)[:, 0]
    synthetic_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) == 0)[:, 0]
    return real_samples_indexes,synthetic_samples_indexes
def generate_fe(trn, tst):
    #tst,target=augment_train(tst,y_train=target)
    real,syn = detect_test(df_test[features])
    al = pd.concat([trn,tst,df_test.iloc[real]],axis=0)
    trn_fe = pd.DataFrame()
    tst_fe = pd.DataFrame()
    for c in features:
        trn[c+"_test"]=trn[c].map(al[c].value_counts())
        trn[c+"_multi"] = trn[c+"_test"]*trn[c]
        #trn[c+"_div"] = trn[c]/trn[c+"_test"]
        trn_fe[c] = trn[c]
        trn_fe[c+"_test"] = trn[c+"_test"]
        trn_fe[c+"_muti"] = trn[c+"_multi"]
        #trn_fe[c+"_div"] = trn[c+"_div"]
        tst[c+"_test"]=tst[c].map(al[c].value_counts())
        #tst[c+"_test"] = tst[c+"_test"]*tst[c]
        tst_fe[c] = tst[c]
        tst[c+"_multi"] = tst[c+"_test"]*tst[c]
        #tst[c+"_div"] = tst[c]/tst[c+"_test"]
        tst_fe[c+"_test"] = tst[c+"_test"]
        tst_fe[c+"_muti"] = tst[c+"_multi"]
        #tst_fe[c+"_div"] = tst[c+"_div"]
    return trn_fe, tst_fe


def generate_fe_test(tst):
    re,sy =  detect_test(tst[features])
    al = pd.concat([df_train,df_test.iloc[re]],axis=0)
    tst_fe = pd.DataFrame()
    for c in features:
        tst[c+"_test"]=tst[c].map(al[c].value_counts())
        #tst[c+"_test"] = tst[c+"_test"]*tst[c]
        tst_fe[c] = tst[c]
        tst[c+"_multi"] = tst[c+"_test"]*tst[c]
        #tst[c+"_div"] = tst[c]/tst[c+"_test"]
        tst_fe[c+"_test"] = tst[c+"_test"]
        tst_fe[c+"_muti"] = tst[c+"_multi"]
        #tst_fe[c+"_div"] = tst[c+"_div"]
    return tst_fe
test_fe = generate_fe_test(df_test[features])


# MODEL DEF
def auc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)
def _Model():
    inp = Input(shape=(200,3))
    d1 = Dense(32,activation='relu')(inp)
    d1 = BatchNormalization()(d1)
    '''d1 = Dense(128,activation='relu')(d1)
    d1 = Dense(64,activation='relu')(d1)
    d1 = Dense(32,activation='relu')(d1)
    d1 = Dense(16,activation='relu')(d1)'''
    #d2 = Lambda(lambda x: x, output_shape=(400,1))(inp)
    #d2 = Dense(256,activation='relu')(inp)
    #d3 = concatenate([d1, d2], axis = 2)
    ''' d3 = Dense(256,activation="relu")(d3)
    d3 = Dense(128,activation="relu")(d3)
    d3 = Dense(64,activation="relu")(d3)
    d3 = Dense(32,activation="relu")(d3)'''
    d1 = Dense(8,activation="relu")(d1)
    d1 = BatchNormalization()(d1)
    d4 = Flatten()(d1)
    preds = Dense(1, activation="sigmoid")(d4)
    model = Model(inputs=inp, outputs=preds)
    adam = optimizers.Adam(lr=0.009)
    model.compile(optimizer=adam, loss=K.binary_crossentropy,metrics=["acc"])
    model.summary()
    return model


# LOGGER
class Logger(callbacks.Callback):
    def __init__(self, out_path='./', patience=30, lr_patience=3, out_fn='', log_fn=''):
        self.auc = 0
        self.path = out_path
        self.fn = out_fn
        self.patience = patience
        self.lr_patience = lr_patience
        self.no_improve = 0
        self.no_improve_lr = 0

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        cv_pred = self.model.predict(self.validation_data[0], batch_size=1024)
        cv_true = self.validation_data[1]
        auc_val = roc_auc_score(cv_true, cv_pred)
        if self.auc < auc_val:
            self.no_improve = 0
            self.no_improve_lr = 0
            print("Epoch %s - best AUC: %s" % (epoch, round(auc_val, 4)))
            self.auc = auc_val
            self.model.save(self.path + self.fn, overwrite=True)
        else:
            self.no_improve += 1
            self.no_improve_lr += 1
            print("Epoch %s - current AUC: %s" % (epoch, round(auc_val, 4)))
            if self.no_improve >= self.patience:
                self.model.stop_training = True
            if self.no_improve_lr >= self.lr_patience:
                lr = float(K.get_value(self.model.optimizer.lr))
                K.set_value(self.model.optimizer.lr, 0.75*lr)
                print("Setting lr to {}".format(0.75*lr))
                self.no_improve_lr = 0

        return


#RUN
preds = []
c = 0
oof_preds = np.zeros((len(df_train), 1))
cv = StratifiedKFold(n_splits=5,shuffle=True, random_state=3263)
for train, valid in cv.split(df_train, y_train):
    print("VAL %s" % c)
    trn = df_train.iloc[train]
    tst = df_train.iloc[valid]
    trn, tst = generate_fe(trn, tst)
    X_train = np.reshape(trn.values, (-1,200,3))
    y_train_ = y_train.iloc[train].values
    X_valid = np.reshape(tst.values, (-1,200,3))
    y_valid = y_train.iloc[valid].values
    model = _Model()
    logger = Logger(patience=30, out_path='./', out_fn='cv_{}.h5'.format(c))
    model.fit(X_train, y_train_, validation_data=(X_valid, y_valid), epochs=150, verbose=2, batch_size=1024,
              callbacks=[logger])
    model.load_weights('cv_{}.h5'.format(c))
    fe = [c for c in test_fe.columns if c not in ["ID_code","target"]]
    X_test = np.reshape(test_fe[fe].values, (200000, 200, 3))
    curr_preds = model.predict(X_test, batch_size=2048)
    oof_preds[valid] = model.predict(X_valid)
    preds.append(curr_preds)
    c += 1
pd.DataFrame(oof_preds).to_csv("NN_oof_preds.csv", index = False)
auc = roc_auc_score(y_train, oof_preds)
print("CV_AUC: {}".format(auc))

# SAVE DATA
preds = np.asarray(preds)
preds = preds.reshape((5, 200000))
preds_final = np.mean(preds.T, axis=1)
submission = pd.read_csv('./../input/santander-customer-transaction-prediction/sample_submission.csv')
submission['target'] = preds_final
submission.to_csv('submission.csv', index=False)


# kkjjjsubmission = pd.read_csv('./../input/santander-customer-transaction-prediction/sample_submission.csv')
# submission['target'] = sub1.target 
# submission.to_csv('submission.csv', index=False)

