# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import keras.backend as K
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D, BatchNormalization, Input, Conv1D
from keras.callbacks import ModelCheckpoint
from keras import metrics
from keras.optimizers import Adam 
from sklearn.preprocessing import StandardScaler
from keras import backend as K
import keras
from keras.models import Model
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import *
from keras import regularizers
import tensorflow as tf
from keras.losses import binary_crossentropy
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
BATCH_SIZE = 1024
NUM_FEATURES = 1200
import seaborn as sns
from sklearn.model_selection import train_test_split
# Any results you write to the current directory are saved as output.


train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")


train1=train.copy(deep=True)
test1=test.copy(deep=True)

print("shape of train data ",train.shape)


train["target"].value_counts()


sns.countplot(train["target"])


count_class_0, count_class_1 = train.target.value_counts()

df_class_0 = train[train['target'] == 0]
df_class_1 = train[train['target'] == 1]


df_class_0_under = df_class_0.sample(count_class_1)
df_test_under = pd.concat([df_class_0_under, df_class_1], axis=0)

print('Random under-sampling:')
print(df_test_under.target.value_counts())

df_test_under.target.value_counts().plot(kind='bar', title='Count (target)');


def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall


def _Model1():
    inp = Input(shape=(200, 1))
    d1 = Dense(128, activation='sigmoid')(inp)
    d2 = Dense(64, activation='relu')(d1)
    d3 = Dense(32, activation='sigmoid')(d2)
    d4 = Dense(16, activation='relu')(d3)
    f2 = Flatten()(d4)
    preds = Dense(1, activation='sigmoid')(f2)
    model = Model(inputs=inp, outputs=preds)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy',recall])
    return model


def _Model2():
    inp = Input(shape=(200, 1))
    d1 = Dense(64, activation='sigmoid')(inp)
    d2 = Dense(128, activation='relu')(d1)
    d3 = Dense(32, activation='sigmoid')(d2)
    d4 = Dense(16, activation='relu')(d3)
    f2 = Flatten()(d4)
    preds = Dense(1, activation='sigmoid')(f2)
    model = Model(inputs=inp, outputs=preds)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy',recall])
    return model


def _Model4():
    inp = Input(shape=(200, 1))
    d1 = Dense(128, activation='sigmoid')(inp)
    d2 = Conv1D(64, 2,activation='relu')(d1)
    d3 = Dense(32, activation='sigmoid')(d2)
    d4 = Dense(16, activation='relu')(d3)
    f2 = Flatten()(d4)
    preds = Dense(1, activation='sigmoid')(f2)
    model = Model(inputs=inp, outputs=preds)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy',recall])
    return model


#print("Summary /n",model.summary())


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
df_train=train.drop(["ID_code","target"],axis=1)
y_train=train["target"]
df_test=test.drop(["ID_code"],axis=1)
scaler=StandardScaler()
df_train=scaler.fit_transform(df_train)
df_test=scaler.fit_transform(df_test)
#y_test=test["target"]


df_train= pd.DataFrame(df_train)
df_test= pd.DataFrame(df_test)


# 128 64 32 16 1
# 64 128 32 1
#32 32 1
# also add dropout
# different activations as well


preds = []
c = 0
oof_preds = np.zeros((len(df_train), 1))
for train, valid in cv.split(df_train, y_train):
    print("VAL %s" % c)
    X_train = np.reshape(df_train.iloc[train].values, (-1, 200, 1))
    y_train_ = y_train.iloc[train].values
    X_valid = np.reshape(df_train.iloc[valid].values, (-1, 200, 1))
    y_valid = y_train.iloc[valid].values
    model = _Model2()
    #logger = Logger(patience=10, out_path='./', out_fn='cv_{}.h5'.format(c))
    history=model.fit(X_train, y_train_, validation_data=(X_valid, y_valid), epochs=20, verbose=2, batch_size=256)
    #print(model.evaluate(X_valid, y_valid))
    #model.load_weights('cv_{}.h5'.format(c))
    
    X_test = np.reshape(df_test.values, (200000, 200, 1))
    curr_preds = model.predict(X_test, batch_size=2048)
    oof_preds[valid] = model.predict(X_valid)
    preds.append(curr_preds)
    c += 1
auc = roc_auc_score(y_train, oof_preds)
print("CV_AUC: {}".format(auc))


prediction=(preds[0]+preds[1]+preds[2]+preds[3]+preds[4])/5


sub = pd.DataFrame() 
sub["ID_code"] = test["ID_code"] 
sub["target"] = prediction
sub.to_csv("submission-cnn-ksplit.csv", index=False)


sub.to_csv("submission-cnn-ksplit.csv", index=False)


history_dict = history.history
history_dict.keys()


import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
recall = history.history['recall']
val_recall = history.history['val_recall']
epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_recall, 'g-', label='Validation Recall')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


df_test_under.head()


x_train.shape


# ## Again trying cross validation on under sampled data


preds = []
c = 0
oof_preds = np.zeros((len(df_train), 1))
df_under_x=df_test_under.drop(["ID_code","target"],axis=1)
df_under_y=df_test_under["target"]
for train, valid in cv.split(df_under_x,df_under_y):
    print("VAL %s" % c)
    X_train = np.reshape(df_under_x.iloc[train].values, (-1, 200, 1))
    y_train_ = df_under_y.iloc[train].values
    X_valid = np.reshape(df_under_x.iloc[valid].values, (-1, 200, 1))
    y_valid = df_under_y.iloc[valid].values
    model = _Model1()
    #logger = Logger(patience=10, out_path='./', out_fn='cv_{}.h5'.format(c))
    history=model.fit(X_train, y_train_, validation_data=(X_valid, y_valid), epochs=20, verbose=2, batch_size=256)
    #print(model.evaluate(X_valid, y_valid))
    #model.load_weights('cv_{}.h5'.format(c))
    
    X_test = np.reshape(df_test.values, (200000, 200, 1))
    curr_preds = model.predict(X_test, batch_size=2048)
    oof_preds[valid] = model.predict(X_valid)
    preds.append(curr_preds)
    c += 1
auc = roc_auc_score(y_train, oof_preds)
print("CV_AUC: {}".format(auc))


# ### We can see in the above case even though our recall is good but the AUC-ROC value is quite low


#model_list=["_Model1","_Model2","_Model3","_Model4"]
#for i in model_list:

#df_test_under
x_train, x_test, y_train, y_test = train_test_split(df_test_under.drop(["ID_code","target"],axis=1),
                                                    df_test_under["target"], test_size=0.3, random_state=42)


    


X_train = np.reshape(x_train.values, (-1, 200, 1))
#y_train 
X_valid = np.reshape(x_test.values, (-1, 200, 1))
y_valid =y_test


model = _Model1()
history1 = model.fit(X_train,
                    y_train,
                    epochs=40,
                    verbose=2,batch_size=256,     
                    validation_data=(X_valid, y_valid))


model = _Model2()
history2 = model.fit(X_train,
                    y_train,
                    epochs=50,
                    verbose=2,batch_size=256,     
                    validation_data=(X_valid, y_valid))


model = _Model3()
history4 = model.fit(X_train,
                    y_train,
                    epochs=20,
                    verbose=2,batch_size=256,     
                    validation_data=(X_valid, y_valid))


import matplotlib.pyplot as plt

hist_list=[history1,history2,history3
j=1
for i in hist_list:
    
    acc = i.history['acc']
    val_acc = i.history['val_acc']
    loss = i.history['loss']
    val_loss = i.history['val_loss']

    epochs = range(1, len(acc) + 1)

    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss of Model '+str(j))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.show()
    j+=1

