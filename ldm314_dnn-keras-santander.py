import os
import gc
import pickle
from pathlib import Path
#math
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import GridSearchCV, StratifiedKFold # Used to use Kfold to train our model
#keras
import keras
from keras.layers import *
from keras.optimizers import *
from keras.models import *
from keras.callbacks import *
from keras import regularizers
from keras.utils.data_utils import *
from keras.layers.normalization import BatchNormalization
from keras import backend as K 
import tensorflow as tf

print(os.listdir("../input"))

#force gpu in my local environment
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# really clears all things
def clear_session(model):
    sess = K.get_session()
    K.clear_session()
    try:
        del model
    except:
        pass
    sess.close()
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    K.set_session(tf.Session(config=config))
    gc.collect()


# config
MODEL_NAME = "santander_1"
RESCALE = True
MIN_MAX = (-1,1)


#file paths
input_path = Path("../input")
train_csv = str(input_path / "train.csv")
test_csv = str(input_path / "test.csv")


#load original features
train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)
train_df.head()


#rescale all feature cols to, MIN_MAX setting
feature_columns = []
for i in range(0,200,1):
    key = 'var_' + str(i)
    feature_columns.append(key)
    if RESCALE:
        test_df[key] = minmax_scale(test_df[key].values.astype(np.float32), feature_range=MIN_MAX, axis=0)
        train_df[key] = minmax_scale(train_df[key].values.astype(np.float32), feature_range=MIN_MAX, axis=0)

train_df.head()


#make a model
def crop(dimension, start, end):
    # Crops (or slices) a Tensor on a given dimension from start to end
    # example : to crop tensor x[:, :, 5:10]
    # call slice(2, 5, 10) as you want to crop on the second dimension
    def func(x):
        if dimension == 0:
            return x[start: end]
        if dimension == 1:
            return x[:, start: end]
        if dimension == 2:
            return x[:, :, start: end]
        if dimension == 3:
            return x[:, :, :, start: end]
        if dimension == 4:
            return x[:, :, :, :, start: end]
    return Lambda(func)

def build_model(dim=(len(feature_columns),)):
    input_tensor = Input(shape=dim)
    x = input_tensor
    x = Dense(75)(x)
    x = Activation('tanh')(x)
    x = Dense(15)(x)
    x = Activation('tanh')(x)
    x = Dense(1)(x)
    x = Activation('hard_sigmoid')(x)
    return Model(input_tensor, x, name='b1')
model = build_model()
model.summary()


#checkpoint reverter, reloads saved if not better val score
import keras
class CheckpointReverter(keras.callbacks.Callback):
    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 mode='auto',skip=3,max_lr=1e-4):
        super(CheckpointReverter, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.max_skip = skip
        self.max_lr = max_lr
        self.current_skip = 0
        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn('Can save best model only with %s available, '
                          'skipping.' % (self.monitor), RuntimeWarning)
        else:
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.monitor_op(current, self.best):
                if self.verbose > 0:
                    print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                          ' dont load model %s'
                          % (epoch + 1, self.monitor, self.best,
                             current, filepath))
                self.best = current
                self.current_skip = 0
            elif current != 0.:
                if self.current_skip >= self.max_skip:
                    self.current_skip = 0
                    try:
                        self.model.load_weights(filepath)
                        old_lr = K.get_value(self.model.optimizer.lr)
                        new_lr = old_lr * 100
                        if new_lr > self.max_lr:
                            new_lr = self.max_lr
                        K.set_value(self.model.optimizer.lr, new_lr)
                    except:
                        print("error loading weights")

                    if self.verbose > 0:
                        print('\nEpoch %05d: %s did not improve from %0.5f,'
                              ' load best from %s, new lr: %f '
                              % (epoch + 1, self.monitor, self.best,filepath,new_lr))
                else:
                    self.current_skip += 1



# custom loss
from keras.losses import *


POS_WEIGHT = 10  # multiplier for positive targets, needs to be tuned

def wbce(target, output):
    """
    Weighted binary crossentropy between an output tensor 
    and a target tensor. POS_WEIGHT is used as a multiplier 
    for the positive targets.

    Combination of the following functions:
    * keras.losses.binary_crossentropy
    * keras.backend.tensorflow_backend.binary_crossentropy
    * tf.nn.weighted_cross_entropy_with_logits
    """
    # transform back to logits
    output = K.clip(output, K.epsilon(), 1-K.epsilon())
    output = tf.log(output / (1 - output))

    # compute weighted loss
    loss = tf.nn.weighted_cross_entropy_with_logits(targets=target,
                                                    logits=output,
                                                    pos_weight=POS_WEIGHT)
    loss = tf.where(tf.is_nan(loss), tf.zeros_like(loss), loss)
    return K.mean(loss)
    
def f1(y_true, y_pred):
#    y_pred = binaryRound(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, K.floatx()), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), K.floatx()), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, K.floatx()), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), K.floatx()), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def f1_loss(y_true,y_pred):
    return tf.cast(1-f1(y_true,y_pred),K.floatx())

def roc_auc_score( y_true,y_pred):
    """ ROC AUC Score.
    Approximates the Area Under Curve score, using approximation based on
    the Wilcoxon-Mann-Whitney U statistic.
    Yan, L., Dodier, R., Mozer, M. C., & Wolniewicz, R. (2003).
    Optimizing Classifier Performance via an Approximation to the Wilcoxon-Mann-Whitney Statistic.
    Measures overall performance for a full range of threshold levels.
    Arguments:
        y_pred: `Tensor`. Predicted values.
        y_true: `Tensor` . Targets (labels), a probability distribution.
    """
    with tf.name_scope("RocAucScore"):

        pos = tf.boolean_mask(y_pred, tf.cast(y_true, tf.bool))
        neg = tf.boolean_mask(y_pred, ~tf.cast(y_true, tf.bool))

        pos = tf.expand_dims(pos, 0)
        neg = tf.expand_dims(neg, 1)

        # original paper suggests performance is robust to exact parameter choice
        gamma = 0.3
        p     = 2

        difference = tf.zeros_like(pos * neg) + pos - neg - gamma

        masked = tf.boolean_mask(difference, difference < 0.0)

    return tf.reduce_sum(tf.pow(-masked, p))

#---------------------------
# AUC for a binary classifier
def auc(y_true, y_pred):
#     ptas = tf.stack([binary_PTA(y_true,y_pred)],axis=0)
#     pfas = tf.stack([binary_PFA(y_true,y_pred)],axis=0)
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 10)],axis=0)
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 10)],axis=0)
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas*binSizes
    return K.sum(s, axis=0)

#---------------------
# PFA, prob false alert for binary classifier
def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)
    return FP/N

#----------------
# P_TA prob true alerts for binary classifier
def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # P = total number of positive labels
    P = K.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = K.sum(y_pred * y_true)
    return TP/P

#---------------------
# PFA, prob false alert for binary classifier
def binary_PFA_l(y_true, y_pred):
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)
    return FP/N

#----------------
# P_TA prob true alerts for binary classifier
def binary_PTA_l(y_true, y_pred):
    # P = total number of positive labels
    P = K.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = K.sum(y_pred * y_true)
    return TP/P

def auc_loss(t,p):
    ptas = tf.stack([binary_PTA_l(t,p)],axis=0)
    pfas = tf.stack([binary_PFA_l(t,p)],axis=0)
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas*binSizes
    return 1-K.sum(s, axis=0)

def brian_loss(t,p):
    return (wbce(t,p) + f1_loss(t,p)) /2
#     return auc_loss(t,p) + binary_crossentropy(t,p)


BATCH_SIZE=50000
EPOCHS=500
FOLDS = 5

# get the train data
X = train_df[feature_columns].values
y = np.array(train_df['target'].values)

#k folds, saving best validation
splits = list(StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=2).split(X, y))
for idx, (train_idx, val_idx) in enumerate(splits):
    print("Beginning fold {}".format(idx+1))
    
    train_X, train_y, val_X, val_y = X[train_idx], y[train_idx], X[val_idx], y[val_idx]

    clear_session(model)
    model = build_model()
        
    for i, layer in enumerate(model.layers):
        model.layers[i].trainable = True


    opt = Adam(5e-3)
    metrics = ["acc",auc,wbce,f1]
    model.compile(optimizer=opt, loss=wbce, metrics=metrics)
    results = model.fit(train_X,train_y, 
                            epochs = EPOCHS, 
                            callbacks = [
                                ReduceLROnPlateau(factor=0.4, patience=40, min_lr=1e-6, min_delta=1e-6, verbose=1, mode='min'),
                                ModelCheckpoint(MODEL_NAME+"_fold_{}.checkpoint".format(idx), 
                                    verbose=0, 
                                    monitor='val_auc',
                                    mode='max',
                                    save_best_only=True, 
                                    save_weights_only=True
                                ),
                                CheckpointReverter(MODEL_NAME+"_fold_{}.checkpoint".format(idx),
                                        verbose=0,
                                        monitor='val_auc',
                                        mode='max',
                                        skip=50
                                       )
                            ],
                            validation_data=[val_X, val_y],
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            verbose=1
                            )


preds_test = []
X = test_df[feature_columns].values

clear_session(model)
model = build_model()
for i in range(FOLDS):
    model.load_weights(MODEL_NAME+"_fold_{}.checkpoint".format(i))
    pred = model.predict(X, batch_size=BATCH_SIZE, verbose=1)
    preds_test.append(pred)


preds = np.mean(np.array(preds_test),axis=0)


#TODO, build this better target ends up being a list of one for each row
submission_df = pd.DataFrame(data={'ID_code': list(test_df['ID_code'].values), 'target': list(preds)})
submission_df['target'] = submission_df['target'].apply(lambda x: float(x[0])).astype('float')
submission_df[submission_df['target'] > 0.5].head(10)



#TODO, try different thresholds? No: submit the fractional instead
# submission_df['target'] = submission_df['target'].apply(lambda x: round(x))
# submission_df[submission_df['target'] > 0.5].head(10)


submission_df.to_csv('submission.csv', index=False)



