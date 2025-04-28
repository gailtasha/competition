# **1) Clean test from fake data**


# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import numpy as np
import gzip
import pickle
import os
import glob
import time
import operator
import pandas as pd
from collections import Counter, defaultdict
from multiprocessing import Process, Manager
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import random
import datetime
from PIL import Image
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.preprocessing import normalize
from tqdm import tqdm
import gc
import psutil
import warnings
import shutil
from keras.callbacks import Callback


INPUT_PATH = '../input/'
MODELS_PATH = './'
OUTPUT_PATH = './'
MODELS_PATH_KERAS = './'
FEATURES_PATH = './'
CACHE_PATH = './'
SUBM_PATH = './'

def read_train():
    train = pd.read_csv(INPUT_PATH + 'train.csv', low_memory=True)
    return train


def read_test():
    test = pd.read_csv(INPUT_PATH + 'test.csv', low_memory=True)
    return test


def get_real_test_data():
    test = read_test()
    ids = test['ID_code'].values.copy()
    test.drop(['ID_code'], axis=1, inplace=True)
    features = test.columns.values
    df_test = test.values

    unique_samples = []
    unique_count = np.zeros_like(df_test)
    for feature in tqdm(range(df_test.shape[1])):
        _, index_, count_ = np.unique(df_test[:, feature], return_counts=True, return_index=True)
        unique_count[index_[count_ == 1], feature] += 1

    # Samples which have unique values are real the others are fake
    real_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) > 0)[:, 0]
    synthetic_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) == 0)[:, 0]

    print(len(real_samples_indexes))
    print(len(synthetic_samples_indexes))

    test1 = read_test()
    ids = ids[real_samples_indexes]
    return ids


# **2) Create Magic features**


def get_magic_features():
    d = dict()
    for i in range(200):
        d['var_{}'.format(i)] = np.str

    train = pd.read_csv(INPUT_PATH + 'train.csv', dtype=d)
    test = pd.read_csv(INPUT_PATH + 'test.csv', dtype=d)

    real_ids = get_real_test_data()
    test = test[test['ID_code'].isin(real_ids)]
    print(len(test))
    features = sorted(list(d.keys()))
    print(train.shape, test.shape)
    table = pd.concat((train[['ID_code'] + features], test[['ID_code'] + features]), axis=0)
    print(table.shape)
    feat_to_store = []
    for f in features:
        print('Go {}'.format(f))
        new_feat = []
        v = dict(pd.value_counts(table[f]))
        for el in table[f].values:
            new_feat.append(v[el])
        table[f + '_counts_sum'] = new_feat
        feat_to_store.append(f + '_counts_sum')

    train = table[['ID_code'] + feat_to_store][:200000]
    test = table[['ID_code'] + feat_to_store][200000:]
    train.to_csv(FEATURES_PATH + 'counts_of_values_train.csv', index=False)
    test.to_csv(FEATURES_PATH + 'counts_of_values_test.csv', index=False)
    return train, test


# 3) Read data with magic features and create additional useful features var_N **mul** magic_N and var_N **div** magic_N


train = read_train()
test = read_test()

t1, t2 = get_magic_features()
train = train.merge(t1, on='ID_code', how='left')
test = test.merge(t2, on='ID_code', how='left')
f_add = list(t1.columns.values)
f_add.remove('ID_code')
test.fillna(-1, inplace=True)

for i in range(200):
    train['var_{}_mul'.format(i)] = train['var_{}'.format(i)] * train['var_{}_counts_sum'.format(i)]
    test['var_{}_mul'.format(i)] = test['var_{}'.format(i)] * test['var_{}_counts_sum'.format(i)]
    train['var_{}_div'.format(i)] = train['var_{}'.format(i)] / train['var_{}_counts_sum'.format(i)]
    test['var_{}_div'.format(i)] = test['var_{}'.format(i)] / test['var_{}_counts_sum'.format(i)]

features = []
for i in range(200):
    features.append('var_{}'.format(i))
    features.append('var_{}_counts_sum'.format(i))
    features.append('var_{}_mul'.format(i))
    features.append('var_{}_div'.format(i))
print('Features: [{}] {}'.format(len(features), features))


# 4) Create model
# 
# 5) Use random shuffle of columns during training


def get_keras_model(input_features):
    from keras.models import Model
    from keras.layers import Input, Dense, BatchNormalization, Conv1D, Reshape, Flatten, MaxPooling1D, Concatenate
    from keras.layers.core import Activation, Dropout, Lambda
    from keras.layers.merge import concatenate

    inp = Input(shape=(input_features, 1))
    x = BatchNormalization(axis=-2)(inp)
    x = Dense(128, activation='relu')(x)
    x = Flatten()(x)
    preds = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inp, outputs=preds)
    return model


def get_kfold_split(folds_number, len_train, target, random_state):
    train_index = list(range(len_train))
    folds = StratifiedKFold(n_splits=folds_number, shuffle=True, random_state=random_state)
    ret = []
    for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train_index, target)):
        ret.append([trn_idx, val_idx])
    return ret


def get_model_memory_usage(batch_size, model):
    import numpy as np
    from keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    number_size = 4.0
    if K.floatx() == 'float16':
         number_size = 2.0
    if K.floatx() == 'float64':
         number_size = 8.0

    total_memory = number_size*(batch_size*shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes


class ModelCheckpoint_AUC(Callback):
    """Save the model after every epoch. """

    def __init__(self, filepath, filepath_static, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='max', period=1, patience=None, validation_data=()):
        super(ModelCheckpoint_AUC, self).__init__()
        self.interval = period
        self.X_val, self.y_val, self.batch_size = validation_data
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.filepath_static = filepath_static
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self.monitor_op = np.greater
        self.best = -np.Inf

        # part for early stopping
        self.epochs_from_best_model = 0
        self.patience = patience

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0

            y_pred = self.model.predict(self.X_val, verbose=0, batch_size=self.batch_size)
            score = roc_auc_score(self.y_val, y_pred)
            filepath = self.filepath.format(epoch=epoch + 1, score=score, **logs)
            print("AUC score: {:.6f}".format(score))
            if score > self.best:
                self.epochs_from_best_model = 0
            else:
                self.epochs_from_best_model += 1

            if self.save_best_only:
                current = score
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                        shutil.copy(filepath, self.filepath_static)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve' %
                                  (epoch + 1, self.monitor))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)
                shutil.copy(filepath, self.filepath_static)

            if self.patience is not None:
                if self.epochs_from_best_model > self.patience:
                    print('Early stopping: {}'.format(self.epochs_from_best_model))
                    self.model.stop_training = True


def batch_generator_train_random_sample(X, y, batch_size):
    rng = list(range(X.shape[0]))
    feature_group_number = 4

    while True:
        index1 = random.sample(rng, batch_size)
        input1 = X[index1, :].copy()
        output1 = y[index1].copy()

        input1_0 = input1[output1 == 0, :]
        input1_1 = input1[output1 == 1, :]
        output1_0 = output1[output1 == 0]
        output1_1 = output1[output1 == 1]

        for i in range(0, input1.shape[1], feature_group_number):
            index = np.arange(0, input1_0.shape[0])
            np.random.shuffle(index)
            for j in range(feature_group_number):
                input1_0[:, i + j] = input1_0[index, i + j]

            index = np.arange(0, input1_1.shape[0])
            np.random.shuffle(index)
            for j in range(feature_group_number):
                input1_1[:, i + j] = input1_1[index, i + j]

        input1 = np.concatenate((input1_0, input1_1), axis=0)
        output1 = np.concatenate((output1_0, output1_1), axis=0)
        input1 = np.expand_dims(input1, axis=2)
        yield input1, output1
                    

from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam

target = train['target'].values
overall_train_predictions = np.zeros(target.shape[0], dtype=np.float64)
model_list = []
all_splits = []
num_fold = 0
ret = get_kfold_split(5, train.shape[0], target, 890)

for train_index, valid_index in ret:
    num_fold += 1
    print('Start fold {}'.format(num_fold))
    X_train = train.loc[train_index].copy()
    X_valid = train.loc[valid_index].copy()
    y_train = target[train_index]
    y_valid = target[valid_index]

    print('Train data:', X_train.shape, y_train.shape)
    print('Valid data:', X_valid.shape, y_valid.shape)

    X_train_matrix = X_train[features].values
    X_valid_matrix = X_valid[features].values

    print('Train data:', X_train_matrix.shape, y_train.shape)
    print('Valid data:', X_valid_matrix.shape, y_valid.shape)
    print('Valid sum: {}'.format(y_valid.sum()))

    optim_name = 'Adam'
    batch_size_train = 1000
    batch_size_valid = 1000
    learning_rate = 0.002
    epochs = 50
    patience = 5
    print('Batch size: {}'.format(batch_size_train))
    print('Learning rate: {}'.format(learning_rate))
    steps_per_epoch = (X_train.shape[0] // batch_size_train)
    validation_steps = 1
    print('Steps train: {}, Steps valid: {}'.format(steps_per_epoch, validation_steps))

    final_model_path = MODELS_PATH_KERAS + '{}_fold_{}.h5'.format('keras', num_fold)
    cache_model_path = MODELS_PATH_KERAS + '{}_temp_fold_{}.h5'.format('keras', num_fold)
    cache_model_path_auc = MODELS_PATH_KERAS + '{}_temp_fold_{}'.format('keras', num_fold) + '_{score:.4f}.h5'

    model = get_keras_model(X_train_matrix.shape[1])
    print(model.summary())
    print('Model memory usage: {} GB'.format(get_model_memory_usage(batch_size_train, model)))

    optim = Adam(lr=learning_rate)
    model.compile(optimizer=optim, loss='binary_crossentropy', metrics=['accuracy'])

    callbacks = [
        # EarlyStopping(monitor='val_loss', patience=patience, verbose=0),
        ModelCheckpoint_AUC(cache_model_path_auc, cache_model_path,
                            validation_data=(np.expand_dims(X_valid_matrix, axis=2), y_valid, batch_size_valid),
                            save_best_only=True,
                            verbose=0,
                            patience=patience),
        # ModelCheckpoint(cache_model_path, save_best_only=False)
        ReduceLROnPlateau(monitor='loss', factor=0.95, patience=5, min_lr=1e-9, min_delta=0.00001,
                          verbose=1, mode='min'),
    ]

    history = model.fit_generator(generator=batch_generator_train_random_sample(X_train_matrix, y_train, batch_size_train),
                                  epochs=epochs,
                                  steps_per_epoch=steps_per_epoch,
                                  validation_data=batch_generator_train_random_sample(X_valid_matrix, y_valid, batch_size_valid),
                                  validation_steps=validation_steps,
                                  verbose=2,
                                  max_queue_size=10,
                                  # class_weight=class_weight,
                                  callbacks=callbacks)

    min_loss = min(history.history['val_loss'])
    print('Minimum loss for given fold: ', min_loss)
    model.load_weights(cache_model_path)
    model.save(final_model_path)

    pred = model.predict(np.expand_dims(X_valid[features].values, axis=2))
    overall_train_predictions[valid_index] += pred[:, 0]
    score = roc_auc_score(y_valid, pred[:, 0])
    print('Fold {} score: {:.6f}'.format(num_fold, score))
    model_list.append(model)
    all_splits.append(ret)

score = roc_auc_score(target, overall_train_predictions)
print('Total AUC score: {:.6f}'.format(score))

train['target'] = overall_train_predictions
train[['ID_code', 'target']].to_csv(SUBM_PATH + 'train_auc_{}.csv'.format(score), index=False, float_format='%.8f')


# **6) Predict on test**


def predict_with_keras_model(test, features, model_list):
    full_preds = []
    for m in model_list:
        preds = m.predict(np.expand_dims(test[features].values, axis=2), batch_size=1000)
        full_preds.append(preds)
    preds = np.array(full_preds).mean(axis=0)
    return preds

overall_test_predictions = predict_with_keras_model(test, features, model_list)
test['target'] = overall_test_predictions
test[['ID_code', 'target']].to_csv(SUBM_PATH + 'test_auc_{}.csv'.format(score), index=False, float_format='%.8f')

