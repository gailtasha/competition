# <h1 align=center><font size = 4>Santander Customer Transaction Prediction</font></h1>
# <h1 align=center><font size = 5>Model binary classifier on imbalanced data</font></h1>


# # Table of Contents
# * [Introduction/Business Problem](#introduction)
# * [Setup](#setup)
# * [Data processing and exploration](#prep)
# * [Model training and evaluation](#modeling)
# * [Submission](#predictions)


# <a id = "introduction"></a>
# # Introduction/Business Problem


# At [Santander](https://www.santanderbank.com/us/personal) our mission is to help people and businesses prosper. We are always looking for ways to help our customers understand their financial health and identify which products and services might help them achieve their monetary goals.
# 
# Our data science team is continually challenging our machine learning algorithms, working with the global data science community to make sure we can more accurately identify new ways to solve our most common challenge, binary classification problems such as: is a customer satisfied? Will a customer buy this product? Can a customer pay this loan?
# 
# In this challenge, we invite Kagglers to help us identify which customers will make a specific transaction in the future, irrespective of the amount of money transacted. The data provided for this competition has the same structure as the real data we have available to solve this problem.


# <a id="setup"></a>
# # Setup


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
import tensorflow as tf
print(tf.__version__)
from tensorflow import keras
import sklearn

import seaborn as sns

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot figures
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['figure.figsize'] = (12, 8)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

print('Libraries imported.')


# <a id="prep"></a>
# # Data processing and exploration


# We are provided with an anonymized dataset containing numeric feature variables, the binary target column, and a string ID_code column.
# 
# The task is to predict the value of target column in the test set.
# 
# **File descriptions**
# * **train.csv** - the training set.
# * **test.csv** - the test set. The test set contains some rows which are not included in scoring.
# * **sample_submission.csv** - a sample submission file in the correct format.


transactions = pd.read_csv('../input/train.csv')
print('train data imported.')
transactions.head()


# The data set contains numeric features type `float64`, besides the binary target column type `int64` and the ID_code column type `object`


print('number of unique ids {}'.format(len(transactions.ID_code.unique())))
print('number of rows {}'.format(len(transactions)))
print('number of columns {}'.format(len(transactions.columns)))
print('missing values: {}'.format(transactions.isna().any().any()))


# Each row represents one transaction. There are 200000 instances in the dataset and the id per transaction is unique as expected. The data has different type of attributes. The majority of the attributes is type numerical. In addition, the dataset contains the target and the ID_code attributes. There are no `NaN` in the transaction dataset.


transactions.describe()


# ### Examine the class label imbalance


neg, pos = np.bincount(transactions['target'])
total = neg + pos
print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(total, pos, 100 * pos / total))


plt.figure(figsize=(6, 6))
sns.countplot(x=transactions.target)
plt.title('Class imbalance')
plt.show()


# From the total number of transactions 10% belongs to 1 (the mean in the `target` attribute), so the classes are imbalanced and accuracy won't be a reasonable metric to evaluate our model


# ### Clean, split and normalize the data


transactions.drop(['ID_code'], axis=1, inplace=True)
transactions.head()


nrows = 2
ncols = 2

fig = plt.gcf()
fig.set_size_inches(ncols*7, nrows*6)

for i, stat in enumerate(['mean', 'std', 'max', 'min']):
    ax = plt.subplot(nrows, ncols, i + 1)
    sns.histplot(transactions.drop(['target'], axis=1).describe().loc[stat], ax= ax)
    ax.set(xlabel=None)
    plt.title('Distribution of numeric features {} values'.format(stat.upper()))
plt.show()


# The above analysis indicates that the numerical features have different scales, thus feature scaling should be considered in ML algorithms.


# Split the dataset into train, validation, and test sets. The validation set is used during the model fitting to evaluate the loss and any metrics, however the model is not fit with this data. The test set is completely unused during the training phase and is only used at the end to evaluate how well the model generalizes to new data. This is especially important with imbalanced datasets where overfitting is a significant concern from the lack of training data.


from sklearn.model_selection import train_test_split

# Use a utility from sklearn to split and shuffle your dataset.
train_df, test_df = train_test_split(transactions, test_size=0.2, stratify=transactions.target)
train_df, valid_df = train_test_split(train_df, test_size=0.2, stratify=train_df.target)


X_train, y_train = train_df.drop(['target'], axis=1).values, train_df.target.values
X_valid, y_valid = valid_df.drop(['target'], axis=1).values, valid_df.target.values
X_test, y_test = test_df.drop(['target'], axis=1).values, test_df.target.values

bool_train_labels = y_train != 0

print('Train features shape: {}'.format(X_train.shape))
print('Valid features shape: {}'.format(X_valid.shape))
print('Test features shape: {}'.format(X_test.shape))

print('\nTrain labels shape: {}'.format(y_train.shape))
print('Valid labels shape: {}'.format(y_valid.shape))
print('Test labels shape: {}'.format(y_test.shape))


# Normalize the input features using the sklearn StandardScaler. This will set the mean to 0 and standard deviation to 1.


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)


# ### Look at the data distribution


pos_df = pd.DataFrame(X_train[bool_train_labels], columns=train_df.drop(['target'], axis=1).columns)
neg_df = pd.DataFrame(X_train[~bool_train_labels], columns=train_df.drop(['target'], axis=1).columns)

sns.jointplot(x=pos_df['var_5'], y=pos_df['var_6'], kind='hex')
plt.suptitle("Positive distribution")

sns.jointplot(x=neg_df['var_5'], y=neg_df['var_6'], kind='hex')
_ = plt.suptitle("Negative distribution")


# <a id="modeling"></a>
# # Model training and evaluation


# ## Define the model and metrics
# Create a simple neural network with two densly connected hidden layers, dropout layers to reduce overfitting, and an output sigmoid layer that returns the probability of a customer being target:


METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]

def build_model(metrics=METRICS, output_bias=None):
    if output_bias is not None:
        output_bias = keras.initializers.Constant(output_bias)

    model = keras.models.Sequential([
        keras.layers.Dense(100, activation='relu', input_shape=[X_train.shape[-1]]),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(50, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias)
    ])
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss=keras.losses.BinaryCrossentropy(),metrics=metrics)
    
    return model


# ### Understanding useful metrics
# 
# Notice that there are a few metrics defined above that can be computed by the model that will be helpful when evaluating the performance.
# 
# 
# *   **False** negatives and **false** positives are samples that were **incorrectly** classified
# *   **True** negatives and **true** positives are samples that were **correctly** classified
# *   **Accuracy** is the percentage of examples correctly classified
# >   $\frac{\text{true samples}}{\text{total samples}}$
# *   **Precision** is the percentage of **predicted** positives that were correctly classified
# >   $\frac{\text{true positives}}{\text{true positives + false positives}}$
# *   **Recall** is the percentage of **actual** positives that were correctly classified
# >   $\frac{\text{true positives}}{\text{true positives + false negatives}}$
# *   **AUC** refers to the Area Under the Curve of a Receiver Operating Characteristic curve (ROC-AUC). This metric is equal to the probability that a classifier will rank a random positive sample higher than a random negative sample.
# *   **AUPRC** refers to Area Under the Curve of the Precision-Recall Curve. This metric computes precision-recall pairs for different probability thresholds. 


# ## Baseline Model
# 
# ### Build the model
# 
# Now create and train your model using the function that was defined earlier. Notice that the model is fit using a larger than default batch size of 2048, this is important to ensure that each batch has a decent chance of containing a few positive samples. If the batch size was too small, they would likely have no positive customers to learn from.
# 
# **Set the correct initial bias**
# 
# These initial guesses are not great. We know the dataset is imbalanced. Set the output layer's bias to reflect that. The correct bias to set can be derived from:
# 
# $$ p_0 = pos/(pos + neg) = 1/(1+e^{-b_0}) $$
# $$ b_0 = -log_e(1/p_0 - 1) $$
# $$ b_0 = log_e(pos/neg)$$


EPOCHS = 100
BATCH_SIZE = 2048

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_prc', 
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True)

# calculate initial bias
initial_bias = np.log([pos/neg])

baseline_model = build_model(output_bias=initial_bias)
baseline_model.summary()


baseline_history = baseline_model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[early_stopping], validation_data=(X_valid, y_valid))


# ### Model Evaluation
# 
# **Check training history**
# 
# I will produce plots of your model's accuracy and loss on the training and validation set. These are useful to check for overfitting.


def plot_metrics(history):
    metrics = ['loss', 'prc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(2,2,n+1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric], color=colors[0], linestyle="--", label='Valid')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':      
            plt.ylim([0.8,1])
        else:
            plt.ylim([0,1])
        plt.legend()

plot_metrics(baseline_history)


# **Evaluate metrics**
# 
# I use a confusion matrix to summarize the actual vs. predicted labels, where the X axis is the predicted label and the Y axis is the actual label:


train_predictions_baseline = baseline_model.predict(X_train, batch_size=BATCH_SIZE)
test_predictions_baseline = baseline_model.predict(X_test, batch_size=BATCH_SIZE)


from sklearn.metrics import confusion_matrix

def plot_cm(labels, predictions, p=0.5):
    cm = confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    
    print('Non-Target Customers Detected (True Negatives): ', cm[0][0])
    print('Non-Target Customers Incorrectly Detected (False Positives): ', cm[0][1])
    print('Target Customers Missed (False Negatives): ', cm[1][0])
    print('Target Customers Detected (True Positives): ', cm[1][1])
    print('Total Target Customers: ', np.sum(cm[1]))


baseline_results = baseline_model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=0)

for name, value in zip(baseline_model.metrics_names, baseline_results):
    print(name, ': ', value)
print()

plot_cm(y_test, test_predictions_baseline)


# ### Plot the ROC
# 
# This plot is useful because it shows, at a glance, the range of performance the model can reach just by tuning the output threshold.


def plot_roc(name, labels, predictions, **kwargs):
    fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)
    plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')
    
plot_roc("Train Baseline", y_train, train_predictions_baseline, color=colors[0])
plot_roc("Test Baseline", y_test, test_predictions_baseline, color=colors[0], linestyle='--')
plt.legend(loc='lower right')
plt.show()


# ## Class weights
# 
# ### Calculate class weights
# 
# The goal is to identify fraudulent transactions, but you don't have very many of those positive samples to work with, so you would want to have the classifier heavily weight the few examples that are available. You can do this by passing Keras weights for each class through a parameter. These will cause the model to "pay more attention" to examples from an under-represented class.


# Scaling by total/2 helps keep the loss to a similar magnitude.
# The sum of the weights of all examples stays the same.
weight_for_0 = (1 / neg) * (total / 2.0)
weight_for_1 = (1 / pos) * (total / 2.0)

class_weight = {0: weight_for_0, 1: weight_for_1}

print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1))


# ### Train a model with class weights
# 
# Now try re-training and evaluating the model with class weights to see how that affects the predictions.


weighted_model = build_model(output_bias=initial_bias)

weighted_history = weighted_model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[early_stopping], 
                                      validation_data=(X_valid, y_valid),
                                      class_weight=class_weight)


# ### Check training history


plot_metrics(weighted_history)


# ### Evaluate metrics


train_predictions_weighted = weighted_model.predict(X_train, batch_size=BATCH_SIZE)
test_predictions_weighted = weighted_model.predict(X_test, batch_size=BATCH_SIZE)


weighted_results = weighted_model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=0)
for name, value in zip(weighted_model.metrics_names, weighted_results):
    print(name, ': ', value)
print()
plot_cm(y_test, test_predictions_weighted)


# ### Plot the ROC


plot_roc("Train Baseline", y_train, train_predictions_baseline, color=colors[0])
plot_roc("Test Baseline", y_test, test_predictions_baseline, color=colors[0], linestyle='--')

plot_roc("Train Weighted", y_train, train_predictions_weighted, color=colors[1])
plot_roc("Test Weighted", y_test, test_predictions_weighted, color=colors[1], linestyle='--')

plt.legend(loc='lower right')
plt.show()


# ## Oversampling
# 
# ### Oversample the minority class using numpy
# 
# A related approach would be to resample the dataset by oversampling the minority class.


pos_features = X_train[bool_train_labels]
neg_features = X_train[~bool_train_labels]

pos_labels = y_train[bool_train_labels]
neg_labels = y_train[~bool_train_labels]

print('Positive features shape: {}'.format(pos_features.shape))
print('Negative features shape: {}'.format(neg_features.shape))

print('\nPositive labels shape: {}'.format(pos_labels.shape))
print('Negative labels shape: {}'.format(neg_labels.shape))


ids = np.arange(len(pos_features))
choices = np.random.choice(ids, len(neg_features))

res_pos_features = pos_features[choices]
res_pos_labels = pos_labels[choices]

print('Resampled Positive features shape: {}'.format(res_pos_features.shape))
print('Resampled Positive labels shape: {}'.format(res_pos_labels.shape))


resampled_features = np.concatenate([res_pos_features, neg_features], axis=0)
resampled_labels = np.concatenate([res_pos_labels, neg_labels], axis=0)

order = np.arange(len(resampled_labels))
np.random.shuffle(order)
resampled_features = resampled_features[order]
resampled_labels = resampled_labels[order]

resampled_features.shape


# ### Train on the oversampled data
# 
# Now try training the model with the resampled data set instead of using class weights to see how these methods compare.


resampled_model = build_model(output_bias=[0])

resampled_history = resampled_model.fit(resampled_features, resampled_labels, batch_size=BATCH_SIZE, 
                                        epochs=EPOCHS, callbacks=[early_stopping], validation_data=(X_valid, y_valid))


# ### Check training history


plot_metrics(resampled_history)


# ### Evaluate metrics


train_predictions_resampled = resampled_model.predict(X_train, batch_size=BATCH_SIZE)
test_predictions_resampled = resampled_model.predict(X_test, batch_size=BATCH_SIZE)


resampled_results = resampled_model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=0)
for name, value in zip(resampled_model.metrics_names, resampled_results):
    print(name, ': ', value)
print()

plot_cm(y_test, test_predictions_resampled)


# ### Plot the ROC


plot_roc("Train Baseline", y_train, train_predictions_baseline, color=colors[0])
plot_roc("Test Baseline", y_test, test_predictions_baseline, color=colors[0], linestyle='--')

plot_roc("Train Weighted", y_train, train_predictions_weighted, color=colors[1])
plot_roc("Test Weighted", y_test, test_predictions_weighted, color=colors[1], linestyle='--')

plot_roc("Train Resampled", y_train, train_predictions_resampled, color=colors[2])
plot_roc("Test Resampled", y_test, test_predictions_resampled, color=colors[2], linestyle='--')
plt.legend(loc='lower right')
plt.show()


# <a id="predictions"></a>
# # Submission


test_df = pd.read_csv('../input/test.csv')
print('test data imported.')
print(test_df.shape)
test_df.head()


X_test = test_df.drop(['ID_code'], axis=1)
X_test = scaler.transform(X_test)
print(X_test.shape)


test_df['target'] = (weighted_model.predict(X_test, batch_size=BATCH_SIZE) > 0.5).astype(int)
test_df.head()


test_df[['ID_code', 'target']].to_csv('submission.csv', index=False)


# # Reference
# 
# [Tensorflow tutorial | imbalanced data](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data)



