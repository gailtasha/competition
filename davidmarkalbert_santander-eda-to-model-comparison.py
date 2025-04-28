# <h1 id="tocheading">Table of Contents</h1>
# <div id="toc"></div>


# This notebook is published as part of a coursera course challenge assignment https://www.coursera.org/learn/advanced-data-science-capstone/home/welcome


# Run the following cell to generate the Table of Contents


%%javascript
$.getScript('https://kmahelona.github.io/ipython_notebook_goodies/ipython_notebook_toc.js')


# # Imports and Dependencies


import sys, os

from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
import plotly.figure_factory as ff
init_notebook_mode()

# use plotly cufflinks
import plotly.tools as tls
tls.embed('https://plot.ly/~cufflinks/8')
import cufflinks as cf
# ensure offline mode
cf.go_offline()
cf.set_config_file(world_readable=False,offline=True, theme='ggplot')

from scipy.stats import spearmanr
import numpy as np
import pandas as pd
from pandas_summary import DataFrameSummary

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import StratifiedKFold, train_test_split


import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers

from keras import backend as K
from keras import regularizers
from keras.layers import Dropout
from keras.constraints import max_norm
from keras.wrappers.scikit_learn import KerasClassifier


# # Load Data


id_col = 'ID_code'
target_col = 'target'
df_play = pd.read_csv('../input/train.csv', index_col=id_col, low_memory=False)
df_comp = pd.read_csv('../input/test.csv', index_col=id_col, low_memory=False)


# The competition test data set in `test.csv` is loaded into `df_comp`. This data will be held aside until after the EDA and then be used for a submission file.
# 
# Then we can split the training data, in `train.csv`, into actual training and validation data.


# ### Split data into local train and local test sets


# We shuffle and split training and test sets and are currently taking only a 10% stratified sample for both train and test sets.
# 
# This train size can be increased after model improvements are made in order to improve overall score.


train_df, test_df = train_test_split(df_play, test_size=.1, train_size=.1, stratify=df_play.target, shuffle=True, random_state=0)


# ### Separate features from targert variable


# Here we create some shorthand convenience variables for later use in model exploration and comparision.


# prepare training and validation dataset
X = train_df.drop(target_col, axis=1)
y = train_df[target_col]
X_val = test_df.drop(target_col, axis=1)
y_val = test_df[target_col]


# # Data Quality Assessment


# limit the columns that are returned from summarize
# restricted to numeric by difference of values in compare_dataframes 
main_cols = ['std', 'min', 'mean', 'max', 'counts', 'missing', 'uniques']


def summarize(df, sample_size=0):
    "sumamrize a dataframe for quality assesment"
    dtypes = pd.DataFrame(df.dtypes, columns=['dtype'])
    stats = DataFrameSummary(df).summary().T
    summary = dtypes.merge(stats, left_index=True, right_index=True)
    summary = summary.merge(dtypes.rename({'dtype':'dtype2'}, axis=1), left_index=True, right_index=True).rename({'dtype':'dtype1'}, axis=1).sort_values('dtype1')
    if sample_size:
        samples = df.sample(sample_size).T
        summary = samples.merge(summary, left_index=True, right_index=True).rename({'dtype':'dtype1'}, axis=1).sort_values('dtype1')
        return summary 
    else:
        return summary[main_cols]
    
def display_all(df):
    "display the entirity of a dataframe in the cell with scroll bars"
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000):
        display(df)


def compare_dataframes(train_df, test_df, target_col):
    "compare the summaries for 2 dataframes"
    # make summaries for the train and test data sets and join them together
    test_summary = summarize(test_df)
    train_summary = summarize(train_df.drop(target_col, axis=1))
    summary = train_summary.merge(test_summary, left_index=True, right_index=True, suffixes=('_train', '_test'))
    # take the differnce of summary values and make a dataframe of them and return it 
    train_test_diff_df = pd.DataFrame(test_summary.values - train_summary.values, index=test_summary.index, columns=[c + '_diff' for c in main_cols])
    summary = summary.merge(train_test_diff_df, left_index=True, right_index=True)
    return summary


# Using `pandas_summary` we can get an easy idea of our dataset's null-count along with other standard descriptive statisics. The `summarize` function collects this info along with the pandas dtypes and possibly some samples if intersted. `compare_dataframes` uses summarize on both and merges the summaries into a single dataframe. It also shows the difference in value for each statistic and feature in a third column.   


summary = compare_dataframes(df_play, df_comp, target_col)


# Using `display_all` we can investigate the full dataframe. It can be seen already from here that there are not any missing values in any column. Also the difference in the given datasets `train.csv` and `test.csv` do not seem to be anything remarkable.   


display_all(summary.sort_index(axis=1, ascending=False).sort_values('std_diff', ascending=False))


# Here we can choose a column to inspect more closely in the next two cells. This should be a column name from the summary dataframe (as opposed to the feature column names).


inspect_col = 'std_train'
summary[inspect_col].iplot(kind='hist', bins=100, title=f'Frequency Histogram for {inspect_col}', 
                          yTitle=f'Number of times value appeared', xTitle=f'Value for {inspect_col}')


hist_data = [list(summary[inspect_col].values - summary[inspect_col].values.mean())]
labels = [inspect_col]    

fig = ff.create_distplot(hist_data, labels)

# update the plot titles
fig.layout.xaxis.update(title=f'Value for {inspect_col}')
fig.layout.yaxis.update(title=f'Probability that value appeared')
fig.layout.update(title=f'Distribution Plot for {inspect_col}');

iplot(fig)


# # Data Exploration


# To check more in depth than the general descriptive statistics above here we can consider column correlation using Spearman's rank correlation.
# 
# We can build a correlation matrix here and check the max values for each feature.


corr = np.round(spearmanr(train_df).correlation, 4)
df_corr = pd.DataFrame(data=corr, index=train_df.columns, columns=train_df.columns)


# The next transforms the matrix to remove duplicate info in the bottom left half of the matrix and also ends with a singele series for the correlataion value and a multiindex for the two features it arises from.   


keep = np.triu(np.ones(df_corr.shape)).astype('bool').reshape(df_corr.size)
c = df_corr.stack()[keep]


# Then we can remove the rows showing correlation between a feature and itself. Also we can remove the correlation between features and the target variable.


c = c.loc[c.index.get_level_values(1)!=c.index.get_level_values(0),]
c = c.loc[c.index.get_level_values(0)!='target',]


# To look at the top N most correlated values


N_corr = 20
c.sort_values()[-N_corr:]


# or the most negatively correlated values.


c.sort_values()[:N_corr]


# All values are quite low and do not require specific treatment at this point.


# # Data Visualization


def dist_plots(var_name='var_1', sample_size=5000):
    "Make a distribution plot for a single variable from the dataset"
    hist_data = [df_play[var_name].sample(sample_size).values, df_comp[var_name].sample(sample_size).values]
    group_labels = ['train', 'test']
    fig = ff.create_distplot(hist_data, group_labels, show_hist=False, show_rug=False)
    return fig


# Here we can make distribution plots for a number of variables in a single figure. I find batches of 25 to be manageable to look at once. 
# 
# The following code will show 25 variables from the dataset at time, beginning from `var_N` where `N` is the `offset` value defined in the next variable assignment. 


offset = 50
plots = [dist_plots(f'var_{i+offset}') for i in range(25)]

for ix, plot in enumerate(plots, 1):
    plot.layout.update(title=f'var {ix+offset}')
    for trace in plot.data:
        trace.showlegend = False


# From this we can visual inspect the difference in distributions for our trainable dataset version the competition dataset.


iplot(cf.subplots(plots, shape=(5, 5), 
                  subplot_titles=[f'var_{i+offset}' for i in range(25)]))


# # Feature Engineering


# Without any missing values for any features we do not need any kind of imputation. We can perfrom standard scaling on the features to see if that improves the model's performance. 


# #### Scale Features


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_sc = sc.fit_transform(X)
X_val_sc = sc.transform(X_val)


# # Model Performance Indicator


# The score function used in the competiton is the Area Under the Reciever Operator Characteristic Curve.
# 
# After training each model to compare, we can save the roc plot data in the folowing `roc_data` variable in order to plot all results together at the end.


roc_data = {}
roc_auc_scores = {}
prediction_df = y_val.to_frame('ground_truth')


def make_roc_fig(roc_data=None):
    "Takes a list of roc_curve results and plots each of the items on the same figure"
    if not roc_data: roc_data = {}
    data = []
    # plot the line for random chance comparison
    trace = go.Scatter(x=[0, 1], y=[0, 1], 
                       mode='lines', 
                       line=dict(width=2, color='black', dash='dash'),
                       name='Luck')
    data.append(trace)
    # plot each of the roc curves given in arg
    for clf_name, roc in roc_data.items():
        fpr, tpr, thresholds = roc
        roc_auc = auc(fpr, tpr)
        trace = go.Scatter(x=fpr, y=tpr, 
                           mode='lines', 
                           line=dict(width=2),
                           name=f'{clf_name} ROC AUC (area = {roc_auc:0.2f})')
        data.append(trace)
    # add layout
    layout = go.Layout(title='Receiver Operating Characteristic',
                       xaxis=dict(title='False Positive Rate', showgrid=False,
                                  range=[-0.05, 1.05]),
                       yaxis=dict(title='True Positive Rate', showgrid=False,
                                  range=[-0.05, 1.05]))
    # create fig then return
    fig = go.Figure(data=data, layout=layout)
    return fig

def score_model(clf_name ,y_pred):
    "collect data from the models prediction for final analysis and model comparison. Return the roc curve data for immediate plotting"
    # Make predictions and add to df for final summary
    prediction_df[clf_name] = y_pred
    # Store score for final judegment
    score = roc_auc_score(y_val, y_pred)
    roc_auc_scores[clf_name] = score
    # Make the ROCs for plotting
    roc = roc_curve(y_val, y_pred)
    roc_data[clf_name] = roc
    print(f'The {clf_name} model has ROC AUC: {score}')
    return roc


# # Model Building


# Here is a comparison of two models, Random Forest and a simple DNN Feed Forward Network. Most of the ways to improve the models have yet to be implemented, but they still have a reasonably good skill. Examples of such are otpimizing other hyperparameters, experiment with these ones further, training the models on the full dataset etc...
# 
# For comparison the model is fitted to both the scaled and unscaled data and the Receiver Operator Characteristic curve will be plotted. Afterwards all curves are shown on a single plot and models are judged by the sum of the area under its ROC curve, calculated with `roc_auc_score` and `dnn_auc`.


# ### Random Forest


rf_param = {
 'min_samples_leaf': 10,
 'max_features': .5,
 'n_estimators': 100}


rfm = RandomForestClassifier(**rf_param, n_jobs=-1, random_state=0)
rfm.fit(X, y)


clf_name = 'rf'

y_pred = rfm.predict_proba(X_val)[:,1]

roc = score_model(clf_name, y_pred.tolist())

iplot(make_roc_fig({clf_name: roc}))


# #### Fit and score the scaled values


rfm = RandomForestClassifier(**rf_param, n_jobs=-1, random_state=0)
rfm.fit(X_sc, y)


clf_name = 'rf_sc'

y_pred = rfm.predict_proba(X_val_sc)[:,1]

roc = score_model(clf_name, y_pred)

iplot(make_roc_fig({clf_name: roc}))


# ## Deep Learning Model


def dnn_auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


def create_dnn():
    model = Sequential()
    model.add(Dense(200, input_dim=200, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[dnn_auc])
    return model

model = create_dnn()


model.fit(X, y, batch_size = 10000, epochs = 200, validation_data = (X_val, y_val), )


clf_name = 'dnn'

y_pred = model.predict(X_val)

roc = score_model(clf_name, y_pred)

iplot(make_roc_fig({clf_name: roc}))


# #### Fit the model with the scaled data


model = create_dnn()
model.fit(X_sc, y, batch_size = 10000, epochs = 200, validation_data = (X_val_sc, y_val), )


clf_name = 'dnn_sc'

y_pred = model.predict(X_val_sc)

roc = score_model(clf_name, y_pred)

iplot(make_roc_fig({clf_name: roc}))


# # Model Performance


# Now we can plot the four curves altogether and see which has the best score.


iplot(make_roc_fig(roc_data))


# We can see see that scaling the features has increased NN's score, whereas it stayed exactly the same for the random forest. This is the expected behavior for each of the models, and the both have given OK results for a start.


# There is lots to be done to improve these models and explore other algorithms, but we can see here  that `dnn_sc`, the simple DNN with feature scaling, seems to have the best curve out of these models, but it ties to the Random Forest overall (under current conditions) with a score of `0.81`. 


# # References


# This notebook is part of a final challenge assignment for the following course.


# https://www.coursera.org/learn/advanced-data-science-capstone/home/welcome


# While working through completing this notebook I found the folling pages very useful.


# https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
# 
# https://machinelearningmastery.com/binary-classification-tutorial-with-the-keras-deep-learning-library/

