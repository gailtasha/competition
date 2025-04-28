# **Santander Customer Transaction Prediction  
# Can you identify who will make a transaction? **    
# ![](https://bit.ly/2BJideW)  
# Santander inivte fellow Kagglers to help them identify which customers will make a specific transaction in the future, irrespective of the amount of money transacted. The data provided for this competition has the same structure as the real data they have available to solve this problem.


# **Analysis Playground**  
# As in every data science prediction problem, I will start with Exploratory Data Analysis (EDA) and move on building model on different machine learning algorithms.


# **Loading the required packages for analysis**


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import ensemble
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings("ignore")
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
import copy

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, LabelEncoder


# **Reading the Training and Testing Dataset**


%%time
sctp_train = pd.read_csv('../input/train.csv')
sctp_test = pd.read_csv('../input/test.csv')
print("=="*45)
print("The training dataset has {0} rows and {1} columns".format(sctp_train.shape[0], sctp_train.shape[1]))
print("The testing dataset has {0} rows and {1} columns".format(sctp_test.shape[0], sctp_test.shape[1]))
print("=="*45)


# This is really odd, as I have never come across a scenario where both the training and testing dataset have the same number of rows. Seems interesting. Here the number of features are bit higer in number. So, we will find which all variables are important based on missing values, correlation analysis etc.


# **Information on the training dataset**


sctp_train.info()


# .info() command in python give the brief glimpse of the dataset. Our traning dataset has three different types of datatypes. most of them are float which are contineous, one feature has integer which is most probably the "Target" column and the final one feature is of object type which i think will be the "ID_code" column.


# **Summary Statistics**


sctp_train.describe()


# **Target Distribution**  
# First let us look at the distribution of the target variable to understand whether the dataset is imbalanced or not.


## target Proportion ##
cnt = sctp_train['target'].value_counts()
tr_prop = go.Bar(
    x=cnt.index,
    y=cnt.values,
    marker=dict(
        color=cnt.values,
        colorscale = 'Picnic',
        reversescale = True
    ),
)

layout = go.Layout(
    title='Target Proportion',
    font=dict(size=18)
)

data = [tr_prop]
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename="TargetProp")

## target distribution ##
labels = (np.array(cnt.index))
sizes = (np.array((cnt / cnt.sum())*100))

tr_pie = go.Pie(labels=labels, values=sizes)
layout = go.Layout(
    title='Target Pie',
    font=dict(size=18),
    width=600,
    height=600,
)
data = [tr_pie]
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename="pie_ty")


# From the target proportion and target pie chart its clearly evident that the target is highly imbalanced with "0" class occupying 90% of the target values and 10% of target values with "1" class.


# **Missing Value Proportion**  
# Now, Let us check the proportion of  many missing values in the training dataset.


def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns

missing_values_table(sctp_train)


# This Looks great as we have no missing values in the dataset.


# **Correlation Coefficient Plot**  
# As there are no missing values in the dataset and all the features are numberic let's try the correlation plot and see how the features are correlated to each other.


labels = []
values = []
for col in sctp_train.columns:
    if col not in ["ID_code", "target"]:
        labels.append(col)
        values.append(spearmanr(sctp_train[col].values, sctp_train["target"].values)[0])
corr_df = pd.DataFrame({'col_labels':labels, 'corr_values':values})
corr_df = corr_df.sort_values(by='corr_values')
 
ind = np.arange(corr_df.shape[0])
width = 0.9
fig, ax = plt.subplots(figsize=(12,30))
rects = ax.barh(ind, np.array(corr_df.corr_values.values), color='g')
ax.set_yticks(ind)
ax.set_yticklabels(corr_df.col_labels.values, rotation='horizontal')
ax.set_xlabel("Correlation coefficient")
ax.set_title("Correlation coefficient of the variables")
plt.show()


# The correlation coefficient values are very low and the maximum value is around 0.08 in negative side of the plot, with respect to positive side the maximum value is around 0.07.
# 
# Overall, the correlation of the features with respect to target are very low.
# 
# So, We will take some of the features which has high correlation values and plot the heatmap for further analysis.


corr_df_sel = corr_df.ix[(corr_df['corr_values'] < -0.05) | (corr_df['corr_values']>0.05)]
corr_df_sel


# Plotting heatmap is done to identify if there are any strong monotonic relationships between these important features. If the values are high, then probably we can choose to keep one of those variables in the model building process. But, we are doing this only for small set of features. we can even try other techniques to explore other features in the dataset.


cols_to_use = corr_df.ix[(corr_df['corr_values'] < -0.05) | (corr_df['corr_values']>0.05)].col_labels.tolist()

temp_df = sctp_train[cols_to_use]
corrmat = temp_df.corr(method='spearman')
f, ax = plt.subplots(figsize=(20, 20))

# Draw the heatmap using seaborn
sns.heatmap(corrmat, vmax=1., square=True, cmap="YlGnBu", annot=True)
plt.title("Important variables correlation map", fontsize=15)
plt.show()


# Seems like none of the selected variables have spearman correlation more than 0.7 with each other.
# 
# The above plots helped us in identifying the important individual variables which are correlated with target. However we generally build many non-linear models in Kaggle competitions. So let us build some non-linear models and get variable importance from them.


# **Feature Importance - eli5 library **  
# For feature importance, I am going to use the Permutation Importance technique that's being used in the [tutorial](https://www.kaggle.com/dansbecker/permutation-importance)


### Get the X and y variables for building model ###
X = sctp_train.drop(["ID_code", "target"], axis=1)
Y = sctp_train["target"]
test_X = sctp_test.drop(["ID_code"], axis=1)

#Train & Validation
from sklearn.model_selection import train_test_split
# create training and testing vars
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=1)
print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)


from sklearn.ensemble import RandomForestClassifier
rfc_model = RandomForestClassifier(random_state=0).fit(X_train, y_train)

#Feature importance by eli5
import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(rfc_model, random_state=1).fit(X_val, y_val)
eli5.show_weights(perm, feature_names = X_val.columns.tolist())


# Interpreting Permutation Importances
# The values towards the top are the most important features, and those towards the bottom matter least.
# 
# The first number in each row shows how much model performance decreased with a random shuffling (in this case, using "accuracy" as the performance metric).
# 
# Like most things in data science, there is some randomness to the exact performance change from a shuffling a column. We measure the amount of randomness in our permutation importance calculation by repeating the process with multiple shuffles. The number after the ± measures how performance varied from one-reshuffling to the next.
# 
# You'll occasionally see negative values for permutation importances. In those cases, the predictions on the shuffled (or noisy) data happened to be more accurate than the real data. This happens when the feature didn't matter (should have had an importance close to 0), but random chance caused the predictions on shuffled data to be more accurate. This is more common with small datasets, like the one in this example, because there is more room for luck/chance.
# 
# In our case, the top 10 most important feature are var_81, var_53, var_139, var_179, var_174, var_40, var_26, var_13, var_24 and var_109. But, Still all the features seems to have value importance close to zero.


# **SMOTE Over-Sampling**  
# As we have more records for target '0', I am going to over sample the target '1' to the same level as target '0' which is basically oversampling the least class.


#Using SMOTE for class imbalance in target
from imblearn.over_sampling import SMOTE
from collections import Counter
sm = SMOTE(random_state=42)
X_resamp_tr, y_resamp_tr = sm.fit_resample(X, Y)
print('Resampled dataset shape %s' % Counter(y_resamp_tr))
X_resamp_tr = pd.DataFrame(X_resamp_tr)
y_resamp_tr = pd.DataFrame({"target": y_resamp_tr})


# **Time for Modelling**  
# **LGBM**  
# Let's try with Lightgbm and see the accuracy.


# https://www.kaggle.com/dromosys/sctp-working-lgb
params = {'num_leaves': 9,
         'min_data_in_leaf': 42,
         'objective': 'binary',
         'max_depth': 16,
         'learning_rate': 0.0123,
         'boosting': 'gbdt',
         'bagging_freq': 5,
         'bagging_fraction': 0.8,
         'feature_fraction': 0.8201,
         'bagging_seed': 11,
         'reg_alpha': 1.728910519108444,
         'reg_lambda': 4.9847051755586085,
         'random_state': 42,
         'metric': 'auc',
         'verbosity': -1,
         'subsample': 0.81,
         'min_gain_to_split': 0.01077313523861969,
         'min_child_weight': 19.428902804238373,
         'num_threads': 4}


%%time
import time
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
fold_n=3
folds = StratifiedKFold(n_splits=fold_n, shuffle=True, random_state=30)
y_pred_lgb = np.zeros(len(test_X))
for fold_n, (train_index, valid_index) in enumerate(folds.split(X_resamp_tr,y_resamp_tr)):
    print('Fold', fold_n, 'started at', time.ctime())
    X_train, X_valid = X_resamp_tr.iloc[train_index], X_resamp_tr.iloc[valid_index]
    y_train, y_valid = y_resamp_tr.iloc[train_index], y_resamp_tr.iloc[valid_index]
    
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid)
        
    lgb_model = lgb.train(params,train_data,num_boost_round=20000,
                    valid_sets = [train_data, valid_data],verbose_eval=300,early_stopping_rounds = 200)
            
    y_pred_lgb += lgb_model.predict(test_X, num_iteration=lgb_model.best_iteration)/3


# **Submission File**


#Submission file
submission_lgb_smote_2 = pd.DataFrame({
        "ID_code": sctp_test["ID_code"],
        "target": y_pred_lgb
    })
submission_lgb_smote_2.to_csv('submission_lgb_smote_2.csv', index=False)

