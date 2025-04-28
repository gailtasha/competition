# # Introduction
# 
# In this challenge, the goal is to help Santander to predict if their clients will make a specific transaction.  The dataset provided by the bank is anonymised, and it is a representation of the real customer data. It is a binary classification problem, with 200 features and 200.000 samples.
# 
# This Notebook presents an overview of how the data was explored, analysed and prepared to be used on an ML model. 


import pandas as pd
import numpy as np
import lightgbm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.scorer import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate as CV
import scipy.stats as stats
warnings.filterwarnings('ignore')
plt.style.use('seaborn')
random_state = 42
np.random.seed(random_state)
from sklearn.model_selection import cross_validate
# visualization
sns.set()


def data_load(filename):
    return pd.read_csv(filename)

def basic_EDA(df):
    size = df.shape
    sum_duplicates = df.duplicated().sum()
    sum_null = df.isnull().sum().sum()
    return print("Number of Samples: %d,\nNumber of Features: %d,\nDuplicated Entries: %d,\nNull Entries: %d" %(size[0],size[1], sum_duplicates, sum_null))

#Plot Bar graph with all classes and percentages - Return number of Classes and Samples per class
def bar_plot(df, target):
    unique, counts = np.unique(target, return_counts = True)
    label = np.zeros(len(unique))
    for i in range(len(unique)):
        label[i] = (counts[i]/df.shape[0])*100
        plt.bar(unique,counts, color = ['burlywood', 'green'], edgecolor='black')
        plt.text(x = unique[i]-0.15, y = counts[i]+0.01*df.shape[0], s = str("%.2f%%" % label[i]), size = 15)
    plt.ylim(0, df.shape[0])
    plt.xticks(unique)
    plt.xlabel("Target")
    plt.ylabel("Count")
    plt.show()
    return unique, counts

#Plots Heatmap and top 10 and bottom correlated features
def feat_corr_analysis(corrmat):
    f, ax = plt.subplots(figsize =(9, 8)) 
    #1 Heatmap
    sns.heatmap(corrmat, vmin=0, vmax=0.2, ax = ax, cmap ="YlGnBu", linewidths = 0.1)
    plt.title("Heatmap - Correlation between data variables")
    #2 Correlation Values and Features
    correlations = corrmat.abs().unstack().sort_values(kind="quicksort").reset_index()
    correlations = correlations[correlations['level_0'] != correlations['level_1']]
    #Top 10 correlated features
    correlations.tail(10)
    #Bottom 10 correlated features
    correlations.head(10)
    return correlations.tail(10)

def feat_corr_distr(train,test):
    #Plot distribution of Feature Correlation
    train_corr_distr = train.values.flatten()
    train_corr_distr = train_corr_distr[train_corr_distr != 1]
    test_corr_distr = test.values.flatten()
    test_corr_distr = test_corr_distr[test_corr_distr != 1]
    plt.figure(figsize=(20,5))
    sns.distplot(train_corr_distr, color="Red", label="Train")
    sns.distplot(test_corr_distr, color="black", label="Test")
    plt.xlabel("Correlation values")
    plt.ylabel("Density")
    plt.title("Feature Correlation"); 
    plt.legend();
    
def prediction(x_train,y_train):
    classifier.fit(x_train,y_train)
    y_proba = classifier.predict_proba(x_train)
    y_pred = classifier.predict(x_train)
    y_proba = classifier.predict_proba(x_train)
    score = roc_auc_score(y_train, y_pred)
    return y_proba, score

def probability_class(y_proba, true_label):
    plt.figure(figsize=(20,5))
    sns.distplot(y_proba[true_label==1,1], label="True Class 1")
    sns.distplot(y_proba[true_label==0,1], label="True Class 0")
    plt.xticks(np.arange(0,1, 0.1))
    plt.xlabel("Predicted probability values of class 1")
    plt.ylabel("Density")
    plt.title("Predicted probability values of class 1 against the true Target"); 
    plt.legend();


# # Part 1 - EDA


# Importing the Training and Test set CSV files.


#Load Data
train = data_load('../input/santander-customer-transaction-prediction/train.csv')
test = data_load('../input/santander-customer-transaction-prediction/test.csv')


# ## Overview
# For an overview of the data, it is analysed the number of samples, features, duplicated and null values:


print("***Train EDA***")
train_EDA = basic_EDA(train)
print("***Test EDA***")
test_EDA  = basic_EDA(test)


# This is a large dataset, with 200.000 samples and over 200 features. No null values or duplicated entries were found, eliminating the need for data cleanse at this stage. It is also fundamental to understand the type of variables present in this dataset:


train.info()
train.head(10)


# The "target" column contains the true labels. The "ID_CODE" is the object dtype mentioned in the info(). Other than this, all values are numeric for train and test sets. Therefore there is no need for enconding.


test.info()
test.head(10)


# At this point, it is possible to remove the ID_code column and replace it with a numerical index that can be easily handled by the ML models. The new column is called Id.


#ID_Code is the only object dtype. It can be replaced by index values
train["Id"] = train.index.values
test["Id"] = test.index.values
init_train_ID = train.ID_code.values
init_test_ID = test.ID_code.values
train.drop("ID_code", axis=1, inplace=True)
test.drop("ID_code", axis=1, inplace=True)
train.head(5)


# ## Samples per Class
# By analysing the samples per class ratio it is possible to see the class imbalance issue. Almost 90% of the samples are clients that did not performed the transaction. 
# 
# * An initial attempt was made to use SMOTE and ADASYN as oversampling methods. Preliminary test have shown that this did not helped the model to generalise to the test set.


##Visualise Class Imbalance - Training Set
num_classes, feat_per_class = bar_plot(train, train["target"])


# ## Feature Correlation
# Since this dataset has a significant amount of features, this analysis helps to develop an understanding of how the variable relate to each other. This can indicate starting points for feature engineering and the most relevant variables for the model. 


train_corrmat = train.drop(["target"], axis=1).corr()
feat_corr_train = feat_corr_analysis(train_corrmat)
test_corrmat = test.corr()
feat_corr_test = feat_corr_analysis(test_corrmat)


# There is none or very little correlation between the features for training and test sets. The plot below shows that the correlation values distribution is similar for the train and test set.


feat_corr_distr(train_corrmat, test_corrmat)


# From the EDA, it was possible to conclude:
# * No major data cleanse method was required, as the data was mostly numerical and there were no missing or duplicated entries;
# * There is little or no variance between the variables, which renders difficult to define the feature engineering path;
# * The data imbalance issue needs to be addressed.


# # Part 2 - Model Baseline
# Random Forest (RF) algorithm is used as a starting point to allow a better understanding of this dataset. RF is quick, and it does not have that many parameters to tune and usually provide reasonable results. By understanding what works and what does not works with RF, it is possible to improve the model quicker and then try other algorithms. This first step is mainly to understand the top essential features,  trial some feature engineering and sampling techniques.


#Prepare DF
X_train = train.drop("target", axis=1).values
y_train = train.target.values


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(test)


# Below is the RF classifier. Grid Search supported the selection of the initial parameters:


#Random Forest Baseline Model
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators = 10, 
                       criterion = 'gini',
                       max_depth = 15, 
                       max_features = 'auto', 
                       min_samples_leaf = 1, 
                       random_state = 0)


# The next block builds the RF model with Cross Validation and outputs and test set scores.


def CV(clf, X, y, metric, cv):
    scores = cross_validate(clf,X,y, scoring = metric, cv = cv)
    return scores

metric = make_scorer(roc_auc_score)
CV_result = CV(classifier, X_train, y_train, metric, 3)
print("Cross Validation Results: \n", (CV_result['test_score']))


# The code below creates the RF model and outputs the AUC result for the training set. It also outputs the probabilities of each sample belonging to class 0 or 1.


RF_y_proba, RF_model_score = prediction(X_train,y_train)
print("Baseline RF: %.2f "%(RF_model_score))
pd.DataFrame(RF_y_proba).describe()


# RF_y_proba contains the probability of class 0 and 1 for every sample. Most values for class 1 are small, and the majority does not reach 0.1, as shown by the table above. One hypothesis can be that the usual threshold of 0.5 is not ideal for this model to assign class 1 or 0. One way to visualise this is to plot the "Class 1" probability distribution for all samples, according to their True label. 


probability_class(RF_y_proba, y_train)


# From the plot above it is possible to extract the following insights:
# * Most of the True Class 1 samples were predicted by our model with a probability lower than 0.5
# * Most of the True Class 0 samples are concentrated within the values of 0 and 0.15
# * For samples with Class 1 probability values higher than approximately 0.15, it is almost safe to say they are True Class 1. Altough there is the orange bump around 0.17
# * It is possible to perform a quick test and see if this helps the model accuracy:


#From the graph, used a threshold of 0.15
threshold = 0.15
y_pred = np.zeros(RF_y_proba.shape[0])
y_pred[RF_y_proba[:,1] >= threshold] = 1
RFT_model_score = roc_auc_score(y_train, y_pred)
print("Baseline RF: %.2f \nThreshold RF: %.2f"%(RF_model_score, RFT_model_score))


# A 30% increase in accuracy was achieved by this simple but effective strategy. This is something to keep in mind in the future and check if other models also present this behaviour.


# ## Feature Importances
# This analysis can helps reduce the number of features while mantaining the minimum impact on model accuracy. Using the RF classifier, the following features are more effective to our model:


importances = classifier.feature_importances_
indices = np.argsort(importances)[::-1]
colors = plt.cm.Reds(importances)
# Plot the feature importances
plt.figure(1, figsize=(40,20))
plt.title("Feature importances")
sns.barplot(x=indices, y=importances[indices], order = indices,palette="Blues_d")
plt.xticks(rotation = 90)
plt.show()


#From the Feature Importances, we can see that 
n_top = 100
idx = np.argsort(importances)[::-1][0:n_top]
feature_names = train.drop("target", axis=1).columns.values
plt.figure(figsize=(20,5))
sns.barplot(x=feature_names[idx], y=importances[idx],palette="Blues_d");
plt.title("Top important features to start");
plt.xticks(rotation = 90)
plt.show()


pd.DataFrame(X_train[:,idx]).describe()


# The next model is built using the top features. Additional aggreagation features were created performing the following operations per row: 


def concat_feat_eng(df,df_feat):
    sum_feat= df_feat.sum(axis=1)
    min_feat = df_feat.min(axis=1)
    max_feat = df_feat.max(axis=1)
    mean_feat = df_feat.mean(axis=1)
    std_feat = df_feat.std(axis=1)
    var_feat = df_feat.var(axis=1)
    per25_feat = np.percentile(df_feat, 25, axis = 1)
    per50_feat = np.percentile(df_feat, 50, axis = 1)
    per75_feat = np.percentile(df_feat, 75, axis = 1)
    iqr_feat = per75_feat - per25_feat
    skw_feat = stats.skew(df_feat, axis = 1)
    kur_feat = stats.kurtosis(df_feat, axis = 1)
    df = np.concatenate((df_feat,                        
                        sum_feat[:,None],
                        min_feat[:,None],
                        max_feat[:,None],
                        mean_feat[:,None],
                        std_feat[:,None],
                        var_feat[:,None],
                        per25_feat[:,None],
                        per50_feat[:,None],
                        per75_feat[:,None],
                        iqr_feat[:,None],
                        skw_feat[:,None],
                        kur_feat[:,None]),
                        axis = 1)
    return df

X_train_top = X_train[:,0:n_top]
X_train_top = concat_feat_eng(X_train_top,X_train_top)

RF1_y_proba, RF1_model_score = prediction(X_train_top, y_train)
probability_class(RF1_y_proba,y_train)


# With the new features, is necessary to revaluate and assign a new threshold 


RF1_y_pred = np.zeros(RF1_y_proba.shape[0])
RF1_y_pred[RF1_y_proba[:,1] >= 0.12] = 1
RF1_model_score = roc_auc_score(y_train, y_pred)
print("Baseline RF %.2f\nThreshold RF %.2f\nThreshold RF with only top features %.2f"%(RF_model_score, RFT_model_score, RF1_model_score))


# ## Initial Submission
# By reducing the number of features to a quarter of the original model, the same accuracy was mantained. 
# 
# At this point is interesting to check our score in the test set and submit:


X_test_top = X_test[:,idx]
X_test_top = X_test[:,0:n_top]
X_test_top = concat_feat_eng(X_test_top,X_test_top)

#####################################################
y_proba = classifier.predict_proba(X_test_top)
y_pred = np.zeros(y_proba.shape[0])
y_pred[y_proba[:,1] >= 0.12] = 1
#submission = pd.concat([pd.DataFrame(init_test_ID),pd.DataFrame(y_pred)],axis = 1)
#submission.columns = ['ID_code', 'Target']
#submission.to_csv("submission_RForest.csv", index=False)


# ## Initial Conclusion
# 
# The result on the test set points to overfitting, since it only achieved **0.59** on the test set while **0.85** on the training set.
# 
# With the learned lessons from the baseline model, it is now possible to devote attention into applying other models.


# # Part 3 - Models


# ## Light GBM 
# LightGBM is a gradient boosting framework that uses tree-based algorithms and follows leaf-wise approach while other algorithms work in a level-wise approach pattern. Light GBM usually demonstrates a good performance for large datasets, if compared to other algorithms (XGBoost, Gradient Boosting and AdaBoost, for example).
# 
# The code below defines my Light GBM model. Grid Search was performed to tune the main parameters, this was commented out to reduce the time to run the full notebook. The "is_unbalance" parameter has proven to be very helpful for this model, it is responsible for a 5% increase on the final auc score. All the dataset features are used, as it did not significantly increased the training time.


#Adds the aggregation features to the original Train and Test set
X_train = concat_feat_eng(X_train, X_train)
X_test = concat_feat_eng(X_test, X_test)

#Creates a Validation Set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
train_data = lightgbm.Dataset(X_train, label=y_train)
val_data = lightgbm.Dataset(X_val, label=y_val)

#Grid Search
#param_grid = {
#    'num_leaves': [16,32,64],
#    'bagging_fraction' :[ 0.5,0.8],
#    'bagging_freq' : [10, 20, 50, 100],
#    'max_depth' : [10,50,100]
#    }

#LGB = lightgbm.LGBMClassifier(boosting = 'gbdt',
#                            objective = 'binary',
#                              is_unbalance = 'true',
#                              learning_rate = 0.005,
#                              metric = 'auc',
#                              num_iterations = 1000,
#                              bagging_fraction = 0.8,
#                              bagging_freq = 20,
#                              feature_fraction = 0.5,
#                              )
#
#LGB_Grid = GridSearchCV(estimator=LGB, param_grid=param_grid, cv= 2, scoring=make_scorer(roc_auc_score), verbose=2)
#LGB_Grid.fit(X_train, y_train)
#print(LGB_Grid.best_params_)
#print(LGB_Grid.best_score_)

#LGB Final Parameters
params = {
    'bagging_fraction': 0.5,
    'bagging_freq': 10,
    'boosting': 'gbdt', 
    'feature_fraction': 0.5, 
    'is_unbalance': 'true',    
    'max_depth': 10,
    'metric': 'auc',    
    'num_leaves': 32,    
    'learning_rate': 0.005,
    'objective': 'binary',    
    'verbose': 0
}

LGB = lightgbm.train(params, 
                   train_data,
                   valid_sets=val_data,
                   num_boost_round=50000,
                   early_stopping_rounds=50)

LGB_y_val = LGB.predict(X_val)


print("Light GBM Validation set AUC: %.2f" %(roc_auc_score(y_val, LGB_y_val)))


# The result is an improvement over the previous RF model. The analysis of the probability distribution also shows that the 0.5 threshold is not the minimum point where the 0 and 1 distributions meet:


sns.distplot(LGB_y_val[y_val==1], label="True Class 1")
sns.distplot(LGB_y_val[y_val==0], label="True Class 0")
plt.xlabel("Predicted probability values")
plt.ylabel("Density")
plt.title("Predicted probability values against the true Target for Validation set"); 
plt.legend();

LGB_y_pred = LGB.predict(X_test)

submission = pd.concat([pd.DataFrame(init_test_ID),pd.DataFrame(LGB_y_pred)],axis = 1)
submission.columns = ['ID_code', 'Target']
submission.to_csv("submission_LGB.csv", index=False)


# From the plot above it is possible to visualise the probability distribtuion between the two classes. Differently from the RF model, the distribution is more balanced and most of the class 1 samples are concentrated on the right corner, where p > 0.5. For this reason, it is not necessary to use the strategy described earlier. The model results can be directly sent for submission.
# 
# The LGB achieved a **0.89** AUC on both training and test sets. This is a good indication that our model is not overfitted to the training data as the RF. 


# # Conclusion
# 
# Light GBM has shown to be an interesting choice for this dataset. The final model auc is near to 90% for both train and test sets. 
# 
# Preliminary tests with Logistic Regression, SVM and XGBoost have not reached auc values above 65%. As future work, an ensemble model would be an interesting approach as well as applying ANN.

