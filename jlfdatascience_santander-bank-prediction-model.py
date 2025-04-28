# <img src="https://storage.googleapis.com/kaggle-organizations/141/thumbnail.jpg?r=890" alt="Kitten" title="Santander Bank"
# width="100" height="100" align="left"/>
# <img src="https://miro.medium.com/max/837/1*Ab299OETAeuTEiGg5TwpMQ.png" alt="Kitten" title="Santander Bank"
# width="150" align="center"/>


# # Santander Customer Transaction Prediction  
# ### Comparison&Performance new generation of GBDT´s prediction model


# ##### Introduction  
# 
# Getting data from a Kaggle's competition, let's compare the performance between classic and new  generation of gradient boosting decision trees (GBDTs).
# 
# Reference: https://www.kaggle.com/c/santander-customer-transaction-prediction
# 
# In this competition proposed by **Santander Bank**, invites Kaggle users to predict which customers will make a specific transaction in the future, regardless of the amount of money made. The data provided for this contest has the same structure as the actual data they have available to solve the problem in the bank, which makes us address a real problem with a demanding dataset by number of records and characteristics, by which will test the performance of classic algorithms versus next-generation algorithms.
# 
# The data is anonymised, where each row contains 200 discrete variables and no categorical variables.
# 
# Next we'll do a data exploration, readiness to apply the model, and analyze which algorithms get the best performance with low overfitting and compare the results between them.


# ### Content
# 1. Libraries
# 2. Data extraction
# 3. Data exploration
# 4. Unbalanced Data and Resampling
# 5. Feature selection
# 6. Binary classification models
# 7. Hyperparameter tuning
# 8. Detection of the most influential variables


# ### 1. Importar las librerías


from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2, f_classif
from sklearn.linear_model import LinearRegression #Selección VIF de características
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier, Pool, cv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from hyperopt import hp
import lightgbm as lgb
import numpy as np
import seaborn as sns
import pandas as pd
import warnings
import imblearn
import zipfile
import time

%pylab
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
warnings.simplefilter("ignore")


# ### 2. Data extraction  
# The data is extracted from the competition opened in Kaggle by the Santander Bank and available for download in:  
# https://www.kaggle.com/c/santander-customer-transaction-prediction/data


#zf = zipfile.ZipFile('datos\santander-customer-transaction-prediction.zip') 
df_train = pd.read_csv('../input/santander-customer-transaction-prediction/train.csv')
df_test = pd.read_csv('../input/santander-customer-transaction-prediction/test.csv')


df_train.head()


# ### 3. Data exploration


df_train.info()


df_train.describe()


# This is an anonymised dataset with 199 discrete numeric variables, with a dependent variable labeled as a binary variable and a column in string format with an identifier label. Two training datasets are provided, a training dataset and evaluation dataset, but no target variable so that for our purpose we won't use it to train the models. The task that is requested in this challenge is to predict the value of the target column in the test set.


#We note that the dependent/target variable is very unbalanced.
df_train.target.value_counts().plot.bar() #.plot(kind="bar")


print("There is {}% of values 1 in the target variable".format(100*df_train['target'].value_counts()[1]/df_train.shape[0], 2))


# We look for possible null values in the dataframe:


df_train.isnull().sum()


#We have many variables, we look for a method to specifically locate null values
null_columns=df_train.columns[df_train.isnull().any()]
df_train[null_columns].isnull().sum()
print(df_train[df_train.isnull().any(axis=1)][null_columns].head())
print('It can´t find null values throughout the df')


# Get an idea of thisdata distribution, we review in the training dataset that we will work with, we review the histogram of the mean values of each record based on the binary target variable.


#Separation of the target variable and the explanatory
target = 'target'
features = list(df_train.columns)
features.remove('target')
features.remove('ID_code')
#Separating the labels from the target variable
t0 = df_train[df_train['target'] == 0]
t1 = df_train[df_train['target'] == 1]
plt.figure(figsize=(16,6))
plt.title("Distribución de la media por fila")
sns.distplot(t0[features].mean(axis=1),color="red", kde=True,bins=120, label='target = 0')
sns.distplot(t1[features].mean(axis=1),color="blue", kde=True,bins=120, label='target = 1', hist_kws={'alpha':0.3})
plt.legend(); plt.show()


# As you can note there is a small variation in the mean of all the features that could explain the target variable, which in any case is a little variation.  
# We try to detect potential correlated variables to decrease high dimensionality. How the correlation matrix would be too large visually, we tried to numerically detect the existence of correlations above 0.5 and below -0.5.


corr_matrix = df_train.corr().abs()
high_corr_var=np.where(corr_matrix>0.5)
high_corr_var=[(corr_matrix.columns[x],corr_matrix.columns[y]) for x,y in zip(*high_corr_var) if x!=y and x<y]
if len(high_corr_var)==0:
    print('There are no correlated variables')


# ### 4. Unbalanced Data and Resampling  
# Note we are dealing with a data set **very unbalanced**, where there is only **10%** of records categorized with target 1, so those customers who have made a financial transaction.  
# To develop a binary classification model we need to have more balanced data since most machine learning algorithms work best when the number of samples in each class is almost the same. This is because most algorithms are designed to maximize accuracy and reduce error, so we'll try to do this in this section before to predict models fit better.
# 
# How we have a large dataset with 200,000 records we could undersampling in the data with the balanced target variable. Initially we will test a resampling in a 1:1 ratio but depending on the results we can use other proportions. Keep in mind that with undersampling we might be removing information that may be valuable. This could lead to a lack of fit and poor generalization of the test set.


# <img src="https://raw.githubusercontent.com/rafjaa/machine_learning_fecib/master/src/static/img/resampling.png" alt="Kitten" title="Santander Bank" align="left"/>


#Generate two variables with the number of records in each class
count_class_0, count_class_1 = df_train.target.value_counts()

#Divide into two df with each class
df_class_0 = df_train[df_train['target'] == 0]
df_class_1 = df_train[df_train['target'] == 1]

#Undersampling with the 'sample' pandas property
df_class_0_under = df_class_0.sample(count_class_1)
df_train_under = pd.concat([df_class_0_under, df_class_1], axis=0)

print('Undersampling is in a number of records:')
print(df_train_under.target.value_counts())

df_train_under.target.value_counts().plot(kind='bar', title='Count (target)');


#Separation of the target variable and the explanatory
target = 'target'
features = list(df_train_under.columns)
features.remove('target')
features.remove('ID_code')
x_train = df_train_under[features]
y_train = df_train_under[target]

#Divide dataset into training and validation
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.75, random_state = 0, stratify=y_train)


#Check result
print(y_train.value_counts())


# We could also use another undersampling strategy from specific libraries such as the `mbalanced-learn` Python module. Through this library we can group the records of the majority class and perform the undersampling by deleting records from each group or category, thus seeking to preserve the maximum information.


import imblearn
from imblearn.under_sampling import RandomUnderSampler

subm = RandomUnderSampler(return_indices=True)
x_subm, y_subm, id_subm = subm.fit_sample(df_train[features], df_train[target])

y_subm_plot = pd.DataFrame(y_subm)
y_subm_plot[0].value_counts().plot(kind='bar', title='Count (target)');


#Check the result
print(y_subm_plot[0].value_counts())


# ### 5. Feature selection  
# ##### Selection of the best features  
# Before creating a model, you can use the `SelectKBest` or `SelectPercentile` constructors to select objects that allow you to select the `k` `feature better`or a **percentage** of them respectively for creating a model. In both cases, the criterion to be used to sort them must be indicated. In *scikit-learn* there are two methods that can be used depending on the type of problem you are working with:
# 
# * `f_regression` for regression models
# * `chi2` or `f_classif` for classification models in this case
# 
# We try to select the top 50 and 100 features to see if the models perform better and generalize better.


var_sk = SelectKBest(f_classif, k = 50)
x_sk = var_sk.fit_transform(x_train, y_train)

print(u"Number of final features:", x_sk.shape[1])
print(u"List of final features: \n", x_train.columns[var_sk.get_support()])
x_train_50best = x_train[x_train.columns[var_sk.get_support()]]


# You can see that the top 50 features have been selected. If we use the `SelectPercentile` constructor, you must be told the percentage of characteristics to be selected from the dataset. For example, you can test by selecting the best 50%, i.e. the top 100 features.


var_pc = SelectPercentile(f_classif, percentile = 50)
x_pc = var_pc.fit_transform(x_train, y_train)

print(u"Number of final features:", x_pc.shape[1])
print(u"List of final features: \n", x_train.columns[var_pc.get_support()])
x_train_100best = x_train[x_train.columns[var_pc.get_support()]]


# ### 6. Binary classification models


# In order to automate the performance measures of the different models, we will factor a function to measure the metrics and be able to make comparisons between the different algorithms applied.


def metricas(y_true, y_pred):
    print(u'La matriz de confusión es ')
    print(confusion_matrix(y_true, y_pred))

    print(u'Precisión:', accuracy_score(y_true, y_pred))
    print(u'Exactitud:', precision_score(y_true, y_pred))
    print(u'Exhaustividad:', recall_score(y_true, y_pred))
    print(u'F1:', f1_score(y_true, y_pred))

    false_positive_rate, recall, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(false_positive_rate, recall)

    print(u'AUC:', roc_auc)

    plot(false_positive_rate, recall, 'b');
    plot([0, 1], [0, 1], 'r--');
    title(u'AUC = %0.2f' % roc_auc);


# #### 6.1. Logistic Regression  
# The **binary classification** of events can be performed from a logistic regression model where the expression is used:
# $$F(x) = \frac{1}{1 + e^{\sum-w_ix_i}}$$
# 
# In *scikit-learn* the constructor with which you can create a logistic regression model is `LogisticRegression`.  
# We take this first model as a **reference** for its easy implementation and in which we can see how the other models behave.


%%time
lr_classifier = LogisticRegression().fit(x_train, y_train)
y_train_pred = lr_classifier.predict(x_train)

print('Métricas de entrenamiento:')
metricas(y_train, y_train_pred);


%%time
#See the overfitting in the test dataset
y_test_pred  = lr_classifier.predict(x_test)
print('Métricas de validación:')
metricas(y_test, y_test_pred);


# We get a **medium performance** of the model but with very **low overfitting** between training and validation.  
# We will take this performance as a base reference to compare it with other models based on Decision Trees and their derivatives as gradient boosting.


# #### 6.2. Random forest  
# We don´t test with the origin decision tree since being a dataset with discrete and non-categorical variables, it is difficult to achieve acceptable performance and stability, so it´s not good for unbalanced classification problems because the generated trees they will be very biased.
# 
# We go straight to testing with the Random forest which is a combination of trained decision trees each with a subset of the original data. This allows for more stable models.
# 
# In `scikit-learn` the constructor with which you can create a Random Forest model is `RandomForestClassifier`. This constructor requires more parameters than the decision tree because it is to be told the number of tree models to use, for which the parameter can be used `n_estimators`. On the other hand, as selecting the data to be used for each submodel it is a good idea to fix the seed to ensure that the results are repeatable.
# 
# With this in mind we can create a model for the resampled training and validation dataset.


%%time
rf_classifier = RandomForestClassifier(n_estimators = 5,
                                       max_depth = 7, #Without limiting the depth of the tree is overfitting is even greater AUC:0.93
                                       random_state = 1)
rf_classifier.fit(x_train, y_train)
y_pred = rf_classifier.predict(x_train)

print('Métricas de entrenamiento:')
metricas(y_train, y_pred);


#Check that this method performs well but we will check the overfitting
#Comprobamos que este método tiene buen rendimiento pero comprobaremos el sobreajuste
y_test_pred = rf_classifier.predict(x_test)
print('Métricas de validación:')
metricas(y_test, y_test_pred);


# You get a model with average performance but as we can see when validating it, you can see a great overfitting of the model so it does not seem that the overall performance is adequate with respect to the reference of the logistic regression (AUC training: 0.78 and validation: 0.77).


# #### 6.3. Gradient Boosting Decision Trees  
# Gradient increase is one of the most powerful techniques for building predictive models. Perhaps the most popular implementation is XGBoost which employs a number of tricks that make it faster and more accurate than the traditional gradient increase (particularly the second order gradient descent).  
# However in this case as a fairly large dataset >10,000 records we chose to apply the two gradient-enhancing algorithms that have been made most popular lately because they are more efficient because of the lower memory usage in large data:
# * Catboost
# * LightGBM
# 
# ####     6.3.1 Catboost  
# The great value of catboost is the optimized handling of categorical variables and in this case we only have discrete variables, we will use this algorithm to have a performance reference with which to compare to LightGBM. 


#Definimos the parameter of the index of categorical properties although 
#in this case as we said by the previous data exploration, we do not have such variables
categorical_features_indices = np.where(x_train.dtypes != np.float)[0]


%%time
#Apply the model by setting some initial generic parameters
cat_model = CatBoostClassifier(
        depth=4,
        custom_loss=['AUC'],
        learning_rate=0.3,
        verbose=50,
        iterations=None,
        od_type='Iter',
        early_stopping_rounds=10
)

cat_model.fit(x_train,y_train,eval_set=(x_test,y_test),use_best_model=True)#para mejorar el procesado paramos en el mejor ajuste

pred = cat_model.predict_proba(x_test)[:,1]
y_train_pred = cat_model.predict(x_train)
#print('AUC de validación: ',roc_auc_score(y_test, pred))
print('Métricas de entrenamiento:')
metricas(y_train, y_train_pred); #Probamos el rendimiento del modelo


y_test_pred  = cat_model.predict(x_test)
print('Métricas de validación:')
metricas(y_test, y_test_pred); #Test on the validation sample to see if it shows overfitting


# In this case we get better performance from both the original reference with the Linear Regression and the Random forest, with a moderate overfitting.


# ####     6.3.2 LightGBM
# . 


#Define the parameter of the index of categorical properties although
#in this case as we said by the previous data exploration, we do not have such variables
categorical_features_indices = np.where(x_train.dtypes != np.float)[0]
feature_names = x_train.columns.tolist()


# LightGBM dataset formatting 
lgtrain = lgb.Dataset(x_train, y_train,
                feature_name=feature_names,
                categorical_feature = categorical_features_indices)
lgvalid = lgb.Dataset(x_test, y_test,
                feature_name=feature_names,
                categorical_feature = categorical_features_indices)


#Set some generic initial parameters
params = {
    'objective' : 'binary',
    #'metric' : 'rmse',
    'num_leaves' : 200,
    'max_depth': 10,
    'learning_rate' : 0.01,
    #'feature_fraction' : 0.6,
    'verbosity' : -1
}
params['metric']=['auc', 'binary_logloss']


%%time
#Apply the model
lgb_clf = lgb.train(
    params,
    lgtrain,
    #num_iterations=2000,
    valid_sets=[lgtrain, lgvalid],
    valid_names=["train", "valid"],
    early_stopping_rounds=500,
    verbose_eval=500
)

#Training Prediction::
y_train_pred = lgb_clf.predict(x_train)

#Convert to binary values because in this model it gives us probabilities
for i in range(len(y_train_pred)):
    if y_train_pred[i]>=.5:       # setting threshold to .5
        y_train_pred[i]=1
    else:
        y_train_pred[i]=0
print('Métricas de entrenamiento:')      
metricas(y_train, y_train_pred);


#Validation prediction:
y_test_pred = lgb_clf.predict(x_test)

#Convert to binary values because in this model it gives us probabilities
for i in range(len(y_test_pred)):
    if y_test_pred[i]>=.5:       # setting threshold to .5
        y_test_pred[i]=1
    else:
        y_test_pred[i]=0
print('Métricas de validación:')
metricas(y_test, y_test_pred);


# Despite what is expected, in this case we get worse performance than in the case of Catboost. Let's check if by adjusting the hyperparameters we reverse the results as you would expect.


# ### 7. Hyperparameters tuning 
# An optimal set of parameters can help achieve greater accuracy. Finding hyperparameters manually is tedious and computationally expensive. Therefore, the automation of hyperparameter tuning is important. RandomSearch, GridSearchCV and Bayesian optimization are generally used to optimize hyperparameters. In this case we will choose a mixed optimization in which we approximate the parameters manually and profile with the `GridSearchCV` method.
# For a comparative idea of the main parameters that most influence the performance of these models, you can refer to the following table:
# ![GBDTs comparation](https://miro.medium.com/max/3400/1*A0b_ahXOrrijazzJengwYw.png)


#    #### 7.1. Catboost


#%%time
#cat_model = CatBoostClassifier(verbose=50)
#Definition parameters space
#params = {'depth'         : [3,4,5,6],
#          'learning_rate' : [0.01,0.05,0.1,0.35,0.4],
#          'iterations'    : [30,50,125,150],
#          'l2_leaf_reg': [3,1,2,5,10]
#          }
#grid = GridSearchCV(estimator=cat_model, param_grid = params, cv = 3, n_jobs=-1)
#grid.fit(x_train,y_train)

#print("\n La mejor métrica de validación cruzada:\n", grid.best_score_)
#print("\n Los mejores parámetros:\n", grid.best_params_)


# Optimization gives us as the best parameters:  
# {'depth': **3**, 'iterations': **150**, 'l2_leaf_reg': **10**, 'learning_rate': **0.4**}


%%time
#Applying these "optimal paramenters" we would get the following model in Catboost
cat_model = CatBoostClassifier(
        depth= 3,
        learning_rate=0.4,
        iterations=150,
        l2_leaf_reg=10,
        custom_loss=['AUC'],
        verbose=50,
        random_seed=501
        )
cat_model.fit(x_train,y_train,eval_set=(x_test,y_test),use_best_model=True)

y_train_pred = cat_model.predict_proba(x_train)[:,1]
y_test_pred = cat_model.predict_proba(x_test)[:,1]
print('AUC_train: ',roc_auc_score(y_train, y_train_pred))
print('AUC_test: ',roc_auc_score(y_test, y_test_pred))


#    #### 7.2. LightGBM


#Complete but very demanding parameters for processing
#lgb_grid_params = {
#    'objetive':['binary'],
#    'boosting_type' : ['gbdt'],
#    'learning_rate':  [0.05, 0.1 , 0.15, 0.2 , 0.255, 0.3], 
#    'max_depth': [ 5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15],
#    'min_child_weight': [1, 2, 3, 4, 5, 6, 7],
#    'num_leaves': [20, 30, 40],
#    'min_child_samples': [20, 33, 40, 50],
#    'colsample_bytree': [0.3, 0.4, 0.5, 0.6, 0.7],
#    'n_estimators': [50, 100, 118, 130],
#    'subsample' : [0.7,0.75],
#    'reg_alpha' : [1,1.2],
#    'reg_lambda' : [1,1.2,1.4],
#    'random_state' : [501]
#}


#%%time
#mdl = lgb.LGBMClassifier(boosting_type= 'gbdt',
#          objective = 'binary',
#          metric= ['binary_logloss', 'auc'],
          #n_jobs = 3, # Updated from 'nthread'
#          silent = True)

#grid = GridSearchCV(estimator=mdl, param_grid = lgb_grid_params, cv = 3, n_jobs=-1)
#grid.fit(x_train,y_train)

#print("\n El mejor estimador:\n", grid.best_estimator_)
#print("\n La mejor métrica de validación cruzada:\n", grid.best_score_)
#print("\n Los mejores parámetros:\n", grid.best_params_)


# La mejor métrica de validación cruzada:
#  0.7835273824924537
# 
#  Los mejores parámetros:
#  {'learning_rate': 0.255, 'min_child_samples': 33, 'n_estimators': 118, 'num_leaves': 20, 'random_state': 501}
# Wall time: 15.8 s


%%time
#LightGBM dataset formatting 
lgtrain = lgb.Dataset(x_train, y_train)
lgvalid = lgb.Dataset(x_test, y_test)

#Applying these optimal paramentros we would get the following model in LightGBM
params = {
    'objective' : 'binary',
    'num_leaves' : 20,
    #'max_depth': 10,
    'learning_rate' : 0.255,
    #'feature_fraction' : 0.6,
    'min_child_samples': 33,
    'n_estimators': 118,
    'verbosity' : 50,
    'random_state':501
}
params['metric']=['auc', 'binary_logloss']

#Training model
lgb_clf = lgb.train(
    params,
    lgtrain,
    #num_iterations=20000,
    valid_sets=[lgtrain, lgvalid],
    valid_names=["train", "valid"],
    early_stopping_rounds=500,
    verbose_eval=500,
    feature_name='auto', 
    categorical_feature='auto'
)

print("RMSE of train:", np.sqrt(mean_squared_error(y_train, lgb_clf.predict(x_train))));
y_train_pred = lgb_clf.predict(x_train);
print('AUC of train: ',roc_auc_score(y_train, y_train_pred ));
y_test_pred = lgb_clf.predict(x_test);
print('AUC of test: ',roc_auc_score(y_test, y_test_pred ));


# As you can see the most efficient model has turned out to be the **LightGBM** algorithm with an AUC of 0.97 in testing versus Catboost 0.91.


# ### 8. Detection of the most influential variables


feature_importances = lgb_clf.feature_importance()
feature_names = x_train.columns
for score, name in sorted(zip(feature_importances, feature_names), reverse=True):
    print('{}: {}'.format(name, score))


fea_imp = pd.DataFrame({'imp': feature_importances, 'col': feature_names})
fea_imp = fea_imp.sort_values(['imp', 'col'], ascending=[True, False]).iloc[-30:]
fea_imp.plot(kind='barh', x='col', y='imp', figsize=(10, 7), legend=None)
plt.title('LightGBM - Feature Importance')
plt.ylabel('Features')
plt.xlabel('Importance');


# ### 9. Submission  
# Submit de Solution


df_test.head()


target = 'target'
features = list(df_test.columns)
features.remove('ID_code')
X_test = df_test[features]

#Prediction with choose model LGBM
Y_prediction = lgb_clf.predict(X_test);

#Convert to binary values because in this model it gives us probabilities
for i in range(len(Y_prediction)):
    if Y_prediction[i]>=.5:       # setting threshold to .5
        Y_prediction[i]=1
    else:
        Y_prediction[i]=0

sub_df = pd.DataFrame({"ID_code":df_test["ID_code"].values})
sub_df["target"] = Y_prediction
sub_df["target"] = sub_df["target"].astype(int)
sub_df.to_csv("submission.csv", index=False)


pd.read_csv("submission.csv").head()

