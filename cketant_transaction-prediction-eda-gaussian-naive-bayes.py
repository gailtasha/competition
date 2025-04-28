import numpy as np # linear algebra
from scipy import stats
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('display.max_columns', 222)

import seaborn as sns
sns.set()
import matplotlib.pyplot as plt 
%matplotlib inline

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

df_train = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')
df_test = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')
df_join = pd.concat([df_train, df_test])


# # EDA


# ## Overview of the Data


print("Rows: %s\nColumns: %s" % (df_train.shape[0], df_train.shape[1]))
print("*" * 30)
print(df_train.head())


group_types = df_train.columns.to_series().groupby(df_train.dtypes)
print(group_types.count())
print("*" * 30)
print(group_types.groups)


# All the feature columns are of type **Float** the prediction target is of type **Int** and the ID columns is of type **Object**.


df_train.info()


miss_vals = df_join.isna().sum()
miss_vals_percent = 100 * miss_vals / len(df_join)
null_df = pd.concat([miss_vals, miss_vals_percent], keys=['Missing Values', 'Missing %'], axis=1)
null_df.sort_values(by='Missing %', inplace=True, ascending=False)
print(null_df)


# It looks like there are no null or empty values the target values that are missing are a part of the test data which is as expected of course.


# ## Gaussian Distrubtion Check


df_train.describe()


df_train_feats = df_train.select_dtypes(include=['float64']) # Grab all the usable features since they are only floats
normality_results = df_train_feats.apply(lambda x: stats.normaltest(x)[1], axis=0)

# The Probability threshold that the feature is normally distributed
alpha = 0.05
normals = normality_results[normality_results > alpha]
print("Number of normally distributed features: %s\nPercentage of Features that are normally distributed: %s\nFeatures that are normally distrubted: %s" % (len(normals), len(normals)/df_train_feats.shape[1], normals.index.values))


# I find it strange that  none of the features are normally distributed given the normality test which uses **D’Agostino's K^2 Normality Test** with a given alpha value of *0.05*. My hunch is that there's something wrong where the normality test are failing or maybe (highly unlikely) that none of the features follow a Guassian distribution. Let's take a visual look. 


# ## Data Visualization


row = 25
col = 8
fig, ax = plt.subplots(row, col, figsize=(col * 7, row * 5))

idx = 0
for r in range(0, row):
    for c in range(0, col):
        x = df_train_feats.iloc[:, idx] # grab column
        sns.distplot(x, axlabel=x.name, ax=ax[r][c])
        idx += 1

plt.show()


# Visually it seems that all the features indicate a Gaussian Distribution so it looks like the **normality test** we tried above failed due to the sample size being too large. See this [article](https://medium.com/data-design/large-amount-of-observations-statistical-test-not-so-statistical-3d8ed0e94be) detailing more. We can perform another visual representation to be sure that are assumption is correct such as a Q-Q plot but I think this will suffice. A Gaussian distribution indicates 2 things that 1) we can use a parameteric modeling method to predict here and 2) most models perform/behave better when data follows a Guassian Distribution which we of course want. 


# # Preprocessing
# Luckily the data is well prepared and will have to do little to now preprocessing.


# ## Feature Selection


THRESHOLD = 0.8
corr = df_train_feats.corr(method='pearson') 
FEATURES_TO_REMOVE = []

i = 0
for j in range(0, len(corr)):
    if i != j:
        if corr.iloc[i,j] >= THRESHOLD:
            FEATURES_TO_REMOVE.append(corr.iloc[:, j].name)

print("Features to Remove\ncount: %s feats: %s" % (len(FEATURES_TO_REMOVE), FEATURES_TO_REMOVE))


# So it doesn't look like any of the features are correlated to each other so no need to remove any features. 


# # Modeling


from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV


df_train_y = df_train[['target']]
df_train_X = df_train.drop(['target', 'ID_code'], axis=1) # only the feature columns are of type float64


train_X, test_X, train_y, test_y = train_test_split(df_train_X, df_train_y, test_size=0.25, random_state=1)


classifiers = [
               'LogisticRegression',
               'GaussianNB'
              ]

scaler = StandardScaler()

results = {
    'classifiers': classifiers,
    'auc': [],
    'acc': [],
    'f1': []
}

for classifier in classifiers:
    pipe = make_pipeline(scaler, eval(classifier)())
    pipe.fit(train_X, train_y)
    pred_y = pipe.predict(test_X)
    
    results['auc'].append(roc_auc_score(test_y, pred_y))
    results['acc'].append(accuracy_score(test_y, pred_y))
    results['f1'].append(f1_score(test_y, pred_y))
    
results_df = pd.DataFrame(data=results)
results_df.sort_values(by='auc', ascending=False, inplace=True)
print(results_df)


# ## Parameter Tuning
# The **GaussianNB** and **LogisticRegressionCV** classifiers perform the best. Let's see if we can improve the scores by parameter tuning. Using the top 2 classifiers, let's run them through the **GridSearch** hyperparameter optimization to see if we can find the optimal parameters to get the best results. 


selected_classifiers = ['LogisticRegression','GaussianNB']
params = {
    'LogisticRegression': {
        'logisticregression__penalty': ['l2'],
        'logisticregression__class_weight': [None, 'balanced'],
        'logisticregression__solver': ['sag', 'saga'],
        'logisticregression__max_iter': [25, 100, 125],
        'logisticregression__tol': [1e-2, 1e-6]
    },
    'GaussianNB': {
        'gaussiannb__var_smoothing': [1e-06, 1e-10, 1e-13]
    }
}
grid_results = {
    "classifiers": selected_classifiers,
    "predict_auc": [],
    "grid_auc": []
}
final_models = {}
scaler = StandardScaler()

for classifier in selected_classifiers:
    pipe = make_pipeline(scaler, eval(classifier)())
    search = GridSearchCV(pipe, params[classifier], scoring = 'roc_auc', n_jobs=2)
    search.fit(train_X, train_y)
    
    grid_results['grid_auc'].append(search.best_score_)
    pred_y = search.predict(test_X)
    grid_results['predict_auc'].append(roc_auc_score(test_y, pred_y))
    
    final_models[classifier] = search.best_estimator_

df_search_results = pd.DataFrame(data=grid_results)
df_search_results.sort_values(by='predict_auc', ascending=False, inplace=True)
print(df_search_results)


# It looks like the winning model with optimized parameters is the **GaussianNB** estimator. 


# # Submission


id_series = df_test['ID_code']
df_test_X = df_test.drop(['ID_code'], axis=1)

best_estimator = final_models['GaussianNB']
pipe = make_pipeline(StandardScaler(), best_estimator)
pipe.fit(df_train_X, df_train_y)

predict_y = pipe.predict(df_test_X)


submission_df = pd.concat([id_series, pd.DataFrame(predict_y, columns=['Target'])], axis=1)
print(submission_df.head())


submission_df.to_csv('Transaction_Prediction_1.csv', index=False)

