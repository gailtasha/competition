#Kaggle's scikit-opt is old, has bugs
!pip install scikit-optimize -U


import numpy as np
import pandas as pd
import seaborn as sns
import shap
import eli5

from matplotlib import pyplot as plt
from tqdm.notebook import tqdm as notebook
from tqdm import tqdm

plt.style.use('ggplot')


shap.initjs();


# # Santander prediction challenge
# 
# This challenge consist on predicting whether a client of the bank will make or not a transaction in the future given a set of values.
# 
# > As you know, Data Science it's an iterative process, so I'll be updating this notebook if I get better result than the posted ones. ()
# 
# The competition was already closed when I created the notebook, but I wanted to give a try and see how far I could get on my own.
# 
# ** What you'll find here **
# 
# 1. Data analysis
# 2. Model baseline: LightGBM (default)
# 3. Feature engineering
# 4. Model tuning (after feature engineering)
# 5. Model training
# 
# Let's begin by loading the data.


PATH_TRAIN_CSV = '../input/santander-customer-transaction-prediction/train.csv'
PATH_TEST_CSV = '../input/santander-customer-transaction-prediction/test.csv'

train_df = pd.read_csv(PATH_TRAIN_CSV)
test_df = pd.read_csv(PATH_TEST_CSV)


# # Data analysis
# 
# First, it's important to see what the data it's about.


train_df.head(3)


# We have 200 predictors (which aren't explained, so knowing what's happening will be more difficult) and a target we have to predict. 


num_samples = train_df.groupby('target')['target'].count()
num_samples.plot.pie(title='Number of samples per category', autopct="%.0f%%");


correlation_matrix = train_df.iloc[:,2:].corr()
fig, ax = plt.subplots(figsize=(7,7))
sns.heatmap(correlation_matrix, ax=ax);


# Several things:
# 
# - There is a great imbalance between the classes. I will take care of that later (if needed).
# - The data has been anonymized (already said), but since there is no correlation between variables, I think that extra processing steps have been taken. Even, it could also be synthetic or fabricated from real data.


# Studying the distribution for each variable...


# Commented for kernel commit speedup
#fig, ax = plt.subplots(2, figsize=(30,10))
#for i in tqdm(range(200)):
#    var_name = f"var_{i}"
#    sns.distplot(train_df[var_name], ax=ax[0], label=var_name)
#    sns.distplot(test_df[var_name], ax=ax[1], label=var_name)
#ax[0].set_title("KDEs of TRAIN variables.")
#ax[1].set_title("KDEs of TEST variables.")


# Distributiuon of several variables:


# for example, this ones
variables = ['var_0', 'var_23', 'var_89', 'var_112', 'var_152', 'var_199']
fig, ax = plt.subplots(2,3, figsize=(16,8))
for i in range(len(variables)):
    var_name = variables[i]
    sns.distplot(train_df[var_name], ax=ax[i//3, i%3], label='train')
    sns.distplot(test_df[var_name], ax=ax[i//3, i%3], label='test')
    ax[i//3, i%3].set_title(f"Distribution of variable {var_name}")
    ax[i//3, i%3].legend()


# Differences between categories (WIDGET)


#from IPython.html.widgets import *
from ipywidgets import *


def plot_kde(var_index):
    sns.distplot(train_df.loc[train_df['target'] == 0, 'var_'+str(var_index)], label='0')
    sns.distplot(train_df.loc[train_df['target'] == 1, 'var_'+str(var_index)], label='1')
    plt.legend()
    plt.show()

interact(plot_kde, var_index=np.arange(200));


# > Above is an interactive IPython widget. If you can't see it, Fork the notebook and run the cell. It shows the difference between distributions of positive/negative transactions with a dropdown to select what variable to display.
# 
# However, we can see that (for example in var_6), there are noticeable differences on the distribution


# Given the data distribution and the fact that ALL the features are uncorrelated, I think that this data has been transformed (projected onto another space/resampled...).
# 
# I suppose that the original data had, at least, some categorical features, so I will check that by counting repeated values on each feature...


nuniq_train_true = train_df[train_df['target'] == 1].iloc[:, 2:].nunique(axis=0)
nuniq_train_false = train_df[train_df['target'] == 0].iloc[:, 2:].nunique(axis=0)

nuniq_train_true /= len(train_df[train_df['target'] == 1])
nuniq_train_false /= len(train_df[train_df['target'] == 0])

fig, ax = plt.subplots(figsize=(30,5))
ax.bar(np.arange(start=0, stop=400, step=2), height=nuniq_train_true, color='red', label='1')
ax.bar(np.arange(stop=400, start=1, step=2), height=nuniq_train_false, color='blue', label='0', alpha=.7)
ax.set_title("Number of different values per varibale on each category (normalized)")
ax.legend();


# Another way of see the same (maybe more clear)


len_true_samples = len(train_df[train_df['target']==1])
unique_ratios = train_df[train_df['target']==1].iloc[:,2:].apply(lambda c: len(c.unique()) / len_true_samples, axis=0)
fig, ax = plt.subplots(figsize=(30,30))
sns.barplot(y=unique_ratios.index.tolist(), x=1-unique_ratios.values, ax=ax)
ax.set_title('Percentage of repeated values for each variable FOR TRUE SAMPLES');


len_false_samples = len(train_df[train_df['target']==0])
unique_ratios = train_df[train_df['target']==0].iloc[:,2:].apply(lambda c: len(c.unique()) / len_false_samples, axis=0)
fig, ax = plt.subplots(figsize=(30,30))
sns.barplot(y=unique_ratios.index.tolist(), x=1-unique_ratios.values, ax=ax)
ax.set_title('Percentage of repeated values for each variable FOR FALSE SAMPLES');


# ### Some Thoughts
# 
# - The data seems to have been generated by some process of taking the real values and then, projecting them onto other space.
#   * There are repeated values (even though they are real numbers), so maybe they originally come categorical variables.
# - The FALSE data could had been augmentated
#     - When discriminating by type, the number of repeated values for the 0 samples is higher. This is an indicator that this data could had augmentated (through bootstraping techniques, for example).
# - var_68 seems to be highly repeated on TRUE and FALSE samples (distributions point that the values are focused around $5.02$)


# ## Data splitting


input_names = [f'var_{i}' for i in range(200)]


from sklearn.model_selection import train_test_split


X_train, X_test = train_test_split(train_df, test_size=0.4, random_state=196)


print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")


# ### Some evaluation utilities


from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

def metrics_summary(y_real, y_pred):
    """ Returns a  figure with the ROC curve, the Accuracy and AUC metrics (in the figure title)"""
    fpr, tpr, thresholds = roc_curve(y_real, y_pred)
    
    fig, ax = plt.subplots(1, figsize=(5,5))
    ax.plot(fpr, tpr, color='red')
    ax.plot([0,1], [0,1], color='black', linestyle='--')
    ax.set_title('ROC Curve')
    plt.close()
    
    acc = accuracy_score(np.array(y_pred>.5, dtype='int'), y_real)
    auc = roc_auc_score(np.array(y_pred>.5, dtype='int'), y_real)
    
    print(f"- ACC {acc}\n- AUC {auc}")
    
    return fig


def predict_submission(model):
    """ Takes a model and predicts on the test split. Returns the submission DataFrame"""
    preds = model.predict_proba(test_df.drop("ID_code", axis=1))[:,1]
    return pd.DataFrame({'ID_code':test_df['ID_code'], 'target':preds})


# ## Baseline: Gradient Boosting (using all features and no CV)


import lightgbm as lgbm


bst = lgbm.LGBMClassifier()


%%time
bst.fit(X_train[input_names], X_train['target']);


preds = bst.predict_proba(X_test[input_names])[:,1]
metrics_summary(X_test['target'], preds)


# 0.87 AUC. Let's see if we can achieve a higher value


# **Studying the result:**
# 
# I will use the built-in explainer by LightGBM and then make an estimation using SHAP


importances = pd.DataFrame({'Feature': input_names, 'Importance':bst.feature_importances_}).sort_values('Importance', ascending=False)

fig, ax = plt.subplots(figsize=(30,30))
sns.barplot(y=importances.Feature, x=importances.Importance, ax=ax)
ax.set_title('Feature importances');


# Using SHAP estimations for 1k training samples


vals = X_train[input_names].sample(1000)
explainer = shap.TreeExplainer(bst, data=vals, model_output='probability', feature_perturbation='interventional')
shap_values = explainer.shap_values(vals)


shap.summary_plot(shap_values, vals)


interact(lambda var: shap.dependence_plot(f"var_{var}", shap_values, vals), var=np.arange(200));


# For example, estimating which features are more important at this particular example
shap.force_plot(explainer.expected_value, shap_values[110,:], vals.iloc[110,:])


# ### Conclussions
# 
# There are a considerable number of variables that have not been used. It would be interesting to remove them.
# 
# Using SHAP, (interact widget), I found that some variables (68, 136, ...) do not affect to the model's outputs.


# # Feature engineering: Applying the lessons from the analysis
# 
# The thing that most catches my attention is the repetition of the values. I have two theories ATM: 
#   - (Several) Original values were categorical
#   - Data has been augmentated using bootstraping
#   
# I will create an extra feature for each one of the originals to encode this repetition information (freq. encoding of the features) and train the same base model to see if that helps


var_names = [f'var_{i}' for i in range(200)] 
var_enc_names = [f'var_{i}_freq' for i in range(200)] 


def get_hist_frequencies(dataframe):
    hist_vars = {}
    for v in var_names:
        hist_vars[v] = dataframe[v].value_counts()
    return hist_vars

def encode_freqs(dataframe, return_calc_hist = False):
    """Adds 200 more feature with the frequency encodings of the variables"""
    # Build histogram of frequencies of each variable
    hist_vars = {}
    for v in var_names:
        hist_vars[v] = dataframe[v].value_counts()
        dataframe[v+"_freq"] = dataframe[v].map(hist_vars[v])
    
    if return_calc_hist:
        return dataframe, hist_vars
    return dataframe

def encode_freqs_with_hist(dataframe, histogram_data):
    dataframe = dataframe.copy()
    for v in var_names:
        dataframe[v+"_freq"] = dataframe[v].map(histogram_data[v])
    return dataframe


histogram_vars = get_hist_frequencies(pd.concat([train_df[var_names], test_df[var_names]]))


train_df_encoded = encode_freqs_with_hist(train_df, histogram_vars)


X_train, X_test = train_test_split(train_df_encoded, test_size=0.4, random_state=196)


# ### Train the same base model on the new data
# 
# > (Not using CV because I want to rapid prototype ideas)


bst = lgbm.LGBMClassifier()


%%time
bst.fit(X_train[var_names+var_enc_names], X_train['target']);


preds = bst.predict_proba(X_test[var_names+var_enc_names])[:,1]
metrics_summary(X_test['target'], preds)


# The frequency encoding made an improvement of 0.02 AUC with respect to the baseline. Let's see if the new features are actually useful


importances = pd.DataFrame({'Feature': var_names+var_enc_names, 'Importance':bst.feature_importances_}).sort_values('Importance', ascending=False)
fig, ax = plt.subplots(figsize=(30,80))
sns.barplot(y=importances.Feature, x=importances.Importance, ax=ax)
ax.set_title('Feature importances');


# It doesn't seem that the new features drastically improve the model. The gain on AUC has to be due to the few new features the model uses, but the difference is so minimal that adding 200 plus features is not worth it.


# (I'm pretty sure of the theories of value repetitions, though)
# 
# 
# Let's fine-tune the model with the base data to see if we can break the 0.9 barrier.


# ### Bayesian optimization
# 
# I'm using a 3-fold CV to search optim params


from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer


search_space = {
    "feature_fraction": Real(0.001, 0.4),
    "max_depth": Integer(10, 25)
}

base_params = {
    "boost_from_average": "false",
    "metric" : "auc",
    "tree_learner": "serial",
}

optimizer_args = {
    'acq_func': "EI",
    'n_initial_points': 15,
    'acq_optimizer': 'sampling'
}


model = lgbm.LGBMClassifier(**base_params)

bayes_search = BayesSearchCV(
    estimator=model,
    search_spaces=search_space,
    n_iter=32,
    cv=3,
    n_jobs=-1,
    scoring='roc_auc',
    optimizer_kwargs=optimizer_args,
    random_state=2343
)

def on_step(optim_result):
    score = bayes_search.best_score_
    print("Best score: %s " % (score,))
    if score >= 0.98:
        print('Interrupting!')
        return True


# disabled for time execution reasons
#%%time
#bayes_search.fit(X_train[var_names+var_enc_names], X_train['target'], callback=on_step);


# I disabled the former cell for commit execution time reasons, but I paste the execution output here:


"""
Best score: 0.8819925481391506 
Best score: 0.8819925481391506 
Best score: 0.8819925481391506 
Best score: 0.8824897736221471 
Best score: 0.8824897736221471 
Best score: 0.8824897736221471 
Best score: 0.8824897736221471 
Best score: 0.8824897736221471 
Best score: 0.8824897736221471 
Best score: 0.8824897736221471 
Best score: 0.8824897736221471 
Best score: 0.8824897736221471 
Best score: 0.8824897736221471 
Best score: 0.8824897736221471 
Best score: 0.8824897736221471 
Best score: 0.8824897736221471 
Best score: 0.8824897736221471 
Best score: 0.8824897736221471 
Best score: 0.8824897736221471 
Best score: 0.8824897736221471 
Best score: 0.8824897736221471 
Best score: 0.8824897736221471 
Best score: 0.8824897736221471 
Best score: 0.8824897736221471 
Best score: 0.8824897736221471 
Best score: 0.8824897736221471 
Best score: 0.8824897736221471 
Best score: 0.8824897736221471 
Best score: 0.8824897736221471 
Best score: 0.8826816117256729 
Best score: 0.8826816117256729 
Best score: 0.8826816117256729 
CPU times: user 1min 18s, sys: 26.5 s, total: 1min 45s
Wall time: 19min 5s
"""


#params_model.update(bayes_search.best_params_)
#params_model

# These are the best hyperparams obtained using bayesian optimization
params_model = {
 'boost_from_average': 'false',
 'metric': 'auc',
 'tree_learner': 'serial',
 'feature_fraction': 0.10927610745498884,
 'max_depth': 10,
 'n_estimators': 100
}



# Train the model on the entire training set


model = lgbm.LGBMClassifier(**params_model)


%%time
model.fit(X_train[var_names+var_enc_names], X_train['target']);


preds = model.predict_proba(X_test[var_names+var_enc_names])[:,1]
metrics_summary(X_test['target'], preds)


# Its interesting how the `feature_fraction` hyperparam shows that it is the most important one. As we force each tree to use less features from the total, the overall ensemble works better. It seems that there are some few important variables (among the 200s) that really matter...


# ## EXTRA: Meta modelling
# 
# On the former part, I found that the GB works best if only few variables are used to build each tree, so I'm going to make one last try :
# 
# I'm going to fit 200 simple models, each one with both a particular var and its frequency encoded representation, and later merge all predictions using a simple logistic regression model (what is known as meta modelling).


from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator

from scipy.special import logit

class MetaModel(BaseEstimator):
    """ Estimator which contains 200 models that are fitted with each var and var_encoded"""
    def __init__(self):
        super(MetaModel)
        self.models = []
        self.merger = LogisticRegression(solver='lbfgs')
    
    def fit(self, X, y, var_names, var_enc_names):
        #Train boosting models
        for v, venc in notebook(zip(var_names, var_enc_names)):
            model = lgbm.LGBMClassifier(**{
                     'boost_from_average': 'false',
                     'metric': 'auc',
                     'tree_learner': 'serial',
                     'max_depth': 10,
                     'n_estimators': 100
                }
            )
            model.fit(X[[v, venc]], y)
            self.models.append(model)
        
        # Train merger
        preds = self._ensemble_predict(X, var_names, var_enc_names)
        self.merger.fit(preds, y)
        

        return self
    
    def predict(self, X, var_names, var_enc_names):
        predictions = self._ensemble_predict(X, var_names, var_enc_names)
        preds = self.merger.predict_proba(predictions)[:, 1]
    
        return preds
    
    def _ensemble_predict(self, X, var_names, var_enc_names):
        index = 0
        predictions = np.zeros((len(X), len(self.models)))
        for v, venc in notebook(zip(var_names, var_enc_names)):
            model = self.models[index]
            predictions[:, index] = model.predict_proba(X[[v, venc]])[:,1]    
            index+=1
        return predictions


var_names = [f"var_{i}" for i in range(200)]
var_enc_names = [i+"_freq" for i in var_names]

mmodel = MetaModel()


mmodel.fit(X_train, X_train['target'], var_names, var_enc_names)


preds = mmodel.predict(X_test, var_names, var_enc_names)


metrics_summary(X_test['target'], preds)


# ## THE END


# The original dataset has been anonymized, and I'm pretty sure that, at least, some of the columns were original categorical variables (because the number of different values on the train/test sets). I tried to exploit that and got around 0.90 AUC, which its not bad. However, this can be further improved (as the competiton LB shows). 



