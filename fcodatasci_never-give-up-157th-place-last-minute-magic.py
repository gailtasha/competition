# With so many impressive codes and extensive explanations of the 'magic', the only thing i can add is the following: never give up.
# I was struggling a lot to find this features that were supposed to improve the score of the model. I tried different things, from feature interactions to ARIMA, from counts of positives and negatives to fourier coefficient, nothing seemed to work. At one point i said in the forum that I did not think i could get into the bronze medal range (being this my first kaggle competition) but that it was worth following the competition becouse of the great kernels and discussions posted from all of you... (so much to learn!!)
# 
# But I did not give up and kept trying new stuff untill the end... since there were so many hints in the forum, I knew the magic has to be on value counts... in employing the frequency is some way. Finally, with just a few hours to competition deadline, I managed to get into silver range (157th place). So here it goes:


# # Imports


import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tqdm import tqdm_notebook as tqdm
from subprocess import check_output
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
random_state = 2019
np.random.seed(random_state)


# # Load Data


df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
numcols = [col for col in df_train.columns if col.startswith('var_')]


# # Remove Fake Test Data


test = df_test.copy()
test.drop(['ID_code'], axis=1, inplace=True)
test = test.values

unique_samples = []
unique_count = np.zeros_like(test)
for feature in tqdm(range(test.shape[1])):
    _, index_, count_ = np.unique(test[:, feature], return_counts=True, return_index=True)
    unique_count[index_[count_ == 1], feature] += 1

# Samples which have unique values are real the others are fake
real_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) > 0)[:, 0]
synthetic_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) == 0)[:, 0]


df_test = df_test.drop(synthetic_samples_indexes)


# # Concatenate train and test, find value frequencies


full = pd.concat([df_train, df_test])
for col in numcols:
    full['count'+col] = full[col].map(full[col].value_counts())


full.head()


plt.figure(figsize=(12,5))
plt.subplot(121)
sns.distplot(full[full['target']==1]['countvar_12'], label='target = 1')
sns.distplot(full[full['target']==0]['countvar_12'], label='target = 0')
plt.legend()
plt.subplot(122)
sns.distplot(full[full['target']==1]['var_12'], label='target = 1')
sns.distplot(full[full['target']==0]['var_12'], label='target = 0')
plt.legend()


# At this point I did not understand how could this work. The magic is here, but how is it usefull? 


plt.figure(figsize=(12,5))
plt.subplot(121)
sns.distplot(full[full['target']==1]['countvar_61'], label='target = 1')
sns.distplot(full[full['target']==0]['countvar_61'], label='target = 0')
plt.legend()
plt.subplot(122)
sns.distplot(full[full['target']==1]['var_61'], label='target = 1')
sns.distplot(full[full['target']==0]['var_61'], label='target = 0')
plt.legend()


# And decided to look for similar features, maybe target encoding on this kind of features could be usefull.


codecols = ['countvar_61','countvar_45','countvar_136','countvar_187',
            'countvar_74','countvar_160', 'countvar_199','countvar_120',
            'countvar_158','countvar_96']


# The plot for var_61 did not have to mean anything, just that there are more unique values, hence the frequency for all of them is lower. But, this could give some kind of categorical features. I decided to divide the actual row value by its corresponding frequency, hoping to get some sort of 'pseudo-categories'. If some specific values repeat to much, then those values will get divided by a large amount and end up as low values. Values that have a low frequency will keep their magnitude... etc. This could be more evident in columns with low standard deviation... had to try. 


for col in numcols:
    full['ratio'+col] = full[col] / full['count'+col]


# And it seemed to work... for some columns better than for others, but i really did not have much time left to explore. I had to use this or i would not have time to train the model before submission deadline


sns.distplot(full['ratiovar_81'])


sns.distplot(full['ratiovar_61'])


sns.distplot(full['ratiovar_146'])


# Define the functions for target encoding:
def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))

def target_encode(trn_series=None,
                  tst_series=None,
                  target=None,
                  min_samples_leaf=1,
                  smoothing=1,
                  noise_level=0):

    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)


# # Split dataset back in train and test


ncols = [col for col in full if col not in ['target', 'ID_code']]
df_train = full[~full['target'].isna()]
df_test = full[full['target'].isna()]
df_test.drop('target', axis=1, inplace=True)


# # Begin the modelling - LightGBM


# From Bayesian Optimization:
# iter | target | featur... | lambda_l1 | lambda_l2 | learni... | max_depth | min_da... | min_ga... | min_su... | num_le... | 
# 26  | 0.9071 | 0.04901    | 0.5563   | 4.772     | 0.1295    | 0.6435    | 99.19     | 0.1414    | 0.3023    | 2.334 |

lgb_params = {
    "objective" : "binary",
    "metric" : "auc",
    "boosting": 'gbdt',
    "max_depth" : -1,
    "num_leaves" : 2,
    "learning_rate" : 0.02, # Lower it for actual submission
    "bagging_freq": 5,
    "bagging_fraction" : 0.4,
    "feature_fraction" : 0.05,
    "min_data_in_leaf": 100,
    "min_sum_hessian_in_leaf": 0.3,
    "lambda_l1":0.556,
    "lambda_l2":4.772,
    "tree_learner": "serial",
    "boost_from_average": "false",
    "bagging_seed" : random_state,
    "verbosity" : 1,
    "seed": random_state}

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
oof = df_train[['ID_code', 'target']]
oof['predict'] = 0
predictions = df_test[['ID_code']]
val_aucs = []
feature_importance_df = pd.DataFrame()


features = [col for col in df_train.columns if col not in ['target', 'ID_code']]
for fold, (trn_idx, val_idx) in enumerate(skf.split(df_train, df_train['target'])):
    d_train = df_train.iloc[trn_idx]
    d_val = df_train.iloc[val_idx]
    d_test = df_test.copy()

    d_val['type'] = 'val'
    d_test['type'] = 'test'
    d_valtest = pd.concat([d_val, d_test])

    for var in codecols:
        d_train['encoded' + var], d_valtest['encoded' + var] = target_encode(d_train[var],
                                                                          d_valtest[var],
                                                                          target=d_train.target,
                                                                          min_samples_leaf=100,
                                                                          smoothing=10,
                                                                          noise_level=0.01)

    real_test = d_valtest[d_valtest['type']=='test'].drop('type', axis=1)
    real_val = d_valtest[d_valtest['type']=='val'].drop('type', axis=1)

    features = [col for col in d_train.columns if col not in ['target', 'ID_code']]
    X_test = real_test[features].values
    X_train, y_train = d_train[features], d_train['target']
    X_valid, y_valid = real_val[features], real_val['target']

    p_valid, yp = 0, 0
    X_t, y_t = X_train.values, y_train.values
    X_t = pd.DataFrame(X_t)
    X_t = X_t.add_prefix('var_')

    trn_data = lgb.Dataset(X_t, label=y_t)
    val_data = lgb.Dataset(X_valid, label=y_valid)
    evals_result = {}
    lgb_clf = lgb.train(lgb_params,
                        trn_data,
                        100000,  # Submission with: 100000
                        valid_sets=[trn_data, val_data],
                        early_stopping_rounds=3000, # Submission with: 3000
                        verbose_eval=1000,
                        evals_result=evals_result
                        )
    p_valid += lgb_clf.predict(X_valid)
    yp += lgb_clf.predict(X_test)
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = lgb_clf.feature_importance()
    fold_importance_df["fold"] = fold + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    oof['predict'][val_idx] = p_valid
    val_score = roc_auc_score(y_valid, p_valid)
    val_aucs.append(val_score)

    predictions['fold{}'.format(fold + 1)] = yp


mean_auc = np.mean(val_aucs)
std_auc = np.std(val_aucs)
all_auc = roc_auc_score(oof['target'], oof['predict'])
print("Mean auc: %.9f, std: %.9f. " % (mean_auc, std_auc))


# # Save the results


# predictions['target'] = np.mean(predictions[[col for col in predictions.columns if col not in ['ID_code', 'target']]].values, axis=1)
# sub_df = pd.DataFrame({"ID_code":df_test["ID_code"].values})
# sub_df["target"] = predictions['target'].values

# df_test = pd.read_csv('../input/test.csv')
# finalsub = df_test[['ID_code']]
# finalsub = finalsub.merge(sub_df, how='left', on='ID_code')
# finalsub = finalsub.fillna(0)


# Sadly, the first time I trained the model there was an error at the end of the code and could not save the predictions. Luckily I was able to recover the information about the last fold and send the submission with just that information, got into silver range with 0.909 and had a confirmation that the method worked... And enough time to run the prediction again. I used only 3 folds since there was really not that much time left. Sent the submission with 0.9117 CV and got the final score of 0.91342 for the 157th place in the private LB and the silver in my first kaggle competition.
# 
# I learnt a lot from all of you, and have to learn much more from the already published kenerls, so a huge thanks. 
# 
# And yeah... nothing new here and nothing to rival the GM uncles, but a good lesson... never give up !

