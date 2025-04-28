# # Oversampling by shuffling
# 
# As has been pointed out by Branden Murray [here](https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/83882) and demonstrated [here](https://www.kaggle.com/brandenkmurray/randomly-shuffled-data-also-works) random shuffling of the numerical values of the features does not seem to hurt the CV and LB scores as long as the shuffling is performed within the same target class (0 or 1). This opens up a possibility to generate a large number of training data sets equivalent to the original one. This might be useful, for example, for balancing the data set -- now we can easily add to the original data set 9 shuffled copies containing only the data corresponding to target = 1. The purpose of this notebook is to see how this idea works in practice.  
# 
# Let's take the LightGBM parameters from one of a [high scoring public kernel](https://www.kaggle.com/marcospcsj/kernel-cod-valid-cruzada-lgbm) and test this idea (spoiler: no, we are not going to break the leaderboard, sorry).  


###############################################################
# Loading libraries
###############################################################

import os
import shutil
import feather
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scikitplot.metrics import plot_confusion_matrix, plot_roc
import seaborn as sns
import matplotlib.pyplot as plt

###############################################################
# Setting parameters
###############################################################
EARLY_STOPPING=4000
NFOLDS=15
NSHUFFLES_1=2
NSHUFFLES_0=1

train_df = pd.read_csv('../input/train.csv') 
test_df = pd.read_csv('../input/test.csv') 
features = [c for c in train_df.columns if c not in ['ID_code', 'target']]
target = train_df['target']

########################################################################
#  Making folds    
########################################################################

num_folds = NFOLDS
folds = StratifiedKFold(n_splits=num_folds, shuffle=False, random_state=2319)

oof_preds = np.zeros((len(train_df), 1))
test_preds = np.zeros((len(test_df), 1))
roc_cv =[]

########################################################################
#  LightGBM parameters    
########################################################################

param = {
    'bagging_freq': 5,          
    'bagging_fraction': 0.335,   
    'boost_from_average':'false',   
    'boost': 'gbdt',
    'feature_fraction': 0.041,   
    'learning_rate': 0.0083,     
    'max_depth': -1,                
    'metric':'auc',
    'min_data_in_leaf': 80,     
    'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 13,           
    'num_threads': 8,
    'tree_learner': 'serial',   
    'objective': 'binary',      
    'verbosity': 1
}


# Next, we need to define a function to do oversampling by shuffling ("overshuffling" :-)):


def overshuff(df, y, n=2, m=1): 
    """
    df - the data frame to process 
    y - target
    n - the number of shuffled copies of the positive class to add
    m - the number of shuffled copies of the negative class to add
    """
    
    df_1 = df[y==1] # Selecting the observations with target=1
    y_1 = y[y==1]  # Target for this observations 
    
    randoms=np.random.randint(0, 999999, size=n) # Random seeds to be used for shuffling
    
    for i, rand in enumerate(randoms): # n shufflings
        # shuffle:
        df_sh = df_1.apply(lambda x: x.sample(n=len(x), random_state=rand).values) 
        df = pd.concat([df, df_sh]) # add to the original data frame
        y = pd.concat([y, y_1]) # add to the target
        print("Step {} of {}. The random state used is {}".format(i+1, n, rand))
    
    #rand = np.random.randint(0, 999999, size=1)[0] # one more random seed 
    #df, y = shuffle(df, y, random_state=rand) # one last shuffling (just in case!)
    
    print("The random state used for the final shuffling is {}".format(rand))
    
    #########################################
    
    df_1 = df[y==0] # Selecting the observations with target=0
    y_1 = y[y==0]  # Target for this observations 
    
    randoms=np.random.randint(0, 999999, size=m) # Random seeds to be used for shuffling
    
    for i, rand in enumerate(randoms): # m shufflings
        # shuffle:
        df_sh = df_1.apply(lambda x: x.sample(n=len(x), random_state=rand).values) 
        df = pd.concat([df, df_sh]) # add to the original data frame
        y = pd.concat([y, y_1]) # add to the target
        print("Step {} of {}. The random state used is {}".format(i+1, n, rand))
    
    rand = np.random.randint(0, 999999, size=1)[0] # one more random seed 
    df, y = shuffle(df, y, random_state=rand) # one last shuffling (just in case!)
    
    print("The random state used for the final shuffling is {}".format(rand))
    
    #########################################
    
    return df, y


# Here is a handy function that we will use to plot the distribution of predictions.


########################################################################
# Function for plotting the distribution of predictions   
########################################################################
    
def plot_prediction_distribution(y_true, y_pred, ax):
    df = pd.DataFrame({'prediction': y_pred, 'ground_truth': y_true})
    
    sns.distplot(df[df['ground_truth'] == 0]['prediction'], 
                 label='negative', ax=ax)
    sns.distplot(df[df['ground_truth'] == 1]['prediction'], 
                 label='positive', ax=ax)

    ax.legend(prop={'size': 16}, title = 'Labels')


########################################################################
#  Training the model    
########################################################################

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values, target.values)):
    print("Fold idx:{}".format(fold_ + 1))
    
    trn_x = train_df.iloc[trn_idx][features]
    trn_y = target.iloc[trn_idx]
    
    # Apply the overshuffling function to the training data 
    # (excluding the validation set!)
    
    trn_x, trn_y = overshuff(trn_x, trn_y, NSHUFFLES_1, NSHUFFLES_0)
    
    print("Converting the data to lgbm format")
    trn_data = lgb.Dataset(trn_x, label=trn_y)
    val_data = lgb.Dataset(train_df.iloc[val_idx][features], label=target.iloc[val_idx])
    
    print("Training the classifier")
    clf = lgb.train(param, trn_data, 1000000, valid_sets = [trn_data, val_data], 
                    verbose_eval=5000, early_stopping_rounds = EARLY_STOPPING)
    
    print("Making predictions for the validation data")
    val_pred = clf.predict(train_df.iloc[val_idx][features], 
                           num_iteration=clf.best_iteration)
    
########################################################################
    
    print("Computing the AUC score")
    roc_cv.append(roc_auc_score(target.iloc[val_idx], val_pred))
    
    print("AUC = {}".format(roc_auc_score(target.iloc[val_idx], val_pred)))
    oof_preds[val_idx, :] = val_pred.reshape((-1, 1))
    
    print("Making predictions for the test data")
    test_fold_pred = clf.predict(test_df[features], 
                                 num_iteration=clf.best_iteration).\
                                 reshape((-1, 1))
        
    test_preds += test_fold_pred
    
   # preds = pd.DataFrame(oof_preds[val_idx, :], columns=['pos_preds'])
   # preds['neg_preds'] = 1.0 - preds['pos_preds']
   # fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(24, 6))
   # plot_prediction_distribution(target.iloc[val_idx], preds['pos_preds'], ax=ax1);
   # plot_roc(target.iloc[val_idx], preds[['neg_preds','pos_preds']], ax=ax2);
   # plot_confusion_matrix(target.iloc[val_idx], oof_preds[val_idx, :]>0.5, ax=ax3);
   # path_fig_full = 'Fold_{}_'.format(fold_ + 1)+'model_diagnostics.png'
   # fig.savefig(path_fig_full) 
    
########################################################################
#  Computing statistics   
########################################################################

test_preds /= num_folds

roc_score_1 = round(roc_auc_score(target.ravel(), oof_preds.ravel()), 5)
roc_cv = np.array(roc_cv)
roc_score = round(sum(roc_cv)/len(roc_cv), 5)
st_dev = round(np.array(roc_cv).std(), 5)

print("Average of the folds' AUCs = {}".format(roc_score))

print("Combined folds' AUC = {}".format(roc_score_1))

print("The standard deviation = {}".format(st_dev))

print("Saving OOF predictions")
oof_preds = pd.DataFrame(np.column_stack((train_df.ID_code.values, 
                                          oof_preds.ravel())), 
                        columns=['ID_code', 'target'])
    
oof_preds.to_csv(("LGBM_{}.csv").format(str(roc_score)), index=False)

print("Saving submission file")
sub = pd.DataFrame({"ID_code": test_df.ID_code.values})
sub["target"] = test_preds
sub.to_csv('submission_{}.csv'.format(str(roc_score)), index=False)

