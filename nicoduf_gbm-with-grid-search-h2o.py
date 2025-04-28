# Hi Kagglers ! 
# 
# This is a notebook to test what you can do in a few minutes with h2o's migthy GBM. I also included some basic grid search and early stopping to make the model more competitive. There is no CV in this kernel, I simply used a train/validation split framework. Unfortunately h2o does not allow you to correct for stratification when splitting (to my knowledge), so I rather used sklearn splitting function. 
# 
# A neat feature is that you can keep an eye on the scoring history along the training with scoring_history() function on many metrics of interest, not only AUC ROC.
# 
# Feel free to comment, fork and upvote, happy kaggling, Cheers!


# # Contents
# 1. [Start h2o and load the data](#step1)
# 2. [Define a grid and train](#step2)
# 3. [Best model](#step3)
# 4. [Submission](#step4)


# ## Start h2o and load the data  <a name="step1"></a>
# 
# Start the h2o cluster and load train, validation and test datasets as h2oFrames. 
# 
# Train/Validation split is done with sklearn function, to correct for stratification on the target column.


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.grid.grid_search import H2OGridSearch

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
%matplotlib inline
import os, gc
print(os.listdir("../input"))

h2o.init()

train_df = pd.read_csv("../input/train.csv")
valid_rate = .15
train_df, valid_df, tr_y, va_y = train_test_split(train_df, train_df['target'], stratify = train_df['target'], test_size=valid_rate, random_state = 42)

train = h2o.H2OFrame(train_df)
valid = h2o.H2OFrame(valid_df)
test = h2o.import_file("../input/test.csv")

import gc
del train_df, valid_df, tr_y, va_y
gc.collect()


y = 'target'
x = train.columns[2:]
train[y] = train[y].asfactor()


# ## Define a grid and train  <a name="step2"></a>
# 
# With the RandomDiscrete strategy you test *n_models*, with parameters randomly chosen from the grid. 
# 
# 2500 trees should be enough here to reach the early stopping *AUC* criteria on the validation frame.


# CHANGE THIS PARAMETER to test as many models as you wish
n_models = 8
grid_params = {
    'max_depth': [2, 3],
    'col_sample_rate': [.6, .7],
    'learn_rate': [.09, .1],
    'learn_rate_annealing': [1],
    'min_rows': [110, 90],
    'sample_rate': [.7]
}

gbm_grid = H2OGridSearch(model=h2o.estimators.H2OGradientBoostingEstimator,
                grid_id='gbm_grid', 
                hyper_params=grid_params,
                search_criteria={'strategy': 'RandomDiscrete', 'max_models': n_models})


gbm_grid.train(x=x, y=y, training_frame=train, validation_frame=valid,
            distribution='bernoulli',
            ntrees=2500,
            score_tree_interval = 20,
            stopping_rounds = 4,
            stopping_metric = "AUC",
            stopping_tolerance = 1e-4,
            seed = 1)


# ## Best model  <a name="step3"></a>
# 
# Now sort the models of the grid from decreasing order on the *auc* criteria, and keep the first. You can check what are the parameters that were used to reach the best score. A detailed history on the different metrics is informative, especially for the *AUC* of the Precision Recall curve since target column is not balanced. 



gridperf = gbm_grid.get_grid(sort_by='auc', decreasing=True)
best_model = gridperf.models[0]
for par in grid_params:
    if par in best_model.params:
        print('par: ' + par); print(best_model.params[par])


best_model.scoring_history()


history = pd.DataFrame(best_model.scoring_history())
history.plot(x='number_of_trees', y = ['validation_auc', 'validation_pr_auc'])


# ## Submission  <a name="step4"></a>
# 
# Here's your submission csv that should get you a LB score around .897 ! 


preds = best_model.predict(test)
submission = pd.read_csv('../input/sample_submission.csv')
submission['target'] = preds['p1'].as_data_frame()
submission.to_csv('gbm_submission.csv', index = False)
submission.head()

