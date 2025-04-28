# ## Import required packages


# The Machine Learning package that we use is h2oAutoML. It's a simplified machine learning tool, which involves various machine learning tools. It will automatically develop GBM, XGBoost, DeepLearning models and have Stacked Ensemble as well.


import numpy as np
import pandas as pd
import h2o
from h2o.automl import H2OAutoML


# #### Initiate a h2o cluster with a maximum memory size of 16G.


h2o.init(max_mem_size='16G')


# #### Set data path for training and testing dataset.


train_path = '../input/train.csv'
test_path = '../input/test.csv'


# #### Our target variable is binary. So we set the column type to 'enum' in h2o.dataframe.


col_types = {'target': 'enum'}


# #### Load training and testing data into h2o dataframe.


train = h2o.import_file(path=train_path, col_types=col_types)


test = h2o.import_file(path=test_path)


# #### Assign target variable 'target' to y, and all other variables but 'ID_code' and 'target' as independent variables to X.


y = 'target'
X = [name for name in train.columns if name not in ['ID_code', y]]


# #### Make 'target' factors.


train[y] = train[y].asfactor()


# ## Train the model
# H2OAutoML is very simple. We decide to build a maximum of 50 models or stop after 5 hours (18000 seconds). Then we train the model with assigned X, y, and our training data.


model = H2OAutoML(max_models=50,
                  max_runtime_secs = 18000,
                seed=12345)
model.train(x=X, y=y, training_frame=train)


# We list all the models that we built.


lb = model.leaderboard
lb.head(rows=lb.nrows)


# model.leader is the best model in terms of auc in all models that we built. model.leader.predict will return the prediction of 'target' on our testing data, and we store the result into a new dataframe 'result'.


model.leader


result = model.leader.predict(test)
result


# Now, we combine our test data with the first column in result, which is the prediction itself.


sub = test.cbind(result[0])


# Then we select 'ID_code' and 'predict' from sub because those two columns are the only ones that we need in our submission file. We rename the 'predict' column to 'target' as required.


sub = sub[['ID_code','predict']]
sub = sub.rename(columns={'predict':'target'})


# We convert our h2o dataframe sub to a pandas dataframe, and write it to csv.


sub = sub.as_data_frame()
sub.to_csv('submission.csv',index=False)


# ## Shut down the h2o cluster


h2o.cluster().shutdown(prompt=True)

