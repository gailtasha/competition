# I have been following this competition since the beginning and it has been a huge learning experience for me as a beginner. I am documenting here all the techniques I learn via kernels and discussions in this competition.They include - 
# * EXPLORATORY DATA ANALYSIS - 
# >      Basic data exploration 
# >      Visualizing target variable distribution
# >      Principal Component Analysis
# *  FEATURE ENGINEERING - 
# >      Genetic features engineering
# >      Data Augmentation
# *  ALGORITHMS - 
# >       LightGBM (with GPU accelaration)
# >       CatBoost (with GPU accelaration)
# *  HYPERPARAMETER TUNING -
# >       Bayesian Optimization
# >       RandomizedSearchCV
# *  RESULT AGGREGATION - 
# >       Ranking
# >       Blending
#      


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics import mean_squared_error,roc_auc_score
import gplearn
from gplearn.genetic import SymbolicTransformer, SymbolicRegressor
from gplearn.functions import make_function
from gplearn.fitness import make_fitness
from sklearn.model_selection import StratifiedKFold,KFold
import random
from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMClassifier
from bayes_opt import BayesianOptimization
import warnings
import lightgbm as lgb
from catboost import Pool, CatBoostClassifier
from scipy.stats import rankdata


# Reading train and test files


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# **Exploratory Data Analysis**


#distribution of classes in our dataset is uneven
sns.countplot(train_df['target'])


train_df.describe()


# PRINCIPAL COMPONENT ANALYSIS


x = train_df.iloc[:, 2:]
y = train_df.iloc[:, 1]


#using randomized PCA and trying to map the input features into 100 components
rpca = PCA(n_components=100, svd_solver='randomized')
rpca.fit(x)
plt.plot(np.cumsum(rpca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')


# Clearly, 100 features can explain only 90 percent of variance in this data.
# It looks like this data was PCA'd already


# # FEATURE ENGINEERING


# GENETIC FEATURE ENGINEERING


#defining a fitness function analogous to loss function
def _my_fit(y, y_pred, w):
    return mean_squared_error(y,y_pred)
my_fit = make_fitness(_my_fit, greater_is_better=False)


# Choose the mathematical functions we will combine together
function_set = ['add', 'sub']
#We can use the below functions as well to create features
#'div', 'log', 'sqrt', 'log', 'abs', 'neg', 'inv',  'max', 'min', 'sin', 'cos', 'tan' 

# Create the genetic learning regressor
#generations is an important parameter in this function
gp = SymbolicRegressor(function_set=function_set, metric = my_fit,
                       verbose=1, generations = 2, 
                       random_state=0, n_jobs=-1)


# Using NUMPY structures, remove one feature (column of data) at a time from the training set
# Use that removed column as the target for the algorithm
# Use the genetically engineered formula to create the new feature
# Do this for both the training set and the test set

X1a = np.array(x)
sam = X1a.shape[0]
col = X1a.shape[1]
X2a = np.zeros((sam, col))

X_test1a = np.array(test_df.drop('ID_code',axis=1))
sam_test = X_test1a.shape[0]
col_test = X_test1a.shape[1]
X_test2a = np.zeros((sam_test, col_test))

for i in range(col) :
    X = np.delete(X1a,i,1)
    y = X1a[:,i]
    gp.fit(X, y) 
    X2a[:,i] = gp.predict(X)
    X = np.delete(X_test1a, i, 1)
    X_test2a[:,i] = gp.predict(X)
    
X2 = pd.DataFrame(X2a)
X_test2 = pd.DataFrame(X_test2a) 


# Add the new features to the existing 200 features
X_test1 = test_df.drop('ID_code',axis=1)
y = train_df['target']
train_df = pd.concat([x, X2], axis=1, sort=False) 
test_df = pd.concat([X_test1, X_test2], axis=1, sort=False)  
train_df = pd.concat([train_df,y],axis=1)
train_df.head()


# DATA AUGMENTATION  --> Generating more samples from existing data to enhance the training set


#augment train in each fold, don't touch valid and test.
#upsample positive instances more.
def augment(x,y,t=2):
    xs,xn = [],[]
    for i in range(t):
        mask = y>0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:,c] = x1[ids][:,c]
        xs.append(x1)

    for i in range(t//2):
        mask = y==0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:,c] = x1[ids][:,c]
        xn.append(x1)

    xs = np.vstack(xs)
    xn = np.vstack(xn)
    ys = np.ones(xs.shape[0])
    yn = np.zeros(xn.shape[0])
    x = np.vstack([x,xs,xn])
    y = np.concatenate([y,ys,yn])
    return x,y


# It prevents overfitting fake interaction appearances. Without data augmentation, it may appear that certain combinations of variables predict target=0 or target=1 but this is just overfitting train. By shuffling values, you remove the possibility of fitting fake interaction appearances in train.


# ALGORITHMS TO USE :
# * LightGBM
# * CatBoost 
# 
# Before running any of the models, we will first learn to exploit them on GPU's


# In Kaggle notebook setting, set the Internet option to Internet connected, and GPU to GPU on.
# We first remove the existing CPU-only lightGBM library and clone the latest github repo.


!rm -r /opt/conda/lib/python3.6/site-packages/lightgbm
!git clone --recursive https://github.com/Microsoft/LightGBM
!apt-get install -y -qq libboost-all-dev


%%bash
cd LightGBM
rm -r build
mkdir build
cd build
cmake -DUSE_GPU=1 -DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=/usr/local/cuda/include/ ..
make -j$(nproc)


!cd LightGBM/python-package/;python3 setup.py install --precompile
!mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd
!rm -r LightGBM


#check if the GPU is blocked or not
!nvidia-smi


# In order to leverage the GPU, we need to set the following parameters:
# *  'device': 'gpu',
# *  'gpu_platform_id': 0,
# * 'gpu_device_id': 0
# 
# We are now good to go on using LightGBM on GPU


# **CATBOOST on GPU** - Add  task_type="GPU" in model params


# **HYPERPARAMETER TUNING**
# 
# Before running the models, we need to tune the parameters. We will be using two techniques to do so :
# * RandomizedSearchCV
# * Bayesian Optimization


#Use Kfold or StratifiedKFold to split the training set for cross validation
#n_splits is kept around 10 in this competition to prevent overfitting and get better results
bayesian_tr_index, bayesian_val_index  = list(StratifiedKFold(n_splits=2, shuffle=True, random_state=1).split(train_df.iloc[:,2:], train_df.target.values))[0]


#Defining the lightgbm function with nfolds 
import lightgbm as lgb
train_df.columns = [str(x) for x in train_df.columns]
test_df.columns = [str(x) for x in test_df.columns]
target = 'target'
predictors = train_df.columns.values.tolist()
predictors.remove('target')
def LGB_bayesian(
    num_leaves,  # int
    min_data_in_leaf,  # int
    learning_rate,
    min_sum_hessian_in_leaf,    # int  
    feature_fraction,
    lambda_l1,
    lambda_l2,
    min_gain_to_split,
    max_depth):
    num_leaves = int(num_leaves)
    min_data_in_leaf = int(min_data_in_leaf)
    max_depth = int(max_depth)

    assert type(num_leaves) == int
    assert type(min_data_in_leaf) == int
    assert type(max_depth) == int

    param = {
        'num_leaves': num_leaves,
        'max_bin': 63,
        'min_data_in_leaf': min_data_in_leaf,
        'learning_rate': learning_rate,
        'min_sum_hessian_in_leaf': min_sum_hessian_in_leaf,
        'bagging_fraction': 1.0,
        'bagging_freq': 5,
        'feature_fraction': feature_fraction,
        'lambda_l1': lambda_l1,
        'lambda_l2': lambda_l2,
        'min_gain_to_split': min_gain_to_split,
        'max_depth': max_depth,
        'save_binary': True, 
        'seed': 1337,
        'feature_fraction_seed': 1337,
        'bagging_seed': 1337,
        'drop_seed': 1337,
        'data_random_seed': 1337,
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'verbose': 1,
        'metric': 'auc',
        'is_unbalance': True,
        'boost_from_average': False,   

    }    
    
    
    xg_train = lgb.Dataset(train_df.iloc[bayesian_tr_index][predictors].values,
                           label=train_df.iloc[bayesian_tr_index][target].values,
                           feature_name=predictors,
                           free_raw_data = False
                           )
    
    xg_valid = lgb.Dataset(train_df.iloc[bayesian_val_index][predictors].values,
                           label=train_df.iloc[bayesian_val_index][target].values,
                           feature_name=predictors,
                           free_raw_data = False
                           )   

    num_round = 50
    clf = lgb.train(param, xg_train, num_round, valid_sets = [xg_valid], verbose_eval=10, early_stopping_rounds = 10)
    print('predict')
    predictions = clf.predict(train_df.iloc[bayesian_val_index][predictors].values, num_iteration=clf.best_iteration)   
    
    score = roc_auc_score(train_df.iloc[bayesian_val_index][target].values, predictions)
    
    return score


#Defining the bounds of parameters for parameter tuning in bayesian optimization
bounds_LGB = {
    'num_leaves': (2, 4), 
    'min_data_in_leaf': (5, 8),  
    'learning_rate': (0.005, 0.01),
    'min_sum_hessian_in_leaf': (0.0001, 0.01),    
    'feature_fraction': (0.5, 0.6),
    'lambda_l1': (1.0, 3.0), 
    'lambda_l2': (0, 1.0), 
    'min_gain_to_split': (0, 1.0),
    'max_depth':(2,4)
}


LGB_BO = BayesianOptimization(LGB_bayesian, bounds_LGB, random_state=13)
init_points = 8
n_iter = 8
print('-' * 130)

with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    LGB_BO.maximize(init_points=init_points, n_iter=n_iter, acq='ucb', xi=0.0, alpha=1e-6)


#Parameter grid for RandomizedSearchCV
param_grid = {
    'is_unbalance': ['True'],
    'boosting_type': ['gbdt'],
    'max_depth' : [2,3],
    'min_sum_hessian_in_leaf' : [0.0005,0.001],
    'min_gain_to_split': list(range(0,5)),
    'max_delta_step' : list(range(0,5)),
    'lambda_l1': list(range(0,10)),
    'lambda_l2': list(range(0,1)),
    'learning_rate': [0.01],
    'min_data_in_leaf': [50,100,200],
    'max_bin': [500,750]
}


lgb = LGBMClassifier(verbose=1,metric='auc',objective= 'binary')
clf = RandomizedSearchCV(estimator = lgb, param_distributions = param_grid, n_iter = 5,
cv = 3, verbose=2, random_state=42, n_jobs = -1)
clf.fit(train_df.iloc[:,2:],train_df.target)
print(clf.best_params_)
print(clf.best_score_)


#Best parameters from Bayesian Optimization
print(LGB_BO.max['target'])
print(LGB_BO.max['params'])


# **ALGORITHMS**
# 
# LIGHTGBM with DATA AUGMENTATION


#Number of iterations has been kept very large in range of 20000 and max_depth,learning rate very small to make the model learn slowly but correctly

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
    'verbosity': 1}


n_split = 11 #kept around 11 for better results
kf = KFold(n_splits=n_split, random_state=432013, shuffle=True)
y_valid_pred = 0 * train_df.target
y_test_pred = 0


import lightgbm as lgb
for idx, (train_index, valid_index) in enumerate(kf.split(train_df)):
    
    y_train, y_valid = train_df['target'].iloc[train_index], train_df['target'].iloc[valid_index]
    X_train, X_valid = train_df[predictors].iloc[train_index,:], train_df[predictors].iloc[valid_index,:]
    X_tr, y_tr = augment(X_train.values, y_train.values)
    X_tr = pd.DataFrame(X_tr)
    
    print( "\nFold ", idx)
    trn_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_valid, label=y_valid)
    #num_iterations are taken very large , here 50 is just for sample purposes
    fit_model = lgb.train(param, trn_data, 1000000, valid_sets = [trn_data, val_data], verbose_eval=5000, early_stopping_rounds = 4000)
    pred = fit_model.predict(X_valid)
    print( "  auc = ", roc_auc_score(y_valid, pred) )
    y_valid_pred.iloc[valid_index] = pred
    y_test_pred += fit_model.predict(test_df)
y_test_pred /= n_split
    


lightgbm_results = y_test_pred


# CATBOOST


#iterations should be taken large for better results
model = CatBoostClassifier(loss_function="Logloss",
                           eval_metric="AUC",
                           task_type="GPU",
                           learning_rate=0.01,
                           iterations=50,
                           l2_leaf_reg=50,
                           random_seed=432013,
                           od_type="Iter",
                           depth=5,
                           early_stopping_rounds=10,
                           border_count=64
                           #has_time= True 
                          )


for idx, (train_index, valid_index) in enumerate(kf.split(train_df)):
    y_train, y_valid = train_df['target'].iloc[train_index], train_df['target'].iloc[valid_index]
    X_train, X_valid = train_df[predictors].iloc[train_index,:], train_df[predictors].iloc[valid_index,:]
    _train = Pool(X_train, label=y_train)
    _valid = Pool(X_valid, label=y_valid)
    print( "\nFold ", idx)
    fit_model = model.fit(_train,
                          eval_set=_valid,
                          use_best_model=True,
                          verbose=5000,
                          plot=True
                         )
    pred = fit_model.predict_proba(X_valid)[:,1]
    print( "  auc = ", roc_auc_score(y_valid, pred) )
    y_valid_pred.iloc[valid_index] = pred
    y_test_pred += fit_model.predict_proba(test_df)[:,1]
y_test_pred /= n_split


catboost_results = y_test_pred


# RANK AVERAGING - 
# 
# Assign ranks to data, dealing with ties appropriately


predict_list = []
predict_list.append(catboost_results)
predict_list.append(lightgbm_results)


print("Rank averaging on ", len(predict_list), " outputs")
predictions = np.zeros_like(predict_list[0])
for predict in predict_list:
    predictions = np.add(predictions, rankdata(predict)/predictions.shape[0])  
predictions /= len(predict_list)

test_df = pd.read_csv('../input/test.csv')
submission = pd.DataFrame({'ID_code' : test_df['ID_code'], 'target' : predictions})
submission.to_csv('rank_average.csv', index=False)


# BLENDING


test_df = pd.read_csv('../input/test.csv')
predictions = 0.5 * (catboost_results + lightgbm_results)
submission = pd.DataFrame({'ID_code' : test_df['ID_code'], 'target' : predictions})
submission.to_csv('blending.csv', index=False)


predictions = lightgbm_results
submission = pd.DataFrame({'ID_code' : test_df['ID_code'], 'target' : predictions})
submission.to_csv('lgb_best_params.csv', index=False)

