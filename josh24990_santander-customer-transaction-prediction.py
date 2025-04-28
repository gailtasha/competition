# # Santander Customer Transaction Prediction


# This kernel is a work in progress, taking inspiration from:<br>
# <b>LightGBM with GPU support:</b> https://www.kaggle.com/vinhnguyen/gpu-acceleration-for-lightgbm<br>
# <b>Data Augmentation:</b> https://www.kaggle.com/jiweiliu/fast-inplace-shuffle-for-augmentation


# ## Re-compile LGBM with GPU support


!rm -r /opt/conda/lib/python3.6/site-packages/lightgbm<br>
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


# ## Onto the challenge


# Data manipulation and set-up
import numpy as np
import pandas as pd

# Statistics
from scipy import stats
from scipy.stats import norm, skew
from statistics import mode
from scipy.special import boxcox1p

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1)

# Modelling (including set-up & evaluation)
from sklearn.model_selection import KFold, StratifiedKFold
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

# Ignore warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')


# Setting random seed
random_state = 42
np.random.seed(random_state)

# Import data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# Feature engineering
# Saving train & test shapes
# ntrain = train.shape[0]
# ntest = test.shape[0]

# New all encompassing dataset
# all_data = pd.concat((train, test)).reset_index(drop=True)

# Transform target to object
# all_data['target'] = all_data['target'].astype('object')

# Extracting continuous features
# numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

# Check for skewness
# skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
# skewness = pd.DataFrame({'Skew' :skewed_feats})
# skewness.head(5)


# Now let's apply the box-cox transformation to correct for skewness
# skewed_features = skewness.index
# lam = 0.3
# for feature in skewed_features:
#     all_data[feature] = boxcox1p(all_data[feature], lam)


# Check
# numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

# skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
# skewness = pd.DataFrame({'Skew' :skewed_feats})
# skewness.head(5)


# Target back to float
# all_data['target'] = all_data['target'].astype('float64')

# Now to return to separate train/test sets for Machine Learning
# train = all_data[:ntrain]
# test = all_data[ntrain:]


# Data augmentation
def disarrange(a, axis=-1):
    """
    Shuffle `a` in-place along the given axis.

    Apply numpy.random.shuffle to the given axis of `a`.
    Each one-dimensional slice is shuffled independently.
    """
    b = a.swapaxes(axis, -1)
    # Shuffle `b` in-place along the last axis.  `b` is a view of `a`,
    # so `a` is shuffled in place, too.
    shp = b.shape[:-1]
    for ndx in np.ndindex(shp):
        np.random.shuffle(b[ndx])
    return

def augment(x,y,t=2):
    xs,xn = [],[]
    for i in range(t):
        mask = y>0
        x1 = x[mask].copy()
        disarrange(x1,axis=0)
        xs.append(x1)

    for i in range(t//2):
        mask = y==0
        x1 = x[mask].copy()
        disarrange(x1,axis=0)
        xn.append(x1)

    xs = np.vstack(xs)
    xn = np.vstack(xn)
    ys = np.ones(xs.shape[0])
    yn = np.zeros(xn.shape[0])
    x = np.vstack([x,xs,xn])
    y = np.concatenate([y,ys,yn])
    return x,y


# Model parameters
lgb_params = {
    "objective" : "binary",
    "metric" : "auc",
    "boosting": 'gbdt',
    "max_depth" : -1,
    "num_leaves" : 13,
    "learning_rate" : 0.01,
    "bagging_freq": 5,
    "bagging_fraction" : 0.4,
    "feature_fraction" : 0.05,
    "min_data_in_leaf": 80,
    "min_sum_hessian_in_leaf": 10,
    "tree_learner": "serial",
    "boost_from_average": "false",
    "bagging_seed" : random_state,
    "verbosity" : 1,
    "seed": random_state,
    'device': 'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 0}


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
oof = train[['ID_code', 'target']]
oof['predict'] = 0
predictions = test[['ID_code']]
val_aucs = []
feature_importance = pd.DataFrame()


features = [col for col in train.columns if col not in ['target', 'ID_code']]
X_test = test[features].values


for fold, (trn_idx, val_idx) in enumerate(skf.split(train, train['target'])):
    X_train, y_train = train.iloc[trn_idx][features], train.iloc[trn_idx]['target']
    X_valid, y_valid = train.iloc[val_idx][features], train.iloc[val_idx]['target']
    
    N = 5
    p_valid,yp = 0,0
    for i in range(N):
        X_t, y_t = augment(X_train.values, y_train.values)
        X_t = pd.DataFrame(X_t)
        X_t = X_t.add_prefix('var_')
    
        trn_data = lgb.Dataset(X_t, label=y_t)
        val_data = lgb.Dataset(X_valid, label=y_valid)
        evals_result = {}
        lgb_clf = lgb.train(lgb_params,
                        trn_data,
                        100000,
                        valid_sets = [trn_data, val_data],
                        early_stopping_rounds=3000,
                        verbose_eval = 1000,
                        evals_result=evals_result
                       )
        p_valid += lgb_clf.predict(X_valid)
        yp += lgb_clf.predict(X_test)
    fold_importance = pd.DataFrame()
    fold_importance["feature"] = features
    fold_importance["importance"] = lgb_clf.feature_importance()
    fold_importance["fold"] = fold + 1
    feature_importance = pd.concat([feature_importance, fold_importance], axis=0)
    oof['predict'][val_idx] = p_valid/N
    val_score = roc_auc_score(y_valid, p_valid)
    val_aucs.append(val_score)
    
    predictions['fold{}'.format(fold+1)] = yp/N


# Submission
predictions['target'] = np.mean(predictions[[col for col in predictions.columns if col not in ['ID_code', 'target']]].values, axis=1)
predictions.to_csv('lgb_all_predictions.csv', index=None)
sub = pd.DataFrame({"ID_code":test["ID_code"].values})
sub["target"] = predictions['target']
sub.to_csv("lgb_submission.csv", index=False)
oof.to_csv('lgb_oof.csv', index=False)

