# Thanks @Yassine Alouini for his great kernel https://www.kaggle.com/yassinealouini/permutation-importance. This kernel is to check my FE based on his kernel: In 3 of the 5 folds, the new feature ranks in top20, better than 90% of the origin ones


import eli5
from eli5.sklearn import PermutationImportance
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold
# Need this to display each ELI5 HTML report within the for loop.
from IPython.display import display
%matplotlib inline


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


train_df['new'] = train_df['var_21'] - train_df['var_22']
test_df['new'] = test_df['var_21'] - test_df['var_22']


SEED = 314
CV = KFold(n_splits=5)
FEATURES = train_df.drop(["target", "ID_code"], axis=1).columns.tolist()
TARGET_COL = "target"


for fold, (train_idx, valid_idx) in enumerate(CV.split(train_df, train_df[TARGET_COL])):
    clf = LGBMClassifier(random_state=SEED, n_threads=-1, 
                         eval_metric="auc", n_estimators=10000)
    clf.fit(train_df.loc[train_idx, FEATURES], 
            train_df.loc[train_idx, TARGET_COL], 
            eval_metric="auc",
            verbose=0,
            early_stopping_rounds=1000,
            eval_set=[(train_df.loc[valid_idx, FEATURES], 
                       train_df.loc[valid_idx, TARGET_COL])])
    permutation_importance = PermutationImportance(clf, random_state=SEED)
    permutation_importance.fit(train_df.loc[valid_idx, FEATURES], 
                               train_df.loc[valid_idx, TARGET_COL])
    print(f"Permutation importance for fold {fold}")
    display(eli5.show_weights(permutation_importance, feature_names = FEATURES))

