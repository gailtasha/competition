# # Deep Learning + LGBM + Weighted Combination
# 
# This kernel will always be running with different parameters and approaches until before the competition deadline.
# 
# Feel free to upvote,fork and test the presented models with different training options, to see if a better score with the following models is possible.
# 
# If forked Please try the different combinations:
# - Only Feature Engineering ( omitting some features maybe)
# - Only Augmented
# - Augmented + Feature Engineering (Augment before or after FE)
# - Augmented + Feature Engineering + folds
# - Augmented + Feature Engineering + full
# - Combination of different prediction weights
# - etc..
# 
# You can also check here for weighted CV approach that will make a minor better prediction that you might need in the competition:
# https://www.kaggle.com/hjd810/introducing-weighted-cross-validation
# 
# Enjoy ! 
# 
# Any comments are appreciated (added motivation <3)
# 
# 1. [Training Options](#options)
# 2. [Sampling](#sample)
# 3. [Feature Engineering](#fe)
# 4. [LGBM Model](#lgbm)
# 5. [Keras Model](#keras)
# 6. [Combination Vis](#vis)
# 7. [Submission](#sub)


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()
import plotly.graph_objs as go
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout, BatchNormalization
from sklearn.model_selection import KFold,StratifiedKFold
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import  train_test_split
from keras import backend as K
from keras import optimizers
import keras as k
import time
from keras.callbacks import EarlyStopping
from sklearn.utils import class_weight
import warnings
warnings.filterwarnings('ignore')


# ## Running Models Options
# <a id='options'></a>
# The options here helps you check the different combinations for training and check which fits best.


sampledf = False #Uses a specific amount of rows , use this for faster training and testing functionalities and features
frac_sample = 0.03 #fraction of the data to use
augmnt = False #use an augmented data set
fold_train = True
agmnt_between = True #augment training data between folds
kfold_shuffle = False
use_perc = True #using percentiles in feature engineering
#--------------------------------------------------------------
#Keras options
#Weighted Classes when training 
weighted = False
balanced = False #balanced weights
#-------------------
#train test Split
tst_size = 0.3

sub_name = 'submission'
print('Options Active: \n\t SampledDF: {} frac: {} \n\t Augmentation: {}\n\t Weighted: {}\n\t Balanced: {}\
      \n\t agmnt_between:{}\n\t Percentiles: {}'.format(sampledf,frac_sample,augmnt,weighted,balanced,agmnt_between,use_perc))


%time
df_t = pd.read_csv('../input/train.csv')
df_tst = pd.read_csv('../input/test.csv')

if sampledf:
    sub_name = sub_name+'_sampled'
    df_train = df_t.sample(frac=(frac_sample))
    df_test = df_tst.sample(frac=(frac_sample))
    print('Loading Sampled df..')
else:
    df_train = df_t.copy()
    df_test = df_tst.copy()

print('Training df shape',df_train.shape)
print('Test df shape',df_test.shape)


def vis_classes(labels,values,title='Target Percentages'):
    trace=go.Pie(labels=labels,values=values)
    layout = go.Layout(
        title=title,
        height=600,
        margin=go.Margin(l=0, r=200, b=100, t=100, pad=4)   # Margins - Left, Right, Top Bottom, Padding
        )
    fig = go.Figure(data=[trace], layout=layout)
    iplot(fig)
    
    
labels = [str(x) for x in list(df_train['target'].unique())]
values = [(len(df_train[df_train['target'] == 0])/len(df_train))*100,(len(df_train[df_train['target'] > 0])/len(df_train))*100]    
vis_classes(labels,values)


# As you can see we are dealing with an unbalanced targets (10% vs 90%)


# ## Sampling from the full dataset (more work on this later)
# <a id='sample'></a>


df_ones = df_train[df_train['target'] > 0]
print('Ones',df_ones.shape)
df_zeros = df_train[df_train['target'] == 0].sample(frac=0.25)
print('Zeros',df_zeros.shape)
#we concat both to the sampling dataframe
df_sampling = pd.concat([df_ones, df_zeros]).sample(frac=1) #shuffling
print(df_sampling.shape)


# ## Simple Feature Engineering
# **Features Used:**
# * sum
# * min
# * max
# * mean
# * std
# * skew
# * kurt
# * med
# * Moving Average
# * percentiles
# <a id='fe'></a>


#part of it Inspired by Gabriel Preda 's Kernel'

def feature_creation(df,idx,use_perc,perc_list,name_num='_1'):
    #data metrics
    print('  * Loading new data metrics: ', name_num)
    df['sum'+name_num] = df[idx].sum(axis=1)  
    df['min'+name_num] = df[idx].min(axis=1)
    df['max'+name_num] = df[idx].max(axis=1)
    df['mean'+name_num] = df[idx].mean(axis=1)
    df['std'+name_num] = df[idx].std(axis=1)
    df['skew'+name_num] = df[idx].skew(axis=1)
    df['kurt'+name_num] = df[idx].kurtosis(axis=1)
    df['med'+name_num] = df[idx].median(axis=1)
    #moving average
    print('  * Loading moving average metric: ', name_num)
    df['ma'+name_num] =  df[idx].apply(lambda x: np.ma.average(x), axis=1)
    #percentiles
    print('  * Loading percentiles: ', name_num)
    if use_perc:
        for i in perc_list:
            df['perc_'+str(i)] =  df[idx].apply(lambda x: np.percentile(x, i), axis=1)
    #interactions
    #coming..
    
perc_list = [1,2,5,10,25,50,60,75,80,85,95,99]
perc_size = len(perc_list)
start_time = time.time()

for i,df in enumerate([df_train, df_test,df_sampling]):
    print('Loading more features for df: {}/{}'.format(i+1,3))
    print('Creating Metrics Part 1')
    features_1 = df_train.columns.values[2:202]
    feature_creation(df,features_1,use_perc,perc_list,name_num='_1') #adding columns using the train features (#200)
    print('Creating Metrics Part 2')
    features_2 = df_train.columns.values[2:211+perc_size] #all features included the ones added
    feature_creation(df,features_2,use_perc,perc_list,name_num='_2') #adding columns using the train features + the new features
    #drop repeated columns
    df.drop(['min_2','max_2'],axis=1,inplace=True)
    print('-'*50)

print('Features loaded !')
print("Execution --- %s seconds ---" % (time.time() - start_time))
print('Train df: ', df_train.columns)
print('Test df: ', df_test.columns)
print('Number of Features: ', len(df_train.columns[2:]))


#thanks to https://www.kaggle.com/jiweiliu/lgb-2-leaves-augment
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


X = df_train.iloc[:,2:]
Y = df_train['target']

if augmnt:
    print('Data Augmentation: Enabled')
    X,Y = augment(X.values,Y.values,t=2)
    X = pd.DataFrame(X,columns=df_train.columns[2:])
    Y = pd.Series(Y)
    print('Augmentation Succeeded')
    labels = ["0","1"]
    values = [(sum(Y == 0)/len(Y))*100,(sum(Y > 0)/len(Y))*100]    
    vis_classes(labels,values,title ='Target Percentages After Augmentation')
    sub_name = sub_name+'_agmted'

#test train split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=tst_size, random_state=6666)

#Sampling train test split
X_smple = df_sampling.iloc[:,2:]
y_smple = df_sampling['target']
X_train_smple, X_test_smple, y_train_smple, y_test_smple = train_test_split(X_smple, y_smple, test_size=0.4, random_state=6)

print("X_train: ", X_train.shape)
print("X_test: " ,X_test.shape)


# # 1. LGBM Model
# <a id='lgbm'></a>


#Model LGBM 
def create_model_lgbm(X_train,y_train,X_val=None,y_val=None,random_state =24):
    dtrain = lgb.Dataset(X_train,label=y_train)
    param = {
    'bagging_freq': 10, #handling overfitting
    'bagging_fraction': 0.6,#handling overfitting - adding some noise
    'boost_from_average':'false',
    'boost': 'gbdt',
    'feature_fraction': 0.05, #handling overfitting
    'learning_rate': 0.01, #the changes between one auc and a better one gets really small thus a small learning rate performs better
    'max_depth':-1, 
    "min_data_in_leaf": 80,
    'metric':'auc',
    'num_threads': 8,
    'tree_learner': 'serial',
    'objective': 'binary', 
    'verbosity': 0,
    "bagging_seed" : random_state,
    "seed": random_state,
     
    }
    if not X_val is None:
        dval = lgb.Dataset(X_val,label=y_val)
        valid_sets = (dtrain,dval)
        valid_names = ['train','valid']
        num_boost_round = 200000
    else:
        valid_sets = (dtrain)
        valid_names = ['train']
        num_boost_round = 60000
    model = lgb.train(param,dtrain,num_boost_round=num_boost_round,valid_sets=valid_sets,valid_names=valid_names,
                      verbose_eval=3000,
                     early_stopping_rounds=3000)
    return model


#Setting up
lgbm_test_x = df_test.iloc[:,1:]

predictions = df_test[['ID_code']]

val_aucs = []
val_pred = 0
target_pred = 0
importand_folds = 0
sub_train_n =3
kf = StratifiedKFold(n_splits=5,shuffle = kfold_shuffle, random_state=1111)
if fold_train:
    for _fold, (trn_idx, val_idx) in enumerate(kf.split(X.values, Y.values)):
            Xtrn, ytrn = X.iloc[trn_idx], Y.iloc[trn_idx]
            Xval, y_val = X.iloc[val_idx], Y.iloc[val_idx]
            #Just for info
            ones_train = (sum(ytrn>0) / len(ytrn))*100
            ones_val = (sum(y_val>0) / len(y_val))*100
            print('-'*50)
            print("Fold num:{}".format(_fold + 1))
            print('\tTrain Perc: 1: {:.2f}%, 0: {:.2f}%'.format(ones_train,100-ones_train))
            print('\tValid Perc: 1 : {:.2f}%, 0:{:.2f}%'.format(ones_val,100-ones_val))
            #augmentation for each training 
            #thanks to https://www.kaggle.com/jiweiliu/lgb-2-leaves-augment
            if agmnt_between:
                val_pred = 0
                target_pred = 0
                for i in range(sub_train_n):
                    print('\tSub-Train: {}'.format(i+1))
                    X_t, y_t = augment(Xtrn.values, ytrn.values)
                    print('\tAugmentation Succeeded..')
                    X_t = pd.DataFrame(X_t)
                    X_t = X_t.add_prefix('var_')
                    # generate some random integers
                    random_state = np.random.randint(0, 300, 1)
                    print('\tFitting Model')
                    clf = create_model_lgbm(X_t,y_t,Xval,y_val,random_state=random_state)
                    target_pred += clf.predict(lgbm_test_x)
                    val_pred += clf.predict(Xval)
            #this part could be used when the augmentation is fully applied to the training data
            else:
                clf = create_model_lgbm(Xtrn,ytrn,Xval,y_val)
                val_pred  += clf.predict(lgbm_X.iloc[val_idx]) / kf.n_splits
                target_pred += clf.predict(df_test.iloc[:,1:]) / kf.n_splits
            
            print('-' * 50)
            importand_folds += clf.feature_importance()
            val_score = roc_auc_score(y_val, val_pred/sub_train_n)
            val_aucs.append(val_score)
            print('\tVal CV score : {}'.format(val_score))
            predictions['fold{}'.format(_fold+1)] = target_pred/sub_train_n

mean_cv_score = np.mean(val_aucs)
print ('----- Mean CV Score: {:.2} ------'.format(mean_cv_score))


num_features = 60
if fold_train:
    indxs = np.argsort(importand_folds/ kf.n_splits)[:num_features]
else:
    indxs = np.argsort(clf.feature_importance())[:num_features]
    
feature_imp = pd.DataFrame(sorted(zip(clf.feature_importance()[indxs],X.columns[indxs])), columns=['Value','Feature'])

plt.figure(figsize=(20, 10))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
plt.title('Top {} LightGBM Features accorss folds'.format(num_features))
plt.tight_layout()
plt.show()


# As we can see , many of the engineered features are present within the top 60 important features


# # 2.Keras NN Model
# <a id='keras'></a>


def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc
#Model NN definition
def create_model_nn(in_dim,layer_size=54):
    model = Sequential()
    model.add(Dense(layer_size,input_dim=in_dim))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    for i in range(7):
        model.add(Dense(layer_size))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    adam = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=adam,loss='binary_crossentropy',metrics = [auc])    
    return model

#Class weights to handle the unbalanced dataset

class_weights = None
if weighted:
    sub_name = sub_name+'_weighted'
    if balanced:
        class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)
    else:
        class_weights = {
            1:50, 
            0:1
                }
model_nn = create_model_nn(X_train.shape[1])
callback = EarlyStopping(monitor="val_auc", patience=20, verbose=0, mode='max')
history = model_nn.fit(X_train, y_train, validation_data = (X_test,y_test),epochs=100,batch_size=256,verbose=1,callbacks=[callback],class_weight=class_weights)
target_pred_nn = model_nn.predict(df_test.iloc[:,1:])[:,0]
print('\n Validation Max score : {}'.format(np.max(history.history['val_auc'])))


# ## Combination Vis
# <a id='vis'></a>


#Ditribution Plots from both models 
nn_val_pred = model_nn.predict(X_test,batch_size=256)[:,0]
predictions['target'] = np.mean(predictions[[col for col in predictions.columns if col not in ['ID_code', 'target']]].values, axis=1) 

comb_approach_test = (0.2*target_pred_nn)+(0.8*predictions['target'])
comb_approach_test[comb_approach_test>1]=1
comb_approach_test[comb_approach_test<0]=0

if fold_train:
    plt.figure(figsize=(13, 9))
    #validations sets
    sns.distplot(nn_val_pred,label='NN Val Score:{:.3f}'.format(roc_auc_score(y_test,nn_val_pred)))
    sns.distplot(val_pred,label='LGBM Val Score : {:.3f}'.format(mean_cv_score))
    plt.title('Validation set target predictions')
    plt.legend()
    plt.show()
    plt.savefig('combination_val.png')

plt.figure(figsize=(13, 9))
#target final test set
sns.distplot(target_pred_nn,label='NN Target')
sns.distplot(predictions['target'],label='LGBM Target')
sns.distplot(comb_approach_test,label='Combination Prediction Target')
plt.title('Test set target predictions')
plt.legend()
plt.show()
plt.savefig('combination_target_test.png')


# **DistPlot Analysis:** 
# 
# We can see from the plots how the predictions for the validation sets, and the final target test set follows closely a similar distribution.
# Which tells us that the test set can be a resemblance of validation sets we are using. 
# Then we can proceed with improving our scores in the validation set knowing that there is a high chance they will also improve in the test set.
# 
# **Combination Analysis**
# 
# We can also see that the combination of both is adding some noise to the prediction, which in some cases can prove helpful when each model
# was able to predict with some features better than the others
# 
# more testing is going on here to see how effecient a combination model can get.


# # Submission
# <a id='sub'></a>


def sub_pred(preds,df_test,name='submission.csv'):
    sub_df = pd.DataFrame({'ID_code':df_test['ID_code'],'target':preds})
    sub_df.to_csv(name, index=False)

sub_file =  sub_name +'.csv'
sub_pred(predictions['target'],df_test,name=sub_file)
print(sub_file+'   --submitted successfully')

print('Submitting Combination File..')
sub_pred(comb_approach_test,df_test,name='comb_submission.csv')
print('comb_submission.csv   --submitted successfully')

