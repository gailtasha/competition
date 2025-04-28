# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from matplotlib import pyplot as plt
import os

# Any results you write to the current directory are saved as output.


random_state = 42
np.random.seed(random_state)


# # Load dataset and separate Validation


df = pd.read_csv('../input/train.csv')
# Uses the las 40000 for Validation (I am assuming that they are already shuffled in CSV)
df_train = df[:160000]
# df_train = df.copy()


df_train.shape


# # Normalization


# Calculate 
def get_stats(df_train, resolution = 501):
    means = df_train.drop(columns=['target', 'ID_code']).mean()
    stds = df_train.drop(columns=['target', 'ID_code']).std()
    mins = df_train.drop(columns=['target', 'ID_code']).min()
    maxs = df_train.drop(columns=['target', 'ID_code']).max()
    max_z_zcore = np.ceil(((maxs-means)/stds).max())
    min_z_zcore = np.floor(((mins-means)/stds).min())
    z_scores = np.linspace(min_z_zcore, max_z_zcore, resolution)
    return means, stds, z_scores, min_z_zcore, max_z_zcore


means, stds, z_scores, min_z_zcore, max_z_zcore = get_stats(df_train, resolution = 501)


N_vars = df_train.shape[1]-2
print('Number of independent variables:', N_vars)


print('Z-score range is from {} to {}'.format(min_z_zcore, max_z_zcore))


# # Bayes Theorem


# Target = 1  
# $\large P(t=1|V_0, V_1, ..., V_{199}) = \frac{P(V_0, V_1, ..., V_{199}|t=1)P(t=1)}{P(V_0,V_1, ..., V_{199})} \quad$
# 
# Target = 0  
# $\large P(t=0|V_0, V_1, ..., V_{199}) = \frac{P(V_0, V_1, ..., V_{199}|t=0)P(t=0)}{P(V_0,V_1, ..., V_{199})} \quad$


# # Naive Bayes


# $\large \large P(t=1|V_0, V_1, ..., V_{199}) = \frac{P(V_0|t=1) P(V_1|t=1) ... P(V_{199}|t=1) P(t=1)}{P(V_0,...,V_{199})} \quad$ Independence of Conditionals
# 
# $\large \large P(t=0|V_0, V_1, ..., V_{199}) = \frac{P(V_0|t=0) P(V_1|t=0) ... P(V_{199}|t=0) P(t=0)}{P(V_0,...,V_{199})} \quad$ Independence of Conditionals


# # Likelihoods calculation:
# 
# $\large P(V_i|t=1)$ and $\large P(V_i|t=0)$


# All the observation where target = 1
df_target_1 = df_train[df_train['target']==1]
# All the observation where target = 0
df_target_0 = df_train[df_train['target']==0]


def likelihoods_frequency(v, i=0, c=3, df_target_1=df_target_1, df_target_0=df_target_0):
    #
    # Counts the observations in region [v-stds[i]/c, v+stds[i]/c] for variable v_i and normalized it with the range
    # 
    """
    v: center of region
    i: variable index (var_i)
    c: smoothing for moving average
    """
    # From the observations where the target is 0 count the number of observations that are between v-stds[i]/c and v+stds[i]/c
    N_interval_0 = len(df_target_0[(df_target_0['var_'+str(i)]>v-stds[i]/c) &(df_target_0['var_'+str(i)]<v+stds[i]/c)])
    
    # From the observations where the target is 1 count the number of observations that are between v-stds[i]/c and v+stds[i]/c
    N_interval_1 = len(df_target_1[(df_target_1['var_'+str(i)]>v-stds[i]/c) &(df_target_1['var_'+str(i)]<v+stds[i]/c)])
    # Returns the estimation of the likelihood at v
    return N_interval_0/(2*stds[i]/c), N_interval_1/(2*stds[i]/c)


def get_pdf(var_i, means=means, stds=stds, z_scores=z_scores, smoothing=1, c=3, N_vars=N_vars, df_target_0=df_target_0, df_target_1=df_target_1):
    # Estimates the likelihood probability density function of variable v_i for all v_i's sample space
    """
    var_i: variable index
    c: smoothing for moving average
    smoothing: laplacian smoothing
    """
    ps_0 = []
    ps_1 = []
    N_0 = len(df_target_0) + smoothing*N_vars
    N_1 = len(df_target_1) + smoothing*N_vars
    for z in z_scores:
        # Unnormalize
        v = z*stds[var_i] + means[var_i]
        l0, l1 = likelihoods_frequency(v, var_i, c=c)
        ps_0.append(l0 + smoothing)
        ps_1.append(l1 + smoothing)
    return np.array(ps_0)/N_0, np.array(ps_1)/N_1


var_i = 0
%time l0_3,l1_3 = get_pdf(var_i, c=3)
plt.plot(z_scores, l0_3, label='$P(V_i|t=0)$')
plt.plot(z_scores, l1_3, label='$P(V_i|t=1)$')
plt.legend()
plt.show()


# ### Effect of moving average smoothing


%time l0_05,l1_05 = get_pdf(var_i, c=0.5)
plt.plot(z_scores, l0_05, label='$P(V_i|t=0)$')
plt.plot(z_scores, l1_05, label='$P(V_i|t=1)$')
plt.title('C = 0.5')
plt.legend()
plt.show()


%time l0_10,l1_10 = get_pdf(var_i, c=10)
plt.plot(z_scores, l0_10, label='$P(V_i|t=0)$')
plt.plot(z_scores, l1_10, label='$P(V_i|t=1)$')
plt.title('C = 10')
plt.legend()
plt.show()


plt.plot(z_scores, l0_3, label='C=3')
plt.plot(z_scores, l0_10, label='C=10')
plt.plot(z_scores, l0_05, label='C=0.5')
plt.title('Moving average compare - likelihood t=0')
plt.legend()
plt.show()


# # Efect of laplacian smoothing


l0_ls1, l1_ls1 = get_pdf(var_i, c=3, smoothing=1)
plt.plot(z_scores, l0_ls1, label='$P(V_i|t=0)$')
plt.plot(z_scores, l1_ls1, label='$P(V_i|t=1)$')
plt.title('laplacian smoothing 1')
plt.legend()
plt.show()


l0_ls10, l1_ls10 = get_pdf(var_i, c=3, smoothing=10)
plt.plot(z_scores, l0_ls10, label='$P(V_i|t=0)$')
plt.plot(z_scores, l1_ls10, label='$P(V_i|t=1)$')
plt.title('laplacian smoothing 10')
plt.legend()
plt.show()


l0_ls01, l1_ls01 = get_pdf(var_i, c=3, smoothing=0.1)
plt.plot(z_scores, l0_ls01, label='$P(V_i|t=0)$')
plt.plot(z_scores, l1_ls01, label='$P(V_i|t=1)$')
plt.title('laplacian smoothing 0.1')
plt.legend()
plt.show()


plt.plot(z_scores, l1_ls01, label='0.1')
plt.plot(z_scores, l1_ls1, label='1')
plt.plot(z_scores, l1_ls10, label='10')
plt.title('Varing laplacian smoothing')
plt.legend()
plt.show()


# # Odds
# $\large \frac{P(t=1|V_0, V_1, ..., V_{199})}{P(t=0|V_0, V_1, ..., V_{199})} > 1\quad$ implies that target 1 is more probable than target 0 
# 
# Doing the quotient from Naive Bayes


# $\huge \frac{\frac{P(V_0|t=1) P(V_1|t=1) ... P(V_{199}|t=1) P(t=1)}{P(V_0,...,V_{199})}}{\frac{P(V_0|t=0) P(V_1|t=0) ... P(V_{199}|t=0) P(t=0)}{P(V_0,...,V_{199})}} = \frac{P(V_0|t=1) P(V_1|t=1) ... P(V_{199}|t=1) P(t=1)}{P(V_0|t=0) P(V_1|t=0) ... P(V_{199}|t=0) P(t=0)}$


# ### Odds for one variable
# $\large \frac{P(V_1|t=1)P(t=1)}{P(V_1|t=0)P(t=0)}$


# Odds for just one variable
# P(t=1) is the estimated as the proportion of observations with target 1
# P(t=0) is the estimated as the proportion of observations with target 0
p_1 = len(df_target_1)/len(df_train)
p_0 = len(df_target_0)/len(df_train)
print(p_1, p_0)


var_i = 50
l0,l1 = get_pdf(var_i, c=3, smoothing=1)
odds = l1/l0 * p_1/p_0
plt.plot(z_scores, odds, label='smoothing=1')
plt.legend()
plt.show()


# ### Effect of laplacian smoothing


l0,l1 = get_pdf(var_i, c=3, smoothing=10)
odds = l1/l0 * p_1/p_0
plt.plot(z_scores, odds, label='smoothing=100')
plt.legend()
plt.show()


l0,l1 = get_pdf(var_i, c=3, smoothing=0.01)
odds = l1/l0 * p_1/p_0
plt.plot(z_scores, odds, label='smoothing=0.01')
plt.legend()
plt.show()


# # Calculate marginal likelihoods for all $V_i$s
# $P(V_0|t=1), P(V_1|t=1), ..., P(V_{199}|t=1)$
# 
# $P(V_0|t=0), P(V_1|t=0), ..., P(V_{199}|t=0)$


# The two functions above: get_pdf and likelihoods_frequency, are inefficient but are easier to understand how the calculations are done. It takes almost 10 seconds in a pentium 7 with 12 processors and 16GB of memory when c=1
# 
# Doing calculations of 200 variables it takes aprox 10s*200/60 = 33 minutes


# ## First: choose c and laplacian smoothing


# Chossen c and smoothing
c = 3
smoothing = 1
l0,l1 = get_pdf(1, c=3, smoothing=0.1)
odds = l1/l0 * p_1/p_0
plt.plot(z_scores, odds, label='smoothing=0.01')
plt.legend()
plt.show()


# ## Second: calculate al V_i's


####
# In a pentium 7 with 12 processors and 16GB of memory
####
# For c=3
# CPU times: user 13min 34s, sys: 103 ms, total: 13min 34s
# Wall time: 13min 34s
####
# For c=0.5
# CPU times: user 32min 46s, sys: 22min 14s, total: 55min 1s
# Wall time: 55min 1s
####

def calculate_all_V_i_inefficient(df_train=df_train, smoothing=smoothing, c=c):
    # Check is the file is already in disk, if not calculate it and save it to disk

    filename_l0 = 'likelihood_matrix_0_smooth_{}_c_{}_{}.npy'.format(smoothing, c, len(df_train))
    filename_l1 = 'likelihood_matrix_1_smooth_{}_c_{}_{}.npy'.format(smoothing, c, len(df_train))

    if os.path.isfile(filename_l0) and os.path.isfile(filename_l1):
        likelihood_matrix_0_np = np.load(filename_l0)
        likelihood_matrix_1_np = np.load(filename_l1)
        print('Skip likelihoods calculations, files {} and {} already exists'.format(filename_l0, filename_l1))
    else:
        likelihood_matrix_0 = []
        likelihood_matrix_1 = []
        for i in range(df_train.shape[1] - 2):
            # This for is very inefficient because it calculates each variable separately
            print('\rCalculating likelihoods for var_'+str(i), end="")
            var_i_0, var_i_1 = get_pdf(i, smoothing=smoothing, c=c)
            likelihood_matrix_0.append(var_i_0)
            likelihood_matrix_1.append(var_i_1)
        # To numpy    
        likelihood_matrix_0_np = np.array(likelihood_matrix_0).T
        likelihood_matrix_1_np = np.array(likelihood_matrix_1).T
        # Save them
        np.save(filename_l0, likelihood_matrix_0_np)
        np.save(filename_l1, likelihood_matrix_1_np)

    # The odds here is not multiplied by p_1/p_0, it will be done later
    odds = likelihood_matrix_1_np/likelihood_matrix_0_np
    filename_odds = 'odds_smooth_{}_c_{}_{}.npy'.format(smoothing, c, len(df_train))
    np.save(filename_odds, odds)
    print()
    return likelihood_matrix_0_np, likelihood_matrix_1_np, odds

# %time likelihood_matrix_0, likelihood_matrix_1, odds = calculate_all_V_i_inefficient(df_train=df_train, smoothing=smoothing, c=c)


# ### Vectorization to accelarate processing


####
# In a pentium 7 with 12 processors and 16GB of memory
####
# For c=3
# CPU times: user 2min 36s, sys: 51.3 s, total: 3min 27s
# Wall time: 3min 27s
####
# For c=0.5
# CPU times: user 2min 55s, sys: 51.7 s, total: 3min 46s
# Wall time: 3min 46s
####
def likelihoods_frequency_vect(v, c=3, df_target_1=df_target_1, df_target_0=df_target_0):
    # This version calculates all variables v_i in one shot
    N_interval_0 = ((df_target_0.drop(columns=['ID_code', 'target'])>v-stds/c) 
                    & (df_target_0.drop(columns=['ID_code', 'target'])<v+stds/c)).sum(axis=0)
    N_interval_1 = ((df_target_1.drop(columns=['ID_code', 'target'])>v-stds/c) 
                    & (df_target_1.drop(columns=['ID_code', 'target'])<v+stds/c)).sum(axis=0)
    return N_interval_0/(2*stds/c), N_interval_1/(2*stds/c)

def get_pdf_vect(means=means, stds=stds, z_scores=z_scores, smoothing=1, c=3, df_target_1=df_target_1, df_target_0=df_target_0):
    # Calculates the probability density function of all V_is
    # Same as calculate_all_V_i_inefficient but in a more efficient way
    """
    var_i: variable index
    c: smoothing for moving average
    smoothing: laplacian smoothing
    """
    N_0 = len(df_target_0) + smoothing*N_vars
    N_1 = len(df_target_1) + smoothing*N_vars
    ps_0 = []
    ps_1 = []
    for z in z_scores:
        print('\r z =', z, end = '')
        # Unnormalize
        v = z*stds + means
        l0, l1 = likelihoods_frequency_vect(v, c=c)
        ps_0.append(l0 + smoothing)
        ps_1.append(l1 + smoothing)
    likelihood_matrix_0 = np.array(ps_0)/N_0
    likelihood_matrix_1 = np.array(ps_1)/N_1
    odds = likelihood_matrix_1/likelihood_matrix_0
    print()
    return likelihood_matrix_0, likelihood_matrix_1, odds
%time likelihood_matrix_0, likelihood_matrix_1, odds = get_pdf_vect(smoothing=smoothing, c=c)


var_i = 0
plt.plot(z_scores, likelihood_matrix_0[:,var_i])
plt.plot(z_scores, likelihood_matrix_1[:,var_i])
plt.show()


# ### Multiprocessing  


from multiprocessing import Pool
from functools import partial


####
# In a pentium 7 with 12 processors and 16GB of memory
####
# For c=3
# CPU times: user 91.6 ms, sys: 144 ms, total: 235 ms
# Wall time: 48.7 s
####
# For c=0.5
# CPU times: user 80.2 ms, sys: 142 ms, total: 222 ms
# Wall time: 48.3 s
####

def get_pdf_vect_parallel(z_scores_interval, means=means, stds=stds, smoothing=1, c=3, df_target_1=df_target_1, df_target_0=df_target_0):
    # Calculates the probability density function of variable v_i for all v's
    """
    var_i: variable index
    c: smoothing for moving average
    smoothing: laplacian smoothing
    """
    N_0 = len(df_target_0) + smoothing*N_vars
    N_1 = len(df_target_1) + smoothing*N_vars
    ps_0 = []
    ps_1 = []
    for z in z_scores_interval:
        # Unnormalize
        v = z*stds + means
        l0, l1 = likelihoods_frequency_vect(v, c=c, df_target_1=df_target_1, df_target_0=df_target_0)
        ps_0.append(l0 + smoothing)
        ps_1.append(l1 + smoothing)
    return np.array(ps_0)/N_0, np.array(ps_1)/N_1



def train_parallel(df, N = 10, smoothing=1, c=3, resolution=501):
    means, stds, z_scores, min_z_zcore, max_z_zcore = get_stats(df_train, resolution = resolution)
    # All the observation where target = 1
    df_target_1 = df_train[df_train['target']==1]
    # All the observation where target = 0
    df_target_0 = df_train[df_train['target']==0]
    
    N_paral = int(len(z_scores)/N)
    z_scores_list = []
    for i in range(N):
        z_scores_min = i*N_paral
        if i == N-1:
            z_scores_max = len(z_scores)
        else:
            z_scores_max = (i+1)*N_paral
        z_scores_list.append(list(z_scores[z_scores_min: z_scores_max]))
    likelihoods_pdfs = []
    with Pool(N) as p:
        likelihoods_pdfs = p.map(partial(get_pdf_vect_parallel, smoothing=smoothing, c=c), z_scores_list)
    likelihood_matrix_0 = np.empty((0, 200))
    likelihood_matrix_1 = np.empty((0, 200))
    for i, (l0, l1) in enumerate(likelihoods_pdfs):
        likelihood_matrix_0 = np.append(likelihood_matrix_0, l0, axis=0)
        likelihood_matrix_1 = np.append(likelihood_matrix_1, l1, axis=0)
    odds = likelihood_matrix_1/likelihood_matrix_0
    p_1 = len(df_target_1)/len(df_train)
    p_0 = len(df_target_0)/len(df_train)
    return likelihood_matrix_0, likelihood_matrix_1, odds, p_1, p_0, means, stds, z_scores

%time likelihood_matrix_0, likelihood_matrix_1, odds, p_1, p_0, means, stds, z_scores = train_parallel(df_train, N = 10, smoothing=smoothing, c=c)


# ### Plot likelihoods


plt.plot(z_scores, likelihood_matrix_0[:,:60])
plt.show()


plt.plot(z_scores, likelihood_matrix_1[:,:60])
plt.show()


var_i = 0
plt.plot(z_scores, likelihood_matrix_0[:,var_i])
plt.plot(z_scores, likelihood_matrix_1[:,var_i])
plt.show()


# ### Plot odds


plt.plot(z_scores, odds * p_1/p_0)
plt.show()


plt.plot(z_scores, odds[:,0] * p_1/p_0)
plt.show()


# # Estimate Observations


from sklearn.metrics import roc_auc_score


# ## Train model (160.000 observations)


df = pd.read_csv('../input/train.csv')
# Uses the las 40000 for Validation (I am assuming that they are already shuffled in CSV)
df_train = df[:160000]
# df_train = df.copy()


_, _, odds, p_1, p_0, means, stds, z_scores = train_parallel(df_train, N = 10, smoothing=1, c=3, resolution=501)


# ### Predict on train


def predict(df, odds, p_1, p_0, means, stds, z_scores, resolution=501):
    min_z_zcore, max_z_zcore = min(z_scores), max(z_scores)
    if 'target' in df:
        observations = df.drop(columns=['target', 'ID_code']).values
    else:
        observations = df.drop(columns=['ID_code']).values
    # Normalize
    observations_normalized = (observations - means.values)/stds.values
    observations_odds = []
    for var_i in range(df.shape[1] - 2):
        indexes = np.array(np.round(((observations_normalized[:,var_i]-min_z_zcore)/(max_z_zcore - min_z_zcore))*resolution), dtype=int)
        observations_odds.append(odds[indexes, var_i])
    observations_odds = np.array(observations_odds).T
    log_odds = np.sum(np.log(observations_odds), axis=1) + np.log(p_1/p_0)
    prod_odds = np.exp(log_odds)
    auc = None
    acc = None
    if 'target' in df:
        auc = roc_auc_score(df['target'], prod_odds)
        acc = (df['target'] == (prod_odds>=1)).sum()/len(df)
        
    return observations_normalized, observations_odds, log_odds, prod_odds, auc, acc


observations_normalized, observations_odds, log_odds, prod_odds, auc, acc = predict(df_train, odds, p_1, p_0, means, stds, z_scores)


print('AUC = {}, for c={} and smothing={}'.format(auc, c, smoothing))
print('Acc = {}, for c={} and smothing={}'.format(acc, c, smoothing))


var_j = 1
plt.plot(z_scores, odds[:,var_j])
plt.scatter(observations_normalized[:,var_j], observations_odds[:,var_j], s=20, marker='.', c='red')
plt.show()


_ = plt.hist(log_odds, 50)


# # Validation


df_valid = pd.read_csv('../input/train.csv')[160000:]


observations_normalized_valid, observations_odds_valid, log_odds_valid, prod_odds_valid, auc_valid, acc_valid = predict(df_valid, odds, p_1, p_0, means, stds, z_scores)


print('Validation AUC = {}, for c={} and smothing={}'.format(auc_valid, c, smoothing))
print('Validation Acc = {}, for c={} and smothing={}'.format(acc_valid, c, smoothing))


var_j = 1
plt.plot(z_scores, odds[:,var_j])
plt.scatter(observations_normalized_valid[:,var_j], observations_odds_valid[:,var_j], s=20, marker='.', c='red')
plt.show()


_ = plt.hist(log_odds_valid, 30)


# Validation AUC = 0.9007052140114506 smoothing = 10 c=3
# Validation AUC = 0.902612244256969 smoothing = 1 c=3
# Validation AUC = 0.9026132128924224 smoothing = 0.5 c=3
# Validation AUC = 0.8951584158287289 smoothing = 0.5 c=10
# Validation AUC = 0.9024698960638358 smoothing = 0.5 c=2


# # Retrain with all dataset


df_train = pd.read_csv('../input/train.csv')


_, _, odds, p_1, p_0, means, stds, z_scores = train_parallel(df_train, N = 10, smoothing=1, c=3, resolution=501)


observations_normalized_full, observations_odds_full, log_odds_full, prod_odds_full, auc_full, acc_full = predict(df_train, odds, p_1, p_0, means, stds, z_scores)


print('Validation AUC = {}, for c={} and smothing={}'.format(auc_full, c, smoothing))


_ = plt.hist(log_odds_full, 30)


# # Test


df_test = pd.read_csv('../input/test.csv')


_, _, log_odds_full, prod_odds_test, _, _ = predict(df_test, odds, p_1, p_0, means, stds, z_scores)


_ = plt.hist(log_odds_full, 50)


sub = pd.read_csv('../input/sample_submission.csv')
sub['target'] = prod_odds_test
sub.to_csv('submission.csv',index=False)



