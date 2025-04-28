# # Introduction
# 
# The aim of this kernel is to help people learn a new method of binning, called Bayesian Blocks. The other reason for releasing this kernel near the end of the competition is that I have not been able to devote that much time to look at other promising methods/discussions that have been shared in this competition so far. So, I hope people can use this new method in their models and learn something new in the process.
# 
# ## What is binning?
# 
# Feature binning is a method of turning continuous variables into categorical variables. This is accomplished by grouping the values into a pre-defined number of bins. The continuous value then gets replaced by a string describing the bin that contains that value.
# 
# ## What is Bayesian Block Binning?
# 
# Bayesian blocks is essentially a method creating histograms with bin sizes that adapt to the data. I am going to quote the whole post from this [link](http://jakevdp.github.io/blog/2012/09/12/dynamic-programming-in-python/). Please feel free to go through either posts to understand more about Bayesian Blocks. 
# 
# I have given a working example of using bayesian blocks at the end of this kernel. You might want to run this kernel on your own machine because of the use of Multiprocessing framework.
# 
# To illustrate that, let's look at some sampled data. 


# Define our test distributions: a mix of Cauchy-distributed variables
import numpy as np
from scipy import stats

np.random.seed(0)

x = np.concatenate([stats.cauchy(-5, 1.8).rvs(500),
                    stats.cauchy(-4, 0.8).rvs(2000),
                    stats.cauchy(-1, 0.3).rvs(500),
                    stats.cauchy(2, 0.8).rvs(1000),
                    stats.cauchy(4, 1.5).rvs(500)])

# Truncate values to a reasonable range:
x = x[(x > -15) & (x < 15)]


# Now, what does this distribution look like? Let's plot a histogram:


import pylab as pl
pl.hist(x, normed=True)


# Not too infomative. The default bins in matplotlib are too wide for this dataset. We might be able to do better by increasing the number of bins:


pl.hist(x, bins=100, normed=True)


# This is better. But having to choose the bin width each time we plot a distribution is not only tiresome, it may lead to missing some important information in our data. In a perfect world, we'd like for the bin width to be learned in an automated fashion, based on the properties of the data itself. There have been many rules-of-thumb proposed for this task (look up Scott's Rule, Knuth's Rule, the Freedman-Diaconis Rule, and others in your favorite statistics text). But all these rules of thumb share a disadvantage: they make the assumption that all the bins are the same size. This is not necessarily optimal. But can we do better?
# 
# Scargle and collaborators showed that the answer is yes. This is their insight: **For a set of histogram bins or blocks, each of an arbitrary size, one can use a Bayesian likelihood framework to compute a fitness function which only depends on two numbers: the width of each block, and the number of points in each block. ** The edges between these blocks (the change-points) can be varied, and the overall block configuration with the maximum fitness is quantitatively the best binning.


# Simple, right?
# 
# Well, no. The problem is, as the number of points N grows large, the number of possible configurations grows as $2^N$. For N=300 points, there are already more possible configurations than the number of subatomic particles in the observable universe! Clearly an exhaustive search will fail in cases of interest. This is where dynamic programming comes to the rescue.
# 
# ### Dynamic Programming
# 
# In our Bayesian Blocks example, we can easily find the optimal binning for a single point. By making use of some mathematical proofs concerning the fitness functions, we can devise a simple step from the optimal binning for $k$ points to the optimal binning for $k + 1$ points (the details can be found in the appendices of the Scargle paper). In this way, Scargle and collaborators showed that the $2^N$ possible states can be explored in $N^2$ time.
# 
# ### Algorithm:
# 
# The resulting algorithm is deceptively simple, but it can be proven to converge to the single best configuration among the $2^N$ possibilities. Below is the basic code written in python. Note that there are a few details that are missing from this version (e.g. priors on the number of bins, other forms of fitness functions, etc.) but this gets the basic job done:


def bayesian_blocks(t):
    """Bayesian Blocks Implementation

    By Jake Vanderplas.  License: BSD
    Based on algorithm outlined in http://adsabs.harvard.edu/abs/2012arXiv1207.5578S

    Parameters
    ----------
    t : ndarray, length N
        data to be histogrammed

    Returns
    -------
    bins : ndarray
        array containing the (N+1) bin edges

    Notes
    -----
    This is an incomplete implementation: it may fail for some
    datasets.  Alternate fitness functions and prior forms can
    be found in the paper listed above.
    """
    # copy and sort the array
    t = np.sort(t)
    N = t.size

    # create length-(N + 1) array of cell edges
    edges = np.concatenate([t[:1],
                            0.5 * (t[1:] + t[:-1]),
                            t[-1:]])
    block_length = t[-1] - edges

    # arrays needed for the iteration
    nn_vec = np.ones(N)
    best = np.zeros(N, dtype=float)
    last = np.zeros(N, dtype=int)

    #-----------------------------------------------------------------
    # Start with first data cell; add one cell at each iteration
    #-----------------------------------------------------------------
    for K in range(N):
        # Compute the width and count of the final bin for all possible
        # locations of the K^th changepoint
        width = block_length[:K + 1] - block_length[K + 1]
        count_vec = np.cumsum(nn_vec[:K + 1][::-1])[::-1]

        # evaluate fitness function for these possibilities
        fit_vec = count_vec * (np.log(count_vec) - np.log(width))
        fit_vec -= 4  # 4 comes from the prior on the number of changepoints
        fit_vec[1:] += best[:K]

        # find the max of the fitness: this is the K^th changepoint
        i_max = np.argmax(fit_vec)
        last[K] = i_max
        best[K] = fit_vec[i_max]

    #-----------------------------------------------------------------
    # Recover changepoints by iteratively peeling off the last block
    #-----------------------------------------------------------------
    change_points =  np.zeros(N, dtype=int)
    i_cp = N
    ind = N
    while True:
        i_cp -= 1
        change_points[i_cp] = ind
        if ind == 0:
            break
        ind = last[ind - 1]
    change_points = change_points[i_cp:]

    return edges[change_points]


# The details of the step from $K$ to $K + 1$ may be a bit confusing from this implementation: it boils down to the fact that Scargle et al. were able to show that given an optimal configuration of $K$ points, the $(K + 1)$^th configuration is limited to one of $K$ possibilities.
# 
# The function as written above takes a sequence of points, and returns the edges of the optimal bins. We'll visualize the result on top of the histogram we saw earlier:


from matplotlib import pyplot

# plot a standard histogram in the background, with alpha transparency
H1 = pyplot.hist(x, bins=200, histtype='stepfilled',
          alpha=0.2, normed=True)
# plot an adaptive-width histogram on top
H2 = pyplot.hist(x, bins=bayesian_blocks(x), color='black',
          histtype='step', normed=True)


# The adaptive-width bins lead to a very clean representation of the important features in the data. More importantly, these bins are quantifiably optimal, and their properties can be used to make quantitative statistical statements about the nature of the data. This type of procedure has proven very useful in analysis of time-series data in Astronomy.


# ### Conclusion of the post from Jake:
# 
# We've just scratched the surface of Bayesian Blocks and Dynamic Programming. Some of the more interesting details of this algorithm require much more depth: the appendicies of the Scargle paper provide these details. Dynamic Programming ideas have been shown to be useful in many optimization problems. One other example I've worked with extensively is Dijkstra's Algorithm for computing the shortest paths on a connected graph. This is available in the scipy.sparse.csgraph submodule, which is included in the most recent release of scipy.


# # Important Links:
# 
# To refer if you want to know more about Bayesian Blocks and its implementation in AstroML:
# 
# 1. [AstroML library and post](http://www.astroml.org/user_guide/density_estimation.html#bayesian-blocks-histograms-the-right-way)
# 2. [Jake's post](https://jakevdp.github.io/blog/2012/09/12/dynamic-programming-in-python/)


# # Use of Bayesian Block in Santander Customer Transaction Prediction Challenge:
# 
# Below is a multiprocessing based approach to calculate bins for the dataset in this competition. As this is a computing intensive task, if you want to run this in a shorter amount of time, please use your machines instead of computing providing by Kaggler kernel as kaggle kernel only have 4 CPU's available for processing. 
# 
# I have not tried to improve the performance of this method (more like I am still learning performance tunning of Python functions, so please feel free to recommend improvements).


from astropy import stats
import multiprocessing as mp
from functools import reduce

def variable_to_bin(var, df_train):
    
    # Lets calculate bin values for a particular column in the dataframe passed to this function
    bin_values = stats.bayesian_blocks(df_train[var],
                                      fitness='events',
                                      p0=0.01)
    
    # Lets create labels for bin values so as to use these labels in dataframe 
    labels = []
    for i, x in enumerate(bin_values):
        labels.append(i)
    
    # delete the last bin label 
    del labels[-1]

    # create a new dataframe to 
    df = pd.DataFrame(index=df_train.index)

    df["ID_code"] = df_train["ID_code"]
    df['new' + var] = pd.cut(df_train[var], 
                               bins = bin_values, 
                               labels = labels)
    
    df.set_index('ID_code')
    
    # Lets delete the bin values and labels to some some space.
    del bin_values, labels
    
    return df


def get_new_feature_train():
    features = [c for c in df_train.columns if c not in ["ID_code", "target"]]

    # Use below line to test whether the binning works or not. 
    #features = ('var_2', 'var_3')
    
    new_df = pd.DataFrame()
    
    # Lets create a multi processing pool but N - 4 CPU's = 4 less CPU's then what your machine has.
    # My machine has 16 CPU's, but I wanted to use 12 of them for calculating bins. 
    pool = mp.Pool(mp.cpu_count() - 4)
    
    # Lets map each CPU to each variable coming out of features list. This line helps in parallel computation
    # of bayesian block bins. 
    results = pool.map(variable_to_bin, features)
    
    pool.close()
    pool.join()
    
    # Lets reduce the series coming out of variable_to_bin function and create a new dataframe.
    results_df = reduce(lambda x, y: pd.merge(x, y, on = 'ID_code'), results)

    return results_df


df_train.set_index('ID_code')
train_df = pd.DataFrame(index=df_train.index)
train_df = get_new_feature_train()


# Please use label encoder to encode the new features generated using the above method as the columns coming out of get_new_feature_train function are of type: category.
# 
# Small snippet on how to use label encoder:


from sklearn.preprocessing import LabelEncoder 

features = [c for c in train_df.columns if c not in ["ID_code", "target"]]

lbl_enc = LabelEncoder()
lbl_enc.fit(train_df[features])

df_Train_cat = lbl_enc.transform(train_df[features])


# # Conclusion of this post:
# 
# I believe this method can be a good way to bin features and use them in models that have been shared in this competition. I would love to see if this method is able to proceed boost to your models/score on the leaderboard.
# 
# Please upvote if you like this kernel and please feel free to provide recommendations.
# 
# Happy Competing!


# (ToDo: This needs to be completed, so please check after 1 day)
# 
# ## Different types of binning described/used in this competition (based on public kernels):
# 
# Going through few popular kernels, I could find these binning approaches used:



