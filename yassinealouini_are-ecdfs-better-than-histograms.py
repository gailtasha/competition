# I have recently checked the following tweet:
# 
# <blockquote class="twitter-tweet" data-lang="fr"><p lang="en" dir="ltr">stop plotting histograms.</p>&mdash; Hugo Bowne-Anderson (@hugobowne) <a href="https://twitter.com/hugobowne/status/1111657955248783366?ref_src=twsrc%5Etfw">29 mars 2019</a></blockquote>
# <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
# 
# 
# 
# Indeed, an [ECDF](https://en.wikipedia.org/wiki/Empirical_distribution_function) 
# is often easier to explore and think about. Here is a [**blog post**](https://ericmjl.github.io/blog/2018/7/14/ecdfs/) explaining some of the logic behind this claim.   
# 
# Let's see how it translates to this competition's dataset!


import pandas as pd
import matplotlib.pylab as plt
%matplotlib inline


train_df = pd.read_csv('../input/train.csv')
N_FEATURES = 200


def ecdf(s):
    """ An ECDF computation function using pandas methods."""
    value_counts_s = s.value_counts()
    return value_counts_s.sort_index().cumsum().div(len(s))




def optimal_fd_bins(s):
    """ 
    Optimal number of bins using the FD rule of thumb: 
    https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule
    """
    # Computeing the interquartile range: 
    # https://en.wikipedia.org/wiki/Interquartile_range
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    width = 2 * iqr / (len(s) ** 0.33)
    return int((s.max() - s.min()) / width)


for i in range(N_FEATURES):
    col = 'var_' + str(i)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # ECDF
    ecdf(train_df.loc[lambda df: df.target == 0, col]).plot(ax=ax[0], label="0")
    ecdf(train_df.loc[lambda df: df.target == 1, col]).plot(ax=ax[0], label="1")
    ax[0].set_title(f"ECDF for {col}")
    ax[0].legend()
    
    # Histogram
    bins = optimal_fd_bins(train_df[col])
    train_df.loc[lambda df: df.target == 0, col].plot(kind="hist", bins=bins, ax=ax[1], 
                                                      label="0")
    train_df.loc[lambda df: df.target == 1, col].plot(kind="hist", bins=bins, ax=ax[1], 
                                                      label="1")
    ax[1].set_title(f"Freedman–Diaconis histogram for {col}")
    ax[1].legend()      
    
    plt.show()
    fig.clf()


# Based on the above plots, it appears that:
# 
# * it is indeed easier to see how much two distriubtions differ by inspecting
# the ecdfs.  
# * median values (and othe statistics) are easier to observe. 
# 
# Something to try: plot the ECDF for a normal distribution having the same mean
# and standard deviation and compare it with the ones plotted above. 
# 
# If you have more suggestions, leave them in the comments section. 
# 
# Thanks. :)

