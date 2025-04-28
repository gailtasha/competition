# In this short notebook, I will explore a [nonparametric statistics](https://en.wikipedia.org/wiki/Nonparametric_statistics) method 
# for fitting densities: [**kernel density estimation**](https://en.wikipedia.org/wiki/Kernel_density_estimation). 
# Notice that this notebook is (mainly) inspired from this great [post](https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/. ). 
# 
# Let's dive in!


# Some imports, as usual.
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
import pandas as pd
import numpy as np
import matplotlib.pylab as plt


train_df = pd.read_csv('../input/train.csv')


fig, ax = plt.subplots(figsize=(12, 8))
train_df['var_81'].plot(kind='hist', bins=100, ax=ax)


var_81_a = train_df['var_81'].values


# # Fitting


# Let's try this approach on the `var_81` feature.


kde = KernelDensity()
kde.fit(var_81_a[:, None])


x_grid = np.linspace(var_81_a.min(), var_81_a.max(), 1000)
pdf = np.exp(kde.score_samples(x_grid[:, None]))


fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(x_grid, pdf, linewidth=3, alpha=0.5, label='bw=%.2f' % kde.bandwidth)
ax.hist(var_81_a, 100, fc='gray', histtype='stepfilled', alpha=0.3, density=True)
ax.legend(loc='upper left')
ax.set_xlim(var_81_a.min(), var_81_a.max());


# => The fitted curve doesn't seem to follow the histogram. Could we do better? 
# 
# Yes, we can. There is a hyperparameter that one can vary: the kernel bandwidth. 
# Let's see how the fitted kde changes when this bandwidth changes. 


# # Different bandwidths


def fit_kde_and_plot(bandwidth):
    """ Fit a kernel density estimator and plot the resulting graph. """
    kde = KernelDensity(bandwidth=bandwidth)
    kde.fit(var_81_a[:, None])
    x_grid = np.linspace(var_81_a.min(), var_81_a.max(), 1000)
    pdf = np.exp(kde.score_samples(x_grid[:, None]))

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.plot(x_grid, pdf, linewidth=3, alpha=0.5, label='bw=%.2f' % kde.bandwidth)
    ax.hist(var_81_a, 100, fc='gray', histtype='stepfilled', alpha=0.3, density=True)
    ax.legend(loc='upper left')
    ax.set_xlim(var_81_a.min(), var_81_a.max())
    return ax


fit_kde_and_plot(0.01)


fit_kde_and_plot(0.1)


fit_kde_and_plot(0.5)


# => A smaller bandwidth results in a more varying estimator and 
# a bigger value results in a smoother one. 
# 
# Have you spotted the [**bias-variance**](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff) trade-off already? ;)
# 
# Let's try [**cross-validation**](https://en.wikipedia.org/wiki/Cross-validation_(statistics)) to find the best one.


# # Grid searching the best bandwidth


# => WIP: The grid search runs for a very long time, maybe I am doing something wrong? Any help is appreciated!


# TODO: Investigate why this takse so much time to run...
# grid = GridSearchCV(KernelDensity(),
#                     {'bandwidth': [0.1, 0.2, 0.3, 0.4, 1.0]},
#                     cv=2, n_jobs=-1)
# grid.fit(var_81_a[:, None])
# print(grid.best_params_)

