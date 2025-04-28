# Since some variable distributions look like Gaussian Mixtures, I have been playing around with the sklearn GMM library to try and extract the number of components using the BIC score: https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html
# 
# Below kernel shows example for one var, but most vars show a different number of expected components. That doesnt make sense intuitively for me, because I would have expected similar noise functions to be added to add noise to something like a categorical variable. 
# 
# Thoughts and comments are welcome!


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.mixture import GaussianMixture as GMM
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
import itertools
import json

from scipy import linalg
import matplotlib as mpl

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


trn = train.filter(regex='var.+')
tst = test.filter(regex='var.+')


trn.head()


#Fit GMM based on number of components, returns labels of predictions
def fit_gmm(X,col,components):
    
    X = trn[col].values
    Xr = X.reshape(-1,1)
    gmm = GMM(n_components=components).fit(Xr)
    labels = gmm.predict(Xr)
    
    return labels


    
#Find best fit for GMM
def calc_gmm(X,col,components):
    
    X = trn[col].values
    X = X.reshape(-1,1)
    
    lowest_bic = np.infty
    best_component = 0
    
    bic = []
    n_components_range = range(1, components)
    cv_types = ['spherical', 'tied', 'diag', 'full']
    
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = GMM(n_components=n_components,
                                      covariance_type=cv_type)
            gmm.fit(X)
            bic.append(gmm.bic(X))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
                best_component = n_components

    bic = np.array(bic)
    color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                              'darkorange'])
    bars = []

    print ('Column: {} Lowest BIC: {} Components:{}'.format(col,lowest_bic,best_component))
           
    # Plot the BIC scores
    plt.figure(figsize=(20, 7))
    spl = plt.subplot(2, 1, 1)
    for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
        xpos = np.array(n_components_range) + .2 * (i - 2)
        bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                  (i + 1) * len(n_components_range)],
                        width=.2, color=color))
    plt.xticks(n_components_range)
    plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
    plt.title('BIC score per model')
    xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
        .2 * np.floor(bic.argmin() / len(n_components_range))
    plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
    spl.set_xlabel('Number of components')
    spl.legend([b[0] for b in bars], cv_types)

    """
    # Plot the winner
    splot = plt.subplot(2, 1, 2)
    Y_ = clf.predict(X)
    for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_,
                                           color_iter)):
        v, w = linalg.eigh(cov)
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan2(w[0][1], w[0][0])
        angle = 180. * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(.5)
        splot.add_artist(ell)

    plt.xticks(())
    plt.yticks(())
    plt.title('Selected GMM: full model, 2 components')
    plt.subplots_adjust(hspace=.35, bottom=.02)
    plt.show()
    """

    
    return best_gmm, best_component    
    




gmm,best_component = calc_gmm(trn,'var_126',15) 


