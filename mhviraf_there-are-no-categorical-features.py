# The search for *Magic* in this competition is still going on. A common suspect is to take advantage of the categorical features in the dataset (quite likely since it's bank data) but we want to show categorical features are not present in this dataset because of the PCA pre-processing.
# 
# Here is why: The data given in this competetion has been pre-processed by Santander and as part of it, PCA was run on the original dataset. The plot in the following confirms this. Since PCA is a linear rotation of the axis, if it's applied to the categorical data and rotates them it is not possible to recover them without the other eigenvectors. If the PCA has been applied on the categorical and numerical at the same time, then the rotation would be worse and heuristics would be much harder to come up with. The plot in the end shows this is the case.
# 
# The close similarity between the PDFs and normal distribution is also due to PCA since it maximizes the variance and combines as many features as possible (Central Limit Theorem).
# 
# Now, there might have been some categorical features in the original dataset we don't have access to, so if you are still interested to see them, look for ways to transform data back to it's original space. If there is any?
# 
# PCA rotates the axes for the optimum explained variance, so if the true dimensionality is smaller than either dimensions, zero eigenvalues show up. However, if there is no dependency between the data, the last eigenvalue always comes out to be in the order of round off error. It is hard to see this unless a log scale is used.
# Other methods from the same family could also have been used (such as FA, NMF, etc) but doing another PCA on the data shows the variances does not change, so the pre-processing has to be PCA.
# 
# We thought it's a cool finding and just wanted to share it with you. Hope you enjoy it and happy kaggling!


# **How to determine wheter PCA was applied on the original dataset or not?**
# 
# 1. run PCA on the given dataset
# 2. calculate variance of each feature (a) before PCA, and (b) after PCA
# 3. sort variances (a) before PCA, and (b) after PCA and compare them to each other (refer to the figure below)
# 4. if they fall exactly on each other (which is the case here) it means that PCA was used in the preprocessing and applying PCA again would not affect variance of features anymore


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


train = pd.read_csv('../input/train.csv')
train = train.iloc[:, 2:]

for col in train:
    train[col] -= train[col].mean()
    
transformer = PCA(n_components=200)

pca_train = transformer.fit_transform(train)

PCA_arr = np.sort(pca_train.var(0))
org_arr = np.sort(train.values.var(0))

plt.figure(figsize=(20,15))
plt.semilogy(PCA_arr[::-1], '-b', label='after PCA', alpha=.5)
plt.semilogy(org_arr[::-1], '--r', label='before PCA', alpha=0.5)
plt.xlabel('features',fontsize=14)
plt.ylabel('variance',fontsize=14)
plt.legend(fontsize=20)



