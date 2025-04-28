# The data looks like it has had Principal Component Analysis applied to it, since each feature looks near Gaussian. Let's generate some data to see what PCA looks like on generated continuous and categorical data, and compare some plots to the Santander data.
# 
# **Update**: CPMP suggested Truncated SVD, so we'll explore that too (https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/87301#503701)


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from sklearn.decomposition import PCA, TruncatedSVD

import matplotlib.pyplot as plt
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Create random data
# Let's create 1000 random variables which are correlated


features = 1000

# Means
m = np.random.normal(size=(features), scale=10)

# Covariance matrix.
A = np.random.normal(size=(features, features), loc=1)
cov = np.dot(A, A.transpose())

generated_raw_data = np.random.multivariate_normal(m, cov, size=10000)
generated_raw_data = pd.DataFrame(generated_raw_data)

generated_raw_data.describe()


sns.distplot(generated_raw_data[48]);


# Check if the variables are correlated
plt.scatter(generated_raw_data[48], generated_raw_data[49]);


# # PCA on continuous variables


pca = PCA(n_components=200)
generated_pca_data = pca.fit_transform(generated_raw_data)

generated_pca_data = pd.DataFrame(generated_pca_data)
generated_pca_data.describe()


# Let's have a look at the distributions and cross plots


sns.pairplot(generated_pca_data[list(range(10))]);


# # Truncated SVD on continuous variables


svd = TruncatedSVD(n_components=200)
generated_svd_data = svd.fit_transform(generated_raw_data)

generated_svd_data = pd.DataFrame(generated_svd_data)
generated_svd_data.describe()


sns.pairplot(generated_svd_data[list(range(10))]);


# # PCA on categorical variables
# Let's convert our continuous dataset to a discrete/categorical dataset by rounding to the nearest 100. Then see what the transformed data looks like


generated_raw_cat_data = generated_raw_data.round(-2)
generated_raw_cat_data.describe()


sns.distplot(generated_raw_cat_data[48]);


plt.scatter(generated_raw_cat_data[48], generated_raw_cat_data[49]);


# So now there should only be a handful of categories per feature


generated_pca_cat_data = pca.fit_transform(generated_raw_cat_data)

generated_pca_cat_data = pd.DataFrame(generated_pca_cat_data)
generated_pca_cat_data.describe()


sns.pairplot(generated_pca_cat_data[list(range(10))]);


# Even with a handful of categories per feature, after performing PCA, the cross plots still look like circular/spherical data clouds
# 
# # Truncated SVD on categorical data


generated_svd_cat_data = svd.fit_transform(generated_raw_cat_data)

generated_svd_cat_data = pd.DataFrame(generated_svd_cat_data)
generated_svd_cat_data.describe()


sns.pairplot(generated_svd_cat_data[list(range(10))]);


# # Comparing to the Santander dataset


train_df = pd.read_csv("../input/train.csv")
train_df.describe()


sns.pairplot(data=train_df[::20], vars=train_df.columns[2:12], hue="target");


# # Discussion
# * By visually comparing the pair plots above with our data for both continuous and categorical features, they look pretty similar. Although we can't prove that PCA or SVD has been applied, we can't rule it out either
# * Applying PCA or SVD on categorical data makes it hard to see that the data was originally categorical after the transform. For example, there are no bands/stripes in the cross plots to suggest discretisation. It looks pretty difficult to recover any categories from the transformed data
# * PCA features typically have a mean close to zero. The Santander features do not. This might be a clue hinting that PCA wasn't applied to the data, or deliberate obfuscation by the organisers. SVD on the otherhand creates features with non-zero mean



