# <h1><center><font size="6">Santander Customer Transaction Prediction</font></center></h1>
# <h1><center><font size="5">Can you identify who will make a transaction?</font></center></h1>
# 
# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/4a/Another_new_Santander_bank_-_geograph.org.uk_-_1710962.jpg/640px-Another_new_Santander_bank_-_geograph.org.uk_-_1710962.jpg" width="500"></img>
# 
# <br>
# 
# <b>
#     
# Our data science team is continually challenging our machine learning algorithms, working with the global data science community to make sure we can more accurately identify new ways to solve our most common challenge, binary classification problems such as: is a customer satisfied? Will a customer buy this product? Can a customer pay this loan?
# 
# In this challenge, we invite Kagglers to help us identify which customers will make a specific transaction in the future, irrespective of the amount of money transacted. The data provided for this competition has the same structure as the real data we have available to solve this problem.
# 
# The data is anonimyzed, each row containing 200 numerical values identified just with a number.</b>


# <h1 style="color:blue;">Loading packages</h1>


# Ignore Warnings
import warnings
warnings.filterwarnings('ignore')


# Import Main Packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Import Main Packages For Visualization
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


# Show Our Fils Data
import os
print(os.listdir("../input"))


train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
sub = pd.read_csv("../input/sample_submission.csv")

print("Data are Ready!!")


train_df.head()


# <h2>Check For Any Missing Data.<h2>


if (train_df.isnull().values.any() == False):
    print("No Missing Data")
else:
    train_df.isnull().sum()


print("Train Data Size {}\nTest Data Size {}".format(train_df.shape, test_df.shape))


# **Then We Have 200k Rows And 202 Features.**


# Show Labels Values
train_df['target'].value_counts()


# **After Show Labels Values Describe It**


# Describe 0 Value
train_df[train_df.target == 0].describe()


# Describe 1 Value
train_df[train_df.target == 1].describe()


# <h2 style='color:red'>Split Training Data To Features And Label.</h2>


features = train_df.drop(['ID_code', 'target'], axis=1)
label = train_df['target']


# EDA
fig, ax = plt.subplots(1, 2, figsize=(20, 8))

sns.countplot(label, ax=ax[0])
sns.violinplot(x=label.values, y=label.index.values, ax=ax[1])


# Linear correlations
# <br>
# I have already seen some correlation heatmaps in public kernels and it seems as if there is almost no correlation between features. Let's check this out by computing all correlation values and plotting the overall distribution:


trn_corr = features.corr()
trn_corr = trn_corr.values.flatten()
trn_corr = trn_corr[trn_corr != 1]

plt.figure(figsize=(20, 8))
sns.distplot(trn_corr, color="Green", label="train")
plt.xlabel("Correlation values found in train (except 1)")
plt.ylabel("Density")
plt.title("Are there correlations between features?"); 
plt.legend();


# **Find Correlations Between Train And Test Features.**


train_correlations = train_df.drop(["target"], axis=1).corr()
train_correlations = train_correlations.values.flatten()
train_correlations = train_correlations[train_correlations != 1]

test_correlations = test_df.corr()
test_correlations = test_correlations.values.flatten()
test_correlations = test_correlations[test_correlations != 1]

plt.figure(figsize=(20,8))
sns.distplot(train_correlations, color="Red", label="train")
sns.distplot(test_correlations, color="Green", label="test")
plt.xlabel("Correlation values found in train (except 1)")
plt.ylabel("Density")
plt.title("Are there correlations between features?"); 
plt.legend();


# <hr>
# <h1><center>Models</center></h1> 


# **First**, We Will Use Simple Gaussian Naive Bayes Model.
# <br>
# **Then** , We Will Use Simple Pipeline With Gaussian Naive Bayes Model.


# Import Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, ComplementNB

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import train_test_split # You Can Comment It

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# **Agian Set Our Features And Labels**


X = features.values.astype('float64')
y = label.values.astype('float64')


X_train, X_test, y_train,  y_test = train_test_split(X, y, test_size=0.5, random_state=42)


# <h2>Set Our Gaussian Naive Bayes Model</h2>


model = GaussianNB() # Set Model
model.fit(X, y) # Fit Features and labels

y_pred = model.predict(X_test)


accuracy_score(y_test, y_pred)


plt.figure(figsize=(12, 8))
mat = confusion_matrix(y_test, y_pred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=True)
plt.xlabel('true label')
plt.ylabel('predicted label');
plt.title("Confusion Matrix");


print(classification_report(y_test, y_pred))


# **Find Receiver Operating Characteristic**


from sklearn.metrics import roc_curve, auc

fpr, tpr, thr = roc_curve(y, model.predict_proba(X)[:,1])
plt.figure(figsize=(12, 8))
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Plot')
auc(fpr, tpr) * 100


# Use Pipeline Algo. For Getting better accuracy.


from sklearn.pipeline import make_pipeline # Import pipeline
from sklearn.preprocessing import QuantileTransformer # For Processing Data.

# Set Model
pipeline = make_pipeline(QuantileTransformer(output_distribution='normal'), GaussianNB())
pipeline.fit(X, y)


p_pred = pipeline.predict(X_test)


accuracy_score(y_test, p_pred)


print(classification_report(y_test, p_pred))


from sklearn.metrics import roc_curve, auc

fpr, tpr, thr = roc_curve(y, pipeline.predict_proba(X)[:,1])
plt.figure(figsize=(12, 8))
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Plot')
auc(fpr, tpr) * 100


# **Ok, After All Of That We Get 1% more accuracy than GaussianNB**
# <br>
# <h1>Good</h1>
# <hr>


# Answer On Question On The Second Line.


# <h1><center>**Let's Identify Who Will Make A Transaction.**</center></h1>


# Show Test File
test_df.head()


# Drop Unused columns
x_test = test_df.drop(['ID_code'], 1).values


# **Now, Predict x_test File By Two Ways**


gnb_pred = model.predict_proba(x_test)[:, 1] # GaussianNB => gnb
pip_pred = pipeline.predict_proba(x_test)[:, 1] # Pipeline => pip


mean_pred = (gnb_pred + pip_pred) / 2.0


# <h1></h1>


# <h1>Submission Our Files</h1>


sub.head()


# **GaussianNB Submission File**


sub['target'] = gnb_pred
sub.to_csv('gnb_submission.csv', index=False)
sub.head()


# **Pipeline Submission File**


sub['target'] = pip_pred
sub.to_csv('pip_submission.csv', index=False)
sub.head()


# <center><h1 style='color:blue'>Thanks For Watching, Hope You Benefit From This Kernel.</h1>
# <h2 style='color:red'>I Would Be So Glad To Answer Your Questions In Comments.</h2></center>

