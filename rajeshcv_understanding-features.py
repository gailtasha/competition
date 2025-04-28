import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.listdir("../input"))


# **Data Visualization**


# Visualizing Data is a very important step in a Data Science project.  As per the recent Kagagle Survey 10-20% of the total Data Science Project time is spent on visualizing data. https://www.kaggle.com/rajeshcv/state-of-data-science-machine-learning-2018
# 
# SaS Data Visualization’s webpage explain Data visualization  beautifully.
# 
# *'The way the human brain processes information, using charts or graphs to visualize large amounts of complex data is easier than poring over spreadsheets or reports. Data visualization is a quick, easy way to convey concepts in a universal manner — and you can experiment with different scenarios by making slight adjustments.'*
# 
# In the Santander Customer Transaction Prediction competition the features are predominently numeric.
# 
# This kernel's objective is to
# *     Understand the value distribution in various features through boxplots and histograms. 
# *     Seggregate features into groups based on range of values.
# *     Identify  features with similiar value distribution.
# *     To understand if there is any difference in values between the two target groups 'transaction done and 'transaction not done'
# *  Check whether feature values in test and train comes from the same sampling  distribution.
# 
# 


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


train.head()


train.info()


# All the 200 features have numeric values . Let's check first if some of these numeric features are categorical  or boolean .In that case they will have less than 500 unique values.


likely_cat = {}
for c in train.columns:
    likely_cat[c] = 1.*train[c].nunique()/train[c].count() < 0.005
likely_cat= pd.Series(likely_cat)
likely_cat[likely_cat==True]


train.var_68.nunique()


# None of the features are categorical or boolean except 'target'. Let's understand the range of values of each of these features by plotting the max, min and median value of the features.


trainvaluedist = pd.DataFrame(train.iloc[:,2:].max(axis=0),columns=["Max_value"])
trainvaluedist['Min_value'] = train.iloc[:,2:].min(axis=0)
trainvaluedist['Median_value'] = train.iloc[:,2:].median(axis=0)
trainvaluedist.head()


sns.set(rc={'figure.figsize':(24,12)})
line=sns.lineplot(data=trainvaluedist )
line= line.set(yticks=[-80,-60,-40,-30,-20,-10,0,10,20,30,40,60,80])


# Combined all values in the features are between 80 and -90.  From the plot it looks like the features can be seggregrated into 10 groups  based on their max and min values.


colzerototen= [c for c in train.iloc[:,2:].columns if (train.loc[:,c].min() >=0) & (train.loc[:,c].max()< 10) ]
print('Number of features with positive values and maximum value less than 10 :',len(colzerototen))
colzerototwenty= [c for c in train.iloc[:,2:].columns if (train.loc[:,c].min() >=0) & (train.loc[:,c].max() >= 10) & (train.loc[:,c].max() < 20)  ]
print('Number of features with positive values maximum value between 10 & 20 :',len(colzerototwenty))
colzeroandtwentyplus= [c for c in train.iloc[:,2:].columns if (train.loc[:,c].min() >=0) & (train.loc[:,c].max() >= 20)]
print('Number of features with positive values maximum value > 20 :',len(colzeroandtwentyplus))
colzerominus= [c for c in train.iloc[:,2:].columns if train.loc[:,c].max() <0 ]
print('Number of features with only negative values :',len(colzerominus))
colplustenminusten= [c for c in train.iloc[:,2:].columns if (train.loc[:,c].max() <= 10) & (train.loc[:,c].min() >=-10 )& (train.loc[:,c].min()< 0 )]
print('Number of features with negative values between 10 and -10 :',len(colplustenminusten))
colplustwentyminusten= [c for c in train.iloc[:,2:].columns if (train.loc[:,c].max() <= 20)& (train.loc[:,c].max() > 10) & (train.loc[:,c].min() >=-10 ) & (train.loc[:,c].min() < 0 )]
print('Number of features with max value between 10 and 20 and min value between  between 0 and -10  :',len(colplustwentyminusten))
colplustwentyminustwenty= [c for c in train.iloc[:,2:].columns if (train.loc[:,c].max() <= 20) &  (train.loc[:,c].min() < -10 ) & (train.loc[:,c].min() >= -20 )]
print('Number of features with max value less than 20 and min value between -10 and -20 :',len(colplustwentyminustwenty))
colplustwentyminustwentyless= [c for c in train.iloc[:,2:].columns if (train.loc[:,c].max() <= 20)& (train.loc[:,c].min() < -20 )]
print('Number of features with max value less than 20 and min value less than -20 :',len(colplustwentyminustwentyless))
colplustwentymoreminusten= [c for c in train.iloc[:,2:].columns if (train.loc[:,c].max() >20)& (train.loc[:,c].min()< 0 ) & (train.loc[:,c].min()>= -10 )]
print('Number of features with max value more than 20 and min value more than -10 :',len(colplustwentymoreminusten))
colplustwentymoreminustwenty= [c for c in train.iloc[:,2:].columns if (train.loc[:,c].max() >20)& (train.loc[:,c].min()< -10 ) & (train.loc[:,c].min()>= -20 )]
print('Number of features with max value more than 20 and min value between -10 and -20:',len(colplustwentymoreminustwenty))
colplustwentymoreminustwentymore= [c for c in train.iloc[:,2:].columns if (train.loc[:,c].max() >20)& (train.loc[:,c].min()< -20 )]
print('Number of features with max value more than 20 and min value less than -20:',len(colplustwentymoreminustwentymore))


# **Features with  positive values and maximum value less than 10**


sns.set(rc={'figure.figsize':(20,8)})
setpositive=train.loc[:,colzerototen].boxplot(rot=90)
setpositive=setpositive.set(yticks=[0,2.5,5,7.5,10],title="Features with  positive values and maximum value less than 10")


# var_68,var_91,var_103,var_148 and var_161 have comparatively lower range of values .
# The histograms below shows the distribution of values in cases of transaction done in green color (target=1) and transaction not done (target=0) in red colour.


sns.set(rc={'figure.figsize':(20,16)})
plotlist =['hist'+ str(col) for col in colzerototen]

for k in range(len(colzerototen)):
     plt.subplot(4,4,k+1)
     plotlist[k] =plt.hist(train[colzerototen[k]])
     #plotlist[k].set(title=colzerototen[k])
    


sns.set(rc={'figure.figsize':(20,16)})
def sephist(col):
    yes = train[train['target'] == 1][col]
    no = train[train['target'] == 0][col]
    return yes, no

for num, alpha in enumerate(colzerototen):
    plt.subplot(4, 4, num+1)
    plt.hist(sephist(alpha)[0], alpha=0.75, label='yes', color='g')
    plt.hist(sephist(alpha)[1], alpha=0.25, label='no', color='r')
    plt.legend(loc='upper right')
    plt.title(alpha)
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)


# var_103 values lie  between 1.1 and 2 , var_148 between 3.4 and 4.6 , var_68 values are in a narrow range between 4.99 and 5.04,   var_161 between 5 and 6.2  &  var_91 between 6.6 and 7.4.  Considering var_166 with values between 2 and 4 and var_169 and var_133  they all appear to be in sequence.
# 
# However there is no significant difference in values between the "transaction done" and "transaction not done" groups


# **Features with  positive values and maximum value between 10 & 20**


sns.set(rc={'figure.figsize':(20,8)})
setpositive20=train.loc[:,colzerototwenty].boxplot(rot=90)
setpositive20=setpositive20.set(yticks=[0,5,10,15,20],title="Features with  positive values and maximum value between 10 & 20")


# var_12,  var_15 ,var_25, var_34,  var_43, var_108, var_125 have very low range of values further elaborated by the histogram below.


sns.set(rc={'figure.figsize':(16,24)})
for num, alpha in enumerate(colzerototwenty):
    plt.subplot(8, 4, num+1)
    plt.hist(sephist(alpha)[0], alpha=0.75, label='yes', color='g')
    plt.hist(sephist(alpha)[1], alpha=0.25, label='no', color='r')
    plt.legend(loc='upper right')
    plt.title(alpha)
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)


# All those variables with a short range of values have values in the range 10 to 15 and as in the earlier group appear to be in some sequence.


# **Features with  positive values and maximum value greater than 20**


sns.set(rc={'figure.figsize':(20,8)})
setpositive20plus=train.loc[:,colzeroandtwentyplus].boxplot(rot=90)
setpositive20plus=setpositive20plus.set(yticks=[0,10,20,30,40],title="Features with  positive values and maximum value more than 20")


sns.set(rc={'figure.figsize':(16,20)})
for num, alpha in enumerate(colzeroandtwentyplus):
    plt.subplot(6, 4, num+1)
    plt.hist(sephist(alpha)[0], alpha=0.75, label='yes', color='g')
    plt.hist(sephist(alpha)[1], alpha=0.25, label='no', color='r')
    plt.legend(loc='upper right')
    plt.title(alpha)
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)


# var_85, var_194 and  var_198 appear to have similiar distribution of values.  var_0, var_46 , var_56, var_175 and var_177 also appear to have a similiar value distribution.


# **Features with  values between 10 and -10**


sns.set(rc={'figure.figsize':(16,8)})
setplustenminusten = train.loc[:,colplustenminusten].boxplot(rot=90)
setplustenminusten = setplustenminusten.set(yticks=[-10,-5,0,5,10],title="Features with  values between 10 and -10")


sns.set(rc={'figure.figsize':(16,16)})
for num, alpha in enumerate(colplustenminusten):
    plt.subplot(4, 4, num+1)
    plt.hist(sephist(alpha)[0], alpha=0.75, label='yes', color='g')
    plt.hist(sephist(alpha)[1], alpha=0.25, label='no', color='r')
    plt.legend(loc='upper right')
    plt.title(alpha)
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)


# **Features with  max value between 10 &  20  and min values between 0 & -10**


sns.set(rc={'figure.figsize':(16,8)})
setplustwentyminusten = train.loc[:,colplustwentyminusten].boxplot(rot=90)
setplustwentyminusten = setplustwentyminusten.set(yticks=[-10,-5,0,5,10,15,20],title="Features with  max value between 10 and 20 and min values between 0 and -10")


sns.set(rc={'figure.figsize':(16,16)})
for num, alpha in enumerate(colplustwentyminusten):
    plt.subplot(5, 4, num+1)
    plt.hist(sephist(alpha)[0], alpha=0.75, label='yes', color='g')
    plt.hist(sephist(alpha)[1], alpha=0.25, label='no', color='r')
    plt.legend(loc='upper right')
    plt.title(alpha)
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)


# From the above histogram for many of the features the ' transaction done'  group in green seems to have  lower range than the 'transaction not done' group in red.


# **Features with  max value between 10 &  20  and min values between -10 & -20**


sns.set(rc={'figure.figsize':(20,8)})
setplustwentyminustwenty=train.loc[:,colplustwentyminustwenty].boxplot(rot=90)
setplustwentyminustwenty=setplustwentyminustwenty.set(yticks=[-20,-15,-10,-5,0,5,10,15,20],title="Features with  max value between 10 and 20 and min values between -10  and -20")


sns.set(rc={'figure.figsize':(16,16)})
for num, alpha in enumerate(colplustwentyminustwenty):
    plt.subplot(4, 4, num+1)
    plt.hist(sephist(alpha)[0], alpha=0.75, label='yes', color='g')
    plt.hist(sephist(alpha)[1], alpha=0.25, label='no', color='r')
    plt.legend(loc='upper right')
    plt.title(alpha)
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)


# var_39,var_65 and var_138 appear to have similiar distribution of values and so is var_63 and var_128
# For some of the features the ' transaction done'  group in green seems to have  lower range than the 'transaction not done' group in red.


# **Features with  max value between 10 &  20  and min values less than  -20**


sns.set(rc={'figure.figsize':(20,8)})
setplustwentyminustwentyless=train.loc[:,colplustwentyminustwentyless].boxplot(rot=90)
setplustwentyminustwentyless=setplustwentyminustwentyless.set(yticks=[-40,-30,-20,-15,-10,-5,0,5,10,15,20],title="Features with  max value between 10 and 20 and min values less than -20")


sns.set(rc={'figure.figsize':(16,16)})
for num, alpha in enumerate(colplustwentyminustwentyless):
    plt.subplot(4, 4, num+1)
    plt.hist(sephist(alpha)[0], alpha=0.75, label='yes', color='g')
    plt.hist(sephist(alpha)[1], alpha=0.25, label='no', color='r')
    plt.legend(loc='upper right')
    plt.title(alpha)
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)


# var_84, var _155, var_157 appear to have similiar distribution of values and so does  var_11, var_180 & var_185


# **Features with  max value greater than 20  and min values less than  -20**


sns.set(rc={'figure.figsize':(20,8)})
setplustwentymoreminusten=train.loc[:,colplustwentymoreminusten].boxplot(rot=90)
setplustwentymoreminusten=setplustwentymoreminusten.set(yticks=[-10,-5,0,5,10,15,20,30,40,60],title="Features with  max value more than 20 and min values between 0 and -10")


sns.set(rc={'figure.figsize':(16,16)})
for num, alpha in enumerate(colplustwentymoreminusten):
    plt.subplot(7, 4, num+1)
    plt.hist(sephist(alpha)[0], alpha=0.75, label='yes', color='g')
    plt.hist(sephist(alpha)[1], alpha=0.25, label='no', color='r')
    plt.legend(loc='upper right')
    plt.title(alpha)
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)


# (var_73 & var_158) , (var_92,var_154,var_159 & var_163) , ( var_20 & var_55) 
# The features within the above groups  appear to have similiar distribution of values . Here also the range  of values for the *'transaction done'* group in green appears to be shorter.


# **Features with  max value more than 20 and min values between -10 and -20**


sns.set(rc={'figure.figsize':(20,8)})
setplustwentymoreminustwenty=train.loc[:,colplustwentymoreminustwenty].boxplot(rot=90)
setplustwentymoreminustwenty=setplustwentymoreminustwenty.set(yticks=[-20,-15,-10,-5,0,5,10,15,20,30,40,60],title="Features with  max value more than 20 and min values between -10 and -20")


sns.set(rc={'figure.figsize':(16,16)})
for num, alpha in enumerate(colplustwentymoreminustwenty):
    plt.subplot(7, 4, num+1)
    plt.hist(sephist(alpha)[0], alpha=0.75, label='yes', color='g')
    plt.hist(sephist(alpha)[1], alpha=0.25, label='no', color='r')
    plt.legend(loc='upper right')
    plt.title(alpha)
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)


# var_21 & var_172 appear to have similiar distribution of values


# **Features with  max value more than 20 and min values less than -20**


sns.set(rc={'figure.figsize':(20,8)})
setplustwentymoreminustwentymore=train.loc[:,colplustwentymoreminustwentymore].boxplot(rot=90)
setplustwentymoreminustwentymore=setplustwentymoreminustwentymore.set(yticks=[-40,-30,-20,-10,0,5,10,15,20,30,40,60],title="Features with  max value more than 20 and min values less than -20")


sns.set(rc={'figure.figsize':(16,16)})
for num, alpha in enumerate(colplustwentymoreminustwentymore):
    plt.subplot(6, 4, num+1)
    plt.hist(sephist(alpha)[0], alpha=0.75, label='yes', color='g')
    plt.hist(sephist(alpha)[1], alpha=0.25, label='no', color='r')
    plt.legend(loc='upper right')
    plt.title(alpha)
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)


# var_47 & var_187 appear to have similiar distribution of values


# **Checking for correlation between feature**s


traincorr = train.iloc[:,2:].corr()
traincorr.head()


sns.set(rc={'figure.figsize':(10,8)})
sns.heatmap(traincorr,xticklabels=traincorr.columns,yticklabels=traincorr.columns,cmap=sns.diverging_palette(240, 10, n=9))


# ***No correlation between any features . Does this mean all features are important?***


# **Kolmogorov-Smirnov test**


# Before concluding let's do a check of whether feature values in test and train comes from the same sampling  distribution.
# Kolmogorov-Smirnov  is a two-sided test for the null hypothesis that 2 independent samples are drawn from the same continuous distribution.
# If the K-S statistic is small or the p-value is high, then we cannot reject the hypothesis that the distributions of the two samples are the same.


from scipy.stats import ks_2samp
from tqdm import tqdm
ks_values =[]
p_values  = []
train_columns = train.iloc[:,2:].columns
for i in tqdm(train_columns):
    ks_values.append(ks_2samp(test[i] , train[i])[0])
    p_values.append(ks_2samp(test[i] , train[i])[1])
p_values_series = pd.Series(p_values, index = train_columns) 


# For the two tailed test at 95% confidence level the pvalue has to be less than 0.05 to reject the null hypothesis that both samples are from same distribution.Let's look for values less than 0.05


dissimiliar_features= p_values_series[p_values_series <0.05].index


# As per the Kolmogorov-Smirnov test 46 features have a high probability of not being from the same sampling distribution.
# Will this affect the models? 
# Let's combine the test and train data to compare these features and understand their didtribution in train and test.


train['is_train'] = 1
test['is_train'] = 0
combined = pd.concat([train,test],sort=False)


sns.set(rc={'figure.figsize':(20,48)})
def  diffcheck(col):
    traindata = combined[combined['is_train'] == 1][col]
    testdata = combined[combined['is_train'] == 0][col]
    return traindata, testdata

for num, alpha in enumerate(dissimiliar_features):
    plt.subplot(12, 4, num+1)
    plt.hist(diffcheck(alpha)[0], alpha=0.75, label='train', color='g')
    plt.hist(diffcheck(alpha)[1], alpha=0.25, label='test', color='r')
    plt.legend(loc='upper right')
    plt.title(alpha)
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)


sns.set(rc={'figure.figsize':(20,48)})
fig, axes = plt.subplots(8,3)
for num, alpha in enumerate(list(dissimiliar_features[0:24])):
    a = sns.boxenplot(x='is_train',y=alpha,data=combined, ax=axes.flatten()[num])
# fig.delaxes(axes[11,2])
# fig.delaxes(axes[11,3])
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0) 
plt.title(alpha)
plt.show()


sns.set(rc={'figure.figsize':(20,48)})
fig, axes = plt.subplots(8,3)
for num, alpha in enumerate(list(dissimiliar_features[24:])):
    a = sns.boxenplot(x='is_train',y=alpha,data=combined, ax=axes.flatten()[num])
fig.delaxes(axes[7,1])
fig.delaxes(axes[7,2])
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0) 
plt.title(alpha)
plt.show()


# sns.set(rc={'figure.figsize':(16,48)})
# set1=train.iloc[:,47:92].hist(layout=(9,5),sharey=True)

