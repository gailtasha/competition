# ![](http://res.cloudinary.com/dyd911kmh/image/upload/f_auto,q_auto:best/v1513868561/output_65_0_knd6e9.png)


# Plots on top of plot on top of plots. It seems that most of the EDA these days is just throwing around fancy plots from fancy libraries. There is no real **insight** or **reusability** from those kinds of notebooks, just to fill in the space.
# 
# 
# # GOAL: This notebook should serve as a reusable template for **INSIGHTFUL** EDA when approaching a DS problem. 
# 
# Ofcourse there is not a universal solution and it always needs to be modified but I feel like that outlining a couple of general ideas and principles will be usefull since they will repeat themselves.


# Dataset will be [Santander Customer Transaction Prediction](https://www.kaggle.com/c/santander-customer-transaction-prediction/data) where we have 200 columns of anonymised data. 


# # What should good EDA be capable of?
# 1. Verify expected relationships actually exist in the data, thus formulating and validating planned techniques of analysis.
# 2. To find some unexpected structure in the data that must be taken into account, thereby suggesting some changes in the planned analysis.
# 3. Deliver data-driven insights to business stakeholders by confirming they are asking the right questions and not biasing the investigation with their assumptions.
# 4. Provide the context around the problem to make sure the potential value of the data scientist’s output can be maximized.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import scipy



train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")


# Main point is gathering and automising as much as possible. So we will plot all of the variables together (modifying the code for different problems), than "zoom in" in case of suspicion. Since there are a lot of indicators I only took some of them to speed up the computation. We can see distributions as well as plots in relationship with other variables.


sns.pairplot(train_df.iloc[:,197:])


# Same for test set.


sns.pairplot(test_df.iloc[:,198:])


# We should also plot other variables in dependence to the dependent variable--**target**. Thats the first column so lets just take a couple of the first columns. (run-time!!!)


sns.pairplot(train_df.iloc[:,:5])


# # Take-off notes from the first analysis:
# 
# **Ofcourse this is individual but we can see unbalanced classes in the target variable, not much correlation (we will check it more subsequently!). Mostly normaly distribution among predictors, tough to distinguish which values are to be associated with 0 & 1 class etc...**
# 
# 
# As already mentioned one ought to "zoom in" and take one special predictor and to subsequent analysis. Doing a 3-d plot with some special interest variables etc. So it really depends on the problem and the domain knowledge of the problem.


# Let us also assume that no pre-processing will be done (often times) before EDA. We will use EDA to help us with that too.


# Dealing with **missing values**:
# 
# This is a bit specific dataset with no missing values:


pd.isna(train_df).sum().sum()


# BUT, this is a boiler plat code that can be re-used later on different projects!


sns.pairplot(pd.isna(train_df.iloc[:,198:]))


# When **dealing with outliers** one should be careful and look also at the distribution of the variable at hand. For example let us say that we have a uniform distributed variable, does 2 points of std really say anything about a potential outlier? Best thing one could do is assume (or better yet test with kolmogorov smirnov test) a distribution of a variable. Than depending on the result just throw away values that are to be found far away on the distribution graph.


# Since distribution of independent variables is mostly normal, lets see what happens with outliers when measured with (different) points of standard deviation. (one can see it as z-score)


# Before excluding certain values in handle_outliers function underneath, we are going to compare two methods and different paramaeters
# All to see which number of outliers seems reasonable, than we are going to exclude entire row that has this outlier
#It will be only a few since we will opt for the most extreme case, where deviation from the mean is really ridiculous.

def out_std(s, nstd=3.0, return_thresholds=False):

    data_mean, data_std = s.mean(), s.std()
    cut_off = data_std * nstd
    lower, upper = data_mean - cut_off, data_mean + cut_off
    if return_thresholds:
        return lower, upper
    else:
        return [False if x < lower or x > upper else True for x in s]
    

    
    
std2 = train_df.iloc[:,198:].apply(out_std, nstd=2.0)
std3 = train_df.iloc[:,198:].apply(out_std, nstd=3.0)
std4 = train_df.iloc[:,198:].apply(out_std, nstd=4.0)

    
    
f, ((ax1, ax2, ax3)) = plt.subplots(ncols=3, nrows=1, figsize=(22, 12));
ax1.set_title('Outliers with 2 standard deviations');
ax2.set_title('Outliers using 3 standard deviations');
ax3.set_title('Outliers using 4 standard deviations');

sns.heatmap(std2, cmap='Blues', ax=ax1);
sns.heatmap(std3, cmap='Blues', ax=ax2);
sns.heatmap(std4, cmap='Blues', ax=ax3);


plt.show()


# Another way to look at the outliers but also in the same time get some more information about distribution (IQR, median, mean etc...) is with the box-plot. But we need to do it efficiently:


melted = pd.melt(train_df.iloc[:,194:])
melted["value"] = pd.to_numeric(melted["value"])


sns_plot1=sns.boxplot(x="variable", y="value", data=melted)
sns_plot1.set_xticklabels(sns_plot1.get_xticklabels(), rotation = 90, fontsize = 10)




# **Correlation map**- after throwing the outliers and missing values away (since it is neccessary before calculating pearson correlation coefficient)


corr = train_df.iloc[:,190:].corr()

# plot the heatmap
sns_plot2=sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)


# As we noticed from the first plots (scattered ones) there is not really much correlation between the variables.


# So what are some other insights that can be gathered using EDA about the data? One interesting thing is the distribution (density) plot of different predicators when in contrast to different classes (0 or 1).


def plot_feature_distribution(df1, df2, label1, label2, features):
    i = 0
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(5,10,figsize=(18,22))

    for feature in features:
        i += 1
        plt.subplot(5,10,i)
        sns.kdeplot(df1[feature], bw=0.5,label=label1)
        sns.kdeplot(df2[feature], bw=0.5,label=label2)
        plt.xlabel(feature, fontsize=9)
        locs, labels = plt.xticks()
        plt.tick_params(axis='x', which='major', labelsize=6, pad=-6)
        plt.tick_params(axis='y', which='major', labelsize=6)
    plt.show();


t0 = train_df.loc[train_df['target'] == 0]
t1 = train_df.loc[train_df['target'] == 1]
features = train_df.columns.values[2:52]
plot_feature_distribution(t0, t1, '0', '1', features)


# **Interesting observation** is its not always normal distribution, in some cases and classes we can observe almost bimodal distribution. **Implication?** Normality assumption is not met, be careful in model choices etc if we were to use these predicators.


# **Another** thing that should be important to us (to ensure could prediction power) is that **test and train sets are the same**, i.e. they come from the same sample and they represent the whole population. Lets plot it for first 50 variables.


features = train_df.columns.values[2:52]
plot_feature_distribution(train_df, test_df, 'train', 'test', features)


# **Additionally** we can speaak about skewness distriibution (here it is normal), additional exploration with some specific variables/domain specific knowledge, contrasting different scatter plots with some categorical variables (here we do not have classes other than the dependent variable), in case of text some word clouds, tf-idf distribution etc etc....
# There are many options but I think these steps are essential no matter what the dataset at hand is. 
# Additional **(part 2) tutorial** can be made concerning purely textual data and good EDA there.



















