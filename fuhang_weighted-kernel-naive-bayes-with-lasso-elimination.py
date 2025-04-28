# # Weighted Kernel Naive Bayes with Lasso Feature Elimination by TF - Santander  
# > *More things should not be used than are necessary. - Occam's razor*
# 
# Is **EVERY** variable IMPORTANT for prediction? ** The answer is not even close to yes by our finding.**
# 
# In this work, we will explore the santander using two upgraded naive bayes inference -- *Kernel NB and Weighted Kernel NB.*  
# 
# Firstly, we are going to introduce an updated version of Gaussian Naive Bayes method called **Kernel Naive Bayes** which release the assumption of every features follow normal distribution.   
# 
# Secondly, we are going to use back-prop & gradient decent to learn an updated model called **Weighted Kernel Naive Bayes** with lasso feature elimination. This is implemented by log transform the naive bayes formular from product terms into additive terms, and use TensorFlow to learn the loss function with L1-norm and ReLU constriants on weights.   
# 
# Our final experiment show that we can achieve **same level of AUC** using **only 75% or even 50% of features**,meaning that nearly half of the features are not informative for prediction.


# -------------------------


#Kernel NB reference: https://www.kaggle.com/jiazhuang/demonstrate-naive-bayes

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import matplotlib.pyplot as plt
import seaborn as sns

#fit arbitary pdf by scipy.stats.kde.gaussian_kde
from scipy.stats.kde import gaussian_kde

from sklearn.metrics import roc_auc_score as AUC
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


train = pd.read_csv('../input/train.csv', index_col=0)
test = pd.read_csv('../input/test.csv', index_col=0)
target = train.target.values
train.drop('target', axis=1, inplace=True)
train.shape, target.shape, test.shape


# Recall naive bayes follows the **feature independence law**, and use logit probability as prediction score. Conclusively, it should have the following fomular:  
# <img src="https://latex.codecogs.com/gif.latex?logit\_prob=\frac{p(y=1|X)}{p(y=0|X)}=\frac{p(y=1)}{p(y=0)}*\frac{p(X|y=1)}{p(X|y=0)}=\frac{p(y=1)}{p(y=0)}*\prod_{i=1}^{200}\frac{p(x_{i}|y=1)}{p(x_{i}|y=0)}" title="logit_p=\frac{p(y=1|X)}{p(y=0|X)}=\frac{p(y=1)}{p(y=0)}*\frac{p(X|y=1)}{p(X|y=0)}=\frac{p(y=1)}{p(y=0)}*\prod_{i=1}^{200}\frac{p(x_{i}|y=1)}{p(x_{i}|y=0)}" />
# 
# where logit_prob is the final prediction. 


#   
# As we know already, in the first term <img src="https://latex.codecogs.com/gif.latex?{p(y=1)}" title="{p(y=1)}" /> is prior probability of positive class. And P(y=0) can also be calculated easily by 1-P(y=1).  


# The problem lies on how to calculate the 200 other terms, that is to say, how to calculate the following:
# <img src="https://latex.codecogs.com/gif.latex?{p(x_{i}|y=1)}" title="{p(x_{i}|y=1)}" /> as well as <img src="https://latex.codecogs.com/gif.latex?{p(x_{i}|y=0)}" title="{p(x_{i}|y=0)}" />


# There are **two** basic way of calculating this.   
# First way to do is assume that ith feature (**xi**) follows a gaussian distribution a.k.a. normal distribution, and calculate <img src="https://latex.codecogs.com/gif.latex?{p(x_{i}|y=1)}" title="{p(x_{i}|y=1)}" /> by the probability density function (PDF) of normal distribution.  
# 
# However,from other kernel we can see clearly that NOT all of these features follow gaussian distribution, for e.g.https://www.kaggle.com/cdeotte/modified-naive-bayes-santander-0-899
# 
# So a Gaussian assumution might **not be the best choice for estimating p(xi|y)**.  
# For the second way we introduce** Kernel Density Estimation** to calculate the pdf of an arbitary distribution of features.


# What's more we use gaussian kernel KDE using scipy.stats.kde.gaussian_kde, and binize the features to reduce the complexity from O(#data) to O(#bins).  


# ## For accelation, instead of calculating **p(xi|y)**, we now calculate **p(bin\_of\_xi|y)** for every bins. To achieve that, we cut every continues value in xi into bins, and map continues xi to its bins' probability:** p(bin\_of\_xi|y)** . This is bining Kernel Naive Bayes. 


pos_idx=(target==1)
neg_idx=(target==0)
prior_pos=pos_idx.sum()/len(target)
prior_neg=1-prior_pos

pos_kdes,neg_kdes=[],[]#kde函数的列表
for col in train.columns:
    pos_kde=gaussian_kde(train.loc[pos_idx,col])
    neg_kde=gaussian_kde(train.loc[neg_idx,col])
    pos_kdes.append(pos_kde)
    neg_kdes.append(neg_kde)

def cal_prob_KDE_col_i(df,i,num_of_bins=100):
    bins=pd.cut(df.iloc[:,i],bins=num_of_bins)
    uniq=bins.unique()
    uniq_mid=uniq.map(lambda x:(x.left+x.right)/2)
    #把每一格uniq_mid映射到kde值
    mapping=pd.DataFrame({
        'pos':pos_kdes[i](uniq_mid),
        'neg':neg_kdes[i](uniq_mid)
    },index=uniq)
    return bins.map(mapping['pos'])/bins.map(mapping['neg'])

ls=[[prior_pos/prior_neg]*len(train)]
for i in range(200):
    ls.append(cal_prob_KDE_col_i(train,i))
train_KernelNB=pd.DataFrame(np.array(ls).T,columns=['prior']+['var_'+str(i) for i in range(200)])


# Here we have already get the Kernel Naive Bayes transformation of original data. It have 201 columns. Be aware of what we have done, these 201 features are not orignal features, but 201 terms in naive bayes formular:
# <img src="https://latex.codecogs.com/gif.latex?logit\_prob=\frac{p(y=1)}{p(y=0)}*\prod_{i=1}^{200}\frac{p(x_{i}|y=1)}{p(x_{i}|y=0)}" title="\frac{p(y=1)}{p(y=0)}*\prod_{i=1}^{200}\frac{p(x_{i}|y=1)}{p(x_{i}|y=0)}" />
# where the first term is logit of prior probability, and last 200 terms are logit of likelyhood.


# We are pretty **close to the final prediction**. If we now apply Naive Bayes, we can get the final prediction by multiplying these 201 terms as following:


pred=train_KernelNB.apply(lambda x:np.prod(x),axis=1)
AUC(target,pred)


# **The AUC is over 0.908.** That is a very impressive score in training set, althought may be a little overfit on the training data.   
# According to the original kernel https://www.kaggle.com/jiazhuang/demonstrate-naive-bayes/notebook, it reported **AUC 0.894** in public leaderboard.


# **Above are explanation & code for Kernel Naive Bayes. **  


# ------------------------
# # Now introduce Weighted Kernel Naive Bayes with Lasso Feature Elimination by gradient decent.


# ## First, in order to simplfy our naive bayes problem, we log the "logit\_prob" term in order to transform series of product into series of sum.
# <img src="https://latex.codecogs.com/gif.latex?log[\frac{p(y=1)}{p(y=0)}*\prod_{i=1}^{200}\frac{p(x_{i}|y=1)}{p(x_{i}|y=0)}]=log\frac{p(y=1)}{p(y=0)}&plus;\sum_{i=1}^{200}log\frac{p(x_{i}|y=1)}{p(x_{i}|y=0)}" title="log[\frac{p(y=1)}{p(y=0)}*\prod_{i=1}^{200}\frac{p(x_{i}|y=1)}{p(x_{i}|y=0)}]=w_{0}\cdot log\frac{p(y=1)}{p(y=0)}+\sum_{i=1}^{200}w_{i}\cdot log\frac{p(x_{i}|y=1)}{p(x_{i}|y=0)}" />


# ## Second, we try to **weighted every term above** -- that is why it is called weighted naive bayes.
# <img src="https://latex.codecogs.com/gif.latex?log\_logit\_prob=w_{0}\cdot&space;log\frac{p(y=1)}{p(y=0)}&plus;\sum_{i=1}^{200}w_{i}\cdot&space;log\frac{p(x_{i}|y=1)}{p(x_{i}|y=0)}" title="w_{0}\cdot log\frac{p(y=1)}{p(y=0)}+\sum_{i=1}^{200}w_{i}\cdot log\frac{p(x_{i}|y=1)}{p(x_{i}|y=0)}" />


# And the constrant is that **every weights wi (201>=i>=0) should be greater or equal to 0.**   


# Actually, the "naive bayes" is just a special case of "weighted naive bayes" **when every weights wi are initlized to 1**. --That insire me to initlize every weights wi to 1 (see the code tf.ones), and let the model update the weights. And experiments show that **initilizations are so vital** that it stablize the model training -- the model would become very hard to learn if we init the weights in random way.


# ## Finally, after we get the sum of weighted log logit probability, we now exp the sum and revocer the raw score!<img src="https://latex.codecogs.com/gif.latex?logit\_prob=exp(log\_logit\_prob)=exp(w_{0}\cdot&space;log\frac{p(y=1)}{p(y=0)}&plus;\sum_{i=1}^{200}w_{0}\cdot&space;log\frac{p(x_{i}|y=1)}{p(x_{i}|y=0)})" title="logit\_prob=exp(w_{0}\cdot log\frac{p(y=1)}{p(y=0)}+\sum_{i=1}^{200}w_{0}\cdot log\frac{p(x_{i}|y=1)}{p(x_{i}|y=0)})" />


# The implement is as follows: We choose **TensorFlow** to do the gradient decent&back-prop to learn the weights wi. Also, in order to elimintate the useless features, we use **Lasso shrikage** method -- also known as L1 normalization. In order to make sure the all weights are greater or equal to zero, **a Rectified Linear Unit(ReLU) are applied to the weights** before its dot product with log of logit.


# **You might wonder why L1-norm produce sparsity? ** you cuold check [this kaggle discussion](https://www.kaggle.com/residentmario/l1-norms-versus-l2-norms) for further machine learning knowledge.


# ## So the main formula are below:


# <img src="https://latex.codecogs.com/gif.latex?logit\_prob=exp(ReLU(w_{0})\cdot&space;log\frac{p(y=1)}{p(y=0)}&plus;\sum_{i=1}^{200}ReLU(w_{i})\cdot&space;log\frac{p(x_{i}|y=1)}{p(x_{i}|y=0)})" title="logit\_prob=exp(ReLU(w_{0})\cdot log\frac{p(y=1)}{p(y=0)}+\sum_{i=1}^{200}ReLU(w_{i})\cdot log\frac{p(x_{i}|y=1)}{p(x_{i}|y=0)})" />


# <img src="https://latex.codecogs.com/gif.latex?L1\_norm=\lambda&space;\cdot&space;\sum_{i=0}^{200}\left&space;|&space;w_{i}&space;\right&space;|" title="L1\_norm=\lambda \cdot \sum_{i=0}^{200}\left | w_{i} \right |" />


# **We choose cross_entropy as target function, and add l1 norm to prevent overfitting as well as doing lasso shrinkage.**


# <img src="https://latex.codecogs.com/gif.latex?loss=cross\_entropy(y\_true,logit\_prob)&space;&plus;L1\_norm" title="loss=cross\_entropy(y\_true,logit\_prob) +L1\_norm" />


# Let's start, Frist we log the train_KernelNB as input data.


log_train_KernelNB=train_KernelNB.apply(np.log,axis=1)#the log transform  is input for our nerual net.


# Just a notice, we can get the same AUC score of 0.908 by **doing sum** on the logified data before doing exp on it. We should have a clear understanding of what we have done: first do a log transformation, and **a sum in log transform means product in original form**.   
# **And the exp of sum actually reverse the transform, making it equal to product of origanl data**. That is to say:


# it is same as temp=train_KernelNB.apply(lambda x:np.prod(x),axis=1)
temp=log_train_KernelNB.apply(lambda x:np.exp(np.sum(x)),axis=1)
AUC(target,temp)


# ## **Let's implement what we have introduce above, and start eliminting the features!**


#NB:score=exp( log(p(y=1)/p(y=0))+∑log(p(xi|y=1)/p(xi|y=0)) )
#weighted NB:score=exp( w0*log(p(y=1)/p(y=0))+ wi*∑log(p(xi|y=1)/p(xi|y=0)) ) 
import tensorflow as tf

n=201 #201=1+200: 1 for prior prob's logit, 200 for likelyhood's logit 
y=tf.placeholder(tf.float32,[None,1])
x=tf.placeholder(tf.float32,[None,n])
w=tf.Variable(tf.ones([n]))#here, we initlize weighted NB making it start as a normal NB.
w=tf.nn.relu(w)#ReLU applied on w the make sure weights are positive or sparse
tf.multiply(w,x).shape

linear_term=tf.reduce_sum(tf.multiply(w,x),axis=1,keepdims=True)#(None,1)
linear_term=tf.math.exp(linear_term)#do the exp transform to reverse log

#define lambda coef for L1-norm, a key parameter to tune.
lambda_w=tf.constant(2*1e-5,name='lambda_w')
l1_norm=tf.multiply(lambda_w,tf.reduce_sum(tf.abs(w),keepdims=True))

error = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=linear_term, labels=y))
loss = error+l1_norm


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(log_train_KernelNB.values, target, test_size=0.2, random_state=42)
print(X_test[:5,:])
def batcher(X_, y_=None, batch_size=-1):
    n_samples = X_.shape[0]
    if batch_size == -1:
        batch_size = n_samples
    if batch_size < 1:
       raise ValueError('Parameter batch_size={} is unsupported'.format(batch_size))
    for i in range(0, n_samples, batch_size):
        upper_bound = min(i + batch_size, n_samples)
        ret_x = X_[i:upper_bound]
        ret_y = None
        if y_ is not None:
            ret_y = y_[i:i + batch_size]
            yield (ret_x, ret_y)

optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
N_EPOCHS=100
batch_size=5000
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(N_EPOCHS):
        if epoch==0:
            for bX, bY in batcher(X_test, y_test):#all sample
                prediction_list=sess.run(linear_term, feed_dict={x: bX.reshape(-1, n)})
                print('Init w:',sess.run(w, feed_dict={x: bX.reshape(-1, n)}))
                print('Init pred:',prediction_list)
                print('Init Test AUC:',AUC(y_test,prediction_list))
               
        perm = np.random.permutation(X_train.shape[0])
        # iterate over batches
        for bX, bY in batcher(X_train[perm], y_train[perm], batch_size):
            sess.run(optimizer, feed_dict={x: bX.reshape(-1, n), y: bY.reshape(-1, 1)})
            
        if (epoch+1)%20==0:
            print("Epoch:",epoch)
            for bX, bY in batcher(X_test, y_test):#all sample
                print('Test CrossEntropy (logloss):',sess.run(error, feed_dict={x: bX.reshape(-1, n), y: bY.reshape(-1, 1)}))
                prediction_list=sess.run(linear_term, feed_dict={x: bX.reshape(-1, n)})
                weights=sess.run(w, feed_dict={x: bX.reshape(-1, n)})
                print('w:',weights)
                print('Test pred:',prediction_list)
                print('Test AUC:',AUC(y_test,prediction_list))
            print('=======')


# You can see the **weights w are becoming sparser and sparser** as train epoch goes on, from all ones to many zeros. So we are making progress in eliminting uesless features.


#  as you see, the **test AUC are really hard to decrease** . However it doesn't matter, because we  want our final model to remain to be simple naive bayes with weights fix to 1.0, and the learned coef is not the key. **We just want to eliminate those useless features** using lasso shrinkage. So we mask those features that are sparse or "almost sparse(<0.01)". And use the mask to reduce the size of columns.


sparse_weights=np.where(np.abs(weights)<1e-2,0,weights)
print ('Number of eliminated feature',(sparse_weights<=0).sum())


# As we can see, after training, 53 of 201 features are eliminated. That is a total of 25% of the original data!   
# **And If we increase the lambda\_w of LASSO from 2e-5 to 3e-5~4e-5, we can best eliminate 50% of the features without lossing too many socres!!!!**


# Let's submit the result with only 148 features and see what score we got.  
# The **AUC on public LB is 0.893** when using only 148 variables, a** very slight drop** from 0.894 using 201 features.


ls=[[prior_pos/prior_neg]*len(test)]
for i in range(200):
    ls.append(cal_prob_KDE_col_i(test,i))
test_KernelNB=pd.DataFrame(np.array(ls).T,columns=['prior']+['var_'+str(i) for i in range(200)])


pred_test=test_KernelNB.loc[:,sparse_weights>0].apply(lambda x:np.prod(x),axis=1)
pd.DataFrame({
    'ID_code': test.index,
    'target': pred_test
}).to_csv('sub01_KernelNB_L1_2E-5_VAL_0.904.csv', index=False)


# Now,Let's see the weights distribution and rank the most important features by their feature weights.


sparse_weights=pd.Series(sparse_weights,index=['prior']+['var_'+str(i) for i in range(200)])
sns.distplot(sparse_weights,bins=50,kde=False)


# As we see, more than 50 features are eliminated when lambda\_w=2e-5. That's why it called feature elimination using LASSO.


print (sparse_weights.sort_values(ascending=False))


# As we saw above, prior is still the most importance term because it have largest weights. Over 25% of features are eliminated, we can check their logit's distribution in this kernel(https://www.kaggle.com/cdeotte/modified-naive-bayes-santander-0-899).   
#   
# These "unimportant" features in our model, such as **var_153, really doesn't have any instructive information for our prediction**,its logit vary from maxium of 0.115 to the minumn of 0.975.It is a very small shake.    
# **However, the "good feature" in our model -- such as var_105, its logit vary from max of 0.19 to min of 0.08. That's a lot of info lying here.**


# Just curious, let's check if the "unimportant" feature have any pattern/order:  


sparse_weights.plot()


# No pattern for the elimintaed features.. So the columns **must have been shuffle**.  
# 
# Moreover, we can **compare** some important features in LightGBM (https://www.kaggle.com/jiweiliu/lgb-2-leaves-augment) and see if they have high score in Lasso Weighted Naive Bayes:


sparse_weights[['var_'+str(i) for i in [108,184,9,80,76,13,166,94,170,154,133,169,174,123,6] ]]


# The weights are **basically consistent** with LGB's importance score. Most LGB's important feature are also important features for Lasso Weighted Naive Bayes.   
# 
# But they are not exactly the same(e.g. var_9), so **it might still be helpful to combine these two algorithms together** because they share some diversity.


# ## more
# 


# We can rethink our weighted naive bayes model and interprete it in this way:


# <img src="https://latex.codecogs.com/gif.latex?w_{0}\cdot&space;log\frac{p(y=1)}{p(y=0)}&plus;\sum_{i=1}^{200}w_{i}\cdot&space;log\frac{p(x_{i}|y=1)}{p(x_{i}|y=0)}&space;=log[\frac{p(y=1)}{p(y=0)}]^{w0}&plus;\sum_{i=1}^{200}log[\frac{p(x_{i}|y=1)}{p(x_{i}|y=0)}]^{w_{i}}" title="w_{0}\cdot log\frac{p(y=1)}{p(y=0)}+\sum_{i=1}^{200}w_{i}\cdot log\frac{p(x_{i}|y=1)}{p(x_{i}|y=0)} =log[\frac{p(y=1)}{p(y=0)}]^{w0}+\sum_{i=1}^{200}log[\frac{p(x_{i}|y=1)}{p(x_{i}|y=0)}]^{w_{i}}" />


# That provide us **a different view about how the weights works** -- if a sparse weight wi=0, then wi=0 make the logit^wi=logit^0 =1, and log(1)=0, making this logit terms equal to zero. So the our final result(as a linear combination) would not be influence by this features any more since adding zero term means adding nothing.


# ## Further  Discussion  
# Is weighted NB model perfect? I don't think so, **althought it is good that our model eliminated useless features,but it still need to be digging why weighted features are not gaining AUC improvment. In theory weighted naive bayes should learn more than naive bayes. Maybe it is an overfitting problem or data noise problem.    **   


# Futher more, weighted on every column a constant might not be as good as** weighting every columns differently by the attribute of this column**.  That model is called ** Attribute weighted Naive Bayesian**, and there have been many research done in that field for recent years. If you are interested you could check [this paper](https://ieeexplore.ieee.org/document/5593445/) as start.


# **And,there are more interesting things to try using naive bayes model, such as using LGB to replace kernel density estimation to estimate p(xi|y) in this kernel: https://www.kaggle.com/b5strbal/lightgbm-naive-bayes-santander-0-900 . **


# # Conclusion:
# In this Kernel, we first explain Kernel Naive Bayes using Kernel Density Estimation(KDE), and illustrate the merits of Kernel NB over Gaussian NB.  
# Secondly, we use log transformation that transform Naive Bayes's product into a sum of 201 logit terms, then **we strengthen the "simple sum of logit term" by giving every term a weight wi(w>=0),making it a standard weighted linear regression problem**. So we use gradient decent to learn these weights. For the purpose of eliminating useless feautures, **Lasso(L1-norm)** are used in the linear model. Also, **ReLU** are applied on the weights to make sure that weights either positive(>0) or sparse(=0).   
# 
# Our Lasso Weighted Naive Bayes model use back-prop&gradient decent to shrikage the weights. And the experiment shows that our model is **expert** in eliminating these useless features by itself.  By tuning the lambda of L1-norm from 2e-5 to 4e-5, at last we are able to gain almost **same level of AUC using only 50% of the features, which actually indicate that around half of the features are not informative (I prefer saying "almost useless") for our prediction.  ** Further analysis show that the feature selection process could be possibly applied to  LGB model.
# 
# -------


# *I would be happy if my post help you understand NB or the data better.*   
# Give me a **thumb up or follow** if this kernel is helpful to you. Appreciate that!

