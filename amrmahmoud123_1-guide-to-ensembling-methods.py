# ![Imgur](https://i.imgur.com/L2Jaqm8.png)


# What is in this tutorial?
# in this tutorial I am trying to illustrate how ensembling techniques work manually and by python code to make a good intuition about why is it  useful and why do we use it.


# **Introduction to ensembling**
# 
# **Types of ensembling :**
# 
# **Basic Ensemble Techniques**
# 
# *     Max Voting
# *     Averaging
# *     Weighted Average
# 
# **Advanced Ensemble Techniques**
# 
# * Stacking
# * Blending
# * Bagging
# * Boosting
# 
# **Algorithms based on Bagging and Boosting**
# 
# > * Bagging meta-estimator
# * Random Forest
# * AdaBoost
# * GBM
# * XGB
# * Light GBM
# * CatBoost


# What is in this tutorial?
# in thi tutorial I am trying to illustrate how ensembling techniques work manually and by python code to make a good intuition about why it is useful and why we use it.
# 


# At first there is a rational we must stabilize that : combination between models increase accuracy and in machine learning combination is **Ensembling** 


# **Introduction to ensembling :**


# **Errors**


# The error emerging from any model can be broken down into three components mathematically. Following are these component :


# ![Imgur](https://i.imgur.com/LmeI08b.png)


# **Why is this important in the current context?**
# To understand what really goes behind an ensemble model, we need to first understand what causes error in the model. We will briefly introduce you to these errors and give an insight to each ensemble learner in this regards.


# **Bias error **
# 
# is useful to quantify how much on an average are the predicted values different from the actual value. A high bias error means we have a under-performing model which keeps on missing important trends.
# 
# **Variance**
# 
# on the other side quantifies how are the prediction made on same observation different from each other. A high variance model will over-fit on your training population and perform badly on any observation beyond training. Following diagram will give you more clarity (Assume that red spot is the real value and blue dots are predictions) :


# ![Imgur](https://i.imgur.com/jFfarvo.png)


# model should maintain a balance between these two types of errors. This is known as the trade-off management of bias-variance errors. **Ensemble learning is one way to execute this trade off analysis.**


# ![Imgur](https://i.imgur.com/ZDZsSr1.png)


# ![Imgur](https://i.imgur.com/Xm5sKxD.png)


# ![Imgur](https://i.imgur.com/GEG80ni.png)


# **A group of predictors is called an ensemble**; thus, this technique is called Ensemble Learning, and an Ensemble Learning algorithm is called an** Ensemble method.**
# 
# Suppose you ask a complex question to thousands of random people, then aggregate their answers. In many cases you will find that this aggregated answer is better than an expert's answer. This is called **the wisdom of the crowd**
# 
# Likewise, if you aggregate the predictions of a group of predictors (e.g. decision tree classifer, SVM, logistic regression), you will often get better predictions than with the best individual predictor.


# **Types of ensembling :****


#  **Basic Ensemble Techniques**
# 
# *     Max Voting
# *     Averaging
# *     Weighted Average
# 
# **Advanced Ensemble Techniques**
# 
# * Stacking
# * Blending
# * Bagging
# * Boosting
# 
# **Algorithms based on Bagging and Boosting**
# 
# > * Bagging meta-estimator
# * Random Forest
# * AdaBoost
# * GBM
# * XGB
# * Light GBM
# * CatBoost
# 


# lets talk first about Max voting


# **Max Voting
# **


# The max voting method is generally used for classification problems. In this technique, multiple models are used to make predictions for each data point. The predictions by each model are considered as a ‘vote’. The predictions which we get from the majority of the models are used as the final prediction.
# 
# For example, when you asked 5 of your colleagues to rate your movie (out of 5); we’ll assume three of them rated it as 4 while two of them gave it a 5. Since the majority gave a rating of 4, the final rating will be taken as 4. You can consider this as taking the mode of all the predictions.
# 
# The result of max voting would be something like this:
# 
# Colleague 1-5
# 
# Colleague 2-4
# 
# Colleague 3-5
# 
# Colleague 4-4
# 
# Colleague 5-4
# 
# Finalrating-4


# **Code in python**


# there are 2 methods :
# 
# 1-Mode
# 
# 2-Voting classifier


model1 = tree.DecisionTreeClassifier()
model2 = KNeighborsClassifier()
model3= LogisticRegression()

model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)

pred1=model1.predict(x_test)
pred2=model2.predict(x_test)
pred3=model3.predict(x_test)

final_pred = np.array([])
for i in range(0,len(x_test)):
    final_pred = np.append(final_pred, mode([pred1[i], pred2[i], pred3[i]]))


from sklearn.ensemble import VotingClassifier
model1 = LogisticRegression(random_state=1)
model2 = tree.DecisionTreeClassifier(random_state=1)
model = VotingClassifier(estimators=[('lr', model1), ('dt', model2)], voting='hard')
model.fit(x_train,y_train)
model.score(x_test,y_test)


# **Majority Voting / Hard Voting
# **


# **Hard voting **
# 
# is the simplest case of majority voting. Here, we predict the class label y^ via majority (plurality) voting of each classifier
# 
# y^=mode{C1(x),C2(x),...,Cm(x)}
# 
# Assuming that we combine three classifiers that classify a training sample as follows:
# 
# classifier 1 -> class 0 classifier 2 -> class 0 classifier 3 -> class 1
# 
# y^=mode{0,0,1}=0
# 
# Via majority vote, we would we would classify the sample as "class 0."


# **Soft Voting
# **


# If all classifiers are able to estimate class probabilities (i.e. they have a predict_proba() method), then you can tell Scikit-Learn to predict the class with the highest class probability, averaged over all the individual classifiers.
# 
# This is called **soft voting** and it often achieves higher performance than hard voting because *it gives more weight to highly confident votes*.
# 
# To perform soft voting, all you need to do is **replace** voting='hard' with voting='soft' **and ensure that all classifiers can estimate class probabilities.
# 
# **The SVC class can't estimate class probabilities by default**, so you'll need to set its probability hyperparameter to **True**, as this will make the SVC class use cross-validation to estimate class probabilities (which slows training down), and it will add a predict_proba() method.
# 
# **In soft voting**, we predict the class labels based on the predicted probabilities p for classifier -- this approach is only recommended if the classifiers are **well-calibrated**.
# 
# *y^=argmaxi∑j=1mwjpij,* where **wj** is the weight that can be assigned to the **jth** classifier.
# 
# Assuming the example in the previous section was a *binary classification* task with class labels i∈{0,1}, our ensemble could make the following prediction:
# 
# C1(x)→[0.9,0.1]
# 
# C2(x)→[0.8,0.2]
# 
# C3(x)→[0.4,0.6]
# 
# Using uniform weights, we compute the average probabilities:
# 
# p(i0∣x)=0.9+0.8+0.43=0.7p(i1∣x)=0.1+0.2+0.63=0.3
# 
# y^=argmaxi[p(i0∣x),p(i1∣x)]=0
# 
# However, assigning the weights {0.1, 0.1, 0.8} would yield a prediction y^=1:
# 
# p(i0∣x)=0.1×0.9+0.1×0.8+0.8×0.4=0.49p(i1∣x)=0.1×0.1+0.2×0.1+0.8×0.6=0.51
# 
# y^=argmaxi[p(i0∣x),p(i1∣x)]=1


# **Averaging**
# 
# Similar to the max voting technique, multiple predictions are made for each data point in averaging. In this method, we take an **average** of predictions from all the models and use it to make the final prediction.Averaging can be used for making predictions in regression problems or while calculating probabilities for classification problems.For example, in the below case, the averaging method would take the average of all the values.
# 
# *i.e. (5+4+5+4+4)/5 = 4.4*


# **Code in python**


model1 = tree.DecisionTreeClassifier()
model2 = KNeighborsClassifier()
model3= LogisticRegression()

model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)

pred1=model1.predict_proba(x_test)
pred2=model2.predict_proba(x_test)
pred3=model3.predict_proba(x_test)

finalpred=(pred1+pred2+pred3)/3


# **Weighted Average
# **


# **This is an extension of the averaging method.** All models are assigned different weights defining the importance of each model for prediction. For instance, if two of your colleagues are critics, while others have no prior experience in this field, then the answers by these two friends are given more importance as compared to the other people.
# 
# The result is calculated as
# 
# *[(50.23) + (40.23) + (50.18) + (40.18) + (4*0.18)] = 4.41.*


# ![Imgur](https://i.imgur.com/J9drqs1.png)


# **Code in python**


model1 = tree.DecisionTreeClassifier()
model2 = KNeighborsClassifier()
model3= LogisticRegression()

model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)

pred1=model1.predict_proba(x_test)
pred2=model2.predict_proba(x_test)
pred3=model3.predict_proba(x_test)

finalpred=(pred1*0.3+pred2*0.3+pred3*0.4)


# **Advanced Ensemble techniques**


# **Bagging**


# **Bagging** is very common in competitions. I don’t think I have ever seen anybody win without using it. But, in order for this to work, your data must have *variance*, otherwise you’re just adding levels after levels of additional iterations with **little benefit** to your score and a big headache for those maintaining your modeling pipeline in production. Even when it does improve things, you have to asked yourself if its worth all that extra work…


# In simple terms, **bagging irons out variance from a data set** . If, after splitting your data into multiple chunks and training them, you find that your predictions are *different*, then your data has *variance*. Bagging can turn a bad thing into a competitive advantage. For more theory behind the magic, check out *Bootstrap Aggregating on Wikipedia.* Bagging was invented by *Leo Breiman* at the University of California. He is also one of the grandfathers of Boosting and Random Forests.


# **Stability and Accuracy**


# By saving each prediction set and averaging them together, you not only lower variance without affecting bias, but your accuracy may be **improved**! In essence, you are creating many slightly different models and ensembling them together; **this avoids over-fitting**, **stabilizes your predictions and increases your accuracy**. Mind you, this assumes your data has variance, if it doesn’t,**bagging won’t help.**


# Bagging is based on the *statistical method of bootstrapping*, Bagging actually refers to (Bootstrap Aggregators). Most any paper or post that references using bagging algorithms will also reference Leo Breiman who wrote a paper in 1996 called “*Bagging Predictors*”.


# 1-we make subsets with replacement: that means every item may appears in different subsets.
# 
# 2-apply model for every subset of the sample.
# 
# 3-The models run in parallel and are independent of each other.
# 
# 4-predict x-text by using each model
# 
# 5-then aggregate their predictions (either by voting or by averaging) to form a final prediction.


# ![Imgur](https://i.imgur.com/eu95V9N.png)


# **Bagging algorithms:**


# 
# * Bagging meta-estimator
# * Random forest


# **Bagging meta-estimator**


# **Bagging meta-estimator** is an ensembling algorithm that can be used for **both** classification (BaggingClassifier) and regression (BaggingRegressor) problems. It follows the typical bagging technique to make predictions. Following are the steps for the bagging meta-estimator algorithm:
# 
# 1-Random subsets are created from the original dataset (Bootstrapping).
# 
# 2-The subset of the dataset includes all features.
# 
# 3-A user-specified base estimator is fitted on each of these smaller sets.
# 
# 4-Predictions from each model are combined to get the final result.


# **Code in python**


final_dt = DecisionTreeClassifier(max_leaf_nodes=10, max_depth=5)                   
final_bc = BaggingClassifier(base_estimator=final_dt, n_estimators=40, random_state=1, oob_score=True)

final_bc.fit(X_train, train_y)
final_preds = final_bc.predict(X_test)


acc_oob = final_bc.oob_score_
print(acc_oob)  


# where :
# 
# **base_estimator:**
# It defines the base estimator to fit on random subsets of the dataset. When nothing is specified, the base estimator is a decision tree.
# 
# **n_estimators:**
# It is the number of base estimators to be created. The number of estimators should be carefully tuned as a large number would take a very long time to run, while a very small number might not provide the best results.
# 
# **max_samples:**
# This parameter controls the size of the subsets. It is the maximum number of samples to train each base estimator.
# 
# **max_features:**
# Controls the number of features to draw from the whole dataset. It defines the maximum number of features required to train each base estimator.
# 
# **n_jobs:**
# The number of jobs to run in parallel. Set this value equal to the cores in your system. If -1, the number of jobs is set to the number of cores.
# 
# **random_state:**
# It specifies the method of random split. When random state value is same for two models, the random selection is same for both models. This parameter is useful when you want to compare different models.


# **Random Forest**


# **Random Forest** is another ensemble machine learning algorithm that follows the bagging technique. It is an extension of the bagging estimator algorithm. The base estimators in random forest are decision trees. Unlike bagging meta estimator, random forest **randomly** selects a set of features which are used to decide the best split at each node of the decision tree.


# step-by-step, this is what a random forest model does:
# 
# 1-Random subsets are created from the original dataset (bootstrapping).
# 
# 2-At each node in the decision tree, only a random set of features are considered to decide the best split.
# 
# 3-A decision tree model is fitted on each of the subsets. The final prediction is calculated by averaging the predictions from all decision trees.
# 
# **Note:** The decision trees in random forest can be built on a subset of data and features. Particularly, the sklearn model of random forest uses all features for decision tree and a subset of features are randomly selected for splitting at each node.
# 
# **To sum up, Random forest randomly selects data points and features, and builds multiple trees (Forest) .**


# **Code in python**


from sklearn.ensemble import RandomForestClassifier
model= RandomForestClassifier(random_state=1)
model.fit(x_train, y_train)
model.score(x_test,y_test)


# You can see feature importance by using **model.featureimportances** in random forest.
# 


# **Sample code for regression problem:**


from sklearn.ensemble import RandomForestRegressor
model= RandomForestRegressor()
model.fit(x_train, y_train)
model.score(x_test,y_test)


# Parameters :
# 
# **n_estimators:**
# It defines the number of decision trees to be created in a random forest. Generally, a higher number makes the predictions stronger and more stable, but a very large number can result in higher training time.
# 
# **criterion:**
# It defines the function that is to be used for splitting. The function measures the quality of a split for each feature and chooses the best split.
# 
# **max_features :**
# It defines the maximum number of features allowed for the split in each decision tree. Increasing max features usually improve performance but a very high number can decrease the diversity of each tree.
# 
# **max_depth:**
# Random forest has multiple decision trees. This parameter defines the maximum depth of the trees. min_samples_split: Used to define the minimum number of samples required in a leaf node before a split is attempted. If the number of samples is less than the required number, the node is not split.
# 
# **min_samples_leaf:**
# This defines the minimum number of samples required to be at a leaf node. Smaller leaf size makes the model more prone to capturing noise in train data.
# 
# **max_leaf_nodes:**
# This parameter specifies the maximum number of leaf nodes for each tree. The tree stops splitting when the number of leaf nodes becomes equal to the max leaf node.
# 
# **n_jobs:**
# This indicates the number of jobs to run in parallel. S


# **Boosting**


# The term ‘Boosting’ refers to a family of algorithms which **converts weak learner to strong learners**. Boosting is an ensemble method for improving the model predictions of any given learning algorithm. The idea of boosting **is to train weak learners sequentially, each trying to correct its predecessor**.
# 
# Boosting is all about “*teamwork*”. Each model that runs, dictates what features the next model will focus on.


# **AdaBoost**


# **Adaptive boosting or AdaBoost** is one of the simplest boosting algorithms. Usually, decision trees are used for modelling. Multiple sequential models are created, each correcting the errors from the last model. AdaBoost assigns weights to the observations which are incorrectly predicted and the subsequent model works to predict these values correctly.
# 
# **steps:**
# 
# 1-all observations in the dataset are given equal weights.
# 
# 2-A model is built on a subset of data.
# 
# 3-Using this model, predictions are made on the whole dataset.
# 
# 4-Errors are calculated by comparing the predictions and actual values.
# 
# 5-While creating the next model, higher weights are given to the data points which were predicted incorrectly.
# 
# 6-Weights can be determined using the error value. For instance, higher the error more is the weight assigned to the observation.
# 
# 7-This process is repeated until the error function does not change, or the maximum limit of the number of estimators is reached.


# **Code in python**


from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier(random_state=1)
model.fit(x_train, y_train)
model.score(x_test,y_test)


#Sample code for regression problem:

from sklearn.ensemble import AdaBoostRegressor
model = AdaBoostRegressor()
model.fit(x_train, y_train)
model.score(x_test,y_test)


# **Parameters**
# 
# **base_estimators:**
# It helps to specify the type of base estimator, that is, the machine learning algorithm to be used as base learner.
# 
# **n_estimators:**
# It defines the number of base estimators.
# The default value is 10, but you should keep a higher value to get better performance.
# 
# **learning_rate:**
# This parameter controls the contribution of the estimators in the final combination.
# There is a trade-off between learning_rate and n_estimators.
# 
# **max_depth:**
# Defines the maximum depth of the individual estimator.
# Tune this parameter for best performance.
# 
# **n_jobs:**
# Specifies the number of processors it is allowed to use.
# Set value to -1 for maximum processors allowed.
# 
# **random_state :**
# An integer value to specify the random data split.
# A definite value of random_state will always produce same results if given with same parameters and training data.


# **stacking
# **


# 
# Stacking is a similar to boosting:
# 
# you also apply several models to your original data. The difference here is, however, that you don't have just an empirical formula for your weight function, rather you introduce a meta-level and use another model/approach to estimate the input together with outputs of every model to estimate the weights or, in other words, to determine what models perform well and what badly given these input data. and finally I get its true illustration.


# consider we have a dataset we splite our data set into 3 parts : training, validation , test


# ![Imgur](https://i.imgur.com/EQYa8C8.png)


# then make this step


# ![Imgur](https://i.imgur.com/fVazCYe.png)


# ![Imgur](https://i.imgur.com/2zjxVBC.png)


# ![Imgur](https://i.imgur.com/lpDhGd1.png)


# train algorythm 0 on **A** and make predictions for **B** and **C** and save to **B1,C1**
# 
# train algorythm 1 on **A** and make predictions for **B** and **C** and save to **B1,C1**


# ![Imgur](https://i.imgur.com/10slay8.png)


# **At this moment we stacked predictions to each others thats where stacking name comes from** and then


# train algorythm 2 on **A** and make predictions for **B** and **C** and save to **B1,C1**


# ![Imgur](https://i.imgur.com/hfp6JGP.png)


# then we take the data from the validation set which we already knew and we are going to feed a new model .
# 
# train algorythm 3 on **B1** and make predictions for **C1**


# ![Imgur](https://i.imgur.com/md3L8yB.png)


# **Code in python**


# ![Imgur](https://i.imgur.com/bfb2qlr.png)


# ![Imgur](https://i.imgur.com/Bo4KItc.png)


# ![Imgur](https://i.imgur.com/hGkZd9T.png)


# I hope that I give you a piece of introduction of ensembling methods and this is not the end of my tutorial but this is only the first episode and I will continue soon illustrating the remaining methods of ensemlbing techniques.


# resources :
# 
# 
# [Google](https://www.google.com/webhp?hl=en&sa=X&ved=0ahUKEwiU0c_cgOLhAhUjQxUIHfetDCwQPAgH)
#     
# 
# [Analytics videa](https://www.analyticsvidhya.com/)
#     
# 
# [youtube](https://www.youtube.com/)
#     
# 
# [wikipedia](https://www.wikipedia.org/)
#     
#   
#   
#   and a lot of other resources .
#     thanks a lot.

