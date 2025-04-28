# 
# Title: SANTANDER CUSTOMER TRANSACTION PREDICTION
# 
# Mikel Kengni / March 2019
# 
# Plan:
# 
# **: )                                     First things First: snacks and chip checked, Coffee checked                                         : )**
# 
# * **Introduction**
# * Competition overview
# * Kernel 1: Importance of Data Balancing
# 
# * Outlines/Progression
# 
#     1- Import the libraries
# 
#     2- Run the data
# 
#     3- Data  exploration and Feature Engineering
#     
#     4- How skewed is our dataset:
#     
#     5- Correlation between all features na dthe target features
#     
#     6- Modelling
#      * Part_1: Using a classifier with the **Class_weight = Balanced** parameter
#      * Part_2: Using the **SMOTE** oversampling technique
#    
#   7- Metric Traps
#   
#   8- Observation
#   
#   9- Conclusion
#   
#   10- Kernels and Materials used


# # Introduction:
# 
# **Competition Overview **:
# 
# In this challenge, we help identify which customers will make a specific transaction in the future, irrespective of the amount of money transacted.
# 
# We are provided an anonymized dataset with each row containing 200 numerical values identified just with a number, the binary target column, and a string ID_code column. 
# 
# The task is to predict the value of target column in the test set.


# **Kernel 1: Importance of Data Balancing**
# 
# This kernel is going to be about the importance of data balancing and the effects of unbalance datasets(uneven classes in this case just 2 classes) on our results. 
# 
# * We will be comparing 2 different ways of data balancing - 
# 
# 1- Using the 'Balanced" parameter in the Class_weight feature 
# 
# 2- Using the SMOTE oversampling technique: 
# 
# * Under the SMOTE Method will determine the right and wrong ways to oversample using SMOTE.
#     
#     * We will do it in 2 ways. 
#     
#         * We will apply SMOTE on the whole predictors features and outcome, then split them into train and validation set, then fit into a model.
#         
#         * We will also do it the other way.  Ie. We will split the datasets into train and validation sets, then we will apply  SMOTE on the X_train and y_train splits.
#             
# 3- Finally, we will find out which of the 2 techniques above score better with this problem and why.


# # 1- Import the Libraries


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
import seaborn as sns
import matplotlib.pyplot as plt

# Any results you write to the current directory are saved as output.


# # 2- Read Train and Test Datasets


data = pd.read_csv("../input/train.csv")
data.head()


test = pd.read_csv("../input/test.csv")
test.head()




# # 3- Data Exploration and Feature Engineering


%%time
print("**"*45)
print('The size of our train and test datasets are: \nTrain : {}\nTest: {}'.format(data.shape, test.shape))
print("**"*45)


# **OBSERVATION:**
# Data contains:
# 
# * ID_code (string)
# * Target
# * 200 numerical variables, var_0 to var_199
# * SHAPE = 200000 ROWS AND 202 COLUMNS
# 
# Test contains:
# 
# * ID_code (string);
# * 200 numerical variables, var_0 to var_199
# * SHAPE = 200000 ROWS AND 201 COLUMNS


data.describe()


data.dtypes.value_counts()


test.dtypes.value_counts()


# We check for any missing or nan values in the data and test sets
null_features = data[data.columns[data.isnull().any()]].sum().sort_values()
missing_train = pd.DataFrame({'Null' : null_features})
missing_train


nulltest_features = test[test.columns[test.isnull().any()]].sum().sort_values()
missing_test = pd.DataFrame({'Null' : nulltest_features})
missing_test


#  :) :) :) Train and test sets have not missing values. GREEEEEEEEEAAAAATTTTT!!!!! :) :) :)


# Lets take a look at the dictribution of the target variables in both the train and test sets.


data['target'].value_counts(normalize = True)


sns.set(style = 'darkgrid')
ax = sns.countplot(x = 'target', data = data)


# Our Target set is very imbalanced. About 90 percent of our target column is 0 while the remianing 10 percent are 1s. This si called Class Imbalanced. It occurs each class does not make up an equal portion of your data-set and It is important to properly adjust your metrics and methods to adjust for your goals. If this is not done, you may end up optimizing for a meaningless metric and hence getting a flawed outcome.


# # 4- How skewed is our a datasets?


# * Why do we need to check for skewness you ask?
# 
# It’s often desirable to transform skewed data and to convert it into values between 0 and 1 because usually, different features in a datasets have values in diffeernet range. In order to have a reliable predictive model, it is important to bring all these features in the same range.
# 
# 
# Let's take a look at skewness in our dataset:


#we check for skewness in  data

skew_limit = 0.75
skew_vals = data.skew()

skew_cols = (skew_vals
             .sort_values(ascending=False)
             .to_frame()
             .rename(columns={0:'Skewness'})
            .query('abs(Skewness) > {0}'.format(skew_limit)))

skew_cols


# # 5- Correlation with the target feature
# https://towardsdatascience.com/data-correlation-can-make-or-break-your-machine-learning-project-82ee11039cc9
# 
# Data correlation is the way in which one set of data may correspond to another set. It is important to determine hoe correlated your features are, as this knowledge may be useful in choosing the right algorithm but also, If you try to train a model on a set of features with no or very little correlation, you will get inaccurate results. 
# 
# Lets se how all the features correlate with the target feature in the train set.


# Correlation between the features and the predictor- SalePrice
predictor = data['target']
features = [x for x in data.columns if x != 'target']
correlations = data[features].corrwith(predictor)
correlations = correlations.sort_values(ascending = False)

# correlations
corrs = (correlations
            .to_frame()
            .reset_index()
            .rename(columns={'level_0':'feature1',
                                0:'Correlations'}))

corrs.head()


# Get the absolute values for sorting
corrs['Abs_correlation'] = corrs.Correlations.abs()
corrs.head()


# Most correlated features wrt the abs_correlations
corrs.sort_values('Correlations', ascending = False).query('Abs_correlation>0.45')


# WAOUWWW!!! Looks like there is not a lot of correlation between the feayures and the predcitor. I will use the random forest classifier later to get the most important features in the future if need arises. Okay!!! lets move on.


# # 6- Preparing for Modelling
#    We are going to do the modelling in 2 part.
#    - Modelling part 1- Metric trap
#    - MOdelling part 2


#  **Modelling part 1:** 
#    
#    - We split the datasets in train and validation sets
#    - We discuss the importance of picking the right metric and why accuracy_score is not the best metric to choose when we have a class imbalance.
#    - Then weuse a simple algorithm in this case a Random Forest Classifier, train our unbalanec dataset on it , calculate the score with the accuarcy_score and then with the roc_auc_score.
#    - In order to proof that the accuracy_score is not the right metric, we will do a small test, pick just on feature and train it, then we calculate the accuracy_score and the roc_auc_score on it.


#Importing my classifiers and scoring metrics
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split


# **Why do we need To balance out dataset:**
# 
# * As we can see from above,the data set is severely imbalanced (90 : 10).
# * The main motivation behind the need to preprocess imbalanced data before we feed them into a classifier is that typically classifiers are more sensitive to detecting the majority class and less sensitive to the minority class.
# * Usually, data imbalance will lead to the classification output being biased, in many cases resulting in always predicting the majority class like we will see in  below.


# We split the data into train and validation set
# We are not going to need the id_code column nor the target column  in X
X = data.drop(columns = ['target', 'ID_code'])
y = data['target']

X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                  test_size = 0.3,
                                                  random_state=42)
print('Training set shape:')
print('X_train.shape:{}\t y_train.shape: {}'.format(X_train.shape, y_train.shape))
print('\nValidation set shape:')
print('X_val.shape:{} \t y_val.shape: {}'.format(X_val.shape, y_val.shape))


# # 8- Metric Trap:
# 
# One of the major issues begginers usually fall into when dealing with unbalanced datasets is the choice of their evalution metrics.  Using simpler metrics like accuracy_score my not always be the correct. 
# 
# In a dataset with highly unbalanced classes, if the classifier always "predicts" the most common class without performing any analysis of the features, it will still have a high accuracy rate. ie, whatever the circumstance, the accuarcy_score will most likely always be the percentage of the majority classe. If you don't get it yet, Hnag on, it will be clearer with examples below..


# But for this little experiement, i will be using the random forest classifier.and for the metric evaluation i will be using the accuarcy_score and the roc_auc_score.


random_forest = RandomForestClassifier(n_estimators=100, random_state=42, verbose=1,
                                      class_weight= None, max_features = 0.5, 
                                       min_samples_leaf = 100)


#Train the model
random_forest.fit(X_train, y_train)


pred_rf = random_forest.predict(X_val)
print ('accuracy_score for Random_forest with unbalanced classses')

accuracy = accuracy_score(y_val, pred_rf)
print("Accuracy: {}".format (accuracy))


#Getting the accuracy_score and ta roc_score for the random_forest
print ('\nroc_auc_score for Random_forest with unbalanced classes')
pred = random_forest.predict_proba(X_val)[:,1]
roc_auc = roc_auc_score(y_val, pred)
print(roc_auc)


# The accuracy_score for this part is 0.8976 while the roc_auc_score is 0.7921. 
# **Do you remeber the percentage of classe distribution in our dataset?** Here we go! look at the percentage dictribution for classe with 0. It is almost the same as the accuarcy_score.


data['target'].value_counts(normalize = True)


#submission 1


test_all = test.drop('ID_code', axis = 1)
test_all.head()


predict_1 = random_forest.predict(test_all)
#random_forest.predict_proba(X_val)[:,1]


solution_1 = pd.DataFrame({'ID_code': test['ID_code'], "target" : predict_1})

# #creating csv file

#solution_1.to_csv("santander_1.csv", index = False)
solution_1.head()


#only predicts the majority class 0 
np.count_nonzero(solution_1['target']==0)


# Lets do something else to confirm what we are already suspecting. Now let's run the same code, but using only one feature. Normally, the accuracy score should be very small given that we are only using one feature. 


rand_f1 = random_forest.fit(X_train[['var_5']], y_train)


accuracy_one = accuracy_score(y_val,  rand_f1.predict(X_val[['var_5']]))
print("Accuracy_one: {}".format (accuracy_one))

roc_predict = rand_f1.predict_proba(X_val[['var_5']])
roc_auc_score_one = roc_auc_score(y_val, (roc_predict)[:,1])

print("roc_one: {}".format (roc_auc_score_one))


# As we can see, The accuracy_score whicch under normal circumstances should be really low, is atill stuck at 0.8976% which is not correct. This goes to show how important the choide of evalution metric especially when dealing with unbalanced datasets. The other metric we used for this(roc_auc_score) is behaving like it should, ie for a single feature, its score actually dropped to 0.51. 


# **Modelling part 2:**
#  
#    - We train the data the same model used in modelling part1 but htis time around we balance the classes before feeding it into an algorithm for training.
#    - then we calculate the score using the same algorithm we used above.
#    - Then map out the importance of always using unbiased datasets(datasets with one classes a lot more present that the other.
#    


# **Modelling part 2: Balancing the classes using the class_weights parameters**
# 
# The paramenter **'class_weight = balanced**' will give equal weights to both classes  irrespective of their reperesentation in the training datase. 


rf = RandomForestClassifier(n_estimators=100, random_state=42, verbose=1,
                                      class_weight= 'balanced', max_features = 0.5, 
                                       min_samples_leaf = 100)


rf.fit(X_train, y_train)


predict_balanced = rf.predict(X_val)
print ('accuracy_score for Random_forest with balanced classes')
print (accuracy_score(y_val, rf.predict(X_val)))

print ('\nroc_auc_score for Random_forest with class_weight = balanced classes')

roc_auc = roc_auc_score(y_val, rf.predict_proba(X_val)[:,1])
print(roc_auc)


# After balancing the classes by balancing their weights, we can see that the accuracy has dropped a little.


#Submission 2 (scores = 0.788 on public leaderboard ~79%)
prediction_2 = rf.predict_proba(test_all)[:,1]
solution_2 = pd.DataFrame({'ID_code': test['ID_code'], "target" : prediction_2 })

#creating csv file

#solution_2.to_csv("santander_2.csv", index = False)
solution_2.shape
  


# **Imbalanced data put accuracy out of business** as we proved above. It is usually not enought to rely on hight accuracy_score to evalute your model because the score may just be illusionary and a simple reflexion of the majority class. Using other evalution metric like the roc_auc_score, f1_score, classification report etc could give us a better evalution of the performance our our model w.r.t the dataset. But it is always a good idea and safer to work with balanced datasets and balancing a dataset can be as easy as just adjusting the class_weight parameters.
# 
# For algorithms with the Class_weight parameter, it sometimes suffices to set set **class_weight = 'Balanced'** like in this case the random_forest classifier.
# 
# With some other algorithms, we may need to set the class weight parameter manually.  We set the class_weight such as to penalize mistakes on the minority class by an amount proportional to how under-represented it is. For example 
# 
# > class_weight = ({0 : "0.25", 1:  "0.85"}).
# 
# Another alternative to using the class_weight parameter is to creat synthetic observations of the minority class using the **SMOTE = Synthetic Minority Oversampling Technique** from the sklearn.imblearn library.


# # - Data Balancing using 'SMOTE'


# **Balancing classes using SMOTE before spltting dataset into train and validaton sets


# **SMOTE Algorithm (Synthetic Minority Oversampling Technique)**
# 
# We will be using the SMOTE algorithm (Synthetic Minority Oversampling Technique) to over-sample our dataset. It is a powerful sampling method that goes beyonds simply increasing or decreasing the number of datas in a dataset. How it works is by.
# 
# 1- Finding the k-nearest-neighbors for minority class observations (finding similar observations)
# 
# 2- Randomly choosing one of the k-nearest-neighbors and using it to create a similar, but randomly tweaked, new observation.


#Using SMOTE for class imbalance in target
from imblearn.over_sampling import SMOTE
from collections import Counter
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X, y)
print('Resampled dataset shape %s' % Counter(y_resampled))


# Now we split the resmpled train set into train and validation sets
x_train_res, x_val_res, y_train_res, y_val_res = train_test_split(X_resampled,
                                                    y_resampled,
                                                    test_size = 0.3,
                                                    random_state=2019)


#Train the model with the same random forest algorithm
rf_1 = random_forest.fit(x_train_res, y_train_res)


#Getting the f1_score and ta classifiaton report for the random_forest

# target_names = ['class 0', 'class 1']

pred_1 = rf_1.predict(x_val_res)
print ('accuracy_score for Random_forest with "smote" oversampling before splitting')
accuracy_smote_1 = accuracy_score(y_val_res, pred_1)
print (accuracy_smote_1)

print ('\nroc_auc_score for Random_forest with "smote" oversampling before splitting')
roc_auc_smote_1 = roc_auc_score(y_val_res, rf_1.predict_proba(x_val_res)[:,1])
print(roc_auc_smote_1) 

# print ('\nClassification report for Random_forest with "smote" oversampling before splitting')
# print(classification_report(y_val_res,pred_1, target_names=target_names))


#scores 0.662 on public leaderboard
prediction_3 = rf_1.predict_proba(test_all)[:,1]


solution_3 = pd.DataFrame({'ID_code': test['ID_code'], "target" : prediction_3 })
#solution_3 = pd.DataFrame({'ID_code': test, "target" : pred_1 })

#creating csv file

#solution_3.to_csv("santander_3.csv", index = False)
#solution_3.sample(10)




# #lets get the accuracy and roc_auc_score for the test data too.
print('The accuracy_score for the test data is:')
accuracy_test_1 = accuracy_score(data['target'], rf_1.predict(test_all))
print(accuracy_test_1)

print('The roc_auc_score for the test data is:')                               
roc_auc_test_1 = roc_auc_score(data['target'], prediction_3)
print(roc_auc_test_1) 

#solution_3 = pd.DataFrame({'ID_code': test['ID_code'], "target" : prediction_3 })
#solution_3 = pd.DataFrame({'ID_code': test, "target" : pred_1 })


  




# # 'SMOTE' on X_train and y_train only
# 
# **Let do a split before we apply smote on x_train amd y_train **
# 


#train the model with the same random forest algorithm
x_tres, y_tres = sm.fit_sample(X_train, y_train)


#fit the random forest classifier on the split train sets
rf_2 = random_forest.fit(x_tres, y_tres)


#Getting the f1_score and ta classifiaton report for the random_forest

# target_names = ['class 0', 'class 1']
pred_2 = rf_2.predict(X_val)
print ('F1_socre for Random_forest with "smote" oversampling after splitting')
accuracy_smote_2 = accuracy_score(y_val, pred_2)
print (accuracy_smote_2)

print ('\nroc_auc_score for Random_forest with "smote" oversampling after splitting')
roc_auc_smote_2 = roc_auc_score(y_val, rf_2.predict_proba(X_val)[:,1])
print(roc_auc_smote_2)  

# print ('\nClassification report for Random_forest with "smote" oversampling after splitting')
# print(classification_report(y_val, pred_2, target_names=target_names))


# # Submission Dataframe


#score 0.670 on public leaderboard
prediction_4 = rf_2.predict_proba(test_all)[:,1]


#Creating a submission file

solution_4 = pd.DataFrame({'ID_code': test['ID_code'], "target" : prediction_4 })
#solution_4 = pd.DataFrame({'ID_code': test, "target" : pred_2 })

# #creating csv file

# solution_4.to_csv("santander_4.csv", index = False)
# solution_4.sample(10)



print('The accuracy_score for the test data resampled after splitting is:')
accuracy_test_2 = accuracy_score(data['target'], rf_2.predict(test_all))
print(accuracy_test_2)

print('The roc_auc_score for the test data resampled after splitting:')                               
roc_auc_test_2 = roc_auc_score(data['target'], prediction_4)
print(roc_auc_test_2) 

  


#putting it all together

# REsults with Smote on data befoe splitting
print ('Validation Results for smote before splitting')
print ('accuracy_score: {} \nroc_auc_score : {}'.format(accuracy_smote_1, roc_auc_smote_1))

print ('\nTest Results for smote before splitting')
print ('accuracy_score: {} \nroc_auc_score : {}'.format(accuracy_test_1, roc_auc_test_1))


#results with smote on x_train and y_train only

print ('Validation Results for smote after splitting i.e on x_train and y_train')
print ('accuracy_score: {} \nroc_auc_score : {}'.format(accuracy_smote_2, roc_auc_smote_2))

print ('\nTest Results for smote after splitting i.e on x_train and y_train')
print ('accuracy_score: {} \nroc_auc_score : {}'.format(accuracy_test_2, roc_auc_test_2))


# We porved earlier that a very high accuracy can sometimes not be a relexion of how the model actually performs. So i will not consider the results from the accuracy in this section. The main purpose of using accuracy_score was to show its flaws especially when dealing with unbalanced datasets. 
# 
# Lets focus on the roc_auc_score for both scenarios( applying SMOTE before splitting and applying SMOTE after splitting.).


print ('Roc_auc_score for scenario 1: applying Smote before splitting:')
print ('roc_auc_score for validation set: {} \nroc_auc_score for test : {}'.format(roc_auc_smote_1, roc_auc_test_1))
print("\n")
print ('Roc_auc_score for scenario 2: applying Smote after splitting(on X_train and y_train):')
print ('roc_auc_score for validation set: {} \nroc_auc_score for test : {}'.format(roc_auc_smote_2, roc_auc_test_2))


# # Observation:
# **Scenario 1: Resamplling then splitting into train - validation sets**
# 
# In this scenario, we oversampled the whole datasets then we split it into train and validation set. From our roc_auc_score evaluation, we can see that we have a 91.0% score on the validation set but a 49.9% score on the test data. That is a huge gap between the scsores and this is so because some information **"bleed"** from the validation set into the training set ofthe model.  
# 
# By oversampling before splitting the dataset into train and validation sets, we ended up with some of the information from  the validation set being used to create some of the synthetic observations in the training set. As a result, the model has already "seen" some of the datait is predicting in the and as such, is able to perfectly predict these data during validation hence increasing the roc_auc_score of the validation set. Hecnce the big gap between the score for the validation set an that of the test set.( 91.0% versus 49.9%)


# **Scenario 2: Splitting the datasets into train and validation sets the resampling the X_train and y_train data**
# 
# In this scenario, we split the dataset into train and validation sets and then resampled the trained data. Here, the validation set is untouched so the result from this scenario is more generalizable. 
# 
# As we see from the roc_auc_scores, the score for the validation set( 66.9 %) is very close to the score from the test set(50.0%). 
# 
# Scenario 2  is the **right way to oversample data.** while scenario 1 is the **wrong way to oversample data.**
# 


# # Conclusion: 
# It is always advisable to **split your datatset into train and validation before oversampling**(Scenario 2). And only **apply your oversampling method on the training sets**. The validation set should be pristine.


# Thanks for reading to the end.


# 
# If you found this kernel helpful, i would really appreciate an upvote. If you did not, please comment below with your suggestion or recommendations and lets make it better together. 
# 


# References used for Data Balancing/ Resampling:
# 
# - https://beckernick.github.io/oversampling-modeling/
# - https://elitedatascience.com/imbalanced-classes
# - https://towardsdatascience.com/handling-imbalanced-datasets-in-deep-learning-f48407a0e758a
# - https://towardsdatascience.com/deep-learning-unbalanced-training-data-solve-it-like-this-6c528e9efea6
# - https://towardsdatascience.com/dealing-with-imbalanced-classes-in-machine-learning-d43d6fa19d2
# - https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/

