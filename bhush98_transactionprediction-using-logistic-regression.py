# A Small example for people that are new to machine learning and don't exactly know the steps so in this
# problem i'm going to example my each step in a simple way.
# So what's the problem here so the problem is given a new data we have to predict whether a transaction is made 
# or not .So we categorize this into a classification problem because we have the inputs and we have to classify whether a transaction is made or not .
# We use Logistic regression here because the classification problem has only two classes so no need to make it complicated .


#We get the libraries that we will need 
import numpy as np                #for matrix operation
import pandas as pd               #for reading data


#Here we get the data by using the pandas inbuil function read_csv because the file is in csv format
training_data = pd.read_csv("../input/train.csv")
#The head function return first five instance of the data
training_data.head()


#The keys function return the column names 

training_data.keys()


#Using the pandas inbuilt describe function to describe our data

training_data.describe()


#using the pandas inbuilt info function to display the information of our data

training_data.info()


#Getting our input data that we will feed to the network into training_data_X

training_data_X = training_data.iloc[:,2:].values
print(training_data_X[0:5,:])


#Getting our output data into training_data_Y

training_data_Y = training_data.iloc[:,1].values
print(training_data_Y[0:5])


#Getting our classifier from sklearn library , we our using Logistic regression classifier

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(solver='lbfgs')


#Providing our input and output data to the classifier

classifier.fit(training_data_X,training_data_Y)


#Getting the accuracy score from the classifier using the inbuilt score method

classifier.score(training_data_X,training_data_Y)


#Reading the testing data using pandas inbuilt function read_csv

testing_data = pd.read_csv("../input/test.csv")
testing_data.head()


#dropping the columns that are not needed

testing_data = testing_data.drop(columns="ID_code")
testing_data.head()


#Storing the predicted results in results variable 

results = classifier.predict(testing_data.iloc[0:10,:])

#Printing the predicted results

print(results)

