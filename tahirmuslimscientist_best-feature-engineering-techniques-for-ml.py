# ## Feature Engineering for ML Projects
# 
# The feature is defined as a distinctive attribute or variable, in layman terms, columns in the dataset. Feature engineering is one of the most difficult and time-consuming phases in building the ML project. It is the process of getting data ready for modeling. It is giving us an opportunity to add some human intelligence to the model we are going to develop. In simple terms adding related features will help to increase the efficiency of the model. It results in improved model accuracy for unseen data. Also, I believe domain knowledge is essential in building the model. Broadly feature engineering techniques are organized in 3 categories.


# ### **Feature Extraction**


# In general, the data we work is stored in the simple comma separated files. However, there are various types of data exists in the real world like image data, time series data, geospatial data, etc.
# Let’s look into different types of data and how we extract different features. Please note that these are only some of the most used ways of extracting data as there are numerous techniques for extraction.
# 
# **Categorical Data:**
# 
# Integer encoding and One Hot encoding:
# The data contains categorical data which needs to convert to numeric vectors for ML models. In Integer encoding where each unique category value is assigned an integer value, and one hot encoding techniques are one in which each binary or multiclass features are converted to many features by assigning dummy variables. Dummy variables are a set of binary variables either 0 or 1 that each represent a single class from a categorical feature.
# 
# **Text data:**
# 
# Bag of words:
# As we know that machine learning algorithms cannot work with raw text, so, therefore, we have to convert the text into numbers, i.e., vectors. First, we have to figure out of Set of all the unique words in reviews, and each word considered as a dimension. However, bigram(2 words) pairs of words considered as a dimension, trigrams take three words as a dimension, and N-grams take n dimensions. We can inference that If n>1, dimensions increases. The disadvantage of a bag of words is it completely discards the sequence of information whereas Bigram, Trigram, N-gram retains sequence of information.
# 
# Term frequency and Inverse Document Frequency: 
# It is used for information retrieval(NLP). Term frequency is calculated by no. of times each word occurs in document divided by total no of words in the document. Inverse document frequency has calculated the log of total no. of documents in the dataset divided by total no of documents which contain that particular word. If one particular word is more frequently occurs in the dataset, then IDF will be low whereas if the word is rarely frequent, then IDF will be high.TF and IDF are multiplied together to fill the elements in the array whereas BOW is used the no of times the individual words occurred.
# 
# **Images Data**:
# 
# Scale Invariant:
# The Major goal of image feature extraction is as an image, or a region within an image, generate the features that will subsequently be fed to a classifier in order to classify the image in one of the possible classes. These image processing techniques are being used heavily in researches and automation of industrial processes. We know that features are raw pixels for simple problems like digit recognition. SIFT, Scale Invariant feature transform is one of the feature extraction methods for images. SIFT( scale invariant feature transform) starts by detecting edges and corners in the image. On the resulted image, SIFT tries to find interesting points that are differentiating that image from the others. From each point, it extracts a histogram where each of the bins is a count of distinct edge or corner orientation. These histograms can be concatenated into some smaller number of groups with a clustering method like K-means.
# 
# **Time Series data:**
# 
# Data/Time features:
# A time series data is a data in which we have a sequence of data points typically in time spaced at uniform time intervals. In general, when working with time series data, it is a good idea to have a calendar with public holidays, abnormal weather conditions, and other important events. For example, if we want to predict the no. of Bart trains run daily, we should know the whether it is weekend or weekday and daytime or nighttime which helps to identify our objective, i.e., When we are creating features based on dates, it is essential to know what is our business goal. Moreover, also we have data from geographical sources where it is critical to normalize the data by time zones as we have different time variations.
# 
# **Fourier Decomposition:**
# One of the simplified methods to represent time series data is Fourier Decomposition/ Transform. Time series data is generally used in statistics, whether forecasting earthquake predictions etc. Fourier decomposition represents time series in the frequency domain base as Fourier series represents time series in the form of the sine waves. The Fourier decomposition explains the time series entirely as a composition of sinusoidal functions.
# 
# **Moving windows:**
# For example, consider we want to predict the customer enters the store during the weekend. We have to define the window width based on the domain experience and identify the features in the window for predicting whether the customer enters the store or not in next 1 hour. Here consider f1, f2 are features where y is the label to identify whether customers enters or does not enter the store


# ### **Feature Transformations**


# Feature transformation is the name given to replace the original features with functions of these respective features. For example, changing the scale of one variable into zero to 0ne.
# 
# **Normalization:**
# 
# Normalization is one of the data transformation technique where attribute data is scaled to fall within a small range and make all the attributes equal. It allows us to understand how the data is assigned across and how manageable regarding scale. For example, k-nearest neighbors (KNN) algorithm with a Euclidean distance measure; it examines the distances between different data points which may lead to having large absolute differences due to different scales.
# 
# Feature normalization used to transform or compress the data in the range between 0 and 1. Here we are making the scale uniform and compressed the data in the hypercube. Normalization refers to the rescaling of the features to a range of [0, 1], which is one of the particular cases of min-max scaling.
# 
# Feature standardization used a lot in practice which transforms or compress the data such that their mean is 0 and the standard deviation is 1. Here we are compressing or expanding the data points in the hypercube to make our standard deviation for any feature is 1. so that the feature columns take the form of a normal distribution, which makes it easier to learn the weights. Also, standardization maintains useful information about outliers and makes the algorithm less sensitive to them in contrast to min-max scaling, which scales the data to a limited range of values.
# 
# **Principle Component Analysis:**
# 
# The principal components of a data set provide a projection onto a rotation of the data space that give ’directions of maximum variance,’ i.e., it identifies the components based on capturing or explaining the maximum variance.
# We can find the principle components for data by finding the eigenvectors of a scaled and centered covariance matrix. The eigenvalues give us a measure of the amount of variance explained by each principal component, and so are monotonically decreasing.
# 
# **Interaction Features:**
# 
# Consider two features f1 and f2 and join them using one hot encoding to create a new feature or multiply each pairwise columns generated by the f1 and f2 encoding. The interaction between the two features is the change in the prediction that occurs by varying the features. These features can applied to categorical and numerical features. For numeric features, we can use different operations like addition, multiplication, etc. Then we can use feature importance to select the essential features.
# 
# **Linear discriminate analysis(LDA):**
# 
# As we know that Logistic regression is used for two-class classification, and if we have more than two classes then Linear Discriminant Analysis is the preferred linear classification technique. It is commonly used as dimensionality reduction technique in the pre-processing step for classifying the patterns and other machine learning applications. It helps to project the data points onto a lower-dimensional space with class label separation in order to reduce the degree of overfitting and also reduce computational costs. In comparison with PCA, LDA is very similar to PCA, the goal in LDA is to find the feature subspace that optimizes class separability, whereas PCA attempts to find the orthogonal component axes of maximum variance in a dataset; the former is an unsupervised algorithm, whereas the latter is supervised. We can perform LDA in python by using a built-in package in sklearn library.
# 
# **Feature Selection**
# It is a process of selecting the best features which improve the accuracy of the model. Even un selection is essential to get rid of unnecessary features due to computational complexity. Features are removed which have low variance, and algorithms like random forest will help us to select the essential features. Also based on the domain knowledge we can select features.
# 
# **Random Forest Algorithm:**
# 
# Random forests are one of the most popular machine learning methods because of their relatively good accuracy, robustness, and ease of use. Mean decrease impurity is defined when training a tree, it can be computed how much each feature decreases the weighted impurity in a tree. The impurity decrease from each feature can be averaged, and the features are ranked as per the average measure. There is a direct attribute called feature importance in the ski kit-learn library to find the features. Also, another method is to directly measure the impact of each feature on the accuracy of the model. The core principle is to permute the values of each feature and measure how much the permutation decreases the accuracy of the model.
# 
# **Sequential Backward Selection (SBS):**
# 
# A classic sequential feature selection algorithm is Sequential Backward Selection (SBS), which aims to reduce the dimensionality of the initial feature subspace with a minimum delay in performance of the classifier to improve upon computational efficiency.
# 
# SBS sequentially removes features from the full feature subset until the new feature subspace contains the desired number of features. In order to determine which feature is to be removed at each stage, we need to define criterion function (say X) that we want to minimize. The criterion calculated by the criterion function can only be the difference in the performance of the classifier after and before the removal of a particular feature. Then the feature to be removed at each stage can merely be defined as the feature that maximizes this criterion, i.e., at each stage, we eliminate the feature that causes the least performance loss after removal.
# 
# **References:**
# https://medium.com/open-machine-learning-course/open-machine-learning-course-topic-6-feature-engineering-and-feature-selection-8b94f870706a
# https://towardsdatascience.com/understanding-feature-engineering-part-1-continuous-numeric-data-da4e47099a7b
# https://www.quora.com/What-is-the-difference-between-bag-of-words-TF-IDF-and-vector-space-model
# https://machinelearningmastery.com/gentle-introduction-bag-words-model/
# https://machinelearningmastery.com/an-introduction-to-feature-selection/
# https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/
# https://www.uio.no/studier/emner/matnat/its/UNIK4690/v16/forelesninger/lecture_9_2_featureextraction.pdf
# https://www.linkedin.com/pulse/feature-engineering-dates-python-summary-ashish-bansal
# http://courses.edsa-project.eu/pluginfile.php/1332/mod_resource/content/0/Module%205%20-%20Feature%20transformation_V1.pdf
# https://www.igi-global.com/dictionary/feature-selection/26525



