# > # NN Approach
# We will discuss how to use a NN approach without overfitting too much. There are several best practices one should keep in mind to avoid overfitting when using this approach:
# * Having as much as data as possible. Here, we will have 200000 data points in the train set.
# * Having as few parameters as possible in the model. It is easy to build a model with millions of parameters, which often leads to overfitting.
# * Using regularization. Basically, this means adding a penalization to having large weights in the layers. This is a similiar concept to lasso/ridge regression. We add the $l^2$, or $l^1$ norm of the weights to our loss function. This will prevent the network from having complex, "large" weights which will be an overfit.
# * Using dropout. In this approach, we randomly set a proportion of the weights of the network to $0$.
# 
# The book "Deep Learning with Python" by François Chollet gives a wonderful introduction to these concept.


import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras import layers
from sklearn.model_selection import train_test_split
from keras import regularizers


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
print(train.shape, test.shape)


# We first split the data into a train/test set and scale it (taking z-scores which is easily done by using sklearn's scale function).


from sklearn.preprocessing import scale

y = train['target']
train = train.drop(['target', 'ID_code'], axis=1)
id_test = test['ID_code']
test = test.drop(['ID_code'], axis=1)

# Scaling the data:
train = scale(train)
test = scale(test)

x_train, x_test, y_train, y_test = train_test_split(train, y, test_size=0.25,
                                                    random_state=42)


# We will have 3 layers in our model. We add $l^2$ regularization, and take relatively small layers.


from keras.models import Sequential
from keras import layers

input_dim = train.shape[1]
print('Input dimension =', input_dim)

model = Sequential()
model.add(layers.Dense(16, input_dim=input_dim, activation='relu', kernel_regularizer=regularizers.l2(0.005)))
model.add(layers.Dense(16, input_dim=input_dim, activation='relu', kernel_regularizer=regularizers.l2(0.005)))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.summary()


history = model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))


# save our prediction
prediction = model.predict(test)
pd.DataFrame({"ID_code":id_test,"target":prediction[:,0]}).to_csv('result_keras.csv',index=False,header=True)

