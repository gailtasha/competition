# ![Imgur](https://i.imgur.com/ZzUxW4t.png)


# **Intuition**


# 
# **Any modification to a learning algorithm to reduce its generalization error but not its training error
# **
# 


# I am here not to review neither mathematical equations nor python codes that run behind regularization, But instead I want to open a window to intuitions from the main equations those controlling shrinkage in order to recognize at a glance what is supposed to do for the model we make and what is not required to be done.
# 


# In mathematics, statistics, and computer science, particularly in machine learning regularization is the process of adding information in order to prevent overfitting.
# 


# there are many methods to overcome overfitting. the one we are about to talk here is regularization or shrinkage.
# 


# when you notices that your model doesn't generalized well on new data by the low accuracy level or AUC or ...etc . Lets imagine that we ask our model this question Oo do you want really to overfit and not generalize well on new data ? Because you want so , I will penalize you by adding a term that makes your coefficients tend to zero (you may reach it or not ),
# 


# ** Cost function = Loss + Regularization term
# **
# 


# By adding regularization term to the equation , we provide a way to preventing you from overfitting by controlling the values of coefficients that are present in the model . Regularization works on assumption that smaller weights generate simpler model and thus helps avoid overfitting.
# 


# Minimization is considered a tug-of-war between the two terms the regularization parameter and the original cost function and no one will beat the other completely . Due to the addition of this regularization term, the values of weight matrices decrease because it assumes that a neural network with smaller weight matrices leads to simpler models. Therefore, it will also reduce overfitting to quite an extent.


# ![Imgur](https://i.imgur.com/a2vFIj8.png)


# as we can see our regularization term is a **norm** which is the penalty term that we use to shrink our coefficients from reaching high values . J is our cost function , L is our error and the 3rd parameter is the complexity parameter. now lets revise on what norm is


# **Norm:
# **


# **norm** is a way to define the magnitude of the vector. The general notation of the norm is like this
# 


# ![Imgur](https://i.imgur.com/WogQMMW.png)


# when p = 1 it is norm-1 and when p=2 it is norm-2 . This is the general equation of forming norms where we can change the value of p and upon changing it will define the norm degree by 1,2,3,…...to infinity
# 


# for example lets consider that you have a vector like this
# 


# ![Imgur](https://i.imgur.com/hzUh2A9.png)


# so for norm-1 the formula will be like this:
# 


# ![Imgur](https://i.imgur.com/2DLfnQc.png)


# and for norm-2 the formula will be like this:
# 


# ![Imgur](https://i.imgur.com/EwWz5X9.png)


# **now lets try to visualize the norms geometrically :
# **
# 
# lets assume that we have L1,L2,L infinity norms . and we have 2 points
# 
# x1=[-1,1]
# 
# x2=[-1,1]
# 
# Q1: visualize norm-L1 for them . define the values of x1 and x2 where all norm-L1 equal 1
# 
# Answer : by trying random values of x1 , we get x2 and then plot the curve between both points .
# 


# ![Imgur](https://i.imgur.com/3ufKmco.png)


# And take a lot of random values varing from -1 to 1 for both x1 and x2 . After finishing , we get a curve like this
# 


# ![Imgur](https://i.imgur.com/d926q1w.png)


# a diamond shape for all possible values in the four quarters. notice the intercept of the lines will be at 1 step(which is the magnitude of the norm-1) with the four coordinates.this gives an intuition that we control the values of x and y axis like restricting them in a specific area which will be after that the thetas of our model that we want control their values . by changing the magnitude of the norm increasing or decreasing , widening or shrinkage of the norm shape is visualized.
# 
# Now lets see what is the curve for L2


# 
# Q2: visualize norm-L2 for them . define the values of x1 and x2 where all norm-L2 equal 1
# 
# Answer : we make the same we did in norm-1


# ![Imgur](https://i.imgur.com/eQKlnbE.png)


# by using a lot of values like in the equation above we can draw our norm-2
# 
# and the resultant will be a ball . it is nothing but a circular shape as it is a result of the euclidean rule or frobenius rule


# ![Imgur](https://i.imgur.com/1BVEdU2.png)


# and this is the one for infinity norm
# 


# ![Imgur](https://i.imgur.com/wIaeKgu.png)


# Swiming into the model :
# 
# Consider a model consisting of the weights (w1,w2,…,wn).
# 
# With L1 regularization, p = 1 and you penalize the model by a loss function (regularization term)


# ![Imgur](https://i.imgur.com/WNXFSEA.png)


# and for L2 is the the part added to the equation
# 


# 
# ![Imgur](https://i.imgur.com/qjgFaY6.png)


# we have now a new cost function.
# 
# Cost function = Loss + Regularization term
# 
# and need to see what gradient descent will do with our new equation


# ![Imgur](https://i.imgur.com/E8GbEK6.png)


# we know that the red circles is the space where gradient descent tend to reach our optimum point . at this point we assume that there will be overfitting , but as we added the regualrization term their will be a problem for the thetas to reach the B^ as they are constrained by a nother space which is the norm space.


# 
# by using gradient descent, we will iteratively make the weights change in the opposite direction of the gradient with a step size η multiplied with the gradient. This means that a more steep gradient will make us take a larger step, while a more flat gradient will make us take a smaller step. And thats why it is a war or a competition.
# 


# ** Let us look at the gradients (subgradient in case of L1):**


# ![Imgur](https://i.imgur.com/YaUu9F6.png)


# where :
# 


# ![Imgur](https://i.imgur.com/uafzynB.png)


# **dL2(w)/dw** can be read as the change of **L2(w)** per change in weight. Since the L2-regularization squares the weights, **L2(w)** will change much more for the same change of weights when we have higher weights. 
# 
# For **L1** however, the change of** L1(w)** per change of weights are the same regardless of what your weights are - this leads to a linear function.
# 
# If we try to plot the loss function and it's derivative for a model consisting of just a single parameter, it looks like this for L1:


# ![Imgur](https://i.imgur.com/f0Q4roP.png)


# Notice that for L1, the gradient is either 1 or -1, except for when w1=0. That means that L1-regularization will move any weight towards 0 with the same step size, regardless the weight's value.
# 
# **and for L2**


# ![Imgur](https://i.imgur.com/Nr4aCeK.png)


# you can see that the L2 gradient is linearly decreasing towards 0 as the weight goes towards 0. Therefore, L2-regularization will also move any weight towards 0, but it will take smaller and smaller steps as a weight approaches 0.
# 


# 
# Try to imagine that you start with a model with w1=5 and using η=1/2. In the following picture, you can see how gradient descent using L1-regularization makes 10 of the updates 


# ![Imgur](https://i.imgur.com/EEvegrt.png)


# ![Imgur](https://i.imgur.com/8t9rVXI.png)


#  with L2-regularization where η=1/2, the gradient is w1, causing every step to be only halfway towards 0. That is, we make the update
# 
# 


# ![Imgur](https://i.imgur.com/bOCO7c4.png)


# Therefore, the model never reaches a weight of 0, regardless of how many steps we take
# 


# ![Imgur](https://i.imgur.com/hWOVlC0.png)


# Note that L2-regularization can make a weight reach zero if the step size η is so high that it reaches zero in a single step. Even if L2-regularization on its own over or undershoots 0, it can still reach a weight of 0 when used together with an objective function that tries to minimize the error of the model with respect to the weights. 


# In that case, finding the best weights of the model is a trade-off between regularizing (having small weights) and minimizing loss (fitting the training data), and the result of that trade-off can be that the best value for some weights are 0.


# ![Imgur](https://i.imgur.com/QvY4mLr.png)


# Regularization can be used in all methods, including both regression and classification. there are not too much difference between regression and classification: the only difference is the loss function. so wherever you noticed loss of classification or regression , the concept is the same. in the previuos pic wh defined the equation of cost function after adding the reg parameter with lampda which is ithe meta parameter we must control.
# 


# **Choice of Regularization Parameter
# **
# 
# Properties of L1 and L2 regularizations
# according to the your goal from the model . some of the points can be considered as advantages but in another model as disadvantages so it is dependent on the case of your model.
# 


# perform cross-validation and select the value of λ that minimizes the cross-validated sum of squared residuals (or some other measure). This is focused on its predictive performance by python it is very easy.


# **L1-regularization properties :
# **
# 
# * Computationally expensive.
# * L1 models can shrink some parameters exactly to zero (sparsity).
# * embedded models : they performs implicit variable selection. this means some variables values will be zeros as if you remove them .
# * one of the correlated predictors has a larger coefficient, while the rest are (nearly) zeroed.
# * Since it provides sparse solutions, it is generally the model of choice for modelling cases where the #features are in millions or more. In such a case, getting a sparse solution is of great computational advantage as the features with zero coefficients can simply be ignored.
# * It arbitrarily selects any one feature among the highly correlated ones and reduced the coefficients of the rest to zero. Also, the chosen variable changes randomly with change in model parameters. generally doesn’t work that well as compared to ridge regression.
# * L1 norm function is more resistant to outliers.


# **L2- regularization properties:
# **


# 
# * Mathematically simple computations :as the derivations of the L2 norm are easily computed. Therefore it is also easy to use gradient based learning methods.
# * L2 norms can shrink parameters close to zero.
# * As a result of shrinking coefficients to exactly zero As a result, it cannot perform variable selection.
# * coefficients of correlated predictors are similar.
# * Since it includes all the features, it is not very useful in case of high # features, say in millions, as it will pose computational challenges.
# * It generally works well even in presence of highly correlated features as it will include all of them in the model but the coefficients will be distributed among them depending on the correlation.
# * very susceptible to outliers, much like the OLS estimator. The reason for that is that we still depend on the least squares minimization technique and this does not allow large residuals. Hence the regression line, plane or hyperplane will be drawn towards the outliers.
# * While L1 regularization can give you a sparse coefficient vector,the non-sparseness of L2 can improve your prediction performance (since you leverage more features instead of simply ignoring them.


# **you can see some of the properties may be considered as advantages and in some cases as disadvantages . To mix between both properties of L1 and L2 we use Elastic Net which will be in the second episode .**

