import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import  LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from sklearn.model_selection import ShuffleSplit

#Implements polinomial regression
def PolynomialRegression(estimator ,degree, X_train, X_test, y_train, y_test, print_coef=False):
    #Pipeline
    model  = Pipeline(steps=[
                         ('poly', PolynomialFeatures(degree=degree)),
                         ('lin_reg', estimator)])

    #Fit the model
    model.fit(X_train, y_train)

    #Plot model
    fig = plt.figure(figsize=(7,6))
    plt.scatter(X_train, y_train)
    plt.scatter(X_test, y_test)
    print('R2_train =', model.score(X_train, y_train).round(5))
    print('R2_test =', model.score(X_test, y_test).round(5))
    print('RMSE_train =', np.sqrt(mean_squared_error(y_train ,model.predict(X_train))).round(5) )
    print('RMSE_test =', np.sqrt(mean_squared_error(y_test ,model.predict(X_test))).round(5) )
    if print_coef:
        print("W = \n" ,model['lin_reg'].coef_)
    x = np.linspace(0,1).reshape(-1,1)
    plt.plot(x, model.predict(x), 'r')
    plt.xlim(-0.01, 1)
    plt.ylim(-0.1, 11)
 


###Sigmoid function
def sigmoid(z):
    return 1/(1+np.exp(-z))

def SGD_LogisticRegression(X, y, learning_rate:float=0.001, epochs:int=100, batch_size:int=10):

    '''
    Implements mini-batch greadient descent to optmizer binary_cross_entropy loss function

    Args: 
        learning_rate:learning rate  (float)
        epochs: number of total epochs
        batch_size: number of instances in each mini-batch

    '''

    m = X.shape[1]
    W = np.random.random(size=m)
    total_epochs = epochs
    total_batches = X.shape[0]//batch_size

    for epoch in range(total_epochs):
        X, y = shuffle(X_train, y_train)

        for i in range(total_batches):

            x_batch = X[(batch_size*i):(batch_size*(i+1))] 
            y_batch = y[(batch_size*i):(batch_size*(i+1))]

            p = sigmoid(x_batch.dot(W))
            grad = x_batch.T.dot(p-y_batch)
            W = W - learning_rate*grad

    return W
